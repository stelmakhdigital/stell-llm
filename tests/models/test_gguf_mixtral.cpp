// ============================================================================
// Тест GGUF mixtral-класса (round 2, задача R9; ADR-09 «Уточнение»):
// конвертер/лоадер с НАТИВНЫМИ именами тензоров формата GGUF:
//   blk.{i}.ffn_gate_exps / ffn_up_exps / ffn_down_exps — стековные
//     экспертные тензоры (срезы по старшей размерности, source_slice);
//   blk.{i}.exp_probs_b — смещение роутера;
//   n_experts / n_experts_active — из GGUF-метаданных (llama.expert_count /
//     llama.expert_used_count).
//
// Roundtrip на синтетическом GGUF mixtral-профиля
// (fixtures/mini_mixtral.gguf, веса — LCG seed 0x00D3, golden/README.md):
//   1. описание модели, сгенерированное конвертером, побайтово равно
//      fixtures/mini_mixtral.json и парсится валидатором (schema 1.0);
//   2. веса читаются лоадером: срезы стековных тензоров побайтово сверены
//      с файлом; роутер/смещение — F32 как есть (dtype — атрибут группы);
//   3. логиты мини-интерпретатора ops-графа (rmsnorm → attention →
//      moe_gather_scatter: роутер+bias → softmax → top-2 → renorm →
//      SwiGLU-эксперты; final norm → head) совпадают с независимым
//      Python-референсом generate_golden.py (golden/logits_mini_mixtral.json)
//      в пределах tolerance_rel; greedy-токен совпадает.
// ============================================================================
#include "stell/models/convert.hpp"
#include "stell/models/registry.hpp"
#include "stell/models/weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "quant.hpp"
#include "tensor_file_formats.hpp"
#include "test_util.hpp"

using namespace stell::models;
using namespace stell_test;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int find_group(const ModelDescription& d, const std::string& name) {
    for (std::size_t i = 0; i < d.groups.size(); ++i)
        if (d.groups[i].name == name) return static_cast<int>(i);
    return -1;
}

// Payload группы в float (F32 — как есть; F16 — референс f16_bits_to_f32).
std::vector<float> group_floats(const stell::tiers::TensorGroup& g,
                                const std::vector<std::uint8_t>& buf) {
    std::size_t n = 1;
    for (auto dim : g.shape) n *= static_cast<std::size_t>(dim);
    std::vector<float> out(n);
    if (g.dtype == stell::tiers::DType::F32) {
        for (std::size_t i = 0; i < n; ++i)
            std::memcpy(&out[i], buf.data() + 4 * i, 4);
    } else if (g.dtype == stell::tiers::DType::F16) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint16_t h = static_cast<std::uint16_t>(buf[2 * i]) |
                                    (static_cast<std::uint16_t>(buf[2 * i + 1]) << 8);
            out[i] = f16_bits_to_f32(h);
        }
    } else {
        throw std::runtime_error("unexpected dtype for " + g.name);
    }
    return out;
}

// --- мини-интерпретатор ops-графа (float32; семантика golden/README.md) ------

using Mat = std::vector<std::vector<float>>;

Mat mat(const std::vector<float>& flat, std::size_t rows, std::size_t cols) {
    Mat m(rows);
    for (std::size_t r = 0; r < rows; ++r)
        m[r].assign(flat.begin() + static_cast<std::ptrdiff_t>(r * cols),
                    flat.begin() + static_cast<std::ptrdiff_t>((r + 1) * cols));
    return m;
}

std::vector<float> rmsnorm_vec(const std::vector<float>& x,
                               const std::vector<float>& w, float eps) {
    std::vector<float> out(x.size());
    float var = 0.0f;
    for (float v : x) var += v * v;
    var /= static_cast<float>(x.size());
    const float inv = 1.0f / std::sqrt(var + eps);
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] * inv * w[i];
    return out;
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

// Каузальная attention, GQA (голова h ← kv h//n_rep), без rope (ops-схема
// fixture не содержит позиционной операции — golden/README.md).
Mat attention_layer(const Mat& x, const Mat& y, const Mat& wq, const Mat& wk,
                    const Mat& wv, const Mat& wo, std::size_t n_heads,
                    std::size_t n_kv_heads, std::size_t d_head) {
    const std::size_t T = x.size(), d_model = wq[0].size();
    const std::size_t n_rep = n_heads / n_kv_heads;
    const std::size_t dq = n_heads * d_head, dkv = n_kv_heads * d_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    // q/k/v: проекции y (матрицы [out, in] — row-major, конвенция safetensors).
    Mat q(T, std::vector<float>(dq, 0.0f)), k(T, std::vector<float>(dkv, 0.0f)),
        v(T, std::vector<float>(dkv, 0.0f));
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t j = 0; j < dq; ++j)
            for (std::size_t dm = 0; dm < d_model; ++dm)
                q[t][j] += y[t][dm] * wq[j][dm];
        for (std::size_t j = 0; j < dkv; ++j) {
            for (std::size_t dm = 0; dm < d_model; ++dm) {
                k[t][j] += y[t][dm] * wk[j][dm];
                v[t][j] += y[t][dm] * wv[j][dm];
            }
        }
    }
    Mat out(T, std::vector<float>(d_model, 0.0f));
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < n_heads; ++h) {
            const std::size_t kvh = h / n_rep;
            std::vector<float> scores;
            scores.reserve(t + 1);
            for (std::size_t s = 0; s <= t; ++s) {
                float sc = 0.0f;
                for (std::size_t d = 0; d < d_head; ++d)
                    sc += q[t][h * d_head + d] * k[s][kvh * d_head + d];
                scores.push_back(sc * scale);
            }
            float maxs = scores[0];
            for (float sc : scores) maxs = std::max(maxs, sc);
            float denom = 0.0f;
            for (float sc : scores) denom += std::exp(sc - maxs);
            std::vector<float> attn(d_head, 0.0f);
            for (std::size_t d = 0; d < d_head; ++d)
                for (std::size_t s = 0; s <= t; ++s)
                    attn[d] += std::exp(scores[s] - maxs) / denom *
                               v[s][kvh * d_head + d];
            for (std::size_t dm = 0; dm < d_model; ++dm)
                for (std::size_t d = 0; d < d_head; ++d)
                    out[t][dm] += attn[d] * wo[dm][h * d_head + d];
        }
        for (std::size_t dm = 0; dm < d_model; ++dm) out[t][dm] += x[t][dm];
    }
    return out;
}

// softmax → top-k (ничья → меньший индекс) → renorm.
std::pair<std::vector<std::size_t>, std::vector<float>> route(
    const std::vector<float>& logits, std::size_t k) {
    const std::size_t E = logits.size();
    std::vector<std::size_t> idx(E);
    for (std::size_t i = 0; i < E; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::size_t a, std::size_t b) {
                         return logits[a] > logits[b];
                     });
    idx.resize(k);
    float maxl = logits[0];
    for (float v : logits) maxl = std::max(maxl, v);
    std::vector<float> denom(E);
    float sum = 0.0f;
    for (std::size_t e = 0; e < E; ++e) {
        denom[e] = std::exp(logits[e] - maxl);
        sum += denom[e];
    }
    std::vector<float> probs(k);
    float picked = 0.0f;
    for (std::size_t j = 0; j < k; ++j) {
        probs[j] = denom[idx[j]] / sum;
        picked += probs[j];
    }
    for (std::size_t j = 0; j < k; ++j) probs[j] /= picked;
    return {idx, probs};
}

// MoE-слой: y = pre-MoE нормированный вход; роутер+bias → softmax → top-2 →
// renorm → SwiGLU-эксперты; residual.
Mat moe_layer(const Mat& x, const Mat& y, const Mat& router,
              const std::vector<float>& bias, const std::vector<Mat>& gates,
              const std::vector<Mat>& ups, const std::vector<Mat>& downs,
              std::size_t n_active) {
    const std::size_t T = x.size(), d_ff = gates[0].size();
    Mat out(T, std::vector<float>(x[0].size(), 0.0f));
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t E = router.size();  // n_experts — строки роутера
        std::vector<float> logits(E);
        for (std::size_t e = 0; e < E; ++e) {
            float acc = 0.0f;
            for (std::size_t d = 0; d < y[t].size(); ++d)
                acc += y[t][d] * router[e][d];
            logits[e] = acc + bias[e];
        }
        auto [topk, probs] = route(logits, n_active);
        std::vector<float> acc(x[t].size(), 0.0f);
        for (std::size_t j = 0; j < n_active; ++j) {
            const std::size_t e = topk[j];
            const std::vector<float>& h2 = y[t];
            std::vector<float> gv(d_ff), uv(d_ff);
            for (std::size_t f = 0; f < d_ff; ++f) {
                float a = 0.0f, b = 0.0f;
                for (std::size_t d = 0; d < h2.size(); ++d) {
                    a += gates[e][f][d] * h2[d];
                    b += ups[e][f][d] * h2[d];
                }
                gv[f] = a;
                uv[f] = b;
            }
            for (std::size_t d = 0; d < h2.size(); ++d)
                for (std::size_t f = 0; f < d_ff; ++f)
                    acc[d] += probs[j] * silu(gv[f]) * uv[f] * downs[e][d][f];
        }
        for (std::size_t d = 0; d < acc.size(); ++d) out[t][d] = x[t][d] + acc[d];
    }
    return out;
}

}  // namespace

void body() {
    const std::string gguf_path = std::string(STELL_FIXTURES_DIR) + "/mini_mixtral.gguf";
    const std::string desc_path = std::string(STELL_FIXTURES_DIR) + "/mini_mixtral.json";
    const std::string golden_path =
        std::string(STELL_GOLDEN_FIXTURE) + "/logits_mini_mixtral.json";

    // 1) Конвертер: описание модели побайтово = committed fixture (roundtrip).
    const std::string json = gguf_to_description(gguf_path);
    const std::string desc_committed = read_file(desc_path);
    CHECK_MSG(json == desc_committed,
              "gguf_to_description(gguf) != committed mini_mixtral.json");
    const ModelDescription d = parse_model_description(json, "mini-mixtral");
    CHECK(d.is_moe);
    CHECK(d.name == "mini-mixtral-gguf");
    CHECK(d.moe && d.moe->n_experts == 4 && d.moe->n_experts_active == 2 &&
          d.moe->expert_d_ff == 16);
    CHECK(d.ops.size() == 3);
    CHECK(d.ops[2].op == "moe_gather_scatter");

    // Нативные имена + срезы: e{NNN}-группы — source_slice = индекс эксперта.
    const int gate0 = find_group(d, "l000.e001.gate");
    CHECK((gate0 >= 0) &&
          (d.group_source_patterns[gate0] == "blk.0.ffn_gate_exps.weight"));
    CHECK(d.group_source_slices[gate0] == 1);
    const int bias0 = find_group(d, "l000.mlp.router_bias");
    CHECK((bias0 >= 0) &&
          (d.group_source_patterns[bias0] == "blk.0.exp_probs_b.weight"));
    CHECK(d.group_source_slices[bias0] == -1);
    const int router0 = find_group(d, "l000.mlp.router");
    CHECK((router0 >= 0) &&
          (d.group_source_patterns[router0] == "blk.0.ffn_gate_inp.weight"));

    // 2) Лоадер: срезы стековных тензоров побайтово сверены с файлом.
    std::FILE* fw = std::fopen(gguf_path.c_str(), "rb");
    TensorFileMap tfm = parse_gguf_header(fw);
    const TensorInfo& stacked = tfm.by_name.at("blk.0.ffn_gate_exps.weight");
    {
        auto loader = make_loader(WeightsFormat::GGUF);
        auto src = loader->open(gguf_path, d);
        const auto& g = d.groups[gate0];
        std::vector<std::uint8_t> buf(g.slot_bytes, 0);
        src->read({g.id, g.role}, buf.data(), buf.size());
        const std::size_t n_slice = 16 * 8;
        std::vector<std::uint8_t> expect(n_slice * 2);
        read_exact(fw, stacked.offset + n_slice * 2 * 1, expect.data(),
                   expect.size());
        CHECK_MSG(std::memcmp(buf.data(), expect.data(), expect.size()) == 0,
                  "slice e001.gate != file bytes (source_slice)");
        for (std::size_t k = expect.size(); k < buf.size(); ++k)
            CHECK(buf[k] == 0);  // хвост слота
        // Роутер/смещение: F32 как есть (dtype — атрибут группы, без догадок).
        for (int gi : {router0, bias0}) {
            const auto& gr = d.groups[gi];
            std::vector<std::uint8_t> bufr(gr.slot_bytes, 0);
            src->read({gr.id, gr.role}, bufr.data(), bufr.size());
            const TensorInfo& ti =
                tfm.by_name.at(d.group_source_patterns[gi]);
            std::vector<std::uint8_t> expectr(ti.bytes);
            read_exact(fw, ti.offset, expectr.data(), expectr.size());
            CHECK_MSG(std::memcmp(bufr.data(), expectr.data(), expectr.size()) == 0,
                      "router/bias payload != file bytes");
        }
        src->close();

        // 3) Веса групп → float (для интерпретатора).
        src = loader->open(gguf_path, d);
        auto read_group = [&](const std::string& name) {
            const int gi = find_group(d, name);
            CHECK_MSG(gi >= 0, "group not found: " + name);
            const auto& gr = d.groups[gi];
            std::vector<std::uint8_t> bufr(gr.slot_bytes, 0);
            src->read({gr.id, gr.role}, bufr.data(), bufr.size());
            return group_floats(gr, bufr);
        };

        const std::size_t T = 3;  // prompt [1, 2, 3]
        const std::vector<int> prompt = {1, 2, 3};
        const std::size_t d_model = 8, n_heads = 2, n_kv_heads = 1, d_head = 4;
        const std::size_t n_experts = 4, n_active = 2, d_ff = 16;
        const float eps = 1e-5f;

        const Mat embed = mat(read_group("embed"), 16, d_model);
        const std::vector<float> norm_final = read_group("norm_final");
        const Mat head = mat(read_group("head"), 16, d_model);

        Mat x(T, std::vector<float>(d_model));
        for (std::size_t t = 0; t < T; ++t) x[t] = embed[prompt[t]];

        for (int layer = 0; layer < 2; ++layer) {
            const std::string ls = std::to_string(layer);
            const std::string lp = "l00" + ls + ".";
            const std::vector<float> w_an = read_group(lp + "attn.norm");
            Mat y(T);
            for (std::size_t t = 0; t < T; ++t)
                y[t] = rmsnorm_vec(x[t], w_an, eps);
            const Mat wq = mat(read_group(lp + "attn.wq"), n_heads * d_head, d_model);
            const Mat wk = mat(read_group(lp + "attn.wk"), n_kv_heads * d_head, d_model);
            const Mat wv = mat(read_group(lp + "attn.wv"), n_kv_heads * d_head, d_model);
            const Mat wo = mat(read_group(lp + "attn.wo"), d_model, d_model);
            x = attention_layer(x, y, wq, wk, wv, wo, n_heads, n_kv_heads, d_head);

            // pre-MoE норма (нативная раскладка: blk.ffn_norm) → h2.
            const std::vector<float> w_fn = read_group(lp + "ffn.norm");
            Mat h2(T);
            for (std::size_t t = 0; t < T; ++t)
                h2[t] = rmsnorm_vec(x[t], w_fn, eps);
            const Mat router =
                mat(read_group(lp + "mlp.router"), n_experts, d_model);
            const std::vector<float> bias = read_group(lp + "mlp.router_bias");
            std::vector<Mat> gates(n_experts), ups(n_experts), downs(n_experts);
            for (std::size_t e = 0; e < n_experts; ++e) {
                char pad[8];
                std::snprintf(pad, sizeof pad, "%03u", static_cast<unsigned>(e));
                const std::string ep = lp + "e" + pad + ".";
                gates[e] = mat(read_group(ep + "gate"), d_ff, d_model);
                ups[e] = mat(read_group(ep + "up"), d_ff, d_model);
                downs[e] = mat(read_group(ep + "down"), d_model, d_ff);
            }
            x = moe_layer(x, h2, router, bias, gates, ups, downs, n_active);
        }

        // final norm → head → логиты последней позиции (prefill, next-token).
        const std::vector<float> yf = rmsnorm_vec(x[T - 1], norm_final, eps);
        std::vector<float> logits(16);
        for (std::size_t v = 0; v < 16; ++v) {
            float acc = 0.0f;
            for (std::size_t dm = 0; dm < d_model; ++dm)
                acc += yf[dm] * head[v][dm];
            logits[v] = acc;
        }
        src->close();

        // 4) Golden-сверка с независимым Python-референсом.
        const JsonValue golden = json_parse(read_file(golden_path));
        CHECK(golden.get_string("model") == "mini-mixtral-gguf");
        const std::vector<JsonValue>& ref = golden.get_array("logits_ref").array_v;
        CHECK(ref.size() == 16);
        const double tol = golden.get_double("tolerance_rel");
        double max_rel = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const double r = ref[i].double_v;
            const double sc = std::max(std::fabs(r), 1.0);
            max_rel = std::max(max_rel,
                               std::fabs(static_cast<double>(logits[i]) - r) / sc);
        }
        CHECK_MSG(max_rel <= tol, "logits rel error vs Python reference (" +
                                      std::to_string(max_rel) + ")");
        int greedy = 0;
        for (std::size_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[greedy]) greedy = static_cast<int>(i);
        CHECK(golden.find("meta")->get_uint("greedy_token") ==
              static_cast<std::uint64_t>(greedy));

        std::printf("mini_mixtral: logits OK (max_rel=%.2e ≤ %.3e), greedy=%d\n",
                    max_rel, tol, greedy);
    }
    std::fclose(fw);
}

TEST_MAIN("test_models_gguf_mixtral")

// [R1] round 6 (planner [Q5], ADR-09 «Уточнение round 6» — оп ffn.shared.gate):
// скалярный сигмоид-гейт shared-эксперта — группа l%03zu.ffn.shared.gate_scalar
// (d_model→1, тензор mlp.shared_expert_gate Qwen2Moe-класса):
//   shared_contrib = sigmoid(dot(w_gate, hidden)) × shared_ffn_out,
// где hidden — вход MoE-контура (тот же, что питает router и shared-FFN);
// порядок суммирования фиксирован: routed_sum накапливается первым,
// shared_contrib добавляется последним (побитовый референс). Отсутствие
// группы — прежнее поведение round 5; частичный набор (гейт без ffn.shared,
// гейт на dense-слое, форма ≠ [1, d_model]) — честный SchemaMismatch.
//
// ИМЯ ГРУППЫ: ffn.shared.gate_scalar (не ffn.shared.gate!) — мастер round 5
// уже связывает ffn.shared.gate со SwiGLU gate_proj shared-эксперта
// ([d_ff_shared, d_model], hf_description.cpp + engine/ops_shared); реальный
// Qwen2Moe содержит ОБА тензора (gate_proj и shared_expert_gate) — обе группы
// плана обязаны сосуществовать с разными именами. Допущение зафиксировано в
// отчёте раунда (docs/ недоступны потоку core для правок).
//
// РЕАЛЬНЫЙ tiny-qwen2moe недоступен локально (чекпоинт вне репозитория,
// постановка [R1]) — тест на синтетической MoE-фикстуре. Референс плана —
// НЕЗАВИСИМАЯ референс-реализация forward одного MoE-слоя (embedding →
// rmsnorm → attention → shared-FFN × sigmoid-гейт → [+0.0 routed] →
// norm_final → head), повторяющая арифметику контракта побитово. На пути
// БЕЗ гейта референс заякорен против уже проверенного round 5 поведения
// (engine/ops_shared: ffn.shared == dense-ffn референсу плана побитово) —
// совпадение референса с интерпретатором на N-фикстуре доказывает faithful-
// копию семантики, после чего совпадение на гейт-фикстурах изолирует именно
// семантику гейта.
//
// Обнуление gate-веса: dot ≡ +0.0 → sigmoid(+0.0) = 0.5 ТОЧНО (независимо от
// hidden) — предсказуемое изменение логитов, проверяется референсом побитово;
// не молчаливое игнорирование: логиты гейт-фикстур отличаются от безгейтовой
// N и друг от друга. routed-вклад обнуляется честно: веса ВСЕХ экспертов
// нулевые (matvec нулевой матрицы ≡ +0.0, silu(±0) ≡ +0.0, scatter_add(w ≥ 0,
// +0.0) ≡ +0.0) — MoE-FFN слоя = shared-вклад независимо от router.
#include "test_util.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/kernels/attention.hpp"

using namespace stell::engine;
using namespace stell_core_test;

namespace {

constexpr std::uint32_t kVocab = 16;
constexpr std::uint32_t kDM = 8;          // d_model
constexpr std::uint32_t kNHeads = 2;
constexpr std::uint32_t kDHead = 4;        // n_heads*d_head == d_model
constexpr std::uint32_t kNKV = 1;
constexpr std::uint32_t kDKV = 4;
constexpr std::uint32_t kDff = 16;
constexpr std::uint32_t kExperts = 4;

// --- фикстура: однослойная MoE (слои dense/MoE-миксы — engine/ops_mixed) ----

enum class GateMode { None, Weighted, Zero };

std::vector<PlainGroup> groups_for(bool with_shared, GateMode gate) {
    std::vector<PlainGroup> out;
    std::uint32_t id = 0;
    const auto add = [&](PlainGroup g) { g.id = id++; out.push_back(std::move(g)); };

    add(make_group(id, "embed", "F16", {kVocab, kDM}, "embedding.weight"));
    add(make_group(id, "norm_final", "F32", {kDM}, "model.norm.weight"));
    add(make_group(id, "head", "F16", {kVocab, kDM}, "head.weight"));
    const std::string p = "l000.";
    add(make_group(id, p + "attn.norm", "F32", {kDM},
                   "layers.0.input_layernorm.weight"));
    add(make_group(id, p + "attn.wq", "F16", {kDM, kDM},
                   "layers.0.self_attn.q_proj.weight"));
    add(make_group(id, p + "attn.wk", "F16", {kNKV * kDKV, kDM},
                   "layers.0.self_attn.k_proj.weight"));
    add(make_group(id, p + "attn.wv", "F16", {kNKV * kDKV, kDM},
                   "layers.0.self_attn.v_proj.weight"));
    add(make_group(id, p + "attn.wo", "F16", {kDM, kDM},
                   "layers.0.self_attn.o_proj.weight"));
    add(make_group(id, p + "kv", "F16", {kNKV * kDKV, kDM},
                   "layers.0.kv_cache", /*kv=*/true));
    add(make_group(id, p + "mlp.router", "F32",
                   {std::int64_t(kExperts), kDM},
                   "layers.0.mlp.router.weight"));
    for (std::uint32_t e = 0; e < kExperts; ++e) {
        char eb[32];
        std::snprintf(eb, sizeof eb, "e%03u", e);
        const std::string base =
            "layers.0.mlp.experts." + std::to_string(e);
        add(make_group(id, p + eb + ".gate", "F16", {kDff, kDM},
                       base + ".gate_proj.weight"));
        add(make_group(id, p + eb + ".up", "F16", {kDff, kDM},
                       base + ".up_proj.weight"));
        add(make_group(id, p + eb + ".down", "F16", {kDM, kDff},
                       base + ".down_proj.weight"));
    }
    // ffn.shared: dense-FFN семантика round 5 ({gate,up,down} — SwiGLU);
    if (with_shared) {
        const std::string base = "layers.0.mlp.shared_expert";
        add(make_group(id, p + "ffn.shared.gate", "F16", {kDff, kDM},
                       base + ".gate_proj.weight"));
        add(make_group(id, p + "ffn.shared.up", "F16", {kDff, kDM},
                       base + ".up_proj.weight"));
        add(make_group(id, p + "ffn.shared.down", "F16", {kDM, kDff},
                       base + ".down_proj.weight"));
    }
    // [R1] round 6: скалярный сигмоид-гейт (d_model→1) — группа
    // ffn.shared.gate_scalar; F32 (неквантованный малый тензор — смешанный
    // план, ADR-09 round 2).
    if (gate != GateMode::None) {
        add(make_group(id, p + "ffn.shared.gate_scalar", "F32", {1, kDM},
                       "layers.0.mlp.shared_expert_gate.weight"));
    }
    return out;
}

PlainSpec make_spec(bool with_shared, GateMode gate, const char* name) {
    PlainSpec s{};
    s.name = name;
    s.is_moe = true;
    s.dtype_weights = "F16";
    s.vocab_size = kVocab;
    s.d_model = kDM;
    s.n_layers = 1;
    s.n_heads = kNHeads;
    s.d_head = kDHead;
    s.n_kv_heads = kNKV;
    s.d_kv = kDKV;
    s.n_experts = kExperts;
    s.n_experts_active = 2;
    s.expert_d_ff = kDff;
    s.router = "softmax-topk";
    s.ops = base_ops(/*with_ffn=*/false, /*with_moe=*/true);
    s.groups = groups_for(with_shared, gate);
    return s;
}

// F32 → F16-половинки значений fill_weights (значения k/16 — точны в F16,
// конверт побитово обратим для фикстуры; материал engine == исходные float).
inline std::vector<std::uint16_t> halves_of(const std::vector<float>& vals) {
    std::vector<std::uint16_t> out(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) out[i] = f32_to_f16(vals[i]);
    return out;
}

// Тензоры фикстуры; expert-группы — НУЛЕВОЙ payload (честное обнуление
// routed-вклада); gate_zero — нулевой payload скалярного гейта.
std::vector<STensor> tensors_for(const PlainSpec& spec, bool zero_gate) {
    std::vector<STensor> out;
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block) continue;
        STensor t;
        t.name = g.source_pattern;
        t.dtype = g.dtype == PlainDTypeF32 ? "F32" : "F16";
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto d : g.shape) n *= static_cast<std::size_t>(d);
        const bool expert = g.name.size() > 6 && g.name[4] == '.' &&
                            g.name[5] == 'e';
        const bool gate_zero = zero_gate &&
                               g.name.find("ffn.shared.gate_scalar") !=
                                   std::string::npos;
        if (expert || gate_zero) {
            t.data.assign(n * (t.dtype == "F32" ? 4 : 2), 0);
        } else {
            const std::vector<float> vals = fill_weights(g.source_pattern, n);
            t.data = t.dtype == "F32" ? f32_payload(vals)
                                      : f16_payload(halves_of(vals));
        }
        out.push_back(std::move(t));
    }
    return out;
}

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load(const PlainSpec& spec, const char* path, bool zero_gate) {
    Loaded l{};
    l.spec = spec;
    write_safetensors(path, tensors_for(spec, zero_gate));
    l.weights = load_plain_weights(path, spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(spec);
    return l;
}

// --- НЕЗАВИСИМЫЙ референс плана (побитовая арифметика контракта) -------------

// Референс-веса: значения fill_weights(source_pattern) — материал engine
// побитово им идентичен (F32-группы — memcpy; F16-конверт точен для значений
// фикстуры k/16, k ∈ [-16, 16]).
struct RefW {
    std::vector<float> embed;        // [kVocab*kDM]
    std::vector<float> attn_norm;    // [kDM]
    std::vector<float> wq, wk, wv, wo;
    std::vector<float> norm_final;   // [kDM]
    std::vector<float> head;         // [kVocab*kDM]
    std::vector<float> sh_g, sh_u, sh_d;   // shared-FFN (SwiGLU gate/up/down)
    std::vector<float> gate;         // скалярный гейт [kDM]; пусто — без гейта
};

RefW ref_weights(bool with_shared, GateMode gate) {
    RefW w{};
    const auto fw = [](const char* pattern, std::size_t n) {
        return fill_weights(pattern, n);
    };
    w.embed = fw("embedding.weight", kVocab * kDM);
    w.attn_norm = fw("layers.0.input_layernorm.weight", kDM);
    w.wq = fw("layers.0.self_attn.q_proj.weight", kDM * kDM);
    w.wk = fw("layers.0.self_attn.k_proj.weight", kNKV * kDKV * kDM);
    w.wv = fw("layers.0.self_attn.v_proj.weight", kNKV * kDKV * kDM);
    w.wo = fw("layers.0.self_attn.o_proj.weight", kDM * kDM);
    w.norm_final = fw("model.norm.weight", kDM);
    w.head = fw("head.weight", kVocab * kDM);
    if (with_shared) {
        w.sh_g = fw("layers.0.mlp.shared_expert.gate_proj.weight", kDff * kDM);
        w.sh_u = fw("layers.0.mlp.shared_expert.up_proj.weight", kDff * kDM);
        w.sh_d = fw("layers.0.mlp.shared_expert.down_proj.weight", kDM * kDff);
    }
    if (gate == GateMode::Weighted)
        w.gate = fw("layers.0.mlp.shared_expert_gate.weight", kDM);
    if (gate == GateMode::Zero)
        w.gate.assign(kDM, 0.0f);        // обнуленный гейт: dot ≡ +0.0
    return w;
}

// matvec контракта (ops_runner::matvec): y[r] = Σ_c m[r*cols + c] * x[c],
// порядок c возрастающий, накопление во float.
std::vector<float> ref_matvec(const std::vector<float>& m, std::size_t rows,
                              std::size_t cols, const std::vector<float>& x) {
    std::vector<float> y(rows, 0.0f);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            y[r] += m[r * cols + c] * x[c];
    return y;
}

float ref_silu(float x) { return x / (1.0f + std::exp(-x)); }

struct RefState {
    std::vector<float> k, v;         // token-major [pos][kDKV]
    std::uint32_t pos = 0;
};

std::vector<float> ref_forward_token(RefState& st, std::int32_t token,
                                     const RefW& w) {
    const float eps = 1e-5f;         // eps base_ops (rmsnorm)
    std::vector<float> hidden(kDM, 0.0f);
    for (std::size_t i = 0; i < kDM; ++i)
        hidden[i] = w.embed[static_cast<std::size_t>(token) * kDM + i];

    // rmsnorm: double-аккумулятор + inv (арифметика op_rmsnorm).
    double ms = 0.0;
    for (float x : hidden) ms += double(x) * double(x);
    const float inv = 1.0f / std::sqrt(static_cast<float>(ms / kDM) + eps);
    std::vector<float> norm_in(kDM);
    for (std::size_t i = 0; i < kDM; ++i)
        norm_in[i] = w.attn_norm[i] * hidden[i] * inv;

    const std::uint32_t T = st.pos + 1;
    const std::vector<float> q = ref_matvec(w.wq, kDM, kDM, norm_in);
    const std::vector<float> k = ref_matvec(w.wk, kNKV * kDKV, kDM, norm_in);
    const std::vector<float> v = ref_matvec(w.wv, kNKV * kDKV, kDM, norm_in);

    // Раскладка kernels [n_kv_heads][T][d_kv] поверх token-major кеша
    // (арифметика op_attention; n_kv_heads = 1).
    std::vector<float> kk(static_cast<std::size_t>(kNKV) * T * kDKV, 0.0f);
    std::vector<float> vk(kk.size(), 0.0f);
    for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t d = 0; d < kDKV; ++d) {
            const std::size_t dst = static_cast<std::size_t>(t) * kDKV + d;
            if (t == st.pos) {
                kk[dst] = k[d];
                vk[dst] = v[d];
            } else {
                kk[dst] = st.k[static_cast<std::size_t>(t) * kDKV + d];
                vk[dst] = st.v[static_cast<std::size_t>(t) * kDKV + d];
            }
        }
    std::vector<float> attn_out(static_cast<std::size_t>(kNHeads) * kDHead,
                                0.0f);
    stell::kernels::attention_decode_f32(q.data(), kk.data(), vk.data(),
                                         kNHeads, kNKV, T, kDHead,
                                         attn_out.data());
    // joined == attn_out (n_heads*d_head == d_model); проекция wo + residual.
    const std::vector<float> proj =
        ref_matvec(w.wo, kDM, kNHeads * kDHead, attn_out);
    for (std::size_t i = 0; i < kDM; ++i) hidden[i] += proj[i];

    // Запись k/v токена в token-major кеш (инкремент позиции — на выходе).
    st.k.insert(st.k.end(), k.begin(), k.end());
    st.v.insert(st.v.end(), v.begin(), v.end());

    // --- MoE-контур: routed ≡ +0.0 (эксперты фикстуры обнулены), shared ----
    // Shared-FFN и скалярный гейт берут hidden ДО residual-добавления
    // routed-суммы (порядок op_moe: shared_out и гейт — до `hidden += y`).
    std::vector<float> shared(kDM, 0.0f);
    if (!w.sh_u.empty()) {
        std::vector<float> g = ref_matvec(w.sh_g, kDff, kDM, hidden);
        for (float& f : g) f = ref_silu(f);
        const std::vector<float> u = ref_matvec(w.sh_u, kDff, kDM, hidden);
        std::vector<float> m_vec(kDff);
        for (std::size_t i = 0; i < kDff; ++i) m_vec[i] = g[i] * u[i];
        shared = ref_matvec(w.sh_d, kDM, kDff, m_vec);
        if (!w.gate.empty()) {
            // [R1] round 6: sigmoid(dot(w_gate, hidden)) × shared_ffn_out.
            const std::vector<float> g1 = ref_matvec(w.gate, 1, kDM, hidden);
            const float gs = 1.0f / (1.0f + std::exp(-g1[0]));
            for (std::size_t i = 0; i < shared.size(); ++i) shared[i] *= gs;
        }
    }
    // Фиксированный порядок суммирования: routed_sum (≡ +0.0; повторяем
    // тождественное добавление — −0.0 → +0.0, как op_moe), затем shared_contrib.
    for (std::size_t i = 0; i < kDM; ++i) hidden[i] += 0.0f;
    for (std::size_t i = 0; i < kDM; ++i) hidden[i] += shared[i];

    // norm_final + head (арифметика forward_token).
    double ms2 = 0.0;
    for (float x : hidden) ms2 += double(x) * double(x);
    const float inv2 = 1.0f / std::sqrt(static_cast<float>(ms2 / kDM) + eps);
    for (std::size_t i = 0; i < kDM; ++i)
        hidden[i] = w.norm_final[i] * hidden[i] * inv2;
    std::vector<float> logits(kVocab, 0.0f);
    for (std::uint32_t t = 0; t < kVocab; ++t) {
        float acc = 0.0f;
        for (std::size_t i = 0; i < kDM; ++i)
            acc += w.head[t * kDM + i] * hidden[i];
        logits[t] = acc;
    }
    ++st.pos;
    return logits;
}

std::vector<std::vector<float>> ref_prompt(const RefW& w,
                                           const std::vector<std::int32_t>& p) {
    RefState st{};
    std::vector<std::vector<float>> rows;
    for (std::int32_t tok : p) rows.push_back(ref_forward_token(st, tok, w));
    return rows;
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

bool throws_schema_mismatch(const PlainSpec& spec, const char* path,
                            const char* tag) {
    try {
        const Loaded l = load(spec, path, /*zero_gate=*/false);
        OpsInterpreter interp(l.spec, l.plan, l.weights);
        (void)interp.run_prompt_logits({1, 2, 3});
    } catch (const std::invalid_argument& e) {
        if (!contains(e.what(), "SchemaMismatch")) {
            std::printf("[FAIL] %s: отказ без пометки SchemaMismatch: %s\n",
                        tag, e.what());
            return false;
        }
        std::printf("schema-mismatch [%s]: %s\n", tag, e.what());
        return true;
    } catch (const std::exception& e) {
        std::printf("[FAIL] %s: неожиданный тип отказа: %s\n", tag, e.what());
        return false;
    }
    std::printf("[FAIL] %s: отказ НЕ произошёл (тихий неверный план)\n", tag);
    return false;
}

void check_bitwise(const std::vector<std::vector<float>>& a,
                   const std::vector<std::vector<float>>& b, const char* what) {
    STELL_CHECK(a.size() == b.size());
    for (std::size_t p = 0; p < a.size(); ++p) {
        STELL_CHECK(a[p].size() == b[p].size());
        for (std::size_t i = 0; i < a[p].size(); ++i)
            if (a[p][i] != b[p][i]) {
                std::printf("[FAIL] %s: позиция %zu, индекс %zu: план=%.9g "
                            "референс=%.9g\n", what, p, i,
                            static_cast<double>(a[p][i]),
                            static_cast<double>(b[p][i]));
                STELL_CHECK(false);
            }
    }
}

bool differs(const std::vector<float>& a, const std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i] != b[i]) return true;
    return false;
}

} // namespace

void run() {
    const std::vector<std::int32_t> prompt = {1, 2, 3, 5, 7, 9, 11, 13};

    // Фикстуры: Z (гейт с весами), Z0 (обнуленный гейт → scalar ≡ 0.5),
    // N (без гейта — прежнее поведение round 5). Эксперты обнулены.
    const Loaded lz = load(make_spec(true, GateMode::Weighted,
                                     "mini-gate-test"),
                           "/tmp/stell_core_test_gate.safetensors",
                           /*zero_gate=*/false);
    const Loaded lz0 = load(make_spec(true, GateMode::Zero,
                                      "mini-gate-zero-test"),
                            "/tmp/stell_core_test_gate_zero.safetensors",
                            /*zero_gate=*/true);
    const Loaded ln = load(make_spec(true, GateMode::None,
                                     "mini-gate-none-test"),
                           "/tmp/stell_core_test_gate_none.safetensors",
                           /*zero_gate=*/false);

    OpsInterpreter iz(lz.spec, lz.plan, lz.weights);
    OpsInterpreter iz0(lz0.spec, lz0.plan, lz0.weights);
    OpsInterpreter in(ln.spec, ln.plan, ln.weights);

    const auto all_z = iz.run_prompt_logits_all(prompt);
    const auto all_z0 = iz0.run_prompt_logits_all(prompt);
    const auto all_n = in.run_prompt_logits_all(prompt);
    STELL_CHECK(all_z.size() == prompt.size());
    for (float v : all_z.back()) STELL_CHECK(std::isfinite(v));

    // --- N (без гейта) == референсу, ПОБИТОВО — ЯКОРЬ: совпадение референса
    // с интерпретатором на уже проверенной round 5 семантике (engine/ops_shared:
    // ffn.shared == dense-ffn референсу плана) доказывает faithful-копию
    // арифметики контракта; последующие совпадения изолируют семантику гейта.
    check_bitwise(all_n, ref_prompt(ref_weights(true, GateMode::None), prompt),
                  "N: логиты != референсу (без гейта)");
    // --- Z (shared+gate) == референсу плана, ПОБИТОВО (все позиции) ---------
    check_bitwise(all_z, ref_prompt(ref_weights(true, GateMode::Weighted),
                                    prompt),
                  "Z: логиты != референсу (гейт с весами)");
    // --- Z0 (обнуленный гейт) == референсу (scalar ≡ 0.5), ПОБИТОВО ---------
    check_bitwise(all_z0, ref_prompt(ref_weights(true, GateMode::Zero), prompt),
                  "Z0: логиты != референсу (обнуленный гейт)");

    // last-путь == последней позиции полного прогона (инвариант round 5).
    const std::vector<float> last_z = iz.run_prompt_logits(prompt);
    check_bitwise({last_z}, {all_z.back()}, "Z: last != all.back()");

    // --- вклад гейта реален: НЕ молчаливое игнорирование --------------------
    // Z != N (гейт с весами меняет логиты), Z0 != N (обнуленный гейт —
    // предсказуемое ×0.5 — тоже меняет), Z != Z0 (разные гейты — разные
    // логиты). Предсказуемость изменения — побитовым совпадением с
    // референсом выше (sigmoid(+0.0) = 0.5 точно).
    STELL_CHECK_MSG(differs(all_z.back(), all_n.back()),
                    "гейт не проявился в логитах (Z == N)");
    STELL_CHECK_MSG(differs(all_z0.back(), all_n.back()),
                    "обнуленный гейт не проявился (Z0 == N)");
    STELL_CHECK_MSG(differs(all_z.back(), all_z0.back()),
                    "Z == Z0 (вклад гейта не зависит от весов)");

    // --- детерминизм повторного прогона --------------------------------------
    OpsInterpreter iz2(lz.spec, lz.plan, lz.weights);
    const auto again = iz2.run_prompt_logits_all(prompt);
    check_bitwise(all_z, again, "Z: повторный прогон");

    // --- частичный набор → честный SchemaMismatch ----------------------------
    const char* kPath = "/tmp/stell_core_test_gate_bad.safetensors";
    std::size_t ok = 0;
    {
        // гейт БЕЗ ffn.shared (MoE-слой): частичный набор, ADR-09 round 6
        PlainSpec bad = make_spec(false, GateMode::Weighted,
                                  "mini-gate-bad-noshared");
        ok += throws_schema_mismatch(bad, kPath, "gate-без-ffn.shared") ? 1 : 0;
    }
    {
        // гейт на dense-слое (без mlp.router): мёртвая группа
        PlainSpec dense = make_spec(false, GateMode::None,
                                    "mini-gate-bad-dense");
        dense.is_moe = false;
        dense.n_experts = 0;
        dense.n_experts_active = 0;
        dense.ops = base_ops(/*with_ffn=*/true, /*with_moe=*/false);
        std::vector<PlainGroup> dg;
        std::uint32_t id = 0;
        const auto add = [&](PlainGroup g) { g.id = id++; dg.push_back(std::move(g)); };
        add(make_group(id, "embed", "F16", {kVocab, kDM}, "embedding.weight"));
        add(make_group(id, "norm_final", "F32", {kDM}, "model.norm.weight"));
        add(make_group(id, "head", "F16", {kVocab, kDM}, "head.weight"));
        add(make_group(id, "l000.attn.norm", "F32", {kDM},
                       "layers.0.input_layernorm.weight"));
        add(make_group(id, "l000.attn.wq", "F16", {kDM, kDM},
                       "layers.0.self_attn.q_proj.weight"));
        add(make_group(id, "l000.attn.wk", "F16", {kNKV * kDKV, kDM},
                       "layers.0.self_attn.k_proj.weight"));
        add(make_group(id, "l000.attn.wv", "F16", {kNKV * kDKV, kDM},
                       "layers.0.self_attn.v_proj.weight"));
        add(make_group(id, "l000.attn.wo", "F16", {kDM, kDM},
                       "layers.0.self_attn.o_proj.weight"));
        add(make_group(id, "l000.kv", "F16", {kNKV * kDKV, kDM},
                       "layers.0.kv_cache", true));
        add(make_group(id, "l000.ffn.gate", "F16", {kDff, kDM},
                       "layers.0.mlp.gate_proj.weight"));
        add(make_group(id, "l000.ffn.up", "F16", {kDff, kDM},
                       "layers.0.mlp.up_proj.weight"));
        add(make_group(id, "l000.ffn.down", "F16", {kDM, kDff},
                       "layers.0.mlp.down_proj.weight"));
        add(make_group(id, "l000.ffn.shared.gate_scalar", "F32", {1, kDM},
                       "layers.0.mlp.shared_expert_gate.weight"));
        dense.groups = dg;
        ok += throws_schema_mismatch(dense, kPath, "gate-на-dense-слое") ? 1 : 0;
    }
    {
        // гейт не [1, d_model]: {2, 8} — не скалярный сигмоид-гейт
        PlainSpec bad = make_spec(true, GateMode::Weighted,
                                  "mini-gate-bad-shape");
        for (PlainGroup& g : bad.groups)
            if (g.name == "l000.ffn.shared.gate_scalar") g.shape = {2, kDM};
        ok += throws_schema_mismatch(bad, kPath, "gate-форма-[2,d_model]") ? 1 : 0;
    }
    {
        // гейт {8, 1} — матрица d_model→1 в транспонированной форме: отказ
        PlainSpec bad = make_spec(true, GateMode::Weighted,
                                  "mini-gate-bad-shape2");
        for (PlainGroup& g : bad.groups)
            if (g.name == "l000.ffn.shared.gate_scalar") g.shape = {kDM, 1};
        ok += throws_schema_mismatch(bad, kPath, "gate-форма-[d_model,1]") ? 1 : 0;
    }
    STELL_CHECK(ok == 4);

    std::printf("ops-gate: MoE + ffn.shared.gate_scalar (sigmoid(dot(w,h)) × "
                "shared_ffn_out, порядок routed→shared) == референсу плана "
                "побитово (все позиции); обнуление гейта — предсказуемое "
                "изменение (scalar ≡ 0.5), не молчаливое игнорирование; без "
                "гейта — прежнее поведение round 5; частичный набор — "
                "SchemaMismatch (4 кейса)\n");
}

STELL_TEST_MAIN(run)

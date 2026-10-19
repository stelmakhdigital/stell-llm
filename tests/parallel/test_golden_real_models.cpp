// tests/parallel: golden-инвариант TP2/TP4/EP2 == TP1 на РЕАЛЬНЫХ фикстурах
// ADR-09 (tests/models/fixtures: mini_dense, mini_moe) — задача раунда 2.
//
// Составляющие:
//   - stell::models: реестр + лоадеры (safetensors для F16/F32 групп,
//     NVFP4W4A16 для квантованных) над синтетическим safetensors-файлом;
//   - stell::kernels: CPU-референсы attention (sm70_fp32_qk_pv) и MoE
//     scatter_add;
//   - мини-интерпретатор ops из ModelDescription (rmsnorm, attention, ffn,
//     moe_gather_scatter): каталога ops engine ещё нет (core-2), интерпретатор
//     написан в тесте; семантика зафиксирована допущениями A1–A7 ниже.
//
// Инвариант: логиты (по всем T токенам промпта) при разбиении голов (TP2),
// колонок FFN (TP4) и экспертов (EP2) совпадают с эталоном ПОБИТОВО: блоки
// (головы / колонки FFN / эксперты) аккумулируются ПОСЛЕДОВАТЕЛЬНО в порядке
// блоков; ранковый порядок allreduce (ADR-10 §2) воспроизводит тот же
// порядок, внутренние матвецы — один и тот же код с одними битами входов.
//
// Эталоны: (а) блоковый эталон сценария — одноранковое исполнение с той же
// блоковой структурой (побитовое равенство с распределённым прогоном);
// (б) «плоский» TP1 (все матвецы целиком) — сверка с блоковым эталоном по
// допуску (порядок суммирования отличается — фиксируем величину расхождения).
//
// Топологии (world == размер редуцируемой группы: эмулятор редуцирует над
// всем world; per-stage sub-communicators — открытый вопрос контракта):
//   mini_dense TP2: головы (world=2); n_kv_heads=1 → KV реплицируется (A4);
//   mini_dense TP4: колонки FFN (world=4); головы полные — n_heads=2 не
//     делится на 4 (D1 в отчёте);
//   mini_moe  TP2: головы (world=2), эксперты реплицированы (A6);
//   mini_moe  EP2: эксперты (world=2), два прохода expert_alltoall.
//
// Допущения (TP-инвариант от них не зависит; зафиксированы в отчёте):
//  A1 silu_mul (одиночная up-матрица фикстуры): y_i = silu(u_i)·u_i;
//  A2 router [n_experts]: logit[e] = router_w[e]·Σ_d h2[t][d] (в фикстуре нет
//     проекции [n_experts,d_model]);
//  A3 slot_id в TokenMeta кодирует (исходный ранок, t, j, позиция скаляра) —
//     обратная маршрутизация (в engine строится по слотам сессий, ADR-08;
//     parallel семантически нейтрален);
//  A4 n_kv_heads % tp != 0 → KV реплицируется (строгая проверка ADR-10 §3 —
//     для случая, когда KV делится);
//  A5 softmax-topk: softmax по всем экспертам, веса top-2 как есть (без
//     перенормировки); выбор top-2 детерминирован (tie → меньший индекс,
//     обход e ↑ со строгим сравнением);
//  A6 EP1 = вычисление всех экспертов локально на каждом ранке (репликация);
//  A7 нормы слоёв: одна норма на слой (attn.norm) перед attention; вход
//     FFN/MoE = пост-attention residual без второй нормы (в фикстурах нет
//     mlp.norm; ops-список mini_dense: rmsnorm, attention, ffn; mini_moe:
//     rmsnorm, attention, moe_gather_scatter);
//  A8 mini_moe не содержит head-группу — выходные логиты = embed·nf
//     (tied embeddings);
//  A3b expert_alltoall: токен = token_elems ЭЛЕМЕНТОВ dtype (уточнение ADR-10
//     round 2): EP-сценарий гонит ЦЕЛЫЕ векторы d_model (token_elems = dm),
//     мета — одна на токен (slot_id кодирует пару источник,(t,j));
//     скалярная семантика token_elems=1 покрыта юнит-тестами эмулятора.
#include "test_harness.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/emulated.hpp"
#include "stell/parallel/plan.hpp"

#include "stell/kernels/attention.hpp"
#include "stell/kernels/moe_gather_scatter.hpp"

#include "stell/models/registry.hpp"
#include "stell/models/weights.hpp"

#include "quant.hpp"      // dequant_nvfp4 — раскладка лоадера models (блок 32)
#include "synth_writers.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

using namespace stell::parallel;
using stell::kernels::attention_forward_f32;
using stell::kernels::moe_scatter_add;
using stell::models::ModelDescription;
using stell::models::Registry;
using stell_test::SynthTensor;
using stell_test::write_safetensors;

namespace {

// --- фикстуры и веса ---------------------------------------------------------

#ifndef STELL_FIXTURES_DIR
#define STELL_FIXTURES_DIR "../tests/models/fixtures"
#endif

constexpr int T = 4;                        // промпт (full attention, без маски)
const int kPrompt[T] = {1, 3, 5, 7};
constexpr int kKTop = 2;                    // n_experts_active mini_moe
constexpr float kEps = 1e-5f;

// Детерминированные dyadic-веса (0.25·k) — точны в F16/F32; NVFP4-квантование
// детерминировано.
float wval(int a, int b) {
    return 0.25f * static_cast<float>(((a * 31 + b * 17 + 7) % 13) - 6);
}

// Тензоры source_pattern по шаблонам описания (смешанная система счисления
// 1:1 с description.cpp::expand_groups: последний ключ cardinality быстрее).
std::vector<SynthTensor> make_tensors(const ModelDescription& d) {
    std::vector<SynthTensor> out;
    for (const auto& tpl : d.group_templates) {
        std::size_t total = 1;
        for (const auto& [key, n] : tpl.cardinality) total *= n;
        for (std::size_t iter = 0; iter < total; ++iter) {
            std::vector<std::pair<std::string, std::uint32_t>> inst;
            std::size_t rem = iter;
            for (std::size_t k = tpl.cardinality.size(); k-- > 0;) {
                inst.emplace_back(tpl.cardinality[k].first,
                                  static_cast<std::uint32_t>(rem % tpl.cardinality[k].second));
                rem /= tpl.cardinality[k].second;
            }
            std::string name = tpl.source_pattern;
            for (const auto& [key, v] : inst) {
                const std::string token = "{" + key + "}";
                const std::string::size_type pos = name.find(token);
                if (pos != std::string::npos) name.replace(pos, token.size(), std::to_string(v));
            }
            SynthTensor t;
            t.name = name;
            t.shape = tpl.shape;
            const bool quant = tpl.dtype == "NVFP4" || tpl.dtype == "W4A16";
            t.dtype = quant ? "F16" : tpl.dtype;   // источник квантования — F16
            std::size_t n = 1;
            for (auto dim : tpl.shape) n *= static_cast<std::size_t>(dim);
            if (t.dtype == "F32") {
                std::vector<float> v(n);
                for (std::size_t k = 0; k < n; ++k) v[k] = wval(int(k), int(k % 8));
                t.data = stell_test::f32_payload(v);
            } else {
                std::vector<std::uint16_t> v(n);
                for (std::size_t k = 0; k < n; ++k)
                    v[k] = float_to_half(wval(int(k / t.shape.back() + 1),
                                              int(k % t.shape.back())));
                t.data = stell_test::f16_payload(v);
            }
            out.push_back(std::move(t));
        }
    }
    return out;
}

// Веса: имя расширенной группы → float-буфер (конвертация по dtype группы).
struct Weights {
    std::map<std::string, std::vector<float>> by_name;
    bool has(const std::string& name) const { return by_name.count(name) != 0; }
    const std::vector<float>& get(const std::string& name) const {
        auto it = by_name.find(name);
        if (it == by_name.end())
            throw std::runtime_error("нет группы '" + name + "' в весах");
        return it->second;
    }
};

Weights load_weights(const ModelDescription& d, const std::string& path) {
    using stell::models::IWeightsLoader;
    using stell::models::IWeightsSource;
    using stell::models::WeightsFormat;
    using stell::tiers::DType;
    using stell::tiers::GroupKey;
    using stell::tiers::Role;

    auto plain = stell::models::make_loader(WeightsFormat::Safetensors);
    auto quant = (d.dtype_weights == "NVFP4" || d.dtype_weights == "W4A16")
                     ? stell::models::make_loader(WeightsFormat::NVFP4W4A16)
                     : std::unique_ptr<IWeightsLoader>{};
    auto plain_src = plain->open(path, d);
    auto quant_src = quant ? quant->open(path, d) : std::unique_ptr<IWeightsSource>{};

    Weights w;
    std::size_t n_groups = 0, n_expected = 0;
    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == Role::Kv) continue;               // KV-группы не резолвятся
        ++n_expected;
        const bool need_quant = g.dtype == DType::NVFP4 || g.dtype == DType::W4A16;
        // Смешанный план NVFP4-профиля: F32/F16 группы читаются plain-лоадером.
        // Найдено раундом 2: NVFP4W4A16-лоадер отклоняет группы не-NVFP4 dtype
        // (UnsupportedQuant) — смешанный план требует двух источников (отчёт).
        IWeightsSource& src = (need_quant && quant_src) ? *quant_src : *plain_src;
        GroupKey key{g.id, g.role};
        std::vector<std::uint8_t> buf(g.slot_bytes, 0xAB);
        src.read(key, buf.data(), buf.size());
        std::size_t n = 1;
        for (auto dim : g.shape) n *= static_cast<std::size_t>(dim);
        std::vector<float> v(n);
        switch (g.dtype) {
            case DType::F16:
                for (std::size_t k = 0; k < n; ++k) {
                    const std::uint16_t h = static_cast<std::uint16_t>(buf[2 * k]) |
                                            (static_cast<std::uint16_t>(buf[2 * k + 1]) << 8);
                    v[k] = half_to_float(h);
                }
                break;
            case DType::F32:
                std::memcpy(v.data(), buf.data(), n * 4);
                break;
            case DType::NVFP4:
                stell::models::dequant_nvfp4(buf.data(), n, v.data());   // блок 32
                break;
            default:
                throw std::runtime_error("dtype группы вне фикстур раунда");
        }
        w.by_name[g.name] = std::move(v);
        ++n_groups;
    }
    CHECK(n_groups == n_expected);
    return w;
}

struct Fixture {
    ModelDescription d;
    Weights w;
};

Fixture load_mini(const char* file) {
    Registry r;
    Fixture f;
    f.d = r.load(std::string(STELL_FIXTURES_DIR) + "/" + file);
    const std::string path = std::string("/tmp/stell_parallel_golden_") + file + ".safetensors";
    write_safetensors(path, make_tensors(f.d));
    f.w = load_weights(f.d, path);
    return f;
}

std::string pad3(int v) {
    char buf[8];
    std::snprintf(buf, sizeof buf, "%03d", v);
    return buf;
}

// --- мини-интерпретатор ------------------------------------------------------

void rmsnorm(const float* x, const float* w, int dm, float* out) {
    float ss = 0.f;
    for (int d = 0; d < dm; ++d) ss += x[d] * x[d];
    const float inv = 1.f / std::sqrt(ss / float(dm) + kEps);
    for (int d = 0; d < dm; ++d) out[d] = x[d] * inv * w[d];
}

inline float silu_mul(float u) { return u / (1.f + std::exp(-u)) * u; }   // A1

// Блок attention-голов [hb, he): q — матвецы строк wq (локальные головы),
// KV реплицируется (A4) — полные матвецы wk/wv; вклад блока — wo[:, hcols]·o_h,
// аккумулированный по головам блока последовательно (h ↑, внутри d ↑).
void attn_heads_block(const ModelDescription& d, const Weights& w, int layer,
                      int hb, int he, const float (*hn)[8], float (*out)[8]) {
    const int dm = int(d.d_model), dh = int(d.d_head);
    const auto& wq = w.get("l" + pad3(layer) + ".attn.wq");
    const auto& wk = w.get("l" + pad3(layer) + ".attn.wk");
    const auto& wv = w.get("l" + pad3(layer) + ".attn.wv");
    const auto& wo = w.get("l" + pad3(layer) + ".attn.wo");
    float kk[T][4], vv[T][4];
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < dh; ++i) {
            float acc_k = 0.f, acc_v = 0.f;
            for (int j = 0; j < dm; ++j) {
                acc_k += wk[std::size_t(i) * dm + j] * hn[t][j];
                acc_v += wv[std::size_t(i) * dm + j] * hn[t][j];
            }
            kk[t][i] = acc_k;
            vv[t][i] = acc_v;
        }
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < dm; ++i) out[t][i] = 0.f;
    std::vector<float> q(std::size_t(T) * dh), oh(std::size_t(T) * dh);
    for (int h = hb; h < he; ++h) {
        for (int t = 0; t < T; ++t)
            for (int dd = 0; dd < dh; ++dd) {
                float acc = 0.f;
                for (int j = 0; j < dm; ++j)
                    acc += wq[std::size_t(h) * dh + dd] * hn[t][j];   // j ↑
                q[std::size_t(t) * dh + dd] = acc;
            }
        attention_forward_f32(q.data(), kk[0], vv[0], 1, T, 4, oh.data());
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < dm; ++i) {
                float acc = 0.f;
                for (int dd = 0; dd < dh; ++dd)
                    acc += wo[std::size_t(i) * dm + h * dh + dd] * oh[std::size_t(t) * dh + dd];
                out[t][i] += acc;
            }
    }
}

// Блок колонок FFN [cb, cend) (строки up): u → silu_mul → down[:, cols]·y;
// вклад аккумулируется по c ↑ внутри блока.
void ffn_block(const ModelDescription& d, const Weights& w, int layer,
               int cb, int cend, const float (*h2)[8], float (*out)[8]) {
    const int dm = int(d.d_model);
    const auto& up = w.get("l" + pad3(layer) + ".ffn.up");
    const auto& dn = w.get("l" + pad3(layer) + ".ffn.down");
    const int inter = int(up.size()) / dm;                  // 16
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < dm; ++i) out[t][i] = 0.f;
    const int width = cend - cb;
    std::vector<float> u(std::size_t(T) * width), y(std::size_t(T) * width);
    for (int t = 0; t < T; ++t)
        for (int c = cb; c < cend; ++c) {
            float acc = 0.f;
            for (int j = 0; j < dm; ++j)
                acc += up[std::size_t(c) * dm + j] * h2[t][j];
            u[std::size_t(t) * width + (c - cb)] = acc;
            y[std::size_t(t) * width + (c - cb)] = silu_mul(acc);
        }
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < dm; ++i) {
            float acc = 0.f;
            for (int c = cb; c < cend; ++c)
                acc += dn[std::size_t(i) * inter + c] *
                       y[std::size_t(t) * width + (c - cb)];
            out[t][i] += acc;
        }
}

// Экспертный FFN: e_out = down_e·silu_mul(up_e·h) — полный матвец (эксперты
// не делятся по колонкам в этих сценариях).
void expert_ffn(const ModelDescription& d, const Weights& w, int layer, int e,
                const float* h, float* out) {
    const int dm = int(d.d_model);
    const auto& up = w.get("l" + pad3(layer) + ".e" + pad3(e) + ".up");
    const auto& dn = w.get("l" + pad3(layer) + ".e" + pad3(e) + ".down");
    std::vector<float> u(dm), y(dm);
    for (int i = 0; i < dm; ++i) {
        float acc = 0.f;
        for (int j = 0; j < dm; ++j) acc += up[std::size_t(i) * dm + j] * h[j];
        u[i] = acc;
        y[i] = silu_mul(acc);
    }
    for (int i = 0; i < dm; ++i) {
        float acc = 0.f;
        for (int j = 0; j < dm; ++j) acc += dn[std::size_t(i) * dm + j] * y[j];
        out[i] = acc;
    }
}

// Роутер (A2) + детерминированный top-2 (A5: softmax по всем, обход e ↑ со
// строгим сравнением — tie даёт меньший индекс).
void router_top2(const ModelDescription& d, const Weights& w, int layer,
                 const float (*h2)[8], std::int32_t (*ids)[kKTop],
                 float (*probs)[kKTop]) {
    const int dm = int(d.d_model), ne = int(d.moe->n_experts);
    const auto& rw = w.get("l" + pad3(layer) + ".mlp.router");
    for (int t = 0; t < T; ++t) {
        float sm[16] = {};                       // ne ≤ 16 в фикстурах
        float sum = 0.f;
        for (int e = 0; e < ne; ++e) {
            float s = 0.f;
            for (int dd = 0; dd < dm; ++dd) s += h2[t][dd];
            sm[e] = std::exp(rw[std::size_t(e)] * s);   // softmax без max-сдвига
            sum += sm[e];
        }
        for (int j = 0; j < kKTop; ++j) {
            int best = -1;
            float bp = -1.f;
            for (int e = 0; e < ne; ++e) {
                bool taken = false;
                for (int jj = 0; jj < j; ++jj) if (ids[t][jj] == e) taken = true;
                if (taken) continue;
                const float p = sm[e] / sum;
                if (p > bp) { bp = p; best = e; }   // строго: tie → меньший e
            }
            ids[t][j] = std::int32_t(best);
            probs[t][j] = bp;
        }
    }
}

// Роль прогона: эталон (distributed=false, все блоки последовательно) либо
// распределённый (rank/world, свой блок + коллективы).
struct Ctx {
    const ModelDescription* d;
    const Weights* w;
    ICommunicator* comm = nullptr;
    rank_t rank = 0;
    rank_t world = 1;
    bool distributed = false;
    int attn_blocks = 1;   // блоков голов (== tp для TP2-голов; 1 — полные)
    int ffn_blocks = 1;    // блоков колонок FFN (== tp для TP4-колонок; 1 — полные)
    int ep = 1;
};

void forward(const Ctx& c, std::vector<float>& logits /* T*vocab */) {
    const ModelDescription& d = *c.d;
    const Weights& w = *c.w;
    const int dm = int(d.d_model);
    const bool moe = d.is_moe;
    CHECK(dm == 8 && int(d.d_head) == 4);      // фикстуры раунда

    float x[T][8];
    const auto& embed = w.get("embed");
    for (int t = 0; t < T; ++t)
        for (int dd = 0; dd < dm; ++dd)
            x[t][dd] = embed[std::size_t(kPrompt[t]) * dm + dd];

    for (int layer = 0; layer < int(d.n_layers); ++layer) {
        // rmsnorm(attn.norm) → attention
        float hn[T][8];
        const auto& anorm = w.get("l" + pad3(layer) + ".attn.norm");
        for (int t = 0; t < T; ++t) rmsnorm(x[t], anorm.data(), dm, hn[t]);

        float o[T][8] = {};
        const int nb = c.attn_blocks;
        if (!c.distributed) {
            for (int b = 0; b < nb; ++b) {
                float pb[T][8];
                attn_heads_block(d, w, layer, b * (int(d.n_heads) / nb),
                                 (b + 1) * (int(d.n_heads) / nb), hn, pb);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < dm; ++i) o[t][i] += pb[t][i];  // блоки ↑
            }
        } else if (nb == int(c.world)) {
            // Свой блок + allreduce в ранковом порядке = порядок блоков эталона.
            const int hb = int(c.rank) * (int(d.n_heads) / nb);
            float pb[T][8];
            attn_heads_block(d, w, layer, hb, hb + int(d.n_heads) / nb, hn, pb);
            c.comm->allreduce(pb, std::size_t(T) * dm, CommDType::F32);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < dm; ++i) o[t][i] = pb[t][i];
        } else {
            // nb == 1 → головы полные, реплицированы (одинаково на всех ранках).
            attn_heads_block(d, w, layer, 0, int(d.n_heads), hn, o);
        }
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < dm; ++i) x[t][i] += o[t][i];

        if (!moe) {
            // mini_dense: вход FFN = пост-attention residual без второй нормы
            // (A7': в фикстуре mini_dense нет mlp.norm — ops-список:
            // rmsnorm, attention, ffn).
            float h2[T][8];
            for (int t = 0; t < T; ++t)
                for (int dd = 0; dd < dm; ++dd) h2[t][dd] = x[t][dd];
            float fo[T][8] = {};
            const int fb = c.ffn_blocks;
            const int inter = 16;
            if (!c.distributed) {
                for (int b = 0; b < fb; ++b) {
                    float pb[T][8];
                    ffn_block(d, w, layer, b * (inter / fb), (b + 1) * (inter / fb), h2, pb);
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < dm; ++i) fo[t][i] += pb[t][i];
                }
            } else if (fb == int(c.world)) {
                const int cb = int(c.rank) * (inter / fb);
                float pb[T][8];
                ffn_block(d, w, layer, cb, cb + inter / fb, h2, pb);
                c.comm->allreduce(pb, std::size_t(T) * dm, CommDType::F32);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < dm; ++i) fo[t][i] = pb[t][i];
            } else {
                ffn_block(d, w, layer, 0, inter, h2, fo);
            }
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < dm; ++i) x[t][i] += fo[t][i];
        } else {
            // mini_moe: вход MoE = пост-attention residual (A7: mlp.norm в
            // фикстуре отсутствует).
            std::int32_t ids[T][kKTop];
            float probs[T][kKTop];
            router_top2(d, w, layer, x, ids, probs);
            if (c.ep == 1 || !c.distributed) {
                // Все эксперты локально (A6 / эталон) — одинаковые биты.
                float eo[8][8];                       // T*k = 8 строк
                std::int32_t eid[T][kKTop];
                float epr[T][kKTop];
                for (int t = 0; t < T; ++t)
                    for (int j = 0; j < kKTop; ++j) {
                        eid[t][j] = ids[t][j];
                        epr[t][j] = probs[t][j];
                        expert_ffn(d, w, layer, ids[t][j], x[t], eo[std::size_t(t) * kKTop + j]);
                    }
                float y[T][8] = {};
                moe_scatter_add(&eo[0][0], &eid[0][0], &epr[0][0], T, kKTop, dm, &y[0][0]);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < dm; ++i) x[t][i] += y[t][i];
            } else {
                // EP: два прохода expert_alltoall (gather → FFN у владельца →
                // scatter назад). Токен = ЦЕЛЫЙ вектор d_model (token_elems = dm,
                // уточнение ADR-10 round 2): max_tokens = T*k токенов, мета —
                // одна на токен (slot_id кодирует пару (источник, t, j)).
                const int ntok = T * kKTop;
                const std::size_t cap = std::size_t(ntok) * std::size_t(c.world); // в токенах
                std::vector<float> send(std::size_t(ntok) * dm),
                    recv(cap * std::size_t(dm));
                std::vector<TokenMeta> smeta(ntok), rmeta(cap);
                const int my_ep = int(c.rank % c.ep);                 // tp=1 (EP2-сценарий)
                std::memset(send.data(), 0, send.size() * sizeof(float));
                for (int t = 0; t < T; ++t)
                    for (int j = 0; j < kKTop; ++j) {
                        const int idx = t * kKTop + j;
                        const int e = ids[t][j];
                        const int owner = int(expert_owner(rank_t(e), rank_t(c.ep)));
                        const bool remote = owner != my_ep;
                        // slot_id: база пары (источник ↑, t, j) — позиция dd
                        // внутри токена не нужна (токен целый).
                        const std::int32_t enc = std::int32_t(c.rank) * 64 +
                                                 t * 16 + j * 8;              // A3
                        if (remote) {
                            smeta[idx] = TokenMeta{std::int32_t(e), enc};
                            for (int dd = 0; dd < dm; ++dd) send[idx * dm + dd] = x[t][dd];
                        } else {
                            smeta[idx] = TokenMeta{-1, -1};   // не отправляется
                        }
                    }
                // --- проход 1: gather к владельцам (все ранки — коллективно) ---
                c.comm->expert_alltoall(send.data(), smeta.data(), recv.data(),
                                        rmeta.data(), std::size_t(ntok), std::size_t(dm),
                                        CommDType::F32);
                // Владелец: восстановление не нужно — входящий токен ГОТОВЫЙ
                // вектор d_model (ключ пары = enc: enc = src*64 + t*16 + j*8);
                // FFN своего эксперта, отправка назад (eid = исходный ранок, A3).
                struct InTok { float h[8]; int e; };   // dm=8 в фикстурах mini_moe
                std::map<std::int32_t, InTok> in;      // ключ пары (src,t,j) → токен
                for (std::size_t i = 0; i < cap; ++i) {
                    if (rmeta[i].expert_id < 0) continue;
                    const std::int32_t key = rmeta[i].slot_id;
                    InTok& r = in[key];
                    r.e = rmeta[i].expert_id;
                    for (int dd = 0; dd < dm; ++dd) r.h[dd] = recv[i * dm + dd];
                }
                std::vector<float> send2(std::size_t(ntok) * dm, 0.f);
                std::vector<TokenMeta> smeta2(ntok, TokenMeta{-1, -1});
                {
                    int idx = 0;
                    for (const auto& [key, r] : in) {
                        (void)key;
                        const std::int32_t src = key / 64;
                        float eout[8];
                        expert_ffn(d, w, layer, r.e, r.h, eout);
                        for (int dd = 0; dd < dm; ++dd) send2[idx * dm + dd] = eout[dd];
                        smeta2[idx] = TokenMeta{src, key};
                        ++idx;
                    }
                    // Хвост — пустые слоты (не отправляется; send2 уже обнулён).
                }
                // --- проход 2: scatter назад к «домашним» ранкам ---
                std::vector<float> recv2(cap * std::size_t(dm), 0.f);
                std::vector<TokenMeta> rmeta2(cap);
                c.comm->expert_alltoall(send2.data(), smeta2.data(), recv2.data(),
                                        rmeta2.data(), std::size_t(ntok), std::size_t(dm),
                                        CommDType::F32);
                // Сборка e_out по (t, j): локальные + полученные; порядок j ↑.
                float eo[8][8];
                std::int32_t eid[T][kKTop];
                float epr[T][kKTop];
                for (int t = 0; t < T; ++t)
                    for (int j = 0; j < kKTop; ++j) {
                        eid[t][j] = ids[t][j];
                        epr[t][j] = probs[t][j];
                        const int owner = int(expert_owner(rank_t(ids[t][j]), rank_t(c.ep)));
                        if (owner == my_ep)
                            expert_ffn(d, w, layer, ids[t][j], x[t], eo[std::size_t(t) * kKTop + j]);
                        else
                            for (int dd = 0; dd < dm; ++dd)
                                eo[std::size_t(t) * kKTop + j][dd] = 0.f;
                    }
                for (std::size_t i = 0; i < cap; ++i) {
                    if (rmeta2[i].expert_id < 0) continue;
                    const std::int32_t enc = rmeta2[i].slot_id;
                    const int t = (enc / 16) % 4, j = (enc / 8) % 2;   // dd = позиция в токене
                    for (int dd = 0; dd < dm; ++dd)
                        eo[std::size_t(t) * kKTop + j][dd] = recv2[i * dm + dd];
                }
                float y[T][8] = {};
                moe_scatter_add(&eo[0][0], &eid[0][0], &epr[0][0], T, kKTop, dm, &y[0][0]);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < dm; ++i) x[t][i] += y[t][i];
            }
        }
    }

    // Финал: rmsnorm(norm_final) → head → logits [T][vocab].
    // A8: mini_moe без head-группы — tied embeddings (logits = embed·nf).
    const auto& fnorm = w.get("norm_final");
    const auto& head = w.has("head") ? w.get("head") : w.get("embed");
    const int vocab = int(d.vocab_size);
    logits.assign(std::size_t(T) * vocab, 0.f);
    for (int t = 0; t < T; ++t) {
        float nf[8];
        rmsnorm(x[t], fnorm.data(), dm, nf);
        for (int v = 0; v < vocab; ++v) {
            float acc = 0.f;
            for (int dd = 0; dd < dm; ++dd)
                acc += head[std::size_t(v) * dm + dd] * nf[dd];
            logits[std::size_t(t) * vocab + v] = acc;
        }
    }
}

// --- сценарии и прогоны ------------------------------------------------------

struct Scenario {
    int tp, ep;             // топология: pp=1; world = tp*ep
    int attn_blocks;        // блоков голов
    int ffn_blocks;         // блоков колонок FFN
};

std::vector<float> run_reference(const Fixture& f, const Scenario& sc) {
    Ctx c;
    c.d = &f.d;
    c.w = &f.w;
    c.distributed = false;
    c.attn_blocks = sc.attn_blocks;
    c.ffn_blocks = sc.ffn_blocks;
    c.ep = sc.ep;
    std::vector<float> logits;
    forward(c, logits);
    return logits;
}

// Распределённый прогон: world = tp*ep ранков (потоки), каждый сравнивает
// свои логиты с эталоном ПОБИТОВО.
void run_distributed(const Fixture& f, const Scenario& sc, int group_id,
               const std::vector<float>& ref) {
    const rank_t world = rank_t(sc.tp * sc.ep);
    SafeFlag ok;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < world; ++r)
        th.emplace_back([&, r] {
            ParallelConfig cfg;
            cfg.mode = CommMode::CpuEmulation;
            cfg.pp = 1;
            cfg.tp = rank_t(sc.tp);
            cfg.ep = rank_t(sc.ep);
            cfg.group_id = group_id;
            cfg.rank = r;
            auto comm = std::make_unique<EmulatedCommunicator>(cfg);
            Ctx c;
            c.d = &f.d;
            c.w = &f.w;
            c.comm = comm.get();
            c.rank = r;
            c.world = world;
            c.distributed = true;
            c.attn_blocks = sc.attn_blocks;
            c.ffn_blocks = sc.ffn_blocks;
            c.ep = sc.ep;
            std::vector<float> logits;
            forward(c, logits);
            for (std::size_t i = 0; i < logits.size(); ++i)
                if (!(logits[i] == ref[i])) ok.fail();
        });
    for (auto& t : th) t.join();
    CHECK(ok.get());   // побитовое равенство логитов TP/EP == эталон
}

// Распределённый прогон сравнивает свои логиты с эталоном ПОБИТОВО внутри
// (SafeFlag в run_distributed); здесь — только допуск «плоский TP1 ↔ блоковый
// эталон» (порядок суммирования отличается).
// «Плоский» TP1 (все матвецы целиком) против блокового эталона — порядок
// суммирования отличается; фиксируем величину расхождения (допуск 1e-4 abs).
void check_flat_vs_blockwise(const std::vector<float>& flat,
                             const std::vector<float>& blocky,
                             const char* what) {
    double maxdiff = 0.0;
    for (std::size_t i = 0; i < flat.size() && i < blocky.size(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(double(flat[i]) - double(blocky[i])));
    std::printf("    %s: макс. |плоский TP1 − блоковый эталон| = %.3e\n", what, maxdiff);
    CHECK(maxdiff <= 1e-4);
}

void run_all_scenarios() {
    const Fixture dense = load_mini("mini_dense.json");
    const Fixture moe = load_mini("mini_moe.json");
    CHECK(dense.d.n_layers == 2 && dense.d.n_heads == 2 && dense.d.kv.n_kv_heads == 1);
    CHECK(moe.d.is_moe && moe.d.moe->n_experts == 4);

    // «Плоские» эталоны TP1/EP1 (все матвецы целиком, один ранок).
    const std::vector<float> flat_dense = run_reference(
        dense, Scenario{1, 1, 1, 1});
    const std::vector<float> flat_moe = run_reference(
        moe, Scenario{1, 1, 1, 1});
    CHECK(flat_dense[0] != 0.f);   // нетривиальность

    // mini_dense TP2 (головы; KV реплицирован, A4) — world=2. Блоковый эталон:
    // две головы-блока последовательно.
    {
        const std::vector<float> ref = run_reference(dense, Scenario{1, 1, 2, 1});
        check_flat_vs_blockwise(flat_dense, ref, "mini_dense TP2-головы");
        run_distributed(dense, Scenario{2, 1, 2, 1}, 601, ref);
    }
    // mini_dense TP4 (колонки FFN; головы полные, D1) — world=4.
    {
        const std::vector<float> ref = run_reference(dense, Scenario{1, 1, 1, 4});
        check_flat_vs_blockwise(flat_dense, ref, "mini_dense TP4-колонки FFN");
        run_distributed(dense, Scenario{4, 1, 1, 4}, 602, ref);
    }
    // mini_moe TP2 (головы; эксперты реплицированы, A6) — world=2.
    {
        const std::vector<float> ref = run_reference(moe, Scenario{1, 1, 2, 1});
        check_flat_vs_blockwise(flat_moe, ref, "mini_moe TP2-головы");
        run_distributed(moe, Scenario{2, 1, 2, 1}, 603, ref);
    }
    // mini_moe EP2 (эксперты, два прохода expert_alltoall) — world=2.
    {
        const std::vector<float> ref = run_reference(moe, Scenario{1, 1, 1, 1});
        run_distributed(moe, Scenario{1, 2, 1, 1}, 604, ref);
    }
}

} // namespace

// Регистрация в харнессе.
TEST(golden_real_mini_dense_tp2_tp4_and_mini_moe_tp2_ep2) {
    run_all_scenarios();
}

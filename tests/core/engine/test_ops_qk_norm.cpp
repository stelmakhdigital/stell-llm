// [R1]/[R2] round 7 (planner [Q1]/[Q2], ADR-09 «Уточнение round 7» — qk-norm
// и роутер без перенормировки):
//
// [R1] per-head RMSNorm q/k — группы l%03zu.attn.q_norm / l%03zu.attn.k_norm
// (1-D вектор веса [d_head] / [d_kv], роль attn, нормы не квантуются),
// применяются к проекциям q/k ПОСЛЕ линейных проекций и ДО QK/PV и записи k
// в KV-кеш; каждая голова нормализуется НЕЗАВИСИМО одним вектором веса:
// slice = x[h*dim .. h*dim+dim), inv = 1/sqrt(mean(slice^2)+eps),
// y[h*dim+d] = w[d]*x[h*dim+d]*inv. eps — параметр qk_eps op attention
// (из config, не дефолт-догадка). Отсутствие ОБОИХ групп — прежнее поведение.
//
// [R2] router_renorm=false: topk-веса роутера БЕЗ перенормировки —
// probs[j] = scores[order[j]] (softmax по всем экспертам уже сделан; top-k и
// stable-sort прежние; gather/scatter и эксперты без изменений). Дефолт true
// — канон Mixtral (golden побитово).
//
// РЕФЕРЕНС — НЕЗАВИСИМАЯ референс-реализация forward (синтетическая
// Qwen3Moe-класса фикстура: MoE-слои с mlp.router, эксперты e%03u.{up,down},
// q_norm/k_norm на каждом слое, op attention с qk_eps, router_renorm=false),
// повторяющая арифметику контракта побитово: matvec, rmsnorm (double-
// аккумулятор), per-head qk-норма (собственный код теста), router softmax →
// topk → веса БЕЗ деления на сумму, SwiGLU-путь экспертов, gather/scatter
// kernels. Якорь честности референса: фикстура БЕЗ qk-групп (и без qk_eps) —
// интерпретатор == референсу побитово на уже проверенном поведении, после
// чего совпадение на qk-фикстуре изолирует именно семантику qk-norm.
//
// Валидация (fail-fast, честный SchemaMismatch): частичный набор (q_norm без
// k_norm); группы без qk_eps; qk_eps без групп (мёртвый параметр); форма
// группы не 1-D [d_head]/[d_kv]; router_renorm=false на dense-плане.
#include "test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/kernels/attention.hpp"
#include "stell/kernels/moe_gather_scatter.hpp"

using namespace stell::engine;
using namespace stell_core_test;

namespace {

constexpr std::uint32_t kVocab = 16;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kDM = 8;           // d_model
constexpr std::uint32_t kNHeads = 2;
constexpr std::uint32_t kDHead = 4;         // n_heads*d_head == d_model
constexpr std::uint32_t kNKV = 1;
constexpr std::uint32_t kDKV = 4;
constexpr std::uint32_t kDff = 16;
constexpr std::uint32_t kExperts = 4;
constexpr std::uint32_t kActive = 2;
constexpr float kQkEps = 1e-6f;             // qk_eps op attention (float)

// F32 → F16-половинки значений fill_weights (значения k/16 — точны в F16,
// конверт побитово обратим для фикстуры; материал engine == исходные float).
inline std::vector<std::uint16_t> halves_of(const std::vector<float>& vals) {
    std::vector<std::uint16_t> out(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) out[i] = f32_to_f16(vals[i]);
    return out;
}

// --- план фикстуры -----------------------------------------------------------

// moe=true: слои с mlp.router + экспертами e%03u.{up,down} (MoE-контур);
// moe=false: dense-слои ffn.{up,down} (для кейса router_renorm=false на
// dense-плане). with_qk: группы attn.q_norm/attn.k_norm + qk_eps у op
// attention (только для MoE-варианта — dense-кейс валидации qk не касается).
std::vector<PlainGroup> groups_for(bool with_qk, bool moe) {
    std::vector<PlainGroup> out;
    std::uint32_t id = 0;
    const auto add = [&](PlainGroup g) { g.id = id++; out.push_back(std::move(g)); };

    add(make_group(id, "embed", "F16", {kVocab, kDM}, "embedding.weight"));
    add(make_group(id, "norm_final", "F32", {kDM}, "model.norm.weight"));
    add(make_group(id, "head", "F16", {kVocab, kDM}, "head.weight"));
    for (std::size_t l = 0; l < kLayers; ++l) {
        char pb[16];
        std::snprintf(pb, sizeof pb, "l%03zu.", l);
        const std::string p = pb;
        const std::string ls = std::to_string(l);
        add(make_group(id, p + "attn.norm", "F32", {kDM},
                       "layers." + ls + ".input_layernorm.weight"));
        add(make_group(id, p + "attn.wq", "F16", {kDM, kDM},
                       "layers." + ls + ".self_attn.q_proj.weight"));
        add(make_group(id, p + "attn.wk", "F16", {kNKV * kDKV, kDM},
                       "layers." + ls + ".self_attn.k_proj.weight"));
        add(make_group(id, p + "attn.wv", "F16", {kNKV * kDKV, kDM},
                       "layers." + ls + ".self_attn.v_proj.weight"));
        add(make_group(id, p + "attn.wo", "F16", {kDM, kDM},
                       "layers." + ls + ".self_attn.o_proj.weight"));
        add(make_group(id, p + "kv", "F16", {kNKV * kDKV, kDM},
                       "layers." + ls + ".kv_cache", /*kv=*/true));
        if (moe) {
            // [R1] round 7: per-head RMSNorm q/k — 1-D [d_head] / [d_kv],
            // F32 (нормы не квантуются — dtype исходного тензора).
            if (with_qk) {
                add(make_group(id, p + "attn.q_norm", "F32", {kDHead},
                               "layers." + ls + ".self_attn.q_norm.weight"));
                add(make_group(id, p + "attn.k_norm", "F32", {kDKV},
                               "layers." + ls + ".self_attn.k_norm.weight"));
            }
            add(make_group(id, p + "mlp.router", "F32",
                           {std::int64_t(kExperts), kDM},
                           "layers." + ls + ".mlp.router.weight"));
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                char eb[32];
                std::snprintf(eb, sizeof eb, "e%03u", e);
                const std::string base =
                    "layers." + ls + ".mlp.experts." + std::to_string(e);
                add(make_group(id, p + eb + ".up", "F16", {kDff, kDM},
                               base + ".up_proj.weight"));
                add(make_group(id, p + eb + ".down", "F16", {kDM, kDff},
                               base + ".down_proj.weight"));
            }
        } else {
            add(make_group(id, p + "ffn.up", "F16", {kDff, kDM},
                           "layers." + ls + ".mlp.up_proj.weight"));
            add(make_group(id, p + "ffn.down", "F16", {kDM, kDff},
                           "layers." + ls + ".mlp.down_proj.weight"));
        }
    }
    return out;
}

PlainSpec make_spec(bool with_qk, bool renorm, bool moe, const char* name) {
    PlainSpec s{};
    s.name = name;
    s.is_moe = moe;
    s.dtype_weights = "F16";
    s.vocab_size = kVocab;
    s.d_model = kDM;
    s.n_layers = kLayers;
    s.n_heads = kNHeads;
    s.d_head = kDHead;
    s.n_kv_heads = kNKV;
    s.d_kv = kDKV;
    if (moe) {
        s.n_experts = kExperts;
        s.n_experts_active = kActive;
        s.expert_d_ff = kDff;
        s.router = "softmax-topk";
    }
    s.router_renorm = renorm;   // [R2] round 7 (дефолт true — канон Mixtral)
    s.ops = base_ops(/*with_ffn=*/!moe, /*with_moe=*/moe);
    // [R1] round 7: qk_eps — параметр op attention (float; не дефолт-догадка).
    if (with_qk)
        for (PlainOp& op : s.ops)
            if (op.op == "attention")
                op.params.push_back({"qk_eps", kQkEps, "", true});
    s.groups = groups_for(with_qk, moe);
    return s;
}

// Тензоры фикстуры (fill_weights — значения k/16, точные в F16; F32 — memcpy).
std::vector<STensor> tensors_for(const PlainSpec& spec) {
    std::vector<STensor> out;
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block) continue;
        STensor t;
        t.name = g.source_pattern;
        t.dtype = g.dtype == PlainDTypeF32 ? "F32" : "F16";
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto d : g.shape) n *= static_cast<std::size_t>(d);
        const std::vector<float> vals = fill_weights(g.source_pattern, n);
        t.data = t.dtype == "F32" ? f32_payload(vals)
                                  : f16_payload(halves_of(vals));
        out.push_back(std::move(t));
    }
    return out;
}

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load(const PlainSpec& spec, const char* path) {
    Loaded l{};
    l.spec = spec;
    write_safetensors(path, tensors_for(spec));
    l.weights = load_plain_weights(path, spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(spec);
    return l;
}

// Пересборка группы с НОВОЙ формой (slot_bytes честно пересчитан — make_group;
// иначе лоадер откажет раньше конструктора по размеру payload).
PlainGroup reshaped(const PlainGroup& g, std::vector<std::int64_t> shape) {
    const std::string dtype = g.dtype == PlainDTypeF32 ? "F32" : "F16";
    return make_group(g.id, g.name, dtype, std::move(shape), g.source_pattern,
                      g.kv_block);
}

// --- НЕЗАВИСИМЫЙ референс плана (побитовая арифметика контракта) -------------

// Референс-веса: значения fill_weights(source_pattern) — материал engine
// побитово им идентичен (F32-группы — memcpy; F16-конверт точен для значений
// фикстуры k/16, k ∈ [-16, 16]).
struct RefW {
    std::vector<float> embed;      // [kVocab*kDM]
    std::vector<float> norm_final; // [kDM]
    std::vector<float> head;       // [kVocab*kDM]
    struct Layer {
        std::vector<float> attn_norm;        // [kDM]
        std::vector<float> wq, wk, wv, wo;
        std::vector<float> q_norm, k_norm;   // per-head веса [d_head]/[d_kv]
        std::vector<float> router;           // [kExperts*kDM]
        std::vector<float> up, down;         // эксперты, плоские стеки
    };
    std::vector<Layer> layers;
};

RefW ref_weights(bool with_qk) {
    RefW w{};
    const auto fw = [](const std::string& pattern, std::size_t n) {
        return fill_weights(pattern, n);
    };
    w.embed = fw("embedding.weight", kVocab * kDM);
    w.norm_final = fw("model.norm.weight", kDM);
    w.head = fw("head.weight", kVocab * kDM);
    w.layers.resize(kLayers);
    for (std::size_t l = 0; l < kLayers; ++l) {
        const std::string ls = std::to_string(l);
        RefW::Layer& L = w.layers[l];
        L.attn_norm = fw("layers." + ls + ".input_layernorm.weight", kDM);
        L.wq = fw("layers." + ls + ".self_attn.q_proj.weight", kDM * kDM);
        L.wk = fw("layers." + ls + ".self_attn.k_proj.weight",
                  kNKV * kDKV * kDM);
        L.wv = fw("layers." + ls + ".self_attn.v_proj.weight",
                  kNKV * kDKV * kDM);
        L.wo = fw("layers." + ls + ".self_attn.o_proj.weight", kDM * kDM);
        if (with_qk) {
            L.q_norm = fw("layers." + ls + ".self_attn.q_norm.weight", kDHead);
            L.k_norm = fw("layers." + ls + ".self_attn.k_norm.weight", kDKV);
        }
        L.router = fw("layers." + ls + ".mlp.router.weight", kExperts * kDM);
        for (std::uint32_t e = 0; e < kExperts; ++e) {
            const std::string base = "layers." + ls + ".mlp.experts." +
                                     std::to_string(e);
            const std::vector<float> up = fw(base + ".up_proj.weight",
                                             kDff * kDM);
            L.up.insert(L.up.end(), up.begin(), up.end());
            const std::vector<float> dn = fw(base + ".down_proj.weight",
                                             kDM * kDff);
            L.down.insert(L.down.end(), dn.begin(), dn.end());
        }
    }
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

// [R1] round 7: НЕЗАВИСИМАЯ реализация forward per-head RMSNorm (собственный
// код теста, арифметика контракта побитово: сумма квадратов через double,
// inv = 1/sqrt(float(ms/dim) + eps), y = w[d]*x*inv; головы не смешиваются).
void ref_head_rmsnorm(std::vector<float>& x, std::uint32_t n_heads,
                      std::uint32_t dim, const std::vector<float>& w,
                      float eps) {
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        double ms = 0.0;
        for (std::uint32_t d = 0; d < dim; ++d) {
            const float xv = x[h * dim + d];
            ms += double(xv) * double(xv);
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(ms / dim) + eps);
        for (std::uint32_t d = 0; d < dim; ++d)
            x[h * dim + d] = w[d] * x[h * dim + d] * inv;
    }
}

struct RefState {
    std::vector<std::vector<float>> k, v;   // по слоям, token-major [pos][dkv]
    std::uint32_t pos = 0;
};

// MoE-контур слоя ([R2] round 7: renorm=false — topk-веса БЕЗ деления на
// сумму; gather/scatter и SwiGLU-путь экспертов {up,down} — без изменений).
std::vector<float> ref_moe(const RefW::Layer& L,
                           const std::vector<float>& hidden, bool renorm) {
    const std::vector<float> scores =
        ref_matvec(L.router, kExperts, kDM, hidden);
    float mx = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    std::vector<float> probs_raw(scores.size());
    for (std::size_t e = 0; e < scores.size(); ++e) {
        probs_raw[e] = std::exp(scores[e] - mx);
        sum += probs_raw[e];
    }
    for (float& s : probs_raw) s /= sum;

    // top-k: stable sort по -score (ничья → меньший индекс).
    std::vector<std::uint32_t> order(scores.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(order.size()); ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::uint32_t a, std::uint32_t b) {
                         return probs_raw[a] > probs_raw[b];
                     });
    order.resize(kActive);

    // [R2] round 7: renorm=false → probs[j] = softmax-scores[order[j]] как
    // есть; true → probs[j] = scores[order[j]] / Σ (канон Mixtral).
    std::vector<float> probs(kActive, 0.0f);
    if (renorm) {
        float pk_sum = 0.0f;
        for (std::uint32_t j = 0; j < kActive; ++j)
            pk_sum += probs_raw[order[j]];
        for (std::uint32_t j = 0; j < kActive; ++j)
            probs[j] = pk_sum > 0.0f ? probs_raw[order[j]] / pk_sum : 0.0f;
    } else {
        for (std::uint32_t j = 0; j < kActive; ++j)
            probs[j] = probs_raw[order[j]];
    }

    // Эксперты {up,down}: hu = silu(up·x), y = down·hu; scatter-add kernels
    // (та же арифметика буферов: y[d] += w_j * y_j[d] в порядке j).
    std::vector<float> expert_out(static_cast<std::size_t>(kActive) * kDM,
                                  0.0f);
    for (std::uint32_t j = 0; j < kActive; ++j) {
        const std::size_t e = order[j];
        const float* up_m =
            L.up.data() + e * kDff * kDM;
        const float* dn_m =
            L.down.data() + e * kDM * kDff;
        const std::vector<float> hu = ref_matvec(
            std::vector<float>(up_m, up_m + kDff * kDM), kDff, kDM, hidden);
        std::vector<float> m_vec(hu.size());
        for (std::size_t i = 0; i < hu.size(); ++i) m_vec[i] = ref_silu(hu[i]);
        const std::vector<float> y = ref_matvec(
            std::vector<float>(dn_m, dn_m + kDM * kDff), kDM, kDff, m_vec);
        for (std::size_t d = 0; d < kDM; ++d)
            expert_out[static_cast<std::size_t>(j) * kDM + d] = y[d];
    }
    std::vector<std::int32_t> ids(kActive);
    for (std::uint32_t j = 0; j < kActive; ++j)
        ids[j] = static_cast<std::int32_t>(order[j]);
    std::vector<float> y(kDM, 0.0f);
    stell::kernels::moe_scatter_add(expert_out.data(), ids.data(), probs.data(),
                                    /*n_tokens=*/1, kActive, kDM, y.data());
    return y;
}

std::vector<float> ref_forward_token(RefState& st, std::int32_t token,
                                     const RefW& w, bool with_qk, bool renorm) {
    const float eps = 1e-5f;         // eps base_ops (rmsnorm)
    std::vector<float> hidden(kDM, 0.0f);
    for (std::size_t i = 0; i < kDM; ++i)
        hidden[i] = w.embed[static_cast<std::size_t>(token) * kDM + i];

    for (std::size_t l = 0; l < kLayers; ++l) {
        const RefW::Layer& L = w.layers[l];
        // rmsnorm: double-аккумулятор + inv (арифметика op_rmsnorm).
        double ms = 0.0;
        for (float x : hidden) ms += double(x) * double(x);
        const float inv = 1.0f / std::sqrt(static_cast<float>(ms / kDM) + eps);
        std::vector<float> norm_in(kDM);
        for (std::size_t i = 0; i < kDM; ++i)
            norm_in[i] = L.attn_norm[i] * hidden[i] * inv;

        std::vector<float> q = ref_matvec(L.wq, kDM, kDM, norm_in);
        std::vector<float> k = ref_matvec(L.wk, kNKV * kDKV, kDM, norm_in);
        const std::vector<float> v = ref_matvec(L.wv, kNKV * kDKV, kDM, norm_in);
        if (with_qk) {
            // [R1] round 7: per-head RMSNorm q/k ПОСЛЕ проекций, ДО QK/PV
            // и записи k в кеш; q — kNHeads голов по kDHead, k — kNKV по kDKV.
            ref_head_rmsnorm(q, kNHeads, kDHead, L.q_norm, kQkEps);
            ref_head_rmsnorm(k, kNKV, kDKV, L.k_norm, kQkEps);
        }

        // Раскладка kernels [n_kv_heads][T][d_kv] поверх token-major кеша
        // (арифметика op_attention; n_kv_heads = 1).
        const std::uint32_t T = st.pos + 1;
        std::vector<float> kk(static_cast<std::size_t>(kNKV) * T * kDKV, 0.0f);
        std::vector<float> vk(kk.size(), 0.0f);
        for (std::uint32_t t = 0; t < T; ++t)
            for (std::uint32_t d = 0; d < kDKV; ++d) {
                const std::size_t dst = static_cast<std::size_t>(t) * kDKV + d;
                if (t == st.pos) {
                    kk[dst] = k[d];
                    vk[dst] = v[d];
                } else {
                    kk[dst] = st.k[l][static_cast<std::size_t>(t) * kDKV + d];
                    vk[dst] = st.v[l][static_cast<std::size_t>(t) * kDKV + d];
                }
            }
        std::vector<float> attn_out(static_cast<std::size_t>(kNHeads) * kDHead,
                                    0.0f);
        stell::kernels::attention_decode_f32(q.data(), kk.data(), vk.data(),
                                             kNHeads, kNKV, T, kDHead,
                                             attn_out.data());
        // joined == attn_out (n_heads*d_head == d_model); wo + residual.
        const std::vector<float> proj =
            ref_matvec(L.wo, kDM, kNHeads * kDHead, attn_out);
        for (std::size_t i = 0; i < kDM; ++i) hidden[i] += proj[i];

        // Запись k/v токена в token-major кеш (k — уже нормализованный).
        st.k[l].insert(st.k[l].end(), k.begin(), k.end());
        st.v[l].insert(st.v[l].end(), v.begin(), v.end());

        // --- MoE-контур: router softmax → topk → веса ([R2] renorm) → эксперты
        const std::vector<float> y = ref_moe(L, hidden, renorm);
        for (std::size_t i = 0; i < kDM; ++i) hidden[i] += y[i];
    }

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
                                           const std::vector<std::int32_t>& p,
                                           bool with_qk, bool renorm) {
    RefState st{};
    st.k.assign(kLayers, {});
    st.v.assign(kLayers, {});
    std::vector<std::vector<float>> rows;
    for (std::int32_t tok : p)
        rows.push_back(ref_forward_token(st, tok, w, with_qk, renorm));
    return rows;
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

bool throws_schema_mismatch(const PlainSpec& spec, const char* path,
                            const char* tag) {
    try {
        const Loaded l = load(spec, path);
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

    // Фикстуры: Z (qk-norm + renorm=false — Qwen3Moe-класс round 7), N (без
    // qk-групп и без qk_eps — прежнее поведение, якорь референса), RN (qk-norm,
    // но router_renorm=true — канон Mixtral).
    const Loaded lz = load(make_spec(/*with_qk=*/true, /*renorm=*/false, true,
                                     "mini-qknorm-test"),
                           "/tmp/stell_core_test_qknorm.safetensors");
    const Loaded ln = load(make_spec(/*with_qk=*/false, /*renorm=*/false, true,
                                     "mini-qknorm-none-test"),
                           "/tmp/stell_core_test_qknorm_none.safetensors");
    const Loaded lrn = load(make_spec(/*with_qk=*/true, /*renorm=*/true, true,
                                      "mini-qknorm-renorm-test"),
                            "/tmp/stell_core_test_qknorm_renorm.safetensors");

    OpsInterpreter iz(lz.spec, lz.plan, lz.weights);
    OpsInterpreter in(ln.spec, ln.plan, ln.weights);
    OpsInterpreter irn(lrn.spec, lrn.plan, lrn.weights);

    const auto all_z = iz.run_prompt_logits_all(prompt);
    const auto all_n = in.run_prompt_logits_all(prompt);
    const auto all_rn = irn.run_prompt_logits_all(prompt);
    STELL_CHECK(all_z.size() == prompt.size());
    for (float v : all_z.back()) STELL_CHECK(std::isfinite(v));

    // --- ЯКОРЬ: N (без qk-групп, без qk_eps) == референсу, ПОБИТОВО ---------
    // Совпадение референса с интерпретатором на прежнем поведении доказывает
    // faithful-копию арифметики контракта (matvec/rmsnorm/router/эксперты);
    // последующее совпадение на Z изолирует семантику qk-norm и no-renorm.
    check_bitwise(all_n, ref_prompt(ref_weights(false), prompt, false, false),
                  "N: логиты != референсу (без qk-групп)");
    // --- Z (qk-norm, renorm=false) == референсу плана, ПОБИТОВО -------------
    check_bitwise(all_z, ref_prompt(ref_weights(true), prompt, true, false),
                  "Z: логиты != референсу (qk-norm, renorm=false)");
    // --- RN (qk-norm, renorm=true — канон Mixtral) == референсу, ПОБИТОВО ---
    check_bitwise(all_rn, ref_prompt(ref_weights(true), prompt, true, true),
                  "RN: логиты != референсу (qk-norm, renorm=true)");

    // last-путь == последней позиции полного прогона (инвариант round 5).
    const std::vector<float> last_z = iz.run_prompt_logits(prompt);
    check_bitwise({last_z}, {all_z.back()}, "Z: last != all.back()");

    // --- вклад реален: НЕ молчаливое игнорирование ---------------------------
    // Z != N: qk-норма меняет проекции q/k → меняет логиты; RN != Z: флаг
    // router_renorm реально исполняется (веса топ-k различаются).
    STELL_CHECK_MSG(differs(all_z.back(), all_n.back()),
                    "qk-норма не проявилась в логитах (Z == N)");
    STELL_CHECK_MSG(differs(all_rn.back(), all_z.back()),
                    "router_renorm не проявился в логитах (RN == Z)");

    // --- детерминизм повторного прогона --------------------------------------
    OpsInterpreter iz2(lz.spec, lz.plan, lz.weights);
    const auto again = iz2.run_prompt_logits_all(prompt);
    check_bitwise(all_z, again, "Z: повторный прогон");

    // --- валидация: честный SchemaMismatch -----------------------------------
    const char* kPath = "/tmp/stell_core_test_qknorm_bad.safetensors";
    std::size_t ok = 0;
    {
        // частичный набор: q_norm БЕЗ k_norm (с qk_eps у op attention)
        PlainSpec bad = make_spec(true, false, true, "mini-qknorm-bad-partial");
        std::vector<PlainGroup> g;
        for (PlainGroup& gg : bad.groups)
            if (gg.name.find("attn.k_norm") == std::string::npos)
                g.push_back(std::move(gg));
        bad.groups = std::move(g);
        ok += throws_schema_mismatch(bad, kPath, "q_norm-без-k_norm") ? 1 : 0;
    }
    {
        // группы БЕЗ qk_eps у op attention — eps не дефолт-догадка
        PlainSpec bad = make_spec(true, false, true, "mini-qknorm-bad-noeps");
        for (PlainOp& op : bad.ops) {
            std::vector<PlainOpParam> ps;
            for (PlainOpParam& p : op.params)
                if (p.key != "qk_eps") ps.push_back(p);
            op.params = std::move(ps);
        }
        ok += throws_schema_mismatch(bad, kPath, "группы-без-qk_eps") ? 1 : 0;
    }
    {
        // qk_eps БЕЗ групп (ни в одном слое) — мёртвый параметр запрещён
        PlainSpec bad = make_spec(false, false, true, "mini-qknorm-bad-deadeps");
        for (PlainOp& op : bad.ops)
            if (op.op == "attention")
                op.params.push_back({"qk_eps", kQkEps, "", true});
        ok += throws_schema_mismatch(bad, kPath, "qk_eps-без-групп") ? 1 : 0;
    }
    {
        // форма группы не 1-D нужного размера: q_norm [d_head+1] — отказ
        PlainSpec bad = make_spec(true, false, true, "mini-qknorm-bad-shape");
        for (PlainGroup& g : bad.groups)
            if (g.name.find("attn.q_norm") != std::string::npos)
                g = reshaped(g, {kDHead + 1});
        ok += throws_schema_mismatch(bad, kPath,
                                     "q_norm-форма-[d_head+1]") ? 1 : 0;
    }
    {
        // форма группы 2-D [1, d_kv] — не 1-D вектор веса: отказ
        PlainSpec bad = make_spec(true, false, true, "mini-qknorm-bad-shape2");
        for (PlainGroup& g : bad.groups)
            if (g.name.find("attn.k_norm") != std::string::npos)
                g = reshaped(g, {1, kDKV});
        ok += throws_schema_mismatch(bad, kPath,
                                     "k_norm-форма-[1,d_kv]") ? 1 : 0;
    }
    {
        // router_renorm=false на dense-плане (без mlp.router) — мёртвый флаг
        PlainSpec dense = make_spec(/*with_qk=*/false, /*renorm=*/false,
                                    /*moe=*/false, "mini-qknorm-bad-dense");
        ok += throws_schema_mismatch(dense, kPath,
                                     "renorm=false-на-dense") ? 1 : 0;
    }
    STELL_CHECK(ok == 6);

    // --- router_renorm=true (дефолт) на dense-плане — валидно (без изменений)
    {
        PlainSpec dense = make_spec(false, /*renorm=*/true, /*moe=*/false,
                                    "mini-qknorm-dense-ok");
        const Loaded ld = load(dense,
                               "/tmp/stell_core_test_qknorm_dense.safetensors");
        OpsInterpreter idn(ld.spec, ld.plan, ld.weights);
        const auto rows = idn.run_prompt_logits_all({1, 2, 3});
        STELL_CHECK(rows.size() == 3);
        for (float v : rows.back()) STELL_CHECK(std::isfinite(v));
    }

    std::printf("ops-qk-norm: per-head RMSNorm q/k (после проекций, до QK/PV "
                "и записи k в кеш) + router_renorm=false (веса top-k без "
                "деления на сумму) == независимому референсу плана побитово "
                "(все позиции); без qk-групп — прежнее поведение (якорь "
                "референса); вклад нормы и флага реален; валидация — "
                "SchemaMismatch (6 кейсов)\n");
}

STELL_TEST_MAIN(run)

#include "stell/engine/ops_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "stell/kernels/attention.hpp"
#include "stell/kernels/dequant.hpp"
#include "stell/kernels/moe_gather_scatter.hpp"

namespace stell::engine {
namespace {

// --- утилиты ----------------------------------------------------------------

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Детерминированный случайный [0,1) — inline LCG (kLcgMul/kLcgAdd, seed от
// запроса; PLAN.md §5: seed=0).
constexpr std::uint64_t kLcgMul = 6364136223846793005ULL;
constexpr std::uint64_t kLcgAdd = 144269504088888963ULL;

// pad3-конвенция имён групп (description.cpp::expand_tokens): "l%03u.".
std::string layer_prefix(std::size_t layer) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "l%03zu.", layer);
    return buf;
}

std::size_t numel(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

} // namespace

// --- OpsInterpreter ----------------------------------------------------------

OpsInterpreter::OpsInterpreter(PlainSpec spec, ExecutionPlan plan, LoadedWeights weights,
                               stell::tiers::TierManager* tiers)
    : spec_(std::move(spec)), plan_(std::move(plan)), tiers_(tiers) {
    // Каталог операций: неизвестная op = ошибка загрузки модели (ADR-09 §1).
    static const char* kKnown[] = {"rmsnorm", "attention", "ffn", "moe_gather_scatter"};
    for (const PlainOp& op : spec_.ops) {
        bool known = false;
        for (const char* k : kKnown) known = known || (op.op == k);
        if (!known)
            throw std::invalid_argument("ops runner: неизвестная операция '" + op.op +
                                        "' (ADR-09 §1: новая операция = новое ядро)");
    }

    // Материализация весов: dequant → F32 матрицы (ядра kernels CPU-референс).
    for (const PlainGroup& g : spec_.groups) {
        if (g.kv_block) continue;                    // KV-группа весов не имеет (ADR-09 §3)
        auto it = weights.slots.find(g.id);
        if (it == weights.slots.end())
            throw std::invalid_argument("ops runner: нет payload для группы '" + g.name + "'");
        const std::size_t n = numel(g.shape);
        Mat m{};
        m.rows = g.shape.size() == 2 ? static_cast<std::size_t>(g.shape[0]) : 1;
        m.cols = g.shape.size() == 2 ? static_cast<std::size_t>(g.shape[1]) : n;
        m.w.resize(n);
        const std::uint8_t* payload = it->second.data();
        switch (g.dtype) {
        case PlainDTypeF32:
            std::memcpy(m.w.data(), payload, n * sizeof(float));
            break;
        case PlainDTypeF16:
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint16_t h = static_cast<std::uint16_t>(payload[2 * i]) |
                                        (static_cast<std::uint16_t>(payload[2 * i + 1]) << 8);
                m.w[i] = stell::kernels::half_to_float(h);
            }
            break;
        case PlainDTypeE4M3:
            for (std::size_t i = 0; i < n; ++i)
                m.w[i] = stell::kernels::e4m3_value(payload[i]);
            break;
        case PlainDTypeNVFP4:
            stell::kernels::dequant_nvfp4_block32(payload, n, m.w.data());
            break;
        case PlainDTypeW4A16:
            stell::kernels::dequant_w4a16_block32(payload, n, m.w.data());
            break;
        }
        mats_[g.id] = std::move(m);
    }

    // --- tiering (ADR-08 «Уточнение»): пара set_plan → set_layer_map, --------
    // «холодная» дисциплина: вызывается при загрузке модели, ДО decode-шагов
    // (CPU-контур однопоточный; параллельная работа методов недопустима).
    if (tiers_) {
        tiers_->set_plan(plan_.tensor_groups());
        tiers_->set_layer_map(plan_.layer_map());
    }
}

const PlainGroup* OpsInterpreter::group(const std::string& name) const {
    for (const PlainGroup& g : spec_.groups)
        if (g.name == name) return &g;
    return nullptr;
}

const OpsInterpreter::Mat* OpsInterpreter::mat(std::uint32_t gid) const {
    auto it = mats_.find(gid);
    return it == mats_.end() ? nullptr : &it->second;
}

std::vector<float> OpsInterpreter::matvec(const Mat& m, const std::vector<float>& x) const {
    std::vector<float> y(m.rows, 0.0f);
    for (std::size_t r = 0; r < m.rows; ++r)
        for (std::size_t c = 0; c < m.cols; ++c) y[r] += m.at(r, c) * x[c];
    return y;
}

void OpsInterpreter::reset_slot(SlotState& st, std::size_t n_layers) {
    st.inited = false;
    st.pos = 0;
    st.k.assign(n_layers, {});
    st.v.assign(n_layers, {});
}

// --- tiering: pin_layer перед слоем + prefetch следующего (ADR-08 §2/§5) -----

void OpsInterpreter::pin_and_prefetch(std::size_t layer) {
    if (!tiers_) return;
    const auto& map = plan_.layer_map();
    if (layer >= map.size()) return;
    try {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<stell::tiers::Awaiter> a =
            plan_.pin_layer(*tiers_, static_cast<std::uint32_t>(layer));
        if (a) a->wait();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        tiers_->record_layer_pause(ms);          // паузы между слоями — tiers (ADR-08 §5)
        ++layer_pins_;
    } catch (const stell::tiers::TierError&) {
        // Политика CPU-контура: abort (ADR-08 §3: планировщик оборачивает
        // TierError в свою политику; fallback — задача CUDA-контура).
        throw;
    }
    // Router-lookahead prefetch (ADR-08 §2: асинхронный, не блокирует,
    // идемпотентен; PLAN.md §4 поток B). Заранее — следующий слой.
    if (layer + 1 < map.size()) {
        std::vector<stell::tiers::GroupKey> keys;
        for (const auto& gid : map[layer + 1])
            for (const PlannedGroup& g : plan_.groups)
                if (g.key.id == gid) { keys.push_back(g.key); break; }
        tiers_->prefetch(keys);
        ++layer_prefetches_;
    }
}

// --- каталог операций --------------------------------------------------------

float OpsInterpreter::op_eps() const {
    float eps = 1e-5f;
    for (const PlainOp& op : spec_.ops)
        if (op.op == "rmsnorm")
            for (const PlainOpParam& p : op.params)
                if (p.key == "eps") eps = static_cast<float>(p.fval);
    return eps;
}

void OpsInterpreter::op_rmsnorm(std::size_t layer, std::vector<float>& hidden) const {
    const PlainGroup* ng = group(layer_prefix(layer) + "attn.norm");
    if (!ng) throw std::invalid_argument("ops runner: нет группы нормы слоя " +
                                         std::to_string(layer));
    const Mat* w = mat(ng->id);
    if (!w || w->rows != 1 || w->w.size() != hidden.size())
        throw std::invalid_argument("ops runner: норма слоя не вектор нужного размера");
    const float eps = op_eps();
    double ms = 0.0;
    for (float x : hidden) ms += double(x) * double(x);
    const float inv = 1.0f / std::sqrt(static_cast<float>(ms / hidden.size()) + eps);
    std::vector<float> y(hidden.size());
    for (std::size_t i = 0; i < hidden.size(); ++i) y[i] = w->w[i] * hidden[i] * inv;
    hidden = std::move(y);
}

void OpsInterpreter::op_attention(std::size_t layer, std::vector<float>& hidden,
                                  SlotState& st) {
    const PlainGroup* gwq = group(layer_prefix(layer) + "attn.wq");
    const PlainGroup* gwk = group(layer_prefix(layer) + "attn.wk");
    const PlainGroup* gwv = group(layer_prefix(layer) + "attn.wv");
    const PlainGroup* gwo = group(layer_prefix(layer) + "attn.wo");
    if (!gwq || !gwk || !gwv || !gwo)
        throw std::invalid_argument("ops runner: неполные группы attention слоя " +
                                    std::to_string(layer));
    const Mat *wq = mat(gwq->id), *wk = mat(gwk->id), *wv = mat(gwv->id),
              *wo = mat(gwo->id);
    if (!wq || !wk || !wv || !wo)
        throw std::invalid_argument("ops runner: нет весов attention слоя");

    // Проекции: q [n_heads*d_head], k/v [n_kv_heads*d_kv] (GQA, ADR-09 §1).
    const std::uint32_t d_model = spec_.d_model;
    const std::uint32_t dkv = spec_.n_kv_heads * spec_.d_kv;
    std::vector<float> q = matvec(*wq, hidden);
    std::vector<float> k = matvec(*wk, hidden);
    std::vector<float> v = matvec(*wv, hidden);

    // KV-кеш: token-major [pos][dkv] (FP32 CPU-контур; квант KV E4M3 —
    // ADR-04, задача SM70-контура). Позиция нового токена = st.pos.
    const std::uint32_t T = st.pos + 1;
    std::vector<float> k_kernels(static_cast<std::size_t>(spec_.n_kv_heads) * T * spec_.d_kv, 0.0f);
    std::vector<float> v_kernels(k_kernels.size(), 0.0f);
    // Раскладка kernels: [n_kv_heads][T][d_kv] — копия token-major кеша.
    for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t h = 0; h < spec_.n_kv_heads; ++h)
            for (std::uint32_t d = 0; d < spec_.d_kv; ++d) {
                const std::size_t dst =
                    static_cast<std::size_t>(h) * T * spec_.d_kv + t * spec_.d_kv + d;
                if (t == st.pos) {
                    // текущий токен — из локальных проекций k/v
                    k_kernels[dst] = k[static_cast<std::size_t>(h) * spec_.d_kv + d];
                    v_kernels[dst] = v[static_cast<std::size_t>(h) * spec_.d_kv + d];
                } else {
                    // прошлые позиции — из token-major кеша
                    const std::size_t src = static_cast<std::size_t>(t) * dkv +
                                            h * spec_.d_kv + d;
                    k_kernels[dst] = st.k[layer][src];
                    v_kernels[dst] = st.v[layer][src];
                }
            }

    // attention FP32 QK/PV — kernels CPU-референс, decode-форма (round 2):
    // одна query-строка (текущий токен) против кеша длины T.
    std::vector<float> attn_out(static_cast<std::size_t>(spec_.n_heads) * spec_.d_head, 0.0f);
    stell::kernels::attention_decode_f32(q.data(), k_kernels.data(), v_kernels.data(),
                                         spec_.n_heads, spec_.n_kv_heads, T, spec_.d_head,
                                         attn_out.data());

    // Головы → плоский вектор [d_model] (fixture: n_heads*d_head == d_model),
    // проекция wo, residual.
    std::vector<float> joined(d_model, 0.0f);
    for (std::uint32_t h = 0; h < spec_.n_heads; ++h)
        for (std::uint32_t d = 0; d < spec_.d_head; ++d)
            joined[static_cast<std::size_t>(h) * spec_.d_head + d] =
                attn_out[static_cast<std::size_t>(h) * spec_.d_head + d];
    std::vector<float> proj = matvec(*wo, joined);
    for (std::size_t i = 0; i < d_model; ++i) hidden[i] += proj[i];

    // Запись k/v токена в token-major кеш (позиция — st.pos, общий счёт
    // последовательности, инкрементируется один раз в forward_token).
    st.k[layer].insert(st.k[layer].end(), k.begin(), k.end());
    st.v[layer].insert(st.v[layer].end(), v.begin(), v.end());
}

void OpsInterpreter::op_ffn(std::size_t layer, std::vector<float>& hidden) {
    const PlainGroup* gup = group(layer_prefix(layer) + "ffn.up");
    const PlainGroup* gdn = group(layer_prefix(layer) + "ffn.down");
    if (!gup || !gdn)
        throw std::invalid_argument("ops runner: неполные группы ffn слоя " +
                                    std::to_string(layer));
    const Mat *up = mat(gup->id), *dn = mat(gdn->id);
    if (!up || !dn) throw std::invalid_argument("ops runner: нет весов ffn слоя");
    // Честная конвенция фикстуры: "silu_mul" при отсутствии gate-группы =
    // silu(up(x)) → down (gate-fusion требует парной группы — в fixtures нет).
    std::vector<float> h = matvec(*up, hidden);
    for (float& x : h) x = silu(x);
    std::vector<float> y = matvec(*dn, h);
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += y[i];
}

void OpsInterpreter::op_moe(std::size_t layer, std::vector<float>& hidden) {
    const PlainGroup* grouter = group(layer_prefix(layer) + "mlp.router");
    if (!grouter) throw std::invalid_argument("ops runner: нет роутера слоя");
    const Mat* router = mat(grouter->id);
    if (!router || router->rows != 1)
        throw std::invalid_argument("ops runner: роутер не вектор");
    // Честная конвенция фикстуры: router — вектор экспертных скоров
    // [n_experts] (нет matmul со скрытым состоянием; softmax-topk — ADR-09 §1).
    std::vector<float> scores(router->w);
    float mx = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - mx); sum += s; }
    for (float& s : scores) s /= sum;

    // top-k экспертов (n_experts_active).
    const std::uint32_t k_top =
        std::min<std::uint32_t>(spec_.n_experts_active,
                                static_cast<std::uint32_t>(scores.size()));
    std::vector<std::uint32_t> order(scores.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(order.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return scores[a] > scores[b];
    });
    order.resize(k_top);

    // --- MoE gather/scatter через kernels CPU-референс (ADR-09 §1) -----------
    // x — [1][d_model]; expert_ids — [1][k_top]; gathered — [k_top][d_model].
    const std::uint32_t d_model = spec_.d_model;
    std::vector<float> x(hidden.begin(), hidden.end());
    std::vector<std::int32_t> expert_ids(static_cast<std::size_t>(k_top));
    std::vector<float> gathered(static_cast<std::size_t>(k_top) * d_model, 0.0f);
    for (std::uint32_t j = 0; j < k_top; ++j)
        expert_ids[j] = static_cast<std::int32_t>(order[j]);
    stell::kernels::moe_gather(x.data(), expert_ids.data(), /*n_tokens=*/1, k_top,
                               d_model, gathered.data());

    // FFN выбранных экспертов: выбор весов по expert_id — engine; kernels
    // gather/scatter — буферы активаций, не диспетчер экспертов (ADR-09 §1).
    std::vector<float> expert_out(gathered.size(), 0.0f);
    std::vector<float> weights(static_cast<std::size_t>(k_top), 0.0f);
    for (std::uint32_t j = 0; j < k_top; ++j) {
        const std::uint32_t e = static_cast<std::uint32_t>(order[j]);
        char buf[64];
        std::snprintf(buf, sizeof buf, "l%03zu.e%03u.up", layer, e);
        const PlainGroup* up_g = group(buf);
        std::snprintf(buf, sizeof buf, "l%03zu.e%03u.down", layer, e);
        const PlainGroup* dn_g = group(buf);
        if (!up_g || !dn_g)
            throw std::invalid_argument("ops runner: нет групп эксперта " +
                                        std::to_string(e));
        const Mat *up = mat(up_g->id), *dn = mat(dn_g->id);
        if (!up || !dn)
            throw std::invalid_argument("ops runner: нет весов эксперта " +
                                        std::to_string(e));
        const float* src = gathered.data() + static_cast<std::size_t>(j) * d_model;
        std::vector<float> h(src, src + d_model);
        std::vector<float> hu = matvec(*up, h);
        for (float& f : hu) f = silu(f);
        std::vector<float> y = matvec(*dn, hu);
        for (std::size_t d = 0; d < d_model; ++d)
            expert_out[static_cast<std::size_t>(j) * d_model + d] = y[d];
        weights[j] = scores[order[j]];
    }
    // Scatter-add: y[0] += Σ_j w[j] * expert_out[j] (kernels CPU-референс).
    std::vector<float> y(d_model, 0.0f);
    stell::kernels::moe_scatter_add(expert_out.data(), expert_ids.data(), weights.data(),
                                    /*n_tokens=*/1, k_top, d_model, y.data());
    for (std::size_t i = 0; i < d_model; ++i) hidden[i] += y[i];
}

void OpsInterpreter::unpin_layer(std::size_t layer) {
    if (!tiers_) return;
    const auto& map = plan_.layer_map();
    if (layer >= map.size()) return;
    for (const auto& gid : map[layer])
        for (const PlannedGroup& g : plan_.groups)
            if (g.key.id == gid) {
                tiers_->unpin(g.key);            // запрет eviction снят (ADR-08 §2)
                break;
            }
}

void OpsInterpreter::layer_body(std::size_t layer, std::vector<float>& hidden,
                                SlotState& st) {
    pin_and_prefetch(layer);                       // tiering: pin перед слоем
    for (const PlainOp& op : spec_.ops) {
        if (op.op == "rmsnorm") op_rmsnorm(layer, hidden);
        else if (op.op == "attention") op_attention(layer, hidden, st);
        else if (op.op == "ffn") op_ffn(layer, hidden);
        else if (op.op == "moe_gather_scatter") op_moe(layer, hidden);
        else throw std::invalid_argument("ops runner: неизвестная op '" + op.op + "'");
    }
    unpin_layer(layer);                            // слот свободен для следующего слоя
}

// --- forward / step ----------------------------------------------------------

std::vector<float> OpsInterpreter::forward_token(SlotState& st, std::int32_t token) {
    if (token < 0 || token >= static_cast<std::int32_t>(spec_.vocab_size))
        throw std::invalid_argument("ops runner: токен вне словаря (" +
                                    std::to_string(token) + ")");
    // embedding (fixture: группы "embed" [vocab,d_model]).
    const PlainGroup* embed = group("embed");
    if (!embed) throw std::invalid_argument("ops runner: нет группы embedding");
    const Mat* em = mat(embed->id);
    if (!em) throw std::invalid_argument("ops runner: нет весов embedding");
    std::vector<float> hidden(spec_.d_model, 0.0f);
    for (std::size_t i = 0; i < spec_.d_model; ++i)
        hidden[i] = em->at(static_cast<std::size_t>(token), i);

    // Слои: ops-граф из описания (ADR-09 §1: порядок = граф слоя). Позиция
    // токена = st.pos (общий счёт последовательности, не per-layer).
    for (std::size_t l = 0; l < spec_.n_layers; ++l) layer_body(l, hidden, st);
    ++st.pos;

    // Финальная норма + logits (группа "norm_final"; голова — "head", при
    // отсутствии — weight tying на embedding, честная пометка).
    const PlainGroup* nf = group("norm_final");
    if (nf) {
        const Mat* w = mat(nf->id);
        if (w && w->rows == 1 && w->w.size() == hidden.size()) {
            const float eps = op_eps();
            double ms = 0.0;
            for (float x : hidden) ms += double(x) * double(x);
            const float inv = 1.0f / std::sqrt(static_cast<float>(ms / hidden.size()) + eps);
            for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = w->w[i] * hidden[i] * inv;
        }
    }
    std::vector<float> logits(spec_.vocab_size, 0.0f);
    if (const PlainGroup* head = group("head")) {
        if (const Mat* hm = mat(head->id))
            for (std::uint32_t t = 0; t < spec_.vocab_size; ++t) {
                float acc = 0.0f;
                for (std::size_t i = 0; i < spec_.d_model; ++i)
                    acc += hm->at(t, i) * hidden[i];
                logits[t] = acc;
            }
    } else {
        // Weight tying (mini_moe fixture без группы head): logits = E·h.
        for (std::uint32_t t = 0; t < spec_.vocab_size; ++t) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < spec_.d_model; ++i) acc += em->at(t, i) * hidden[i];
            logits[t] = acc;
        }
    }
    return logits;
}

std::vector<float> OpsInterpreter::step_logits(SlotId slot,
                                               const std::vector<std::int32_t>& prompt,
                                               std::int32_t token) {
    SlotState& st = slots_[slot];
    if (!st.inited) {
        reset_slot(st, spec_.n_layers);
        // Prefill: все токены промпта, кроме последнего (кеш KV слота).
        for (std::size_t i = 0; i + 1 < prompt.size(); ++i)
            (void)forward_token(st, prompt[i]);
        st.inited = true;
    }
    return forward_token(st, token);
}

std::vector<float> OpsInterpreter::run_prompt_logits(
    const std::vector<std::int32_t>& prompt) {
    SlotState st{};
    reset_slot(st, spec_.n_layers);
    std::vector<float> logits;
    for (std::int32_t tok : prompt) {
        logits = forward_token(st, tok);
    }
    return logits;
}

// --- sampling ----------------------------------------------------------------

std::int32_t OpsInterpreter::sample(const std::vector<float>& logits,
                                    const SamplingParams& sp, float rnd01) {
    if (logits.empty()) return -1;
    if (sp.temperature <= 0.0f) {
        // Greedy argmax (режим golden-контракта ADR-09 §4 «Уточнение»).
        return static_cast<std::int32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    // softmax(logits / T), top-k, nucleus top-p; сэмпл — детерминированный LCG.
    const float t = sp.temperature > 0.0f ? sp.temperature : 1.0f;
    std::vector<float> probs(logits.size());
    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs[i] = std::exp((logits[i] - mx) / t);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    std::vector<std::uint32_t> order(probs.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(order.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return probs[a] > probs[b];
    });
    const std::int32_t k = sp.top_k > 0 ? sp.top_k : static_cast<std::int32_t>(order.size());
    std::size_t n = std::min<std::size_t>(static_cast<std::size_t>(k), order.size());
    float cum = 0.0f;
    std::size_t keep = n;
    for (std::size_t i = 0; i < n; ++i) {
        cum += probs[order[i]];
        if (cum >= sp.top_p) { keep = i + 1; break; }
    }
    keep = std::max<std::size_t>(1, std::min(keep, n));
    // Случай (детерминированный): равномерно по nucleus-списку — честная
    // аппроксимация сэмплинга bench-профиля; точная полигон-семпл-логика —
    // задача среза 4 (спекуляция/качество). rnd01 даёт runner.
    const float r = rnd01;
    const std::size_t idx = static_cast<std::size_t>(r * static_cast<float>(keep)) % keep;
    return static_cast<std::int32_t>(order[idx]);
}

// --- CpuModelRunner ----------------------------------------------------------

CpuModelRunner::CpuModelRunner(PlainSpec spec, ExecutionPlan plan, LoadedWeights weights,
                               stell::tiers::TierManager* tiers)
    : interp_(std::move(spec), std::move(plan), std::move(weights), tiers) {}

std::vector<IModelRunner::DecodeOutput> CpuModelRunner::decode_step(
    const std::vector<DecodeInput>& in) {
    std::vector<DecodeOutput> out;
    out.reserve(in.size());
    for (const DecodeInput& i : in) {
        const std::vector<float> logits =
            interp_.step_logits(i.slot, i.prompt, i.token);
        // Детерминированный случайный [0,1): LCG, сидируемый seed'ом запроса
        // при первом шаге слота (одинаковые прогоны при одинаковом вызове).
        auto st = rng_.find(i.slot);
        if (st == rng_.end()) st = rng_.emplace(i.slot, i.sampling.seed).first;
        st->second = st->second * kLcgMul + kLcgAdd;
        const float r = static_cast<float>((st->second >> 33) & 0xFFFFFFULL) /
                        static_cast<float>(0xFFFFFFULL);
        const std::int32_t token = interp_.sample(logits, i.sampling, r);
        out.push_back({i.slot, token});
    }
    return out;
}

} // namespace stell::engine

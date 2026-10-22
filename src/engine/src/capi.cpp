// Реализация C ABI (ADR-11 §1) поверх Scheduler (round 1 CPU-контур; round 2 —
// вклейка реального CPU-раннера: каталог операций + лоадеры models + tiers).
// runner:"demo" — детерминированный встроенный раннер для CPU CI (round 1);
// runner:"ops" (round 2) — реальное CPU-исполнение модели (срез 1, A×D):
//   cfg: {"runner":"ops",
//         "engine":{"max_slots":16,"batch_step_max":16},
//         "model":{"description":"<path>","weights":"<uri>",
//                  "weights_format":"safetensors|gguf|nvfp4w4a16|stelltier",
//                  "tokenizer":"byte",
//                  "tiering":{"vram_slots":4,"ram_capacity":...,
//                             "nvme_layout":"<path>"}   // опционально
//          }}
// Честные пометки: tokenizer — byte-level fallback (прод-токенизатор —
// отдельная задача); tiering в CPU-контуре даёт residency/prefetch/паузы,
// сами веса материалены из лоадера (см. ops_runner.hpp).
#include "stell/engine/capi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "stell/engine/json.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/plain_spec.hpp"
#include "stell/engine/scheduler.hpp"
#include "stell/engine/spec_decode.hpp"
#include "stell/engine/tokenizer.hpp"

using namespace stell::engine;

namespace {

class DemoRunner : public IModelRunner {
public:
    std::vector<DecodeOutput> decode_step(const std::vector<DecodeInput>& in) override {
        std::vector<DecodeOutput> out;
        for (const auto& i : in) {
            const int c = counter_[i.slot]++;
            out.push_back({i.slot, c});        // детерминированно: 0,1,2,...
        }
        return out;
    }
    std::map<SlotId, int> counter_;            // детерминизм по слоту
};

PlainWeightsFormat weights_format_from_name(const std::string& name) {
    if (name == "safetensors") return PlainWeightsSafetensors;
    if (name == "gguf") return PlainWeightsGGUF;
    if (name == "nvfp4w4a16") return PlainWeightsNVFP4W4A16;
    if (name == "stelltier") return PlainWeightsStellTier;
    throw std::runtime_error("capi: неизвестный weights_format '" + name + "'");
}

// round 3 [R5] (planner Q5, ADR-11 §4): статистика вызовов stell_logits для
// снапшота /metrics. Семантика счётчиков — по словарю секции serve
// (python/stell_serve/metrics.py: record_request только при успехе,
// record_error при ошибке): requests_total — число УСПЕШНЫХ вызовов
// (rc==0; ошибочный вызов не даёт payload и в requests не попадает — иначе
// errors_total дублировал бы те же вызовы), errors_total — число ошибочных
// (rc==-1). Сумма = полное число вызовов. last_n_vocab / last_positions —
// параметры последнего успешного вызова (ошибочный вызов валидного ответа
// не имеет и статистику не двигает); до первого успеха — «нет данных».
// last_positions хранится как код (0=нет данных, 1=last, 2=all) — потокобез-
// опасно без мьютекса; в JSON выдаётся строкой в словаре ABI-запроса
// stell_logits (ADR-11 §1: "positions":"last"|"all") — стиль ADR-11: строковые
// enum'ы в наружном JSON ("state":"serving", "kv_quant":"F16").
struct LogitsStats {
    std::atomic<std::uint64_t> requests{0};
    std::atomic<std::uint64_t> errors{0};
    std::atomic<std::int64_t> last_n_vocab{-1};   // -1 — успешных вызовов не было
    std::atomic<int> last_positions{0};           // 0 — вызовов не было
};

struct EngineCtx {
    FixedSlotPool* pool = nullptr;
    Scheduler* sched = nullptr;
    std::unique_ptr<IModelRunner> runner;          // DemoRunner | CpuModelRunner
    std::unique_ptr<stell::tiers::TierManager> tiers;
    ByteTokenizer tok;
    std::string last_error;
    LogitsStats lstats;                            // round 3 [R5]: /metrics "logits"
    // round 8 (срез 4): спекулятивный контур на реальном runner'е
    // (spec_decode.hpp): decoder — только при runner=ops и конфиге "spec"
    // (drafter — отдельная модель-плагин, свой ops-раннер); slot — вне слотов
    // планировщика (резервный id). Телеметрия — decoder->telemetry().
    std::unique_ptr<IModelRunner> spec_draft_runner;
    std::unique_ptr<SpeculativeDecoder> spec;
};

    // round 8 (срез 4): сборка ops-раннера из секции описания модели (цель и
    // драфтер — один путь; у драфтера tiers нет — маленькая модель).
    std::unique_ptr<CpuModelRunner> make_ops_runner(const Json& model,
                                                    std::unique_ptr<stell::tiers::TierManager>& tiers_owner) {
        const std::string desc_path =
            model.find("description") ? model.find("description")->as_string("") : std::string();
        const std::string weights_uri =
            model.find("weights") ? model.find("weights")->as_string("") : std::string();
        const std::string fmt_name =
            model.find("weights_format") ? model.find("weights_format")->as_string("safetensors")
                                         : std::string("safetensors");
        if (desc_path.empty() || weights_uri.empty())
            throw std::runtime_error(
                "capi: секция модели требует description и weights");
        const PlainSpec spec = load_plain_spec(desc_path);
        const LoadedWeights weights =
            load_plain_weights(weights_uri, spec, weights_format_from_name(fmt_name));
        const ExecutionPlan plan = build_execution_plan(spec);
        stell::tiers::TierManager* tiers = nullptr;
        if (const Json* t = model.find("tiering")) {
            stell::tiers::TierConfig tc{};
            if (const Json* v = t->find("vram_slots"))
                tc.vram_slots = static_cast<std::size_t>(v->as_int(4));
            if (const Json* v = t->find("ram_capacity"))
                tc.ram_capacity = static_cast<std::size_t>(v->as_int(0));
            if (const Json* v = t->find("nvme_layout"))
                tc.nvme_layout = v->as_string("");
            tiers_owner = std::make_unique<stell::tiers::TierManager>(tc);
            tiers = tiers_owner.get();
        }
        return std::make_unique<CpuModelRunner>(spec, plan, weights, tiers);
    }
} // namespace

extern "C" {

int stell_engine_create(const char* cfg_json, stell_engine_t** out) {
    if (!cfg_json || !out) return -1;
    *out = nullptr;
    try {
        Json cfg = json_parse(cfg_json);
        auto* ctx = new EngineCtx{};

        std::uint32_t max_slots = 16, batch_step_max = 16;
        if (const Json* e = cfg.find("engine")) {
            if (const Json* v = e->find("max_slots"))
                max_slots = static_cast<std::uint32_t>(v->as_int(16));
            if (const Json* v = e->find("batch_step_max"))
                batch_step_max = static_cast<std::uint32_t>(v->as_int(16));
        }
        ctx->pool = new FixedSlotPool(max_slots, 4096);

        std::string runner_name = "demo";
        if (const Json* v = cfg.find("runner")) runner_name = v->as_string("demo");

        if (runner_name == "ops") {
            // Round 2: реальное CPU-исполнение модели (срез 1 A×D, ADR-09/03).
            const Json* model = cfg.find("model");
            if (!model)
                throw std::runtime_error("capi: runner=ops требует секцию model");

            ctx->runner = make_ops_runner(*model, ctx->tiers);
            ctx->sched = new Scheduler(*ctx->runner, *ctx->pool);
            ctx->sched->set_tokenizer(&ctx->tok);          // byte-level fallback

            // round 8 (срез 4): спекулятивный контур — конфиг "spec"
            // (drafter — отдельная модель-плагин, свой ops-раннер; slot —
            // резервный, вне слотов планировщика). Честно: без секции "spec"
            // контура нет (features.spec_draft=false в снапшоте).
            if (const Json* sp = cfg.find("spec")) {
                const bool sp_enabled =
                    sp->find("enabled") ? (sp->find("enabled")->t == Json::T::Bool ? sp->find("enabled")->boolean : false) : false;
                if (sp_enabled) {
                    const Json* dm = sp->find("draft_model");
                    if (!dm)
                        throw std::runtime_error(
                            "capi: spec.enabled требует spec.draft_model "
                            "(драфтер — отдельная модель-плагин)");
                    if (!dynamic_cast<CpuModelRunner*>(ctx->runner.get()))
                        throw std::runtime_error(
                            "capi: spec.enabled требует runner=ops (верификация — "
                            "FP32-логиты цели через ops-раннер)");
                    ctx->spec_draft_runner = make_ops_runner(*dm, ctx->tiers);
                    auto* draft_cpu =
                        dynamic_cast<CpuModelRunner*>(ctx->spec_draft_runner.get());
                    auto* target_cpu = dynamic_cast<CpuModelRunner*>(ctx->runner.get());
                    SpeculativeDecoder::Config sc{};
                    if (const Json* v = sp->find("n_draft"))
                        sc.n_draft = static_cast<std::uint32_t>(v->as_int(4));
                    if (const Json* v = sp->find("accept_target"))
                        sc.accept_target = v->t == Json::T::Number ? v->num : 0.5;
                    if (const Json* v = sp->find("window"))
                        sc.window = static_cast<std::uint32_t>(v->as_int(32));
                    if (const Json* v = sp->find("warmup_rounds"))
                        sc.warmup_rounds = static_cast<std::uint32_t>(v->as_int(4));
                    sc.enabled = true;
                    // Драфтер-плагин: greedy-продолжение драфт-модели по полному
                    // контексту (stateless пересчёт — честное ограничение round 8;
                    // stateful-драфтер с KV-rollback — отдельная задача).
                    OpsInterpreter* draft_interp = &draft_cpu->interpreter();
                    const SlotId kDraftSlot = static_cast<SlotId>(0xFFFFFFF0);
                    SpeculativeDecoder::DrafterFn drafter =
                        [draft_interp, kDraftSlot](
                            const std::vector<std::int32_t>& context,
                            std::size_t n_draft) {
                            if (context.empty() || n_draft == 0)
                                return std::vector<std::int32_t>{};
                            draft_interp->release_slot(kDraftSlot);
                            std::vector<float> lg = draft_interp->step_logits(
                                kDraftSlot, context, context.back());
                            std::vector<std::int32_t> out;
                            for (std::size_t i = 0; i < n_draft; ++i) {
                                const std::int32_t c = static_cast<std::int32_t>(
                                    std::max_element(lg.begin(), lg.end()) - lg.begin());
                                out.push_back(c);
                                lg = draft_interp->step_logits(kDraftSlot, {}, c);
                            }
                            return out;
                        };
                    ctx->spec = std::make_unique<SpeculativeDecoder>(
                        target_cpu->interpreter(), drafter, sc,
                        static_cast<SlotId>(0xFFFFFFFE));
                }
            }
        } else {
            ctx->runner = std::make_unique<DemoRunner>();
            Scheduler::Config cfg2{};
            cfg2.batch_step_max = batch_step_max;
            ctx->sched = new Scheduler(*ctx->runner, *ctx->pool, nullptr, cfg2);
            // demo: токенизатора нет — text пуст (round-1 поведение, честно).
        }
        *out = reinterpret_cast<stell_engine_t*>(ctx);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[engine.capi] create: %s\n", ex.what());
        return -1;
    }
}

void stell_engine_destroy(stell_engine_t* e) {
    if (!e) return;
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    delete ctx->sched;
    ctx->spec.reset();
    ctx->spec_draft_runner.reset();
    ctx->runner.reset();
    ctx->tiers.reset();
    delete ctx->pool;
    delete ctx;
}

uint64_t stell_submit(stell_engine_t* e, const char* request_json, char** msg_out) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    auto fail = [&](const char* what) -> uint64_t {
        if (msg_out) *msg_out = strdup(what);
        return 0;
    };
    if (!ctx || !request_json) return fail("bad arguments");
    try {
        Json req = json_parse(request_json);
        SubmitRequest r{};
        if (const Json* s = req.find("sampling")) {
            if (const Json* v = s->find("max_tokens"))
                r.sampling.max_tokens = static_cast<std::int32_t>(v->as_int(256));
            if (const Json* v = s->find("temperature"))
                r.sampling.temperature = static_cast<float>(v->num);
            if (const Json* v = s->find("top_p"))
                r.sampling.top_p = static_cast<float>(v->num);
            if (const Json* v = s->find("top_k"))
                r.sampling.top_k = static_cast<std::int32_t>(v->as_int(20));
            if (const Json* v = s->find("seed"))
                r.sampling.seed = static_cast<std::uint32_t>(v->as_int(0));
        }
        if (const Json* v = req.find("prompt_tokens")) {
            if (v->t != Json::T::Array) return fail("prompt_tokens: not array");
            for (const Json& t : v->arr)
                r.prompt_tokens.push_back(static_cast<std::int32_t>(t.as_int()));
        }
        if (const Json* v = req.find("eos_token"))
            r.eos_token = static_cast<std::int32_t>(v->as_int(2));

        if (r.prompt_tokens.empty()) return fail("empty prompt_tokens");

        const SessionId id = ctx->sched->submit(r);
        if (id == 0) return fail("no free slot (slot pool exhausted)");
        if (msg_out) *msg_out = strdup("");
        return id;
    } catch (const std::exception& ex) {
        return fail(ex.what());
    }
}

// событие → JSON-строка (ADR-11 §3); text экранируется минимально (ASCII-safe:
// byte-fallback токенизатор даёт printable-байты; escape-синтаксис —
// ограничение парсера serve-стороны, см. engine/json.hpp).
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04X", c);
            out += buf;
        } else out += c;
    }
    return out;
}

static std::string event_to_json(const Event& ev) {
    char buf[512];
    switch (ev.type) {
    case Event::Type::Token:
        std::snprintf(buf, sizeof buf,
                      "{\"seq\":%llu,\"type\":\"token\",\"token_id\":%d,"
                      "\"text\":\"%s\"}",
                      (unsigned long long)ev.seq, ev.token_id,
                      json_escape(ev.text).c_str());
        break;
    case Event::Type::Finish:
        std::snprintf(buf, sizeof buf,
                      "{\"seq\":%llu,\"type\":\"finish\",\"reason\":\"%s\","
                      "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%u}}",
                      (unsigned long long)ev.seq, ev.finish_reason.c_str(),
                      ev.prompt_tokens, ev.completion_tokens);
        break;
    case Event::Type::Error:
        std::snprintf(buf, sizeof buf,
                      "{\"seq\":%llu,\"type\":\"error\",\"code\":\"%s\","
                      "\"message\":\"%s\"}",
                      (unsigned long long)ev.seq, ev.error_code.c_str(),
                      ev.error_message.c_str());
        break;
    }
    return std::string(buf);
}

int stell_poll(stell_engine_t* e, uint64_t session_id, char** out_json) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    if (!ctx || !out_json) return -1;
    const std::vector<Event> evs = ctx->sched->poll(session_id);
    std::string arr = "[";
    for (std::size_t i = 0; i < evs.size(); ++i) {
        if (i) arr += ",";
        arr += event_to_json(evs[i]);
    }
    arr += "]";
    *out_json = strdup(arr.c_str());
    return static_cast<int>(evs.size());
}

int stell_cancel(stell_engine_t* e, uint64_t session_id) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    if (!ctx) return -1;
    // Конвенция ADR-11 §1: 0 = успех, -1 = ошибка (сессия не найдена/уже
    // закрыта). Внутренний Scheduler::cancel → bool маппится здесь (round 2:
    // было 1/0 bool — случайное отступление от конвенции модуля).
    return ctx->sched->cancel(session_id) ? 0 : -1;
}

void stell_step(stell_engine_t* e) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    if (!ctx) return;
    ctx->sched->step();
}

int stell_metrics(stell_engine_t* e, char** out_json) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    if (!ctx || !out_json) return -1;
    // минимальный снапшот (ADR-11 §4): engine.sessions_active/slots_busy;
    // round 3 [R5]: секция "logits" — строгий формат (planner Q5): статистика
    // вызовов stell_logits БЕЗ payload (массивы логитов в /metrics не выдаются
    // никогда — размер). До первого успешного вызова last_* — null («честно:
    // нет данных», как в serve-секции).
    // round 8 (срез 4): spec-поля (ADR-11 §4, bench harness.py их читает):
    //   features.spec_draft — контур сконфигурирован и включён арбитром;
    //   latency.tok_s.spec_decode — tok/s спек-декода по телеметрии (0 —
    //   контур не вооружён/не использовался — честный ноль, не пропуск);
    //   spec.accept_rate/n_draft/rounds — телеметрия драфтера.
    // round 9 (ADR-13 [R1]): «configured ≠ executed» (ADR-13 [R1]) —
    //   features.spec_draft_configured / _executed / _reason различают
    //   настроенный путь, фактическое исполнение и причину неактивности.
    //   Обратная совместимость: features.spec_draft остаётся bool (его читают
    //   bench harness.py и python control-plane).
    const auto pos = ctx->lstats.last_positions.load();
    const bool spec_on = ctx->spec && ctx->spec->enabled();
    const auto& st = ctx->spec ? ctx->spec->telemetry()
                               : SpeculativeDecoder::Telemetry{};
    // configured ≠ executed (ADR-13 [R1]): configured — контур собран и
    // подключён (ctx->spec); executed — был хотя бы один состоявшийся
    // спек-раунд. reason — почему фича не активна ПРЯМО СЕЙЧАС.
    const bool spec_configured = ctx->spec != nullptr;
    const bool spec_executed = spec_configured && st.rounds > 0;
    const char* spec_reason =
        !spec_configured ? "not_configured"
        : ctx->spec->auto_disabled() ? "arbiter_disabled"
        : spec_on ? "enabled"
                  : "disabled_by_config";
    const double spec_tok_s = st.wall_s > 0.0
                                  ? static_cast<double>(st.tokens) / st.wall_s
                                  : 0.0;
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "{\"engine\":{\"state\":\"serving\",\"sessions_active\":%zu,"
                  "\"slots_busy\":%zu,\"slots_total\":%u,\"batch_size_max\":16,"
                  // [R3] ADR-13: шаги, прижатые mixed-prefill бюджетом
                  // (0 при default без лимита — честный ноль, не пропуск).
                  "\"prefill_budget_deferrals\":%llu},"
                  "\"logits\":{\"requests_total\":%llu,\"errors_total\":%llu,"
                  "\"last_n_vocab\":%s,\"last_positions\":%s},"
                  "\"features\":{\"spec_draft\":%s,"
                  "\"spec_draft_configured\":%s,\"spec_draft_executed\":%s,"
                  "\"spec_draft_reason\":\"%s\"},"
                  "\"latency\":{\"tok_s\":{\"spec_decode\":%.3f}},"
                  "\"spec\":{\"accept_rate\":%.4f,\"n_draft\":%u,"
                  "\"n_draft_effective\":%u,\"rounds\":%llu}}",
                  ctx->sched->active(), ctx->pool->busy(),
                  (unsigned)ctx->pool->capacity(),
                  (unsigned long long)ctx->sched->prefill_deferrals(),
                  (unsigned long long)ctx->lstats.requests.load(),
                  (unsigned long long)ctx->lstats.errors.load(),
                  // -1 → null (успешных вызовов ещё не было)
                  [&]{ const auto v = ctx->lstats.last_n_vocab.load();
                       return v < 0 ? std::string("null") : std::to_string(v); }().c_str(),
                  pos == 2 ? "\"all\"" : pos == 1 ? "\"last\"" : "null",
                  spec_on ? "true" : "false",
                  spec_configured ? "true" : "false",
                  spec_executed ? "true" : "false",
                  spec_reason,
                  spec_tok_s,
                  ctx->spec ? ctx->spec->acceptance_rate() : 0.0,
                  ctx->spec ? ctx->spec->config().n_draft : 0u,
                  ctx->spec ? ctx->spec->effective_n_draft() : 0u,
                  (unsigned long long)st.rounds);
    *out_json = strdup(buf);
    return 0;
}

// --- round 8 (срез 4): спекулятивный контур на реальном runner'е ------------
// ABI-расширение ADR-11 §1 (по образцу stell_logits round 2): новый символ,
// существующие не меняются. Контракт — spec_decode.hpp; драфтер — модель-плагин
// из конфига "spec"; верификация — FP32-логиты цели; последовательность на
// выходе побитово == plain greedy (спекуляция без потери качества).
int stell_spec_step(stell_engine_t* e, const char* request_json, char** out_json) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    auto fail = [&](const std::string& what) -> int {
        if (out_json)
            *out_json = strdup(("{\"error\":\"" + what + "\"}").c_str());
        return -1;
    };
    if (!ctx || !request_json || !out_json) return -1;
    if (!ctx->spec)
        return fail("stell_spec_step: спекулятивный контур не сконфигурирован "
                    "(нужна секция spec с draft_model, runner=ops)");
    try {
        Json req = json_parse(request_json);
        std::vector<std::int32_t> prompt;
        if (const Json* v = req.find("prompt_tokens")) {
            if (v->t != Json::T::Array) return fail("prompt_tokens: not array");
            for (const Json& t : v->arr)
                prompt.push_back(static_cast<std::int32_t>(t.as_int()));
        }
        if (prompt.empty()) return fail("empty prompt_tokens");
        std::size_t n_steps = 16;
        if (const Json* v = req.find("n_steps"))
            n_steps = static_cast<std::size_t>(v->as_int(16));

        const auto before = ctx->spec->telemetry();
        const std::vector<std::int32_t> tokens = ctx->spec->run(prompt, n_steps);
        const auto after = ctx->spec->telemetry();

        std::string arr;
        for (std::size_t i = 0; i < tokens.size(); ++i)
            arr += (i ? "," : "") + std::to_string(tokens[i]);
        const double tok_s = after.wall_s > 0.0
                                 ? static_cast<double>(after.tokens) / after.wall_s
                                 : 0.0;
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "{\"tokens\":[%s],\"accepted\":%llu,\"proposed\":%llu,"
                      "\"rounds\":%llu,\"emitted\":%llu,\"wall_s\":%.6f,"
                      "\"tok_s\":%.3f,\"enabled\":%s}",
                      arr.c_str(),
                      (unsigned long long)(after.accepted - before.accepted),
                      (unsigned long long)(after.proposed - before.proposed),
                      (unsigned long long)(after.rounds - before.rounds),
                      (unsigned long long)tokens.size(),
                      after.wall_s - before.wall_s,
                      tok_s,
                      ctx->spec->enabled() ? "true" : "false");
        *out_json = strdup(buf);
        return 0;
    } catch (const std::exception& ex) {
        return fail(std::string("stell_spec_step: ") + ex.what());
    }
}

// --- round 2: stell_logits (ADR-11 §1, новое ABI-расширение) -----------------
// Логиты финального слоя (до softmax) на готовых token id; runner:"ops"
// (OpsInterpreter) исполняет prefill всех prompt_tokens на свежем состоянии;
// "positions":"last" — только последняя позиция, "all" — все (teacher-
// forcing за один вызов). Сериализация float — %.9g (float32 ≈ 9 значащих
// цифр — round-trip без потери; json_dump использует %g — 6 цифр, мало для
// quality-гейтов с rel ≤ 1e-6).
int stell_logits(stell_engine_t* e, const char* request_json, char** out_json) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    auto fail = [&](const std::string& what) -> int {
        ctx->lstats.errors++;          // [R5]: ошибочный вызов (rc==-1)
        if (out_json)
            *out_json = strdup(("{\"error\":\"" + json_escape(what) + "\"}").c_str());
        return -1;
    };
    if (!ctx) return -1;
    if (!request_json || !out_json) { ctx->lstats.errors++; return -1; }
    try {
        Json req = json_parse(request_json);
        std::vector<std::int32_t> prompt;
        if (const Json* v = req.find("prompt_tokens")) {
            if (v->t != Json::T::Array) return fail("prompt_tokens: not array");
            for (const Json& t : v->arr)
                prompt.push_back(static_cast<std::int32_t>(t.as_int()));
        }
        if (prompt.empty()) return fail("empty prompt_tokens");
        std::string positions = "last";
        if (const Json* v = req.find("positions"))
            positions = v->as_string("last");
        if (positions != "last" && positions != "all")
            return fail("positions: допустимо last или all");

        // Логиты даёт только ops-раннер (demo — детерминированные токены без
        // модели; честный отказ, не заглушка).
        auto* cpu = dynamic_cast<CpuModelRunner*>(ctx->runner.get());
        if (!cpu) return fail("stell_logits: runner не поддерживает логиты (нужен ops)");
        OpsInterpreter& interp = cpu->interpreter();

        std::vector<std::vector<float>> rows;
        if (positions == "all") {
            rows = interp.run_prompt_logits_all(prompt);
        } else {
            rows.push_back(interp.run_prompt_logits(prompt));
        }
        const unsigned n_vocab = static_cast<unsigned>(interp.spec().vocab_size);

        std::string out = "{\"logits\":[";
        char buf[40];
        for (std::size_t p = 0; p < rows.size(); ++p) {
            if (p) out += ',';
            out += '[';
            for (std::size_t i = 0; i < rows[p].size(); ++i) {
                if (i) out += ',';
                std::snprintf(buf, sizeof buf, "%.9g",
                              static_cast<double>(rows[p][i]));
                out += buf;
            }
            out += ']';
        }
        std::snprintf(buf, sizeof buf, "],\"n_positions\":%zu,\"n_vocab\":%u}",
                      rows.size(), n_vocab);
        out += buf;
        *out_json = strdup(out.c_str());
        // [R5]: успешный вызов — обновляем статистику для /metrics
        ctx->lstats.requests++;
        ctx->lstats.last_n_vocab = n_vocab;
        ctx->lstats.last_positions = (positions == "all") ? 2 : 1;
        return 0;
    } catch (const std::exception& ex) {
        return fail(ex.what());
    }
}

} // extern "C"

// Реализация C ABI (ADR-11 §1) поверх Scheduler (round 1, CPU-контур).
// runner:"demo" — детерминированный встроенный раннер для CPU CI; реальный
// runner приходит от models/kernels при вклейке (потоки C/D/A-срезы).
#include "stell/engine/capi.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "stell/engine/json.hpp"
#include "stell/engine/scheduler.hpp"

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

struct EngineCtx {
    FixedSlotPool* pool = nullptr;
    Scheduler* sched = nullptr;
    DemoRunner* runner = nullptr;
    std::string last_error;
};

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
        ctx->runner = new DemoRunner{};
        Scheduler::Config cfg2{};
        cfg2.batch_step_max = batch_step_max;
        ctx->sched = new Scheduler(*ctx->runner, *ctx->pool, nullptr, cfg2);
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
    delete ctx->runner;
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

// событие → JSON-строка (ADR-11 §3)
static std::string event_to_json(const Event& ev) {
    char buf[256];
    switch (ev.type) {
    case Event::Type::Token:
        std::snprintf(buf, sizeof buf,
                      "{\"seq\":%llu,\"type\":\"token\",\"token_id\":%d}",
                      (unsigned long long)ev.seq, ev.token_id);
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
    return ctx->sched->cancel(session_id) ? 1 : 0;
}

void stell_step(stell_engine_t* e) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    if (!ctx) return;
    ctx->sched->step();
}

int stell_metrics(stell_engine_t* e, char** out_json) {
    auto* ctx = reinterpret_cast<EngineCtx*>(e);
    if (!ctx || !out_json) return -1;
    // минимальный снапшот (ADR-11 §4): engine.sessions_active/slots_busy
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "{\"engine\":{\"state\":\"serving\",\"sessions_active\":%zu,"
                  "\"slots_busy\":%zu,\"slots_total\":%u,\"batch_size_max\":16}}",
                  ctx->sched->active(), ctx->pool->busy(),
                  (unsigned)ctx->pool->capacity());
    *out_json = strdup(buf);
    return 0;
}

} // extern "C"

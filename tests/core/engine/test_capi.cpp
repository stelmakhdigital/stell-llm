// Стык engine ↔ serve (ADR-11): C ABI — create/submit/poll/cancel/metrics.
#include "test_util.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "stell/engine/capi.h"

namespace {

void free_str(char* p) { std::free(p); }

std::string poll_json(stell_engine_t* e, uint64_t sid, int* n_out) {
    char* out = nullptr;
    const int n = stell_poll(e, sid, &out);
    if (n_out) *n_out = n;
    std::string s(out ? out : "");
    if (out) free_str(out);
    return s;
}

} // namespace

void run() {
    // --- create: минимальный конфиг (round 1 CPU-контур) ---------------------
    stell_engine_t* e = nullptr;
    char* msg = nullptr;
    const int rc = stell_engine_create(
        "{\"runner\":\"demo\",\"engine\":{\"max_slots\":16,\"batch_step_max\":16}}",
        &e);
    STELL_CHECK_MSG(rc == 0, "create ok");
    STELL_CHECK(e != nullptr);

    // --- submit + poll: события token → finish(length) ----------------------
    uint64_t s1 = stell_submit(e,
        "{\"sampling\":{\"max_tokens\":3,\"temperature\":1.0,\"top_p\":0.95,"
        "\"top_k\":20,\"seed\":0},\"prompt_tokens\":[1,2,3],\"eos_token\":1000000}",
        &msg);
    STELL_CHECK_MSG(s1 != 0, "submit ok");
    if (msg) free_str(msg);

    int n = 0;
    std::string evs;

    stell_step(e);                      // шаг 1: token(0)
    evs = poll_json(e, s1, &n);
    STELL_CHECK(n == 1);
    STELL_CHECK_MSG(evs.find("\"type\":\"token\"") != std::string::npos, "token event");
    STELL_CHECK(evs.find("\"seq\":1") != std::string::npos);

    stell_step(e);                      // шаг 2: token(1)
    poll_json(e, s1, &n);

    stell_step(e);                      // шаг 3: token(2) + finish(length)
    evs = poll_json(e, s1, &n);
    STELL_CHECK(n == 2);
    STELL_CHECK_MSG(evs.find("\"type\":\"finish\"") != std::string::npos, "finish event");
    STELL_CHECK(evs.find("\"reason\":\"length\"") != std::string::npos);
    STELL_CHECK(evs.find("\"completion_tokens\":3") != std::string::npos);

    evs = poll_json(e, s1, &n);         // запись сессии удалена после выкачивания
    STELL_CHECK(n == 0 && evs == "[]");

    // --- исчерпание слотов: submit → 0 + msg --------------------------------
    // (16 сессий с max_tokens=100 — все слоты заняты; шаги не ведём)
    std::vector<uint64_t> ids;
    for (int i = 0; i < 16; ++i) {
        const uint64_t id = stell_submit(e,
            "{\"sampling\":{\"max_tokens\":100},\"prompt_tokens\":[9],\"eos_token\":1000000}",
            nullptr);
        STELL_CHECK(id != 0);
        ids.push_back(id);
    }
    {
        char* m = nullptr;
        const uint64_t id = stell_submit(e,
            "{\"sampling\":{\"max_tokens\":1},\"prompt_tokens\":[1],\"eos_token\":-1}",
            &m);
        STELL_CHECK_MSG(id == 0, "нет слота → 0");
        STELL_CHECK(m != nullptr && std::strlen(m) > 0);
        if (m) free_str(m);
    }

    // --- metrics -------------------------------------------------------------
    char* mj = nullptr;
    STELL_CHECK(stell_metrics(e, &mj) == 0);
    const std::string metrics(mj);
    free_str(mj);
    STELL_CHECK(metrics.find("\"slots_total\":16") != std::string::npos);
    STELL_CHECK(metrics.find("\"slots_busy\":16") != std::string::npos);

    // освобождаем слот: cancel одной из 16 сессий
    STELL_CHECK(stell_cancel(e, ids.back()) == 1);
    (void)poll_json(e, ids.back(), &n);

    // --- cancel + finish(cancelled) ------------------------------------------
    uint64_t s2 = stell_submit(e,
        "{\"sampling\":{\"max_tokens\":10},\"prompt_tokens\":[4,5],\"eos_token\":-1}",
        nullptr);
    STELL_CHECK(s2 != 0);
    STELL_CHECK(stell_cancel(e, s2) == 1);
    evs = poll_json(e, s2, &n);
    STELL_CHECK_MSG(evs.find("\"reason\":\"cancelled\"") != std::string::npos,
                    "cancel → finish cancelled");
    STELL_CHECK(stell_cancel(e, s2) == 0);           // повторный cancel — 0

    stell_engine_destroy(e);
}

STELL_TEST_MAIN(run)

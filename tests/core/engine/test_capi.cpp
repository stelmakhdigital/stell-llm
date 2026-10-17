// Стык engine ↔ serve (ADR-11): C ABI — create/submit/poll/cancel/metrics.
#include "test_util.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/synth.hpp"
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

void run_demo_case() {
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

// --- round 2: runner:"ops" — реальное CPU-исполнение модели через ABI -------
// mini_dense fixture → лоадер safetensors → каталог операций → логиты;
// byte-токенизатор: text в token-событиях (ADR-11 §3, честная пометка: прод-
// токенизатор — отдельная задача).
void run_ops_case() {
    using namespace stell_core_test;
    const char* weights = "/tmp/stell_core_test_capi_ops.safetensors";
    {
        stell::engine::PlainSpec spec =
            stell::engine::load_plain_spec(STELL_FIXTURES_DIR "/mini_dense.json");
        write_safetensors(weights, fixture_tensors(spec));
    }

    const std::string cfg = std::string("{\"runner\":\"ops\",") +
        "\"engine\":{\"max_slots\":4}," +
        "\"model\":{\"description\":\"" STELL_FIXTURES_DIR "/mini_dense.json\"," +
        "\"weights\":\"" + weights + "\"," +
        "\"weights_format\":\"safetensors\",\"tokenizer\":\"byte\"}}";

    stell_engine_t* e = nullptr;
    STELL_CHECK_MSG(stell_engine_create(cfg.c_str(), &e) == 0, "ops engine created");
    STELL_CHECK(e != nullptr);

    char* msg = nullptr;
    const uint64_t id = stell_submit(e,
        "{\"prompt_tokens\":[1,2],\"sampling\":{\"max_tokens\":3,"
        "\"temperature\":0},\"eos_token\":-1}", &msg);
    if (msg) free_str(msg);
    STELL_CHECK(id != 0);

    for (int i = 0; i < 3; ++i) stell_step(e);
    int n = 0;
    const std::string evs = poll_json(e, id, &n);
    STELL_CHECK(n >= 1);
    // 3 token-события + finish(length); text заполняется byte-токенизатором
    int tokens = 0, with_text = 0;
    std::size_t pos = 0;
    while ((pos = evs.find("\"type\":\"token\"", pos)) != std::string::npos) {
        ++tokens;
        ++pos;
    }
    pos = 0;
    while ((pos = evs.find("\"text\":", pos)) != std::string::npos) {
        ++with_text;
        ++pos;
    }
    STELL_CHECK(tokens == 3);
    STELL_CHECK(with_text == 3);
    STELL_CHECK_MSG(evs.find("\"type\":\"finish\"") != std::string::npos, "finish");
    STELL_CHECK(evs.find("\"reason\":\"length\"") != std::string::npos);
    STELL_CHECK(evs.find("\"completion_tokens\":3") != std::string::npos);

    // детерминизм: вторая сессия с тем же промптом → те же токены
    const uint64_t id2 = stell_submit(e,
        "{\"prompt_tokens\":[1,2],\"sampling\":{\"max_tokens\":3,"
        "\"temperature\":0},\"eos_token\":-1}", nullptr);
    STELL_CHECK(id2 != 0);
    for (int i = 0; i < 3; ++i) stell_step(e);
    const std::string evs2 = poll_json(e, id2, &n);
    STELL_CHECK(evs2 == evs);

    char* mj = nullptr;
    STELL_CHECK(stell_metrics(e, &mj) == 0);
    const std::string metrics(mj);
    free_str(mj);
    STELL_CHECK(metrics.find("\"engine\":{") != std::string::npos);

    stell_engine_destroy(e);
}

void run() {
    run_demo_case();
    run_ops_case();
}

STELL_TEST_MAIN(run)

// CPU-исполнение реальной модели (round 2, срез 1 A×D, CPU-вариант):
// mini_dense fixture → лоадер safetensors → тензорные группы → OpsInterpreter
// (каталог операций: embedding/rmsnorm/attention FP32 QK/PV/ffn/logits) →
// логиты; e2e через Scheduler (continuous batching) + byte-токенизатор.
#include "test_util.hpp"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/scheduler.hpp"
#include "stell/engine/tokenizer.hpp"

using namespace stell::engine;
using namespace stell_core_test;

namespace {

const char* kWeightsPath = "/tmp/stell_core_test_dense.safetensors";

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load_fixture() {
    Loaded l{};
    l.spec = load_plain_spec(STELL_FIXTURES_DIR "/mini_dense.json");
    write_safetensors(kWeightsPath, fixture_tensors(l.spec));
    l.weights = load_plain_weights(kWeightsPath, l.spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(l.spec);
    return l;
}

} // namespace

void run() {
    const Loaded l = load_fixture();
    STELL_CHECK(l.spec.name == "mini-dense-test");
    STELL_CHECK(l.plan.vocab_size == 16);
    STELL_CHECK(l.plan.slot_bytes_total > 0);

    // --- ops-каталог: неизвестная операция = ошибка загрузки (ADR-09 §1) ----
    bool threw = false;
    try {
        PlainSpec bad = l.spec;
        bad.ops.push_back(PlainOp{});
        bad.ops.back().op = "unknown_op_xyz";
        OpsInterpreter ib(bad, l.plan, l.weights);
        (void)ib;
    } catch (const std::invalid_argument&) { threw = true; }
    STELL_CHECK(threw);

    // --- интерпретатор: логиты промпта -------------------------------------
    OpsInterpreter interp(l.spec, l.plan, l.weights);
    const std::vector<std::int32_t> prompt = {1, 2, 3};
    const std::vector<float> logits = interp.run_prompt_logits(prompt);
    STELL_CHECK(logits.size() == l.plan.vocab_size);
    for (float v : logits) STELL_CHECK(std::isfinite(v));

    // детерминизм: повторный прогон — побитово тот же результат
    OpsInterpreter interp2(l.spec, l.plan, l.weights);
    const std::vector<float> again = interp2.run_prompt_logits(prompt);
    STELL_CHECK(again.size() == logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i)
        STELL_CHECK(again[i] == logits[i]);

    // greedy argmax: sample == argmax
    SamplingParams sp{};
    sp.temperature = 0;                      // golden-режим (ADR-09 §4 «Уточнение»)
    STELL_CHECK(interp.sample(logits, sp) ==
                static_cast<std::int32_t>(
                    std::max_element(logits.begin(), logits.end()) - logits.begin()));

    // --- e2e через Scheduler (continuous batching) + byte-токенизатор --------
    CpuModelRunner runner(l.spec, l.plan, l.weights);
    FixedSlotPool pool(4, l.plan.slot_bytes_total);
    Scheduler sched(runner, pool);
    ByteTokenizer tok;
    sched.set_tokenizer(&tok);

    SubmitRequest req{};
    req.prompt_tokens = prompt;
    req.sampling.max_tokens = 4;
    req.sampling.temperature = 0.0f;         // greedy
    req.eos_token = -1;                      // не останавливаем по eos
    const SessionId id = sched.submit(req);
    STELL_CHECK(id != 0);
    for (int i = 0; i < 4; ++i) sched.step();

    std::vector<Event> evs = sched.poll(id);
    STELL_CHECK(!evs.empty());
    std::uint32_t tokens = 0;
    bool finish = false;
    for (const Event& ev : evs) {
        if (ev.type == Event::Type::Token) {
            ++tokens;
            STELL_CHECK(ev.token_id >= 0 && ev.token_id < 16);
            // round 2: text заполняется byte-токенизатором (1 байт на токен)
            STELL_CHECK(ev.text.size() == 1);
            STELL_CHECK(ev.seq > 0);
        }
        if (ev.type == Event::Type::Finish) {
            finish = true;
            STELL_CHECK(ev.finish_reason == "length");
            STELL_CHECK(ev.completion_tokens == 4);
            STELL_CHECK(ev.prompt_tokens == 3);
        }
    }
    STELL_CHECK(tokens == 4);
    STELL_CHECK(finish);

    // --- два слота: одинаковый промпт → одинаковые токены (детерминизм) -----
    FixedSlotPool pool2(4, l.plan.slot_bytes_total);
    Scheduler sched2(runner, pool2);
    sched2.set_tokenizer(&tok);
    const SessionId a = sched2.submit(req);
    const SessionId b = sched2.submit(req);
    STELL_CHECK(a != 0 && b != 0 && a != b);
    sched2.step();
    sched2.step();
    const std::vector<Event> eva = sched2.poll(a);
    const std::vector<Event> evb = sched2.poll(b);
    STELL_CHECK(eva.size() == evb.size());
    for (std::size_t i = 0; i < eva.size(); ++i)
        if (eva[i].type == Event::Type::Token && evb[i].type == Event::Type::Token)
            STELL_CHECK(eva[i].token_id == evb[i].token_id);
}

STELL_TEST_MAIN(run)

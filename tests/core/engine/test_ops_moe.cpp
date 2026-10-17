// CPU-исполнение MoE-модели (round 2, mini_moe fixture, профиль NVFP4/ADR-03):
// лоадер NVFP4W4A16 (квантование на read) → block32 dequant (kernels) →
// ops-каталог: rmsnorm/attention FP32 QK/PV/moe_gather_scatter (kernels CPU-
// референс) → логиты; e2e через Scheduler.
#include "test_util.hpp"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/scheduler.hpp"

using namespace stell::engine;
using namespace stell_core_test;

namespace {

const char* kWeightsPath = "/tmp/stell_core_test_moe.safetensors";

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load_fixture() {
    Loaded l{};
    l.spec = load_plain_spec(STELL_FIXTURES_DIR "/mini_moe.json");
    STELL_CHECK(l.spec.is_moe);
    STELL_CHECK(l.spec.dtype_weights == "NVFP4");
    write_safetensors(kWeightsPath, fixture_tensors(l.spec));
    l.weights = load_plain_weights(kWeightsPath, l.spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(l.spec);
    return l;
}

} // namespace

void run() {
    const Loaded l = load_fixture();
    STELL_CHECK(l.spec.n_experts == 4);
    STELL_CHECK(l.spec.n_experts_active == 2);
    STELL_CHECK(l.spec.expert_d_ff == 8);

    // --- интерпретатор: MoE-граф, логиты ------------------------------------
    OpsInterpreter interp(l.spec, l.plan, l.weights);
    const std::vector<std::int32_t> prompt = {3, 1, 4};
    const std::vector<float> logits = interp.run_prompt_logits(prompt);
    STELL_CHECK(logits.size() == l.plan.vocab_size);
    for (float v : logits) STELL_CHECK(std::isfinite(v));

    // детерминизм (повторная материализация + прогон)
    Loaded l2 = load_fixture();
    OpsInterpreter interp2(l2.spec, l2.plan, l2.weights);
    const std::vector<float> again = interp2.run_prompt_logits(prompt);
    STELL_CHECK(again.size() == logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i)
        STELL_CHECK(again[i] == logits[i]);

    // greedy argmax через sample
    SamplingParams sp{};
    sp.temperature = 0;
    STELL_CHECK(interp.sample(logits, sp) ==
                static_cast<std::int32_t>(
                    std::max_element(logits.begin(), logits.end()) - logits.begin()));

    // --- e2e Scheduler: MoE-декод с expert-группами -------------------------
    CpuModelRunner runner(l.spec, l.plan, l.weights);
    FixedSlotPool pool(4, l.plan.slot_bytes_total);
    Scheduler sched(runner, pool);
    SubmitRequest req{};
    req.prompt_tokens = prompt;
    req.sampling.max_tokens = 3;
    req.sampling.temperature = 0.0f;
    req.eos_token = -1;
    const SessionId id = sched.submit(req);
    STELL_CHECK(id != 0);
    for (int i = 0; i < 3; ++i) sched.step();
    const std::vector<Event> evs = sched.poll(id);
    std::uint32_t tokens = 0;
    bool finish = false;
    for (const Event& ev : evs) {
        if (ev.type == Event::Type::Token) ++tokens;
        if (ev.type == Event::Type::Finish) {
            finish = true;
            STELL_CHECK(ev.finish_reason == "length");
            STELL_CHECK(ev.completion_tokens == 3);
        }
    }
    STELL_CHECK(tokens == 3);
    STELL_CHECK(finish);
}

STELL_TEST_MAIN(run)

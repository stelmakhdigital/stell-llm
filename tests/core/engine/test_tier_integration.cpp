// Интеграция engine ↔ tiers (round 2, ADR-08): реальный TierManager в
// декодовом контуре engine — set_plan по реальным группам плана, set_layer_map
// (ADR-08 «Уточнение»), pin_layer перед каждым слоем, prefetch следующего,
// снятие пина после слоя; интеграционный тест decode-шага с проверкой
// residency групп в VRAM-тире и метрик пауз; ADR-09 §4: tiering не меняет
// вычисление (логиты с tiers == логиты без tiers, побитово).
#include "test_util.hpp"

#include <cstdio>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/scheduler.hpp"
#include "stell/tiers/layout.hpp"
#include "stell/tiers/tier_manager.hpp"

using namespace stell::engine;
using namespace stell_core_test;
using stell::tiers::GroupId;
using stell::tiers::TierLayoutHeader;
using stell::tiers::TierLayoutWriter;
using stell::tiers::TierStats;
using stell::tiers::TensorGroup;

namespace {

const char* kWeightsPath = "/tmp/stell_core_test_tier.safetensors";
const char* kTierPath = "/tmp/stell_core_test_tier.stelltier";

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

// stell-tier layout с реальными payload групп ( tiers'овский writer — ADR-12;
// честно: файл читается только tiers'овским reader — см. ADR-12 «Уточнение»
// о несовместимости CRC заголовка с конвертером models).
std::string write_layout(const Loaded& l) {
    std::vector<TensorGroup> tgs = l.plan.tensor_groups();
    TierLayoutHeader h;
    h.n_groups = static_cast<std::uint32_t>(tgs.size());
    h.dtype_profile = 1;                       // F16 (ADR-12 §2)
    h.model_name = "mini-dense-test";
    TierLayoutWriter w(kTierPath, h);
    for (const auto& g : tgs) {
        const auto it = l.weights.slots.find(g.id.value);
        const std::size_t len = it != l.weights.slots.end() ? it->second.size() : g.slot_bytes;
        w.add_group(g, /*tensor_count=*/1,
                    it != l.weights.slots.end() ? it->second.data() : nullptr, len);
    }
    w.finalize();
    return kTierPath;
}

} // namespace

void run() {
    const Loaded l = load_fixture();
    const std::string tier_path = write_layout(l);

    // --- TierManager по плану реальных групп (ADR-08 §2) --------------------
    stell::tiers::TierConfig cfg{};
    cfg.vram_slots = 16;
    cfg.ram_capacity = 1u << 20;
    cfg.nvme_layout = tier_path;
    cfg.lookahead = 2;
    stell::tiers::TierManager tm(cfg);

    // set_plan делает runner при загрузке модели (пара set_plan →
    // set_layer_map — «холодная» дисциплина ADR-08 «Уточнение»).
    CpuModelRunner runner(l.spec, l.plan, l.weights, &tm);
    const std::uint32_t version_before = l.plan.layout_version;

    // --- decode-шаг через Scheduler: pin_layer/prefetch внутри runner --------
    FixedSlotPool pool(4, l.plan.slot_bytes_total);
    Scheduler sched(runner, pool);
    SubmitRequest req{};
    req.prompt_tokens = {1, 2, 3};
    req.sampling.max_tokens = 3;
    req.sampling.temperature = 0.0f;
    req.eos_token = -1;
    const SessionId id = sched.submit(req);
    STELL_CHECK(id != 0);
    sched.step();
    sched.step();
    sched.step();

    // --- residency: слои прошли через pin_layer → группы в VRAM-тире ---------
    const auto& map = l.plan.layer_map();
    STELL_CHECK(map.size() == l.plan.n_layers);
    for (std::size_t layer = 0; layer < map.size(); ++layer)
        for (const auto& gid : map[layer])
            STELL_CHECK_MSG(tm.resident_in_vram(gid),
                            "группа слоя " + std::to_string(layer) + " не resident в VRAM");

    // --- метрики пауз (ADR-08 §2 TierStats; ADR-11 §4 passthrough) -----------
    const stell::tiers::TierStats st = tm.stats();
    STELL_CHECK(st.layer_pause.p50_ms >= 0.0);
    STELL_CHECK(st.layer_pause.p95_ms >= st.layer_pause.p50_ms ||
                st.layer_pause.p95_ms == 0.0);          // честно: пустое окно → 0
    STELL_CHECK(st.prefetch.hit_rate >= 0.0 && st.prefetch.hit_rate <= 1.0);
    STELL_CHECK(st.io_ops > 0);                          // NVMe-чтения состоялись
    STELL_CHECK(st.io_bytes > 0);

    // --- метрики runner'а: pin_layer/prefetch состоялись --------------------
    STELL_CHECK(runner.interpreter().layer_pins() >= l.plan.n_layers);
    STELL_CHECK(runner.interpreter().layer_prefetches() >= l.plan.n_layers - 1);

    // --- ADR-09 §4: tiering не меняет вычисление (побитовое равенство) ------
    OpsInterpreter plain(l.spec, l.plan, l.weights, /*tiers=*/nullptr);
    const std::vector<float> ref = plain.run_prompt_logits({1, 2, 3});
    OpsInterpreter tiered(l.spec, l.plan, l.weights, &tm);
    const std::vector<float> got = tiered.run_prompt_logits({1, 2, 3});
    STELL_CHECK(ref.size() == got.size());
    for (std::size_t i = 0; i < ref.size(); ++i)
        STELL_CHECK_MSG(ref[i] == got[i], "tiering изменил вычисление логитов");

    // layout_version не изменился (переснятие графа не требуется)
    STELL_CHECK(l.plan.layout_version == version_before);
}

STELL_TEST_MAIN(run)

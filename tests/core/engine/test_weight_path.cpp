// Физический путь весов (round 2, задача T-20261017200354 [R8], ADR-08/09/12):
//   прямая загрузка плана (safetensors, лоадер models)
//     == stell-tier layout (writer — tiers TierLayoutWriter, физический файл
//        на NVMe-тире: tmp-файл, CPU-конфигурация)
//        → читатель tiers (TierLayoutReader: индекс + CRC32 слота, ADR-12 §2.3)
//        → слоты VRAM/RAM tiers (TierManager: set_plan/set_layer_map,
//          pin_layer, NVMe→RAM→VRAM с тесными слотами/ёмкостью)
//        → интерпретатор ops engine (OpsInterpreter)
//   — логиты ПОБИТОВО равны прямой загрузке того же плана (ADR-09 §4).
//
// Честная фиксация границы (планировщик раунда: «не молчи»):
//   * публичного API чтения СОДЕРЖИМОГО VRAM/RAM-слота TierManager'а нет
//     (resident_in_vram/stats — только факты residency/IO) — поэтому байты
//     читателя снимаются с NVMe-тира через TierLayoutReader (публичный API
//     чтения слота tiers) и проверяются на побитовую идентичность payload'у
//     прямой загрузки; резидентность в VRAM/переносы фиксируются
//     resident_in_vram/io_ops/io_bytes;
//   * читатель models (StellTierSource) на этом файле ОТКАЗАЛСЯ бы: он
//     требует, чтобы в layout не было записей вне некешевых групп плана,
//     а tiers set_plan требует наличия ВСЕХ групп плана (включая KV) —
//     расхождение контрактов readers зафиксировано в отчёте раунда (тест
//     идёт через reader tiers, как и требует [R8]).
#include "test_util.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/plain_spec.hpp"
#include "stell/tiers/layout.hpp"
#include "stell/tiers/tier_manager.hpp"

using namespace stell::engine;
using namespace stell_core_test;
using stell::tiers::TierLayoutHeader;
using stell::tiers::TierLayoutReader;
using stell::tiers::TierLayoutWriter;
using stell::tiers::TensorGroup;

namespace {

const char* kWeightsPath = "/tmp/stell_core_test_weight_path.safetensors";
const char* kTierPath = "/tmp/stell_core_test_weight_path.stelltier";

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load_direct() {
    Loaded l{};
    l.spec = load_plain_spec(STELL_FIXTURES_DIR "/mini_dense.json");
    write_safetensors(kWeightsPath, fixture_tensors(l.spec));
    l.weights = load_plain_weights(kWeightsPath, l.spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(l.spec);
    return l;
}

// stell-tier layout со всеми группами плана (set_plan tiers требует наличия
// всех, включая KV; payload — из прямой загрузки).
void write_layout(const Loaded& l) {
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
}

// Физическое чтение слотов групп из .stelltier-файла через публичный reader
// tiers (CRC32 слота при первом чтении — ADR-12 §2.3) → LoadedWeights.
std::uint64_t g_reader_io_ops = 0, g_reader_io_bytes = 0;

LoadedWeights read_weights_via_tiers(const PlainSpec& spec) {
    TierLayoutReader reader(kTierPath);
    LoadedWeights w{};
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block) continue;              // KV-группа весов не имеет (ADR-09 §3)
        const auto* e = reader.find(g.id);
        STELL_CHECK_MSG(e != nullptr, "группы " + g.name + " нет в layout");
        STELL_CHECK_MSG(e->slot_bytes == g.slot_bytes, "slot_bytes " + g.name);
        std::vector<std::uint8_t> slot(static_cast<std::size_t>(g.slot_bytes), 0xAA);
        reader.read_group(g.id, slot.data());  // бросит при CRC-мismatch
        w.slots[g.id] = std::move(slot);
    }
    STELL_CHECK(reader.io_ops() > 0);          // физические чтения состоялись
    STELL_CHECK(reader.io_bytes() > 0);
    g_reader_io_ops = reader.io_ops();
    g_reader_io_bytes = reader.io_bytes();
    return w;
}

} // namespace

void run() {
    const Loaded direct = load_direct();
    write_layout(direct);

    // --- физическое чтение: reader tiers, байты слотов → weights ------------
    Loaded tiered{};
    tiered.spec = load_plain_spec(STELL_FIXTURES_DIR "/mini_dense.json");
    tiered.plan = build_execution_plan(tiered.spec);
    tiered.weights = read_weights_via_tiers(tiered.spec);

    // payload каждой группы из физического файла побитово == прямой загрузке
    for (const PlainGroup& g : tiered.spec.groups) {
        if (g.kv_block) continue;
        const auto ia = direct.weights.slots.find(g.id);
        const auto ib = tiered.weights.slots.find(g.id);
        STELL_CHECK_MSG(ia != direct.weights.slots.end() && ib != tiered.weights.slots.end(),
                        "нет payload группы " + g.name);
        STELL_CHECK_MSG(ia->second.size() == ib->second.size(),
                        "размер payload " + g.name);
        STELL_CHECK_MSG(std::memcmp(ia->second.data(), ib->second.data(),
                                    ia->second.size()) == 0,
                        "payload группы " + g.name + " не побитово равен");
    }

    // --- TierManager: слоты VRAM/RAM над NVMe-тиром (физический tmp-файл) ---
    stell::tiers::TierConfig tc{};
    tc.vram_slots = 16;                // ≥ числа групп слоя (pin всех групп слоя)
    tc.ram_capacity = 2048;            // тесный RAM-тир → потоки NVMe→RAM→VRAM
    tc.nvme_layout = kTierPath;
    tc.lookahead = 2;
    stell::tiers::TierManager tm(tc);

    CpuModelRunner runner(tiered.spec, tiered.plan, tiered.weights, &tm);
    FixedSlotPool pool(4, tiered.plan.slot_bytes_total);
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

    // Группы слоёв прошли через pin_layer → resident в VRAM-тире (ADR-08 §2).
    const auto& map = tiered.plan.layer_map();
    for (std::size_t layer = 0; layer < map.size(); ++layer)
        for (const auto& gid : map[layer])
            STELL_CHECK_MSG(tm.resident_in_vram(gid),
                            "группа слоя " + std::to_string(layer) +
                            " не resident в VRAM");

    // Физические переносы состоялись (NVMe-чтения tiers).
    const stell::tiers::TierStats st = tm.stats();
    STELL_CHECK(st.io_ops > 0);
    STELL_CHECK(st.io_bytes > 0);
    STELL_CHECK(runner.interpreter().layer_pins() >= tiered.plan.n_layers);

    // --- логиты через tiers-путь == прямая загрузка, ПОБИТОВО ----------------
    const std::vector<std::int32_t> prompt = {1, 2, 3};
    OpsInterpreter tiered_interp(tiered.spec, tiered.plan, tiered.weights, &tm);
    const std::vector<float> logits_tier = tiered_interp.run_prompt_logits(prompt);

    OpsInterpreter plain(direct.spec, direct.plan, direct.weights, /*tiers=*/nullptr);
    const std::vector<float> logits_direct = plain.run_prompt_logits(prompt);

    STELL_CHECK(logits_tier.size() == logits_direct.size());
    for (std::size_t i = 0; i < logits_direct.size(); ++i)
        STELL_CHECK_MSG(logits_tier[i] == logits_direct[i],
                        "логиты tiers-пути != прямой загрузке (индекс " +
                            std::to_string(i) + ")");

    // И логиты всех позиций (режим "all" stell_logits) тоже побитово равны:
    // физический путь не меняет ни один шаг forward.
    const auto all_tier = tiered_interp.run_prompt_logits_all(prompt);
    const auto all_direct = plain.run_prompt_logits_all(prompt);
    STELL_CHECK(all_tier.size() == all_direct.size());
    for (std::size_t p = 0; p < all_direct.size(); ++p)
        for (std::size_t i = 0; i < all_direct[p].size(); ++i)
            STELL_CHECK(all_tier[p][i] == all_direct[p][i]);

    std::printf(
        "weight-path: mini_dense — payload групп побитово равен (файл → "
        "TierLayoutReader, %llu IO-оп/%llu байт reader'а), логиты tiers-пути "
        "(NVMe→RAM→VRAM, %llu IO-оп tiers) побитово == прямой загрузке\n",
        (unsigned long long)g_reader_io_ops, (unsigned long long)g_reader_io_bytes,
        (unsigned long long)st.io_ops);
}

STELL_TEST_MAIN(run)

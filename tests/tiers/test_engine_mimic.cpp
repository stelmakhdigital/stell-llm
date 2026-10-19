// Тест-двойник вклейки core(engine) → tiers: мини-симуляция вызовов
// engine по контракту ADR-08 (set_plan → set_layer_map → decode-цикл
// pin_layer/prefetch/unpin → stats) и «холодная» дисциплина §4.
// Реальная вклейка — задача интеграции (worker/следующий раунд); здесь —
// фиксация совместимости API-контракта (ADR-08 «Уточнение»):
//   - engine вправе работать с TierManager напрямую (set_layer_map —
//     расширение на реализации, НЕ новый метод ITierManager);
//   - layout_version — величина engine (tiers её не знает);
//   - pin_layer вне карты → PlanMismatch; pin неизвестной группы → PlanMismatch.
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <memory>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

static Role role_of(const std::vector<TensorGroup>& plan, GroupId id) {
    for (const auto& g : plan)
        if (g.id == id) return g.role;
    throw std::runtime_error("role_of: unknown GroupId");
}

static const std::size_t kSlot = 4096;

static std::vector<TensorGroup> engine_plan() {
    // 3 слоя × (attn + e0 + e1) + kv на слой (в плане, НЕ в карте слоя —
    // kv-кеш не pinится через pin_layer, ADR-08 §5).
    std::vector<TensorGroup> plan;
    for (std::uint32_t l = 0; l < 3; ++l) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "l%03u.attn", l);
        plan.push_back(make_group(plan.size(), buf, Role::Attn, DType::F16, kSlot, {8, 8}));
        for (int e = 0; e < 2; ++e) {
            char b2[32];
            std::snprintf(b2, sizeof b2, "l%03u.e%03d", l, e);
            plan.push_back(make_group(plan.size(), b2, Role::Expert, DType::F16, kSlot, {8, 8}));
        }
        char b3[32];
        std::snprintf(b3, sizeof b3, "l%03u.kv", l);
        plan.push_back(make_group(plan.size(), b3, Role::Kv, DType::F16, kSlot, {4, 8}));
    }
    return plan;
}

static std::string write_layout(const std::string& path,
                                const std::vector<TensorGroup>& groups) {
    TierLayoutHeader h;
    h.dtype_profile = 1;
    h.n_groups = groups.size();
    h.model_name = "engine-mimic";
    TierLayoutWriter w(path, h);
    for (const auto& g : groups) {
        std::vector<std::uint8_t> payload(g.slot_bytes, std::uint8_t(g.id.value));
        w.add_group(g, 1, payload.data(), payload.size());
    }
    w.finalize();
    return path;
}

T_CASE(engine_contract_mimic) {
    const auto plan = engine_plan();
    const auto path = write_layout("/tmp/stell_tier_engine_mimic.stelltier", plan);

    TierConfig cfg;
    // Bugfix раунда 3 (корректный учёт VRAM-слотов в acquire_slot): для
    // детерминизма prefetch-хитов нужно ≥ 2 слоёв групп (3 группы/слой):
    // при 3 слотах (1 слой) загрузка слоя l+1 вытесняла слой l → все pin —
    // промахи. Раньше тест проходил на aliasing-баге слотов (resident
    // завышался, вытеснения не происходило фактически).
    cfg.vram_slots = 6;                 // churn: 3 слоя × 3 группы > 6 слотов
    cfg.vram_slot_size = kSlot;
    cfg.ram_capacity = 6 * kSlot;
    cfg.nvme_layout = path;
    cfg.lookahead = 0;                  // предсказатель выкл.: детерминизм метрик
    TierManager mgr(cfg);

    // Engine держит tiers через интерфейсный указатель (ADR-08 §2); расши-
    // рение set_layer_map доступно только на конкретном TierManager.
    ITierManager& tm = mgr;

    // pin_layer без карты слоя → PlanMismatch (до set_layer_map).
    bool thrown = false;
    try {
        tm.pin_layer(0);
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "pin_layer до set_layer_map → PlanMismatch");

    // Холодная дисциплина §4: set_plan ДО decode-цикла.
    tm.set_plan(plan);

    // Карта слоя: слой l → {attn_l, e_l0, e_l1} (kv — вне карты). Группы
    // упорядочены по 4 на слой: id/4 = слой.
    std::vector<std::vector<GroupId>> layers(3);
    for (const auto& g : plan) {
        if (g.role == Role::Kv) continue;
        layers[g.id.value / 4].push_back(g.id);
    }
    mgr.set_layer_map(layers);

    // Карта слоя с группой вне плана → pin_layer(l) → PlanMismatch.
    {
        auto bad = layers;
        bad[0].push_back(GroupId{999});
        TierManager probe(cfg);
        probe.set_plan(plan);
        probe.set_layer_map(bad);
        thrown = false;
        try {
            probe.pin_layer(0);
        } catch (const TierError& e) {
            thrown = e.code() == TierError::Code::PlanMismatch;
        }
        T_REQUIRE(thrown, "группа слоя вне плана → PlanMismatch");
    }

    // Decode-цикл: prefetch(l+1) заранее → pin_layer(l) → wait → unpin(l).
    for (unsigned l = 0; l < layers.size(); ++l) {
        if (l + 1 < layers.size()) {
            std::vector<GroupKey> keys;
            for (GroupId gid : layers[l + 1])
                keys.push_back(GroupKey{gid, role_of(plan, gid)});
            tm.prefetch(keys);
        }
        auto a = tm.pin_layer(l);
        a->wait();
        for (GroupId gid : layers[l]) tm.unpin(GroupKey{gid, role_of(plan, gid)});
    }
    const TierStats st = tm.stats();
    T_REQUIRE(st.prefetch.hits >= layers.size() - 1,
              "prefetch следующего слоя → hit при pin_layer (router-lookahead)");
    T_REQUIRE(st.io_bytes > 0 && st.io_ops > 0, "NVMe-тир отработал");
    T_REQUIRE(st.tier_bytes[2] == plan.size() * kSlot,
              "NVMe занято = сумме слотов layout'а");
    T_REQUIRE(st.layer_pause.p50_ms >= 0.0, "паузы слоёв измеряются");

    // Холодная дисциплина §4: ResetPlan при pinned-группе → PlanMismatch;
    // после unpin → допустим (повторный вызов = ResetPlan).
    mgr.pin(GroupKey{plan[0].id, plan[0].role})->wait();
    bool thrown2 = false;
    try {
        tm.set_plan(plan);
    } catch (const TierError& e) {
        thrown2 = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown2, "set_plan при pinned-группе → PlanMismatch");
    mgr.unpin(GroupKey{plan[0].id, plan[0].role});
    tm.set_plan(plan); // ResetPlan — допустим после drain pinned/queue
    // Карта слоёв сброшена ResetPlan'ом → pin_layer снова PlanMismatch.
    thrown = false;
    try {
        tm.pin_layer(0);
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "после ResetPlan карта слоёв пуста → PlanMismatch");
    mgr.set_layer_map(layers);
    mgr.pin_layer(1)->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{4}), "слоёвая карта жива после ResetPlan");
}

T_MAIN

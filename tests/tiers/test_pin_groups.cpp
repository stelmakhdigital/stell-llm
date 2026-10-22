// [R1] round 10 (ADR-13 §[R3], planner T-20261022183913 R2): публичный
// групповой API pin_groups — пиннинг ПРОИЗВОЛЬНОГО подмножества групп
// (экспертно-гранулярный tier-flow MoE-контура), при НЕИЗМЕННОЙ семантике
// pin_layer. Проверки: fan-in подмножества, PlanMismatch, пауза — на
// вызывающем (stats().layer_pause не растёт из pin_groups), снятие пина
// ровно пиннутого (unpin непиннутого → NotPinned).
#include <stell/tiers/tier_manager.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

static const std::size_t kSlot = 4096;

static std::vector<TensorGroup> moe_plan() {
    // Мини-план MoE-слоя: attn + kv + router + 2 экспертные группы.
    return {
        make_group(0, "l000.attn.wq", Role::Attn, DType::F16, kSlot),
        make_group(1, "l000.kv", Role::Kv, DType::F16, kSlot),
        make_group(2, "l000.mlp.router", Role::Expert, DType::F32, kSlot),
        make_group(3, "l000.e000.up", Role::Expert, DType::F16, kSlot),
        make_group(4, "l000.e001.up", Role::Expert, DType::F16, kSlot),
    };
}

static std::string write_layout(const std::string& path,
                                const std::vector<TensorGroup>& groups) {
    TierLayoutHeader h;
    h.n_groups = groups.size();
    TierLayoutWriter w(path, h);
    for (const auto& g : groups) {
        std::vector<std::uint8_t> payload(kSlot, std::uint8_t(g.id.value));
        w.add_group(g, 1, payload.data(), payload.size());
    }
    w.finalize();
    return path;
}

static TierConfig cfg_with(const std::string& path, std::size_t slots) {
    TierConfig cfg;
    cfg.vram_slots = slots;
    cfg.vram_slot_size = kSlot;
    cfg.ram_capacity = 8 * kSlot;
    cfg.nvme_layout = path;
    cfg.lookahead = 0;
    return cfg;
}

T_CASE(pin_groups_subset_fan_in) {
    const auto groups = moe_plan();
    const auto path = write_layout("/tmp/stell_tier_test_pingroups1.stelltier", groups);
    TierManager mgr(cfg_with(path, 4));
    mgr.set_plan(groups);

    // Подмножество слоя: attn + kv + router (БЕЗ экспертных групп) —
    // как MoE-контур до маршрутизации ([R1]).
    const std::vector<GroupKey> subset = {
        GroupKey{GroupId{0}, Role::Attn},
        GroupKey{GroupId{1}, Role::Kv},
        GroupKey{GroupId{2}, Role::Expert},
    };
    auto a = mgr.pin_groups(subset);
    a->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{0}), "attnresident после fan-in");
    T_REQUIRE(mgr.resident_in_vram(GroupId{1}), "kv resident после fan-in");
    T_REQUIRE(mgr.resident_in_vram(GroupId{2}), "router resident после fan-in");
    T_REQUIRE(!mgr.resident_in_vram(GroupId{3}),
              "непиннутый эксперт НЕ resident (вытесняем — гранулярность)");
    T_REQUIRE(!mgr.resident_in_vram(GroupId{4}),
              "второй непиннутый эксперт НЕ resident");

    // Снятие пина ровно пиннутого; unpin непиннутого → NotPinned.
    for (const auto& k : subset) mgr.unpin(k);
    bool thrown = false;
    try {
        mgr.unpin(GroupKey{GroupId{3}, Role::Expert});
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::NotPinned;
    }
    T_REQUIRE(thrown, "unpin группы без пина → NotPinned (честный контракт)");
}

T_CASE(pin_groups_expert_granular_flow) {
    // Гранулярный flow: пин подмножества, затем ВЫБРАННЫЙ эксперт; далее
    // ДРУГОЙ выбранный эксперт — вытеснение по политике работает на уровне
    // ГРУПП (непиннутые resident-группы — кандидаты eviction), не слоёв:
    // матрица {LRU,LCU}×{k} на экспертной гранулярности не вырождена ([R2]).
    const auto groups = moe_plan();
    const auto path = write_layout("/tmp/stell_tier_test_pingroups2.stelltier", groups);
    TierManager mgr(cfg_with(path, 2));   // 2 VRAM-слота: вытеснение неизбежно
    mgr.set_plan(groups);

    const std::vector<GroupKey> base = {GroupKey{GroupId{0}, Role::Attn},
                                        GroupKey{GroupId{1}, Role::Kv}};
    auto a = mgr.pin_groups(base);
    a->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{0}) && mgr.resident_in_vram(GroupId{1}),
              "базовое подмножество слоя resident");
    for (const auto& k : base) mgr.unpin(k);

    // Выбран роутером эксперт e000: пин + wait (загрузка; вытесняется
    // непиннутый resident-кандидат — групповая гранулярность).
    auto ae = mgr.pin_groups({GroupKey{GroupId{3}, Role::Expert}});
    ae->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{3}), "выбранный эксперт resident");
    mgr.unpin(GroupKey{GroupId{3}, Role::Expert});

    // Повторный гранулярный проход с ДРУГИМ экспертом: e001 грузится.
    auto b = mgr.pin_groups({GroupKey{GroupId{4}, Role::Expert}});
    b->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{4}), "второй выбранный эксперт resident");
    mgr.unpin(GroupKey{GroupId{4}, Role::Expert});

    // Per-group hit/miss учёт ([R2]): e000 и e001 — по cold-miss (первый пин).
    const auto gs = mgr.group_prefetch_stats();
    T_REQUIRE(gs.size() == groups.size(), "снапшот покрывает весь план");
    std::size_t expert_misses = 0;
    for (const auto& s : gs)
        if (s.gid == GroupId{3} || s.gid == GroupId{4})
            expert_misses += s.misses;
    T_REQUIRE(expert_misses == 2,
              "каждый выбранный эксперт — честный cold-miss per-group учёт");
}

T_CASE(pin_groups_unknown_group) {
    const auto groups = moe_plan();
    const auto path = write_layout("/tmp/stell_tier_test_pingroups3.stelltier", groups);
    TierManager mgr(cfg_with(path, 2));
    mgr.set_plan(groups);
    bool thrown = false;
    try {
        mgr.pin_groups({GroupKey{GroupId{99}, Role::Expert}});
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "pin_groups: группа вне плана → PlanMismatch");
}

T_CASE(pin_groups_pause_on_caller) {
    // Пауза — на вызывающем (engine меряет и пишет record_layer_pause);
    // pin_groups НЕ пишет паузу слоя (в отличие от pin_layer) — иначе сумма
    // пауз групп слоя ([R2]) задваивалась бы.
    const auto groups = moe_plan();
    const auto path = write_layout("/tmp/stell_tier_test_pingroups4.stelltier", groups);
    TierManager mgr(cfg_with(path, 2));
    mgr.set_plan(groups);
    const TierStats before = mgr.stats();
    auto a = mgr.pin_groups({GroupKey{GroupId{0}, Role::Attn}});
    a->wait();
    const TierStats after = mgr.stats();
    T_REQUIRE(after.layer_pause.p99_ms == before.layer_pause.p99_ms &&
              after.layer_pause.p95_ms == before.layer_pause.p95_ms,
              "pin_groups не пишет паузу слоя (пауза — на вызывающем)");
    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
}

T_MAIN

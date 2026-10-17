// Тест router-lookahead prefetch-предиктора и hit-rate (PLAN.md §4, поток B).
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

T_CASE(predictor_markov_top_k) {
    MarkovPredictor p;
    // Контролируемая история: 0→1 трижды, 0→2 один раз.
    for (int i = 0; i < 3; ++i) p.observe(GroupId{0}, GroupId{1});
    p.observe(GroupId{0}, GroupId{2});
    const auto next = p.predict(GroupId{0}, 2);
    T_REQUIRE(next.size() == 2, "top-2");
    T_REQUIRE(next[0].value == 1, "наиболее вероятное продолжение — 1");
    T_REQUIRE((next[1].value == 2), "второй кандидат — 2");
    T_REQUIRE(p.predict(GroupId{5}, 2).empty(), "неизвестный префикс → пусто");
    T_REQUIRE(p.predict(GroupId{0}, 0).empty(), "k=0 → пусто");
    // Самопереходы не регистрируются.
    MarkovPredictor p2;
    p2.observe(GroupId{7}, GroupId{7});
    T_REQUIRE(p2.predict(GroupId{7}, 1).empty(), "self-transition игнорируется");
}

T_CASE(predictor_history_cap) {
    MarkovPredictor p(3);
    for (int i = 0; i < 100; ++i) p.observe(GroupId{0}, GroupId{1});
    T_REQUIRE(!p.predict(GroupId{0}, 1).empty(), "история жива после деления счётчиков");
}

// Hit-rate предиктора через менеджер: контролируемая история обращений
// 0 → 1 → 0 → 1 ... с lookahead 1 → prefetch следующей группы до pin.
T_CASE(predictor_hit_rate_via_manager) {
    const std::size_t kSlot = 4096;
    const std::string path = "/tmp/stell_tier_test_pred.stelltier";
    std::vector<TensorGroup> groups;
    for (std::uint32_t i = 0; i < 4; ++i)
        groups.push_back(make_group(i, "g" + std::to_string(i), Role::Expert,
                                    DType::F16, kSlot));
    TierLayoutHeader h;
    h.n_groups = groups.size();
    TierLayoutWriter w(path, h);
    for (const auto& g : groups) {
        std::vector<std::uint8_t> payload(kSlot, std::uint8_t(g.id.value));
        w.add_group(g, 1, payload.data(), payload.size());
    }
    w.finalize();

    TierConfig cfg;
    cfg.vram_slots = 4; // все resident одновременно — eviction не мешает prefetch
    cfg.vram_slot_size = kSlot;
    cfg.ram_capacity = 8 * kSlot;
    cfg.nvme_layout = path;
    cfg.lookahead = 1;
    TierManager mgr(cfg);
    mgr.set_plan(groups);

    const auto pin_and_wait = [&](std::uint32_t id) {
        auto a = mgr.pin(GroupKey{GroupId{id}, Role::Expert});
        a->wait();
    };

    // Прогрев: первые два обращения — холодные промахи.
    pin_and_wait(0);
    pin_and_wait(1); // блокируется до завершения загрузки (job от pin(1))

    // Цикл 0 → 1 двадцать раз; предиктор prefetch'ит следующую группу.
    for (int i = 0; i < 20; ++i) {
        pin_and_wait(0);
        pin_and_wait(1);
    }
    const TierStats st = mgr.stats();
    T_REQUIRE(st.prefetch.hit_rate >= 0.6, "hit_rate предиктора ≥ 60% (PLAN.md §4)");
    std::printf("    hit_rate=%.3f hits=%llu misses=%llu\n", st.prefetch.hit_rate,
                (unsigned long long)st.prefetch.hits, (unsigned long long)st.prefetch.misses);
}

T_MAIN

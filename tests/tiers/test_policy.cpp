// Тесты политики eviction LRU/LCU (со статистикой маршрутизатора).
#include <stell/tiers/policy.hpp>

#include <algorithm>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

static std::vector<ResidentCandidate> sample() {
    return {
        ResidentCandidate{GroupId{1}, /*last_use*/5, /*use_count*/10, /*router*/0},
        ResidentCandidate{GroupId{2}, /*last_use*/9, /*use_count*/10, /*router*/0},
        ResidentCandidate{GroupId{3}, /*last_use*/3, /*use_count*/30, /*router*/0},
    };
}

T_CASE(policy_lru_picks_least_recent) {
    const auto c = sample();
    T_REQUIRE(choose_victim(c, EvictionPolicy::Lru).value == 3, "LRU: жертва = минимальный last_use");
    // Обновим last_use кандидата 3 → жертва 1.
    auto c2 = sample();
    c2[2].last_use_tick = 100;
    T_REQUIRE(choose_victim(c2, EvictionPolicy::Lru).value == 1, "LRU пересчитывается");
}

T_CASE(policy_lcu_uses_router_statistics) {
    const auto c = sample();
    // LCU: сумма use_count + router_count. Кандидат 3 имеет наибольший use_count,
    // но его можно «спасти» статистикой маршрутизатора наоборот — проверим базу.
    T_REQUIRE(choose_victim(c, EvictionPolicy::Lcu).value == 1,
              "LCU: жертва = минимальный суммарный счётчик (1 и 2 по 10)");
    auto c2 = sample();
    c2[0].router_count = 100; // маршрутизатор часто выбирает группу 1 → она дольше живёт в VRAM
    T_REQUIRE(choose_victim(c2, EvictionPolicy::Lcu).value == 2,
              "статистика маршрутизатора меняет жертву LCU");
}

T_CASE(policy_empty_candidates_slot_exhausted) {
    bool thrown = false;
    try {
        choose_victim({}, EvictionPolicy::Lru);
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::SlotExhausted;
    }
    T_REQUIRE(thrown, "нет кандидатов → SlotExhausted");
}

T_MAIN

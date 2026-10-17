// Политика выгрузки resident-групп: LRU / LCU (со статистикой маршрутизатора).
#include <stell/tiers/policy.hpp>

#include <algorithm>
#include <cassert>

namespace stell::tiers {

GroupId choose_victim(const std::vector<ResidentCandidate>& candidates, EvictionPolicy policy) {
    if (candidates.empty())
        throw TierError(TierError::Code::SlotExhausted, "eviction: no candidates");
    assert(!candidates.empty());
    const ResidentCandidate* best = &candidates[0];
    for (const auto& c : candidates) {
        if (policy == EvictionPolicy::Lru) {
            // LRU: жертва с наименьшим tick последнего обращения.
            if (c.last_use_tick < best->last_use_tick) best = &c;
        } else {
            // LCU: наименьший суммарный счётчик использований (обращения
            // планировщика + статистика маршрутизатора); тай-брейк — LRU.
            const std::uint64_t score = c.use_count + c.router_count;
            const std::uint64_t best_score = best->use_count + best->router_count;
            if (score < best_score || (score == best_score && c.last_use_tick < best->last_use_tick))
                best = &c;
        }
    }
    return best->id;
}

} // namespace stell::tiers

// Политика выгрузки resident-групп из VRAM-тира: LRU / LCU (least count use).
// LCU учитывает статистику маршрутизатора (частоты выбора экспертов), поданную
// через TierManager::report_router_counts() — интерфейс подачи статистики
// по ADR-08 §5 (метрики/политика tiers; ветвлений по role в tiers нет).
#pragma once

#include <cstdint>
#include <vector>

#include <stell/tiers/errors.hpp>
#include <stell/tiers/types.hpp>

namespace stell::tiers {

enum class EvictionPolicy {
    Lru, // least recently used: жертва = минимум last_use_tick
    Lcu, // least count use: жертва = минимум (use_count + router_count), тай-брейк по last_use
};

struct ResidentCandidate {
    GroupId     id;
    std::uint64_t last_use_tick = 0;
    std::uint64_t use_count     = 0;
    std::uint64_t router_count  = 0;
};

// Выбирает жертву eviction из resident-кандидатов (все кандидаты — unpinned).
GroupId choose_victim(const std::vector<ResidentCandidate>& candidates, EvictionPolicy policy);

} // namespace stell::tiers

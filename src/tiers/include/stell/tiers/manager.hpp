// ADR-08 §2: интерфейс Tier-менеджера (контракт engine ↔ tiers).
// Сигнатуры менять нельзя; конкретная реализация — stell/tiers/tier_manager.hpp.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <stell/tiers/types.hpp>

namespace stell::tiers {

struct PrefetchStats {
    std::uint64_t hits;                 // prefetch попал: группа уже в VRAM к использованию
    std::uint64_t misses;
    double        hit_rate;             // hits / (hits + misses)
};

struct PauseStats {                     // паузы между слоями (ожидание весов)
    double p50_ms;
    double p95_ms;                      // целевой порог ≤ 5 мс на FP16-профиле (PLAN.md §4)
    double p99_ms;
};

struct TierStats {
    PrefetchStats prefetch;
    PauseStats    layer_pause;
    std::array<std::uint64_t, 3> tier_bytes;   // [VRAM, pinned RAM, NVMe] занято
    std::uint64_t io_bytes;                     // суммарно перенесено тир↔тир
    std::uint64_t io_ops;                       // количество I/O-операций
};

// Асинхронный handle ожидания готовности групп (для планировщика).
class Awaiter {
public:
    virtual ~Awaiter() = default;
    virtual bool ready() const = 0;                    // неблокирующая проверка
    virtual void wait() = 0;                            // блокирующее ожидание
};

class ITierManager {
public:
    virtual ~ITierManager() = default;

    // Разметка: engine передаёт полный план групп одной модели (см. ADR-09/ADR-12).
    // Должна вызываться ДО старта decode-потоков; повторный вызов = ResetPlan.
    virtual void set_plan(const std::vector<TensorGroup>& groups) = 0;

    // --- pin / unpin -----------------------------------------------------
    virtual std::unique_ptr<Awaiter> pin(const GroupKey& key) = 0;
    virtual void unpin(const GroupKey& key) = 0;       // снять запрет eviction
    virtual std::unique_ptr<Awaiter> pin_layer(std::uint32_t layer) = 0;

    // --- prefetch ----------------------------------------------------------
    // Асинхронный, идемпотентный, НЕ блокирует.
    virtual void prefetch(const std::vector<GroupKey>& keys) = 0;

    // --- stats --------------------------------------------------------------
    virtual TierStats stats() const = 0;
};

} // namespace stell::tiers

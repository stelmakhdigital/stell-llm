// Router-lookahead prefetch-предиктор (PLAN.md §4, поток B).
// Предсказывает следующие группы по истории обращений (марковская модель 1-го
// порядка по GroupId). Tiers не знает ни модели, ни роутера — только историю
// идентификаторов групп, поданную через pin/prefetch.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <stell/tiers/types.hpp>

namespace stell::tiers {

class IPredictor {
public:
    virtual ~IPredictor() = default;
    // Зафиксировать переход prev → next в истории.
    virtual void observe(GroupId prev, GroupId next) = 0;
    // Топ-k наиболее вероятных продолжений от last (без учёта самого last).
    virtual std::vector<GroupId> predict(GroupId last, std::size_t k) const = 0;
};

class MarkovPredictor : public IPredictor {
public:
    explicit MarkovPredictor(std::size_t history_cap = 4096);

    void observe(GroupId prev, GroupId next) override;
    std::vector<GroupId> predict(GroupId last, std::size_t k) const override;

private:
    std::size_t history_cap_;                  // предел суммарных наблюдений (ring)
    std::uint64_t observed_ = 0;
    std::unordered_map<std::uint32_t,
        std::unordered_map<std::uint32_t, std::uint64_t>> counts_;
};

} // namespace stell::tiers

// Пул слотов фиксированного размера (ADR-05: фиксированные пулы, константная
// разметка; ADR-08: slot_bytes групп фиксируется ДО старта графа).
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace stell::engine {

using SlotId = std::uint32_t;

// Интерфейс пула: scheduler работает против него; CUDA-конфигурация поверх
// добавляет device-буферы, CPU-конфигурация — только bookkeeping.
class ISlotPool {
public:
    virtual ~ISlotPool() = default;

    virtual std::uint32_t capacity() const = 0;
    // Слот выделяется или отказ (нет исключений на горячем пути планировщика:
    // отказ == «сессия встаёт в очередь/ошибка слот-пула», ADR-11 §3 error).
    virtual bool acquire(SlotId& out) = 0;
    virtual void release(SlotId slot) = 0;
    virtual bool occupied(SlotId slot) const = 0;
    virtual std::size_t busy() const = 0;
    // Фиксированный размер слота в байтах (ADR-05): константа пула, не меняется.
    virtual std::size_t slot_bytes() const = 0;
};

// Реализация: фиксированная ёмкость, LIFO free-list, потокобезопасность
// (планировщик + фоновые постановки запросов — ADR-11 §2).
class FixedSlotPool : public ISlotPool {
public:
    FixedSlotPool(std::uint32_t capacity, std::size_t slot_bytes)
        : capacity_(capacity), slot_bytes_(slot_bytes),
          occupied_(capacity, false), free_()
    {
        free_.reserve(capacity);
        for (std::uint32_t i = 0; i < capacity; ++i) free_.push_back(i);
    }

    std::uint32_t capacity() const override { return capacity_; }
    std::size_t   slot_bytes() const override { return slot_bytes_; }

    bool acquire(SlotId& out) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (free_.empty()) return false;
        out = free_.back();
        free_.pop_back();
        occupied_[out] = true;
        return true;
    }

    void release(SlotId slot) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (slot >= capacity_ || !occupied_[slot]) return;
        occupied_[slot] = false;
        free_.push_back(slot);
    }

    bool occupied(SlotId slot) const override {
        std::lock_guard<std::mutex> lk(mu_);
        return slot < capacity_ && occupied_[slot];
    }

    std::size_t busy() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<std::size_t>(occupied_.size() - free_.size());
    }

private:
    std::uint32_t capacity_;
    std::size_t slot_bytes_;
    std::vector<bool> occupied_;
    std::vector<SlotId> free_;
    mutable std::mutex mu_;
};

} // namespace stell::engine

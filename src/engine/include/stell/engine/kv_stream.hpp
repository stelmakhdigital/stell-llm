// KV-streaming для длинного контекста (PLAN.md §4 поток A; ADR-04 dtype;
// ADR-10: KvStream поверх chain-send/кольцевой очереди).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace stell::engine::kv {

// Синк с семантикой parallel::ICommunicator::kv_stream_push/pop (ADR-10 §2):
// push — неблокирующий; pop — false при пустой очереди или несовпадении размера.
class IKvStreamSink {
public:
    virtual ~IKvStreamSink() = default;
    virtual void push(const void* block, std::size_t bytes) = 0;
    virtual bool pop(void* dst, std::size_t bytes) = 0;
    virtual std::size_t pending() const = 0;
};

// CPU-реализация: кольцевая очередь блоков (SPSC-семантика + mutex для тестов).
class KvRingBuffer : public IKvStreamSink {
public:
    void push(const void* block, std::size_t bytes) override {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(std::vector<char>(
            static_cast<const char*>(block),
            static_cast<const char*>(block) + bytes));
    }

    bool pop(void* dst, std::size_t bytes) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (queue_.empty()) return false;
        auto& front = queue_.front();
        if (front.size() != bytes) return false;    // mismatch — контракт нарушен вызывающим
        std::memcpy(dst, front.data(), bytes);
        queue_.pop_front();
        return true;
    }

    std::size_t pending() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return queue_.size();
    }

private:
    mutable std::mutex mu_;
    std::deque<std::vector<char>> queue_;
};

// Менеджер длинного контекста: KV-блоки слота сверх локального бюджета
// уезжают в поток (синк), выгрузка/загрузка — в порядке seq (ADR-04: tiers
// переносит KV-блоки, не зная про модель).
class KvStreamer {
public:
    KvStreamer(IKvStreamSink& sink, std::size_t block_bytes,
               std::size_t local_budget_blocks)
        : sink_(&sink), block_bytes_(block_bytes), budget_(local_budget_blocks) {}

    std::size_t block_bytes() const { return block_bytes_; }

    // Записать KV-блок слота (seq — монотонный номер блока). Если локальный
    // бюджет исчерпан, самый старый локальный блок выталкивается в поток.
    void push_block(std::uint32_t slot, std::uint32_t seq, const void* data) {
        std::lock_guard<std::mutex> lk(mu_);
        Local b{};
        b.slot = slot; b.seq = seq;
        b.data.assign(static_cast<const char*>(data),
                      static_cast<const char*>(data) + block_bytes_);
        local_.push_back(std::move(b));
        while (local_.size() > budget_) {
            const Local& old = local_.front();
            sink_->push(old.data.data(), old.data.size());
            local_.pop_front();
            ++streamed_;
        }
    }

    // Вытянуть готовый блок из потока (в порядке отправки). false — пусто.
    bool pull_block(void* dst) { return sink_->pop(dst, block_bytes_); }

    std::size_t local_blocks() const {
        std::lock_guard<std::mutex> lk(mu_);
        return local_.size();
    }

    std::size_t streamed_blocks() const {
        std::lock_guard<std::mutex> lk(mu_);
        return streamed_;
    }

private:
    struct Local {
        std::uint32_t slot = 0;
        std::uint32_t seq = 0;
        std::vector<char> data;
    };

    IKvStreamSink* sink_;
    std::size_t block_bytes_;
    std::size_t budget_;
    std::deque<Local> local_;
    std::uint64_t streamed_ = 0;
    mutable std::mutex mu_;
};

} // namespace stell::engine::kv

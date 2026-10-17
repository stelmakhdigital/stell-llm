// FAKE/STUB header (engine-owned build aid) — ИНТЕРФЕЙС ВЫПИСАН ДОСЛОВНО из ADR-10 §2.
// Единственный источник истины — docs/adr/ADR-10-engine-parallel.md.
// После вклейки реального модуля parallel (src/parallel/include/stell/parallel/comm.hpp)
// заменить на реальный заголовок. Только объявления — реализация (NCCL и CPU-эмулятор)
// не здесь (поток C).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace stell::parallel {

using rank_t = std::int32_t;

// Тип данных комм — только те, что реально гоняются по NVLink (ADR-04).
enum class CommDType { F16, F32, E4M3 };

class ICommunicator {
public:
    virtual ~ICommunicator() = default;
    virtual rank_t world_size() const = 0;
    virtual rank_t rank() const = 0;

    // Все операции синхронны по семантике, запуск на переданном CUDA-stream
    // (CPU-эмуляция stream игнорирует).
    virtual void allreduce(void* buf, std::size_t count, CommDType t) = 0;
    virtual void allgather(void* send, void* recv, std::size_t count, CommDType t) = 0;

    // EP: send_tokens[i] уходит на ранок owner(expert_id[i]); recv — локальные
    // токены других ранков.
    virtual void expert_alltoall(
        void* send_tokens, void* send_meta,
        void* recv_tokens, void* recv_meta,
        std::size_t max_tokens, CommDType t) = 0;

    // PP: point-to-point по цепочке (peer = rank±1).
    virtual void chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) = 0;
    virtual void chain_recv(void* buf, std::size_t count, CommDType t, rank_t peer) = 0;

    // KV-streaming: неблокирующий кольцевой канал KV-блоков (ADR-04 dtype).
    virtual void kv_stream_push(const void* kv_block, std::size_t bytes) = 0;
    virtual bool  kv_stream_pop(void* dst, std::size_t bytes) = 0;
};

// Конфиг параллелизма — из конфига engine (ADR-10 §2: { tp, pp, ep, device_ids }).
struct ParallelConfig {
    std::uint32_t tp = 1;
    std::uint32_t pp = 1;
    std::uint32_t ep = 1;
    std::vector<int> device_ids;
};

// Ошибки — исключениями (единая политика с ADR-08 §3).
class CommError : public std::runtime_error {
public:
    enum class Code { NcclFailure, RankMismatch, BufferTooSmall, Timeout };
    CommError(Code c, const std::string& what)
        : std::runtime_error(what), code_(c) {}
    Code code() const noexcept { return code_; }
private:
    Code code_;
};

// Фабрика (ADR-10 §2/§4): make_communicator(const ParallelConfig&) — реализует поток C.
// Здесь НЕ реализована; engine против неё только абстрагируется.

} // namespace stell::parallel

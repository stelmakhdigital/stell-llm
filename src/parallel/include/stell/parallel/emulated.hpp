// stell-llm — parallel: CPU-эмулятор комм-паттернов (ADR-10 §4).
//
// Никаких NCCL/GPU/CUDA: ранки = потоки одного процесса; инстансы с одинаковым
// (group_id, world) образуют процессную группу. Коллективы — барьерные
// (rendezvous всех ранков), expert_alltoall — общие очереди, chain_send/recv —
// FIFO-каналы парами ранков, KvStream — кольцевые SPSC-подобные очереди.
// Тайм-ауты поддерживаются той же ошибкой Timeout (тесты deadlock-свободности).
#pragma once

#include <memory>
#include <vector>
#include "stell/parallel/comm.hpp"

namespace stell::parallel {

// Состояние коллективов группы/подгруппы (детали в .cpp). Полная группа и
// каждая подгруппа имеют собственное состояние — коллективы подгруппы не
// пересекаются с коллективами родителя.
struct SubgroupState;

// Детали группы скрыты в .cpp; заголовок нужен фабрике и тестам.
// Под-комм (subgroup, ADR-10 round 2) создаётся только методом subgroup()
// родителя и принадлежит ему — публичный ctor всегда создаёт группу целиком.
class EmulatedCommunicator final : public ICommunicator {
public:
    explicit EmulatedCommunicator(const ParallelConfig& cfg);
    ~EmulatedCommunicator() override;

    rank_t world_size() const override;
    rank_t rank() const override;

    void allreduce(void* buf, std::size_t count, CommDType t) override;
    void allgather(void* send, void* recv, std::size_t count, CommDType t) override;
    void expert_alltoall(void* send_tokens, void* send_meta,
                         void* recv_tokens, void* recv_meta,
                         std::size_t max_tokens, std::size_t token_elems,
                         CommDType t) override;
    void chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) override;
    void chain_recv(void* buf, std::size_t count, CommDType t, rank_t peer) override;
    void kv_stream_push(const void* kv_block, std::size_t bytes) override;
    bool kv_stream_pop(void* dst, std::size_t bytes) override;

    ICommunicator* subgroup(const std::vector<rank_t>& ranks) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Приватный ctor подгруппы: члены (отсортированы), позиция данного ранка
    // в членах и общее состояние коллективов подгруппы.
    EmulatedCommunicator(const ParallelConfig& cfg,
                         const std::vector<rank_t>& members, rank_t my_pos,
                         const std::shared_ptr<struct SubgroupState>& rs);
};

} // namespace stell::parallel

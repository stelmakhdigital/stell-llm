// stell-llm — parallel: CPU-эмулятор комм-паттернов (ADR-10 §4). Реализация.
//
// Ранки = потоки одного процесса; инстансы с одинаковым group_id и world
// образуют процессную группу. Коллективы — барьерные (рандеву всех ранков),
// редукция детерминирована (порядок ранков, аккумулятор F32, см. ADR-10 §4).
// После ошибки Timeout состояние группы не гарантируется — пересоздайте
// группу (новый group_id); тесты это соблюдают.
#include "stell/parallel/emulated.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/plan.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace stell::parallel {
namespace {

constexpr rank_t kMaxRanks = 64;

// --- процессная группа --------------------------------------------------------
struct GroupState {
    std::mutex m;                     // охраняет registered/world
    rank_t world = 0;
    rank_t registered = 0;

    // chain-каналы: ключ (src, dst) -> FIFO блоков байтов.
    struct Chain {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::vector<std::uint8_t>> q;
    };
    std::map<std::pair<rank_t, rank_t>, Chain*> chains;

    // KV-кольца: ключ — ранок-получатель (блоки приходят от rank-1).
    struct KvRing {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::vector<std::uint8_t>> q;
        std::size_t capacity = 16;
    };
    std::map<rank_t, KvRing*> kv_rings;

    // Рандеву коллективов: по состоянию на тип коллектива; epoch — номер
    // последней завершённой операции. Слоты буферов — по ранкам.
    struct Rendez {
        std::mutex m;
        std::condition_variable cv;
        std::uint64_t epoch = 0;
        int arrived = 0;
        CommError::Code error = CommError::Code::NcclFailure;
        void* bufs[kMaxRanks] = {};         // allreduce in-place | allgather send | alltoall tokens
        void* bufs2[kMaxRanks] = {};        // allgather recv | alltoall send_meta
        void* bufs3[kMaxRanks] = {};        // alltoall recv_tokens
        void* bufs4[kMaxRanks] = {};        // alltoall recv_meta
        std::size_t counts[kMaxRanks] = {}; // count | alltoall max_tokens
        std::size_t caps[kMaxRanks] = {};   // alltoall: ёмкость recv (max_tokens*world)
    };
    Rendez allreduce_r, allgather_r, alltoall_r;

    ~GroupState() {
        for (auto& kv : chains) delete kv.second;
        for (auto& kv : kv_rings) delete kv.second;
    }

    Chain& chain(rank_t src, rank_t dst) {
        std::lock_guard<std::mutex> lk(m);   // защита ленивого создания канала
        auto& slot = chains[{src, dst}];
        if (!slot) slot = new Chain();
        return *slot;
    }
    KvRing& kv_ring(rank_t dst, std::size_t capacity = 0) {
        std::lock_guard<std::mutex> lk(m);   // защита ленивого создания кольца
        auto& slot = kv_rings[dst];
        if (!slot) {
            slot = new KvRing();
            if (capacity) slot->capacity = capacity;
        }
        return *slot;
    }
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}
std::map<int, std::shared_ptr<GroupState>>& group_registry() {
    static std::map<int, std::shared_ptr<GroupState>> map;
    return map;
}

std::shared_ptr<GroupState> get_group(int group_id) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto& slot = group_registry()[group_id];
    if (!slot) slot = std::make_shared<GroupState>();
    return slot;
}

// Редукция по ранкам 0,1,...,world-1; аккумулятор F32; неассоциативность
// float фиксируется этим порядком (golden-тест TP2 == TP1 опирается на него).
float reduce_in_rank_order(GroupState::Rendez& r, rank_t world, std::size_t idx, CommDType t) {
    const std::size_t es = dtype_size(t);
    float acc = comm_to_float(t, static_cast<std::uint8_t*>(r.bufs[0]) + idx * es);
    for (rank_t rr = 1; rr < world; ++rr)
        acc += comm_to_float(t, static_cast<std::uint8_t*>(r.bufs[rr]) + idx * es);
    return acc;
}

template <class Reg, class Exec>
// Общая схема рандеву: reg() регистрирует слоты ранка (под mutex, до
// ожидания); последний прибывший исполняет exec (пишет результаты напрямую
// в буферы всех участников — один процесс); при ошибке все получают ту же
// CommError; при тайм-ауте ранок получает Timeout.
void rendez_collective(GroupState::Rendez& r, rank_t /*my_rank*/, rank_t world,
                       int timeout_ms, Reg&& reg, Exec&& exec) {
    std::unique_lock<std::mutex> lk(r.m);
    const std::uint64_t my_epoch = r.epoch;
    reg(lk);
    if (++r.arrived < world) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        while (r.epoch == my_epoch) {
            if (r.cv.wait_until(lk, deadline) == std::cv_status::timeout) {
                throw CommError(CommError::Code::Timeout, "collective: peer did not arrive");
            }
        }
        if (r.error != CommError::Code::NcclFailure)
            throw CommError(r.error, "collective failed on another rank");
        return;
    }
    try {
        exec();
    } catch (const CommError& e) {
        r.error = e.code();
    }
    r.arrived = 0;
    ++r.epoch;
    r.cv.notify_all();
}

} // namespace

// --- EmulatedCommunicator ------------------------------------------------------

struct EmulatedCommunicator::Impl {
    ParallelConfig cfg;
    std::shared_ptr<GroupState> g;
};

EmulatedCommunicator::EmulatedCommunicator(const ParallelConfig& cfg) : impl_(new Impl) {
    check_topology(cfg);
    if (cfg.mode != CommMode::CpuEmulation)
        throw CommError(CommError::Code::NcclFailure,
                        "EmulatedCommunicator требует mode=CpuEmulation");
    if (cfg.rank < 0 || cfg.rank >= cfg.world())
        throw CommError(CommError::Code::RankMismatch, "rank вне [0, world)");
    if (cfg.world() > kMaxRanks)
        throw CommError(CommError::Code::RankMismatch,
                        "CPU-эмулятор ограничен 64 ранками (kMaxRanks)");
    impl_->cfg = cfg;
    impl_->g = get_group(cfg.group_id);
    GroupState& g = *impl_->g;
    std::lock_guard<std::mutex> lk(g.m);
    if (g.registered == 0) g.world = cfg.world();
    if (g.world != cfg.world())
        throw CommError(CommError::Code::RankMismatch,
                        "world не совпал с ранее зарегистрированной группой (group_id)");
    ++g.registered;
}

EmulatedCommunicator::~EmulatedCommunicator() = default;

rank_t EmulatedCommunicator::world_size() const { return impl_->cfg.world(); }
rank_t EmulatedCommunicator::rank() const { return impl_->cfg.rank; }

void EmulatedCommunicator::allreduce(void* buf, std::size_t count, CommDType t) {
    GroupState& g = *impl_->g;
    GroupState::Rendez& r = g.allreduce_r;
    const rank_t world = impl_->cfg.world(), my = impl_->cfg.rank;
    rendez_collective(
        r, my, world, impl_->cfg.timeout_ms,
        [&](std::unique_lock<std::mutex>&) {
            r.bufs[my] = buf;
            r.counts[my] = count;
        },
        [&] {
            for (rank_t q = 0; q < world; ++q)
                if (r.counts[q] != count)
                    throw CommError(CommError::Code::RankMismatch,
                                    "allreduce: count не одинаков по ранкам");
            const std::size_t es = dtype_size(t);
            for (std::size_t i = 0; i < count; ++i) {
                const float acc = reduce_in_rank_order(r, world, i, t);
                for (rank_t d = 0; d < world; ++d)
                    comm_store_float(t, static_cast<std::uint8_t*>(r.bufs[d]) + i * es, acc);
            }
        });
}

void EmulatedCommunicator::allgather(void* send, void* recv, std::size_t count, CommDType t) {
    GroupState& g = *impl_->g;
    GroupState::Rendez& r = g.allgather_r;
    const rank_t world = impl_->cfg.world(), my = impl_->cfg.rank;
    rendez_collective(
        r, my, world, impl_->cfg.timeout_ms,
        [&](std::unique_lock<std::mutex>&) {
            r.bufs[my] = send;
            r.bufs2[my] = recv;
            r.counts[my] = count;
        },
        [&] {
            for (rank_t q = 0; q < world; ++q)
                if (r.counts[q] != count)
                    throw CommError(CommError::Code::RankMismatch,
                                    "allgather: count не одинаков по ранкам");
            const std::size_t es = dtype_size(t);
            for (rank_t d = 0; d < world; ++d)
                for (rank_t s = 0; s < world; ++s)
                    std::memcpy(static_cast<std::uint8_t*>(r.bufs2[d]) +
                                    std::size_t(s) * count * es,
                                r.bufs[s], count * es);
        });
}

void EmulatedCommunicator::expert_alltoall(
    void* send_tokens, void* send_meta, void* recv_tokens, void* recv_meta,
    std::size_t max_tokens, CommDType t) {
    GroupState& g = *impl_->g;
    GroupState::Rendez& r = g.alltoall_r;
    const rank_t world = impl_->cfg.world(), my = impl_->cfg.rank;
    const ParallelConfig cfg = impl_->cfg;
    const std::size_t es = dtype_size(t);
    rendez_collective(
        r, my, world, impl_->cfg.timeout_ms,
        [&](std::unique_lock<std::mutex>&) {
            r.bufs[my] = send_tokens;
            r.bufs2[my] = send_meta;
            r.bufs3[my] = recv_tokens;
            r.bufs4[my] = recv_meta;
            r.counts[my] = max_tokens;
            r.caps[my] = max_tokens * std::size_t(world);
        },
        [&] {
            for (rank_t q = 0; q < world; ++q)
                if (r.counts[q] != max_tokens)
                    throw CommError(CommError::Code::RankMismatch,
                                    "expert_alltoall: max_tokens не одинаков по ранкам");
            // Входящие на каждый ранок: список (байты токена, метка) в порядке
            // (ранок-источник ↑, индекс ↑) — детерминировано.
            std::vector<std::uint8_t> in_tokens[static_cast<std::size_t>(kMaxRanks)];
            std::vector<TokenMeta> in_meta[static_cast<std::size_t>(kMaxRanks)];
            for (rank_t s = 0; s < world; ++s) {
                const RankDecomp sd = decompose_rank(s, cfg);
                auto* meta = static_cast<TokenMeta*>(r.bufs2[s]);
                for (std::size_t i = 0; i < max_tokens; ++i) {
                    const std::int32_t eid = meta[i].expert_id;
                    if (eid < 0) continue;                   // пустой слот — не отправляется
                    rank_t dst[static_cast<std::size_t>(kMaxRanks)];
                    expert_alltoall_route(rank_t(eid), sd.pp_rank, cfg, dst);
                    for (rank_t k = 0; k < cfg.tp; ++k) {
                        const rank_t d = dst[k];
                        if (d >= world)
                            throw CommError(CommError::Code::RankMismatch, "route вне world");
                        const std::uint8_t* tok =
                            static_cast<std::uint8_t*>(r.bufs[s]) + i * es;
                        in_tokens[d].insert(in_tokens[d].end(), tok, tok + es);
                        in_meta[d].push_back(meta[i]);
                    }
                }
            }
            // Запись в recv-буферы: входящие, затем заполнение TokenMeta{-1}.
            for (rank_t d = 0; d < world; ++d) {
                const std::size_t cap = r.caps[d];
                if (in_meta[d].size() > cap)
                    throw CommError(CommError::Code::BufferTooSmall,
                                    "expert_alltoall: recv-ёмкость меньше входящего потока");
                auto* rt = static_cast<std::uint8_t*>(r.bufs3[d]);
                auto* rm = static_cast<TokenMeta*>(r.bufs4[d]);
                std::size_t off = 0;
                for (std::size_t i = 0; i < in_meta[d].size(); ++i) {
                    std::memcpy(rt + off, in_tokens[d].data() + i * es, es);
                    off += es;
                    rm[i] = in_meta[d][i];
                }
                for (std::size_t i = in_meta[d].size(); i < cap; ++i) {
                    std::memset(rt + off, 0, es);
                    off += es;
                    rm[i] = TokenMeta{};                     // expert_id = -1
                }
            }
        });
}

void EmulatedCommunicator::chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) {
    GroupState& g = *impl_->g;
    const rank_t my = impl_->cfg.rank;
    if (peer < 0 || peer >= impl_->cfg.world() || peer == my)
        throw CommError(CommError::Code::RankMismatch, "chain_send: неверный peer");
    const std::size_t bytes = count * dtype_size(t);
    GroupState::Chain& ch = g.chain(my, peer);
    std::vector<std::uint8_t> data(bytes);
    std::memcpy(data.data(), buf, bytes);
    {
        std::lock_guard<std::mutex> lk(ch.m);
        ch.q.push_back(std::move(data));
    }
    ch.cv.notify_all();
}

void EmulatedCommunicator::chain_recv(void* buf, std::size_t count, CommDType t, rank_t peer) {
    GroupState& g = *impl_->g;
    const rank_t my = impl_->cfg.rank;
    if (peer < 0 || peer >= impl_->cfg.world() || peer == my)
        throw CommError(CommError::Code::RankMismatch, "chain_recv: неверный peer");
    GroupState::Chain& ch = g.chain(peer, my);
    const std::size_t bytes = count * dtype_size(t);
    std::unique_lock<std::mutex> lk(ch.m);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(impl_->cfg.timeout_ms);
    while (ch.q.empty()) {
        if (ch.cv.wait_until(lk, deadline) == std::cv_status::timeout)
            throw CommError(CommError::Code::Timeout, "chain_recv: блок не пришёл");
    }
    if (ch.q.front().size() != bytes)
        throw CommError(CommError::Code::BufferTooSmall, "chain_recv: размер блока не совпал");
    std::memcpy(buf, ch.q.front().data(), bytes);
    ch.q.pop_front();
}

void EmulatedCommunicator::kv_stream_push(const void* kv_block, std::size_t bytes) {
    GroupState& g = *impl_->g;
    const rank_t my = impl_->cfg.rank;
    const rank_t dst = rank_t((my + 1) % std::size_t(impl_->cfg.world()));
    GroupState::KvRing& ring =
        g.kv_ring(dst, impl_->cfg.kv_ring_capacity ? impl_->cfg.kv_ring_capacity : 16);
    std::unique_lock<std::mutex> lk(ring.m);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(impl_->cfg.timeout_ms);
    while (ring.q.size() >= ring.capacity) {
        if (ring.cv.wait_until(lk, deadline) == std::cv_status::timeout)
            throw CommError(CommError::Code::Timeout, "kv_stream_push: кольцо переполнено");
    }
    ring.q.emplace_back(static_cast<const std::uint8_t*>(kv_block),
                        static_cast<const std::uint8_t*>(kv_block) + bytes);
    ring.cv.notify_all();
}

bool EmulatedCommunicator::kv_stream_pop(void* dst, std::size_t bytes) {
    GroupState& g = *impl_->g;
    GroupState::KvRing& ring = g.kv_ring(impl_->cfg.rank);
    std::lock_guard<std::mutex> lk(ring.m);
    if (ring.q.empty()) return false;
    if (ring.q.front().size() != bytes)
        throw CommError(CommError::Code::BufferTooSmall, "kv_stream_pop: размер блока не совпал");
    std::memcpy(dst, ring.q.front().data(), bytes);
    ring.q.pop_front();
    ring.cv.notify_all();
    return true;
}

} // namespace stell::parallel

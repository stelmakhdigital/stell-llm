// stell-llm — parallel: CPU-эмулятор комм-паттернов (ADR-10 §4). Реализация.
//
// Ранки = потоки одного процесса; инстансы с одинаковым group_id и world
// образуют процессную группу. Коллективы — барьерные (рандеву всех ранков),
// редукция детерминирована (порядок ранков, аккумулятор F32, см. ADR-10 §4).
// После ошибки Timeout состояние группы не гарантируется — пересоздайте
// группу (новый group_id); тесты это соблюдают.
//
// Раунд 2 (уточнения ADR-10): expert_alltoall несёт token_elems (элементов
// dtype на токен); subgroup(ranks) возвращает под-комм только по своим ранкам
// (перенумерация по возрастанию глобального ранка, владение у родителя);
// нехватка recv-ёмкости — CommError(BufferTooSmall), молчаливое сокращение
// пересылки запрещено (в т.ч. маршруты за пределы подгруппы).
#include "stell/parallel/emulated.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/plan.hpp"

#include <algorithm>
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

// Рандеву коллективов: по состоянию на тип коллектива; epoch — номер
// последней завершённой операции. Слоты буферов — по ПОЗИЦИЯМ в группе/
// подгруппе (для полной группы позиция == глобальному ранку; для подгруппы
// — по возрастанию глобального ранка, перенумерация ADR-10 round 2).
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
    std::size_t te[kMaxRanks] = {};     // alltoall: token_elems (одинаков по ранкам)
    std::size_t caps[kMaxRanks] = {};   // alltoall: ёмкость recv в токенах (max_tokens*n)
};

} // namespace

// Состояние коллективов группы/подгруппы (ADR-10 round 2): полная группа
// и каждая подгруппа имеют собственное состояние — коллективы подгруппы
// не пересекаются с коллективами родителя и с другими подгруппами. Члены
// хранятся по возрастанию глобального ранка (перенумерация подгруппы).
struct SubgroupState {
    std::vector<rank_t> members;
    Rendez allreduce_r, allgather_r, alltoall_r;
};

namespace {

// --- процессная группа --------------------------------------------------------
struct GroupState {
    std::mutex m;                     // охраняет registered/world/subgroups
    rank_t world = 0;
    rank_t registered = 0;

    // chain-каналы: ключ (src, dst) -> FIFO блоков байтов (глобальные ранки).
    struct Chain {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::vector<std::uint8_t>> q;
    };
    std::map<std::pair<rank_t, rank_t>, Chain*> chains;

    // KV-кольца: ключ — глобальный ранок-получатель (блоки приходят от rank-1).
    struct KvRing {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::vector<std::uint8_t>> q;
        std::size_t capacity = 16;
    };
    std::map<rank_t, KvRing*> kv_rings;

    std::shared_ptr<SubgroupState> full;    // вся группа; создаётся в ctor GroupState
    // Ключ — отсортированный список членов; под мьютексом m.
    std::map<std::vector<rank_t>, std::shared_ptr<SubgroupState>> subgroups;

    GroupState() : full(std::make_shared<SubgroupState>()) {}

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

// Редукция по позициям 0,1,...,n-1 (для полной группы — по возрастанию
// глобального ранка); аккумулятор F32; неассоциативность float фиксируется
// этим порядком (golden-тест TP2 == TP1 опирается на него).
float reduce_in_rank_order(Rendez& r, rank_t n, std::size_t idx, CommDType t) {
    const std::size_t es = dtype_size(t);
    float acc = comm_to_float(t, static_cast<std::uint8_t*>(r.bufs[0]) + idx * es);
    for (rank_t rr = 1; rr < n; ++rr)
        acc += comm_to_float(t, static_cast<std::uint8_t*>(r.bufs[rr]) + idx * es);
    return acc;
}

template <class Reg, class Exec>
// Общая схема рандеву: reg() регистрирует слоты ранка (под mutex, до
// ожидания); последний прибывший исполняет exec (пишет результаты напрямую
// в буферы всех участников — один процесс); при ошибке все получают ту же
// CommError; при тайм-ауте ранок получает Timeout.
void rendez_collective(Rendez& r, rank_t /*my_rank*/, rank_t world,
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
    // Контракт (ADR-10 §2/§4): при ошибке ВСЕ ранки получают ту же CommError —
    // включая последнего прибывшего (исполнителя exec); NcclFailure — страж
    // «ошибки нет».
    if (r.error != CommError::Code::NcclFailure)
        throw CommError(r.error, "collective failed on this rank");
}

// Валидация списка членов подгруппы: уникальные глобальные ранки из текущего
// состава, содержащие собственный ранок my_global. Возвращает отсортированный
// список. Бросает CommError(RankMismatch).
std::vector<rank_t> validated_members(const std::vector<rank_t>& ranks,
                                      const std::vector<rank_t>& cur_members,
                                      rank_t my_global) {
    std::vector<rank_t> sorted = ranks;
    std::sort(sorted.begin(), sorted.end());
    const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
    if (dup != sorted.end())
        throw CommError(CommError::Code::RankMismatch, "subgroup: повторяющиеся ранки");
    for (rank_t r : sorted) {
        if (r < 0)
            throw CommError(CommError::Code::RankMismatch, "subgroup: ранок вне world");
        if (!std::binary_search(cur_members.begin(), cur_members.end(), r))
            throw CommError(CommError::Code::RankMismatch,
                            "subgroup: ранок не входит в текущий комм");
    }
    if (!std::binary_search(sorted.begin(), sorted.end(), my_global))
        throw CommError(CommError::Code::RankMismatch,
                        "subgroup: собственный ранок не входит в подгруппу");
    return sorted;
}

} // namespace

// --- EmulatedCommunicator ------------------------------------------------------

struct EmulatedCommunicator::Impl {
    ParallelConfig cfg;                                // родительская топология (мир, таймауты)
    std::shared_ptr<GroupState> g;                     // состояние процессной группы
    std::shared_ptr<SubgroupState> rs;                 // коллективы: вся группа или подгруппа
    std::vector<rank_t> members;                       // глобальные ранки-участники (по возрастанию)
    rank_t my_pos = 0;                                 // позиция данного инстанса в members
    std::vector<rank_t> pos_of;                        // глобальный ранок -> позиция (или -1)
    // Владение под-коммами у родителя (ADR-10 round 2): живут в деструкторе
    // родителя; повторный subgroup() с тем же ranks возвращает тот же объект.
    std::map<std::vector<rank_t>, std::unique_ptr<EmulatedCommunicator>> owned_subs;

    rank_t sub_world() const { return rank_t(members.size()); }
    rank_t my_global() const { return members[std::size_t(my_pos)]; }
    // Позиция глобального ранка d в подгруппе; -1 — не член (для полной группы
    // всегда >= 0 в пределах world).
    rank_t position_of(rank_t d) const {
        if (std::size_t(d) >= pos_of.size()) return -1;
        return pos_of[std::size_t(d)];
    }
    rank_t resolve_peer(rank_t peer) const {
        if (peer < 0 || peer >= sub_world())
            throw CommError(CommError::Code::RankMismatch,
                            "peer вне подгруппы (перенумерация ADR-10 round 2)");
        return members[std::size_t(peer)];
    }
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
    impl_->rs.reset();   // заполнится ниже (full)
    GroupState& g = *impl_->g;
    std::lock_guard<std::mutex> lk(g.m);
    if (g.registered == 0) g.world = cfg.world();
    if (g.world != cfg.world())
        throw CommError(CommError::Code::RankMismatch,
                        "world не совпал с ранее зарегистрированной группой (group_id)");
    ++g.registered;
    // Полная группа: члены 0..world-1, позиция == глобальному ранку.
    impl_->rs = g.full;
    for (rank_t r = 0; r < cfg.world(); ++r) impl_->members.push_back(r);
    impl_->my_pos = cfg.rank;
    impl_->pos_of.assign(std::size_t(cfg.world()), 0);
    for (rank_t r = 0; r < cfg.world(); ++r) impl_->pos_of[std::size_t(r)] = r;
}

EmulatedCommunicator::EmulatedCommunicator(const ParallelConfig& cfg,
                                           const std::vector<rank_t>& members,
                                           rank_t my_pos,
                                           const std::shared_ptr<SubgroupState>& rs)
    : impl_(new Impl) {
    impl_->cfg = cfg;
    impl_->g = get_group(cfg.group_id);
    impl_->rs = rs;
    impl_->members = members;
    impl_->my_pos = my_pos;
    // pos_of: глобальный ранок -> позиция в members (остальное — -1).
    rank_t maxr = 0;
    for (rank_t r : members) maxr = std::max(maxr, rank_t(r + 1));
    impl_->pos_of.assign(std::size_t(maxr), -1);
    for (std::size_t i = 0; i < members.size(); ++i)
        impl_->pos_of[members[i]] = rank_t(i);
}

EmulatedCommunicator::~EmulatedCommunicator() = default;

rank_t EmulatedCommunicator::world_size() const { return impl_->sub_world(); }
rank_t EmulatedCommunicator::rank() const { return impl_->my_pos; }

void EmulatedCommunicator::allreduce(void* buf, std::size_t count, CommDType t) {
    SubgroupState& rs = *impl_->rs;
    Rendez& r = rs.allreduce_r;
    const rank_t n = impl_->sub_world(), my = impl_->my_pos;
    rendez_collective(
        r, my, n, impl_->cfg.timeout_ms,
        [&](std::unique_lock<std::mutex>&) {
            r.bufs[my] = buf;
            r.counts[my] = count;
        },
        [&] {
            for (rank_t q = 0; q < n; ++q)
                if (r.counts[q] != count)
                    throw CommError(CommError::Code::RankMismatch,
                                    "allreduce: count не одинаков по ранкам");
            const std::size_t es = dtype_size(t);
            for (std::size_t i = 0; i < count; ++i) {
                const float acc = reduce_in_rank_order(r, n, i, t);
                for (rank_t d = 0; d < n; ++d)
                    comm_store_float(t, static_cast<std::uint8_t*>(r.bufs[d]) + i * es, acc);
            }
        });
}

void EmulatedCommunicator::allgather(void* send, void* recv, std::size_t count, CommDType t) {
    SubgroupState& rs = *impl_->rs;
    Rendez& r = rs.allgather_r;
    const rank_t n = impl_->sub_world(), my = impl_->my_pos;
    rendez_collective(
        r, my, n, impl_->cfg.timeout_ms,
        [&](std::unique_lock<std::mutex>&) {
            r.bufs[my] = send;
            r.bufs2[my] = recv;
            r.counts[my] = count;
        },
        [&] {
            for (rank_t q = 0; q < n; ++q)
                if (r.counts[q] != count)
                    throw CommError(CommError::Code::RankMismatch,
                                    "allgather: count не одинаков по ранкам");
            const std::size_t es = dtype_size(t);
            for (rank_t d = 0; d < n; ++d)
                for (rank_t s = 0; s < n; ++s)
                    std::memcpy(static_cast<std::uint8_t*>(r.bufs2[d]) +
                                    std::size_t(s) * count * es,
                                r.bufs[s], count * es);
        });
}

void EmulatedCommunicator::expert_alltoall(
    void* send_tokens, void* send_meta, void* recv_tokens, void* recv_meta,
    std::size_t max_tokens, std::size_t token_elems, CommDType t) {
    if (token_elems == 0)
        throw CommError(CommError::Code::RankMismatch,
                        "expert_alltoall: token_elems должен быть > 0");
    SubgroupState& rs = *impl_->rs;
    Rendez& r = rs.alltoall_r;
    const rank_t n = impl_->sub_world(), my = impl_->my_pos;
    const ParallelConfig cfg = impl_->cfg;      // маршрутизация — по родительской топологии
    const auto& members = impl_->members;
    const std::size_t es = dtype_size(t);
    rendez_collective(
        r, my, n, impl_->cfg.timeout_ms,
        [&](std::unique_lock<std::mutex>&) {
            r.bufs[my] = send_tokens;
            r.bufs2[my] = send_meta;
            r.bufs3[my] = recv_tokens;
            r.bufs4[my] = recv_meta;
            r.counts[my] = max_tokens;
            r.te[my] = token_elems;
            r.caps[my] = max_tokens * std::size_t(n);   // ёмкость recv в токенах
        },
        [&] {
            for (rank_t q = 0; q < n; ++q) {
                if (r.counts[q] != max_tokens)
                    throw CommError(CommError::Code::RankMismatch,
                                    "expert_alltoall: max_tokens не одинаков по ранкам");
                if (r.te[q] != token_elems)
                    throw CommError(CommError::Code::RankMismatch,
                                    "expert_alltoall: token_elems не одинаков по ранкам");
            }
            // Входящие на каждую позицию: список (байты токена, метка) в порядке
            // (ранок-источник ↑, индекс ↑) — детерминировано. Маршрутизация —
            // по РОДИТЕЛЬСКОЙ топологии (глобальные ранки); позиция назначения
            // ищется в составе подгруппы.
            std::vector<std::uint8_t> in_tokens[static_cast<std::size_t>(kMaxRanks)];
            std::vector<TokenMeta> in_meta[static_cast<std::size_t>(kMaxRanks)];
            for (rank_t p = 0; p < n; ++p) {
                const rank_t s = members[std::size_t(p)];
                const RankDecomp sd = decompose_rank(s, cfg);
                auto* meta = static_cast<TokenMeta*>(r.bufs2[p]);
                for (std::size_t i = 0; i < max_tokens; ++i) {
                    const std::int32_t eid = meta[i].expert_id;
                    if (eid < 0) continue;                   // пустой слот — не отправляется
                    rank_t dst[static_cast<std::size_t>(kMaxRanks)];
                    expert_alltoall_route(rank_t(eid), sd.pp_rank, cfg, dst);
                    for (rank_t k = 0; k < cfg.tp; ++k) {
                        const rank_t d = dst[k];
                        if (d < 0 || d >= cfg.world())
                            throw CommError(CommError::Code::RankMismatch, "route вне world");
                        const rank_t dpos = impl_->position_of(d);
                        if (dpos < 0)
                            // Маршрут уходит за пределы подгруппы: приёмник не
                            // входит в состав — входящий поток не помещается в
                            // recv-ёмкость подгруппы. Молчаливое сокращение
                            // пересылки запрещено (ADR-10, round 2).
                            throw CommError(CommError::Code::BufferTooSmall,
                                            "expert_alltoall: маршрут вне подгруппы "
                                            "(recv-ёмкость подгруппы не покрывает поток)");
                        const std::uint8_t* tok =
                            static_cast<std::uint8_t*>(r.bufs[p]) + i * token_elems * es;
                        in_tokens[dpos].insert(in_tokens[dpos].end(), tok,
                                               tok + token_elems * es);
                        in_meta[dpos].push_back(meta[i]);
                    }
                }
            }
            // Запись в recv-буферы: входящие токены (token_elems элементов
            // каждый), затем заполнение нулями и TokenMeta{-1}.
            for (rank_t p = 0; p < n; ++p) {
                const std::size_t cap = r.caps[p];
                if (in_meta[p].size() > cap)
                    throw CommError(CommError::Code::BufferTooSmall,
                                    "expert_alltoall: recv-ёмкость меньше входящего потока");
                auto* rt = static_cast<std::uint8_t*>(r.bufs3[p]);
                auto* rm = static_cast<TokenMeta*>(r.bufs4[p]);
                const std::size_t tb = token_elems * es;
                std::size_t off = 0;
                for (std::size_t i = 0; i < in_meta[p].size(); ++i) {
                    std::memcpy(rt + off, in_tokens[p].data() + i * tb, tb);
                    off += tb;
                    rm[i] = in_meta[p][i];
                }
                for (std::size_t i = in_meta[p].size(); i < cap; ++i) {
                    std::memset(rt + off, 0, tb);
                    off += tb;
                    rm[i] = TokenMeta{};                     // expert_id = -1
                }
            }
        });
}

void EmulatedCommunicator::chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) {
    GroupState& g = *impl_->g;
    const rank_t my = impl_->my_global();
    const rank_t dst = impl_->resolve_peer(peer);
    if (dst == my)
        throw CommError(CommError::Code::RankMismatch, "chain_send: неверный peer");
    const std::size_t bytes = count * dtype_size(t);
    GroupState::Chain& ch = g.chain(my, dst);
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
    const rank_t my = impl_->my_global();
    const rank_t src = impl_->resolve_peer(peer);
    if (src == my)
        throw CommError(CommError::Code::RankMismatch, "chain_recv: неверный peer");
    GroupState::Chain& ch = g.chain(src, my);
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
    const rank_t n = impl_->sub_world();
    const rank_t dst = impl_->members[std::size_t((impl_->my_pos + 1) % n)];
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
    // Ёмкость кольца задаётся симметрично (kv_ring_capacity обоих ранков):
    // кто первым создаст кольцо (push/pop), тот и фиксирует ёмкость.
    GroupState::KvRing& ring =
        g.kv_ring(impl_->my_global(),
                  impl_->cfg.kv_ring_capacity ? impl_->cfg.kv_ring_capacity : 16);
    std::lock_guard<std::mutex> lk(ring.m);
    if (ring.q.empty()) return false;
    if (ring.q.front().size() != bytes)
        throw CommError(CommError::Code::BufferTooSmall, "kv_stream_pop: размер блока не совпал");
    std::memcpy(dst, ring.q.front().data(), bytes);
    ring.q.pop_front();
    ring.cv.notify_all();
    return true;
}

ICommunicator* EmulatedCommunicator::subgroup(const std::vector<rank_t>& ranks) {
    GroupState& g = *impl_->g;
    const rank_t my_global = impl_->my_global();
    const std::vector<rank_t> sorted =
        validated_members(ranks, impl_->members, my_global);
    // Владение у родителя (ADR-10 round 2): под-комм хранится в родителе,
    // удаляется его деструктором; повторный вызов с тем же ranks возвращает
    // ТОТ ЖЕ под-комм.
    auto& slot = impl_->owned_subs[sorted];
    if (!slot) {
        std::lock_guard<std::mutex> lk(g.m);
        auto& st = g.subgroups[sorted];
        if (!st) {
            st = std::make_shared<SubgroupState>();
            st->members = sorted;
        }
        const rank_t my_pos = rank_t(std::find(sorted.begin(), sorted.end(), my_global) -
                                     sorted.begin());
        slot.reset(new EmulatedCommunicator(impl_->cfg, sorted, my_pos, st));
    }
    return slot.get();
}

} // namespace stell::parallel

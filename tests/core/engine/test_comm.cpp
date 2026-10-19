// Стык engine ↔ parallel (ADR-10 §3): планировщик на ранке 0 рассылает мету
// шага через allgather; слоты/сессии одинаковы на ранках. FakeCommunicator
// (CPU-эмуляция — реализация comm принадлежит потоку C).
#include "test_util.hpp"

#include <cstring>
#include <map>
#include <vector>

#include "stell/engine/scheduler.hpp"

using namespace stell::engine;

namespace {

class FakeComm : public stell::parallel::ICommunicator {
public:
    stell::parallel::rank_t world_size() const override { return 1; }
    stell::parallel::rank_t rank() const override { return 0; }

    void allreduce(void* buf, std::size_t count, stell::parallel::CommDType t) override {
        (void)buf; (void)count; (void)t;
        ++allreduce_calls;
    }
    void allgather(void* send, void* recv, std::size_t count,
                   stell::parallel::CommDType t) override {
        (void)t;
        ++allgather_calls;
        // CPU-эмуляция: recv = send (ранки в одной памяти)
        std::memcpy(recv, send, count * sizeof(float));
        // мета шага: [шаг, размер батча, ...] — фиксируем для проверок
        const float* m = static_cast<const float*>(send);
        if (count >= 2) {
            last_step = static_cast<std::uint64_t>(m[0]);
            last_batch = static_cast<std::uint32_t>(m[1]);
        }
    }
    void expert_alltoall(void*, void*, void*, void*, std::size_t, std::size_t,
                         stell::parallel::CommDType) override { ++ea_calls; }
    void chain_send(void*, std::size_t, stell::parallel::CommDType,
                    stell::parallel::rank_t) override { ++cs_calls; }
    void chain_recv(void*, std::size_t, stell::parallel::CommDType,
                    stell::parallel::rank_t) override { ++cr_calls; }
    void kv_stream_push(const void*, std::size_t) override { ++push_calls; }
    bool kv_stream_pop(void* dst, std::size_t bytes) override {
        (void)dst; (void)bytes;
        ++pop_calls;
        return false;
    }
    // Подгруппа (ADR-10 round 2): world=1 — единственная валидная подгруппа
    // {0}; возвращаем сам объект (владение у родителя — это и есть родитель).
    stell::parallel::ICommunicator* subgroup(const std::vector<stell::parallel::rank_t>& ranks) override {
        if (ranks.size() == 1 && ranks[0] == 0) return this;
        throw stell::parallel::CommError(stell::parallel::CommError::Code::RankMismatch,
                                         "FakeComm: подгруппа world=1 — только {0}");
    }

    int allreduce_calls = 0, allgather_calls = 0, ea_calls = 0;
    int cs_calls = 0, cr_calls = 0, push_calls = 0, pop_calls = 0;
    std::uint64_t last_step = 0;
    std::uint32_t last_batch = 0;
};

class FakeRunner : public IModelRunner {
public:
    std::vector<DecodeOutput> decode_step(const std::vector<DecodeInput>& in) override {
        std::vector<DecodeOutput> out;
        for (const auto& i : in) out.push_back({i.slot, 42});
        return out;
    }
};

} // namespace

void run() {
    FakeComm comm;
    FakeRunner runner;
    FixedSlotPool pool(16, 1024);

    // broadcast_step_meta=false: планировщик не трогает comm
    Scheduler sched1(runner, pool, &comm, Scheduler::Config{});
    sched1.submit({{1, 2}, {3, 1.0f, 0.95f, 20, 0}, -1});
    sched1.step();
    STELL_CHECK(comm.allgather_calls == 0);
    sched1.cancel(sched1.poll(1).empty() ? 1 : 1);   // cancel сессии id=1
    (void)sched1.poll(1);

    // broadcast_step_meta=true: каждый шаг рассылает мету (ADR-10 §3)
    Scheduler::Config cfg{};
    cfg.broadcast_step_meta = true;
    Scheduler sched2(runner, pool, &comm, cfg);
    SessionId a = sched2.submit({{5, 6, 7}, {3, 1.0f, 0.95f, 20, 0}, -1});
    SessionId b = sched2.submit({{8, 9}, {3, 1.0f, 0.95f, 20, 0}, -1});
    sched2.step();
    STELL_CHECK(comm.allgather_calls == 1);
    STELL_CHECK(comm.last_batch == 2);               // мета: размер батча шага
    sched2.step();
    STELL_CHECK(comm.allgather_calls == 2);
    STELL_CHECK(comm.last_step == 1);                // мета: индекс шага

    sched2.step();                                   // шаг 3 → finish(length), слоты свободны
    (void)sched2.poll(a);
    (void)sched2.poll(b);
    STELL_CHECK(pool.busy() == 0);
}

STELL_TEST_MAIN(run)

// Интеграция engine ↔ parallel (round 2, ADR-10 + кларификация): реальный
// CPU-эмулятор ICommunicator в scheduler meta-протоколе — allgather F32
// 32 байта (8 слотов: [шаг, размер батча, 6 зарезервированных]; ранг 0 —
// источник меты). Тест multi-rank эмуляции со слотами сессий: world=2
// (tp=2), два Scheduler'а-ранка в lockstep-барье; детерминизм: одинаковые
// сессии/слоты на ранках дают одинаковые события декода.
#include "test_util.hpp"

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "stell/engine/scheduler.hpp"
#include "stell/parallel/emulated.hpp"

using namespace stell::engine;
using namespace stell::parallel;
using stell::parallel::ParallelConfig;

namespace {

class DetRunner : public IModelRunner {
public:
    std::vector<DecodeOutput> decode_step(const std::vector<DecodeInput>& in) override {
        std::vector<DecodeOutput> out;
        for (const auto& i : in) {
            const int c = counter_[i.slot]++;
            out.push_back({i.slot, c});      // детерминированно: 0,1,2,... по слоту
        }
        return out;
    }
    std::unordered_map<SlotId, int> counter_;
};

struct RankPipe {
    FixedSlotPool pool;
    DetRunner runner;
    Scheduler sched;
    RankPipe(stell::parallel::ICommunicator* comm, Scheduler::Config cfg)
        : pool(4, 4096), sched(runner, pool, comm, cfg) {}
};

} // namespace

void run() {
    // --- CPU-эмулятор: world=2, tp=2 (ADR-10 §4: никаких NCCL/GPU) -----------
    ParallelConfig pc{};
    pc.mode = stell::parallel::CommMode::CpuEmulation;
    pc.tp = 2;
    pc.pp = 1;
    pc.ep = 1;
    pc.group_id = 771;                        // свежая процессная группа
    pc.rank = 0;
    pc.timeout_ms = 8000;
    auto comm0 = stell::parallel::make_communicator(pc);
    STELL_CHECK(comm0 != nullptr);
    STELL_CHECK(comm0->world_size() == 2);
    STELL_CHECK(comm0->rank() == 0);

    pc.rank = 1;
    auto comm1 = stell::parallel::make_communicator(pc);
    STELL_CHECK(comm1 != nullptr);
    STELL_CHECK(comm1->rank() == 1);

    Scheduler::Config cfg{};
    cfg.batch_step_max = 8;
    cfg.broadcast_step_meta = true;           // ADR-10 §3: мета шага через allgather

    RankPipe r0{comm0.get(), cfg};
    RankPipe r1{comm1.get(), cfg};

    // Одинаковые сессии/слоты на обоих ранках (ADR-10 §3: слоты одинаковы,
    // содержимое слота шардировано по ранкам — здесь CPU-эмуляция).
    SubmitRequest req{};
    req.prompt_tokens = {7, 11};
    req.sampling.max_tokens = 4;
    req.eos_token = -1;
    req.sampling.seed = 0;
    const SessionId s0 = r0.sched.submit(req);
    const SessionId s1 = r1.sched.submit(req);
    STELL_CHECK(s0 != 0 && s1 != 0);

    // --- мета-протокол: шаги ранков в lockstep (двухфазный барьер: рандеву
    // allgather требует одновременных вызовов коллективов обоих ранков) -------
    struct Barrier {
        std::atomic<int> count{0};
        std::atomic<int> gen{0};
        void wait() {
            const int g = gen.load();
            if (count.fetch_add(1) == 1) {
                count.store(0);
                gen.store(g + 1);
            } else {
                while (gen.load() == g) std::this_thread::yield();
            }
        }
    };
    Barrier barrier{};
    const int kSteps = 4;
    auto step_loop = [&](RankPipe* rp) {
        for (int i = 0; i < kSteps; ++i) {
            barrier.wait();                       // ранки синхронны перед каждым шагом
            rp->sched.step();
        }
    };
    std::thread t0(step_loop, &r0);
    std::thread t1(step_loop, &r1);
    t0.join();
    t1.join();

    // --- детерминизм: события декода ранков совпадают (ADR-10 §3) -----------
    const std::vector<Event> ev0 = r0.sched.poll(s0);
    const std::vector<Event> ev1 = r1.sched.poll(s1);
    STELL_CHECK(ev0.size() == ev1.size());
    for (std::size_t i = 0; i < ev0.size(); ++i) {
        STELL_CHECK(ev0[i].type == ev1[i].type);
        if (ev0[i].type == Event::Type::Token)
            STELL_CHECK(ev0[i].token_id == ev1[i].token_id);
        if (ev0[i].type == Event::Type::Finish)
            STELL_CHECK(ev0[i].finish_reason == ev1[i].finish_reason);
    }
    STELL_CHECK(r0.sched.steps_done() == r1.sched.steps_done());

    // --- allgather F32 32 байта: прямая проверка мета-протокола --------------
    {
        ParallelConfig p2{};
        p2.mode = stell::parallel::CommMode::CpuEmulation;
        p2.tp = 3;
        p2.group_id = 772;
        p2.rank = 0;
        p2.timeout_ms = 8000;
        auto c0 = stell::parallel::make_communicator(p2);
        ParallelConfig p3 = p2;
        p3.rank = 2;
        auto c2 = stell::parallel::make_communicator(p3);
        ParallelConfig p1 = p2;
        p1.rank = 1;
        auto c1 = stell::parallel::make_communicator(p1);

        // мета: ранг 0 — источник [шаг=5, батч=2, остальные 0]; 8×F32 = 32 байта
        float send0[8] = {5, 2, 0, 0, 0, 0, 0, 0};
        float send1[8] = {0};
        float send2[8] = {0};
        float recv0[24] = {}, recv1[24] = {}, recv2[24] = {};   // 8*world
        std::thread ta([&] { c0->allgather(send0, recv0, 8, stell::parallel::CommDType::F32); });
        std::thread tb([&] { c2->allgather(send2, recv2, 8, stell::parallel::CommDType::F32); });
        c1->allgather(send1, recv1, 8, stell::parallel::CommDType::F32);
        ta.join();
        tb.join();
        // ранг 0 — источник меты: слоты [0..7] в recv каждого ранка = мета ранка 0
        for (int i = 0; i < 8; ++i) {
            STELL_CHECK(recv0[i] == send0[i]);
            STELL_CHECK(recv1[i] == send0[i]);
            STELL_CHECK(recv2[i] == send0[i]);
        }
    }
}

STELL_TEST_MAIN(run)

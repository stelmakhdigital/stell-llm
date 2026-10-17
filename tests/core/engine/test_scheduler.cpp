// Планировщик continuous batching: ≥8 сессий, события ADR-11 §3, finish по
// eos/max_tokens, cancel, poll, слоты освобождаются.
#include "test_util.hpp"

#include <map>
#include <vector>

#include "stell/engine/scheduler.hpp"

using namespace stell::engine;

namespace {

class FakeRunner : public IModelRunner {
public:
    // per-slot счётчик: токен = base + counter
    std::vector<DecodeOutput> decode_step(const std::vector<DecodeInput>& in) override {
        std::vector<DecodeOutput> out;
        for (const auto& i : in) {
            int& c = counter_[i.slot];
            out.push_back({i.slot, base_ + c});
            ++c;
        }
        return out;
    }
    std::map<SlotId, int> counter_;
    std::int32_t base_ = 0;
};

} // namespace

void run() {
    // --- ≥8 одновременных сессий --------------------------------------------
    FixedSlotPool pool(16, 1024);
    FakeRunner runner;
    Scheduler sched(runner, pool);
    STELL_CHECK(pool.capacity() >= 8);

    SubmitRequest req;
    req.prompt_tokens = {0, 1, 2};
    req.sampling.max_tokens = 3;
    req.eos_token = 1000000;                 // не достигается

    std::vector<SessionId> ids;
    for (int i = 0; i < 12; ++i) {           // 12 > 8: continuous batching держит
        SessionId id = sched.submit(req);
        STELL_CHECK(id != 0);
        ids.push_back(id);
    }
    STELL_CHECK(sched.active() == 12);
    STELL_CHECK(pool.busy() == 12);

    // --- шаг декода: события token в порядке батча, seq монотонен -----------
    sched.step();
    for (SessionId id : ids) {
        std::vector<Event> evs = sched.poll(id);
        STELL_CHECK(evs.size() == 1);
        STELL_CHECK(evs[0].type == Event::Type::Token);
        STELL_CHECK(evs[0].seq == 1);
        STELL_CHECK(evs[0].token_id == 0);   // base=0, counter=0
    }
    STELL_CHECK(sched.steps_done() == 1);

    sched.step();
    for (SessionId id : ids) {
        std::vector<Event> evs = sched.poll(id);
        STELL_CHECK(evs.size() == 1);
        STELL_CHECK(evs[0].token_id == 1);
        STELL_CHECK(evs[0].seq == 2);
    }

    sched.step();                            // completion=3 == max_tokens → length
    for (SessionId id : ids) {
        std::vector<Event> evs = sched.poll(id);
        STELL_CHECK(evs.size() == 2);        // token(2) + finish(length)
        STELL_CHECK(evs[0].type == Event::Type::Token);
        STELL_CHECK(evs[0].token_id == 2);
        STELL_CHECK(evs[1].type == Event::Type::Finish);
        STELL_CHECK(evs[1].finish_reason == "length");
        STELL_CHECK(evs[1].prompt_tokens == 3);
        STELL_CHECK(evs[1].completion_tokens == 3);
    }
    // слоты освобождены
    STELL_CHECK(pool.busy() == 0);
    // запись сессии удалена после выкачивания
    STELL_CHECK(sched.active() == 0);
    // повторная poll — пусто
    STELL_CHECK(sched.poll(ids[0]).empty());

    // --- finish по eos -------------------------------------------------------
    FakeRunner runner2;                      // base=2: первый же токен = 2 == eos
    runner2.base_ = 2;
    Scheduler sched3(runner2, pool);
    SubmitRequest req2;
    req2.prompt_tokens = {7, 8};
    req2.sampling.max_tokens = 100;
    req2.eos_token = 2;
    SessionId id = sched3.submit(req2);
    sched3.step();
    std::vector<Event> evs = sched3.poll(id);
    STELL_CHECK(evs.size() == 2);            // token + finish(stop)
    STELL_CHECK(evs[1].type == Event::Type::Finish);
    STELL_CHECK(evs[1].finish_reason == "stop");
    STELL_CHECK(evs[1].completion_tokens == 1);
    STELL_CHECK(pool.busy() == 0);

    // --- cancel --------------------------------------------------------------
    Scheduler sched4(runner, pool);
    SessionId c1 = sched4.submit(req);
    sched4.step();
    (void)sched4.poll(c1);
    STELL_CHECK(sched4.cancel(c1));
    std::vector<Event> evs2 = sched4.poll(c1);
    STELL_CHECK(evs2.size() == 1);
    STELL_CHECK(evs2[0].type == Event::Type::Finish);
    STELL_CHECK(evs2[0].finish_reason == "cancelled");
    STELL_CHECK(!sched4.cancel(c1));         // повторный cancel — false

    // --- исчерпание слотов: submit → 0 --------------------------------------
    FixedSlotPool pool1(1, 64);
    Scheduler sched5(runner, pool1);
    SessionId a = sched5.submit(req);
    STELL_CHECK(a != 0);
    STELL_CHECK(sched5.submit(req) == 0);    // нет слотов
    STELL_CHECK(sched5.cancel(a));
    STELL_CHECK(sched5.submit(req) != 0);    // слот вернулся
}

STELL_TEST_MAIN(run)

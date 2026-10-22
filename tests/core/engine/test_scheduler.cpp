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

    // =========================================================================
    // [R3] round 10 (ADR-13 [R4]): mixed-prefill budget. Входящий большой
    // префилл не блокирует resident-декодеры: декодер делает ТЕ ЖЕ шаги в
    // тех же раундах, что и без бюджета; префилл откладывается; телеметрия
    // счётчика прижатых шагов: >0 при сжатии, ==0 при default (без лимита).
    // =========================================================================
    {
        FixedSlotPool poolB(8, 4096);
        FakeRunner runnerB;
        const std::vector<std::int32_t> small_prompt = {1, 2};       // 2 токена
        const std::vector<std::int32_t> big_prompt(100, 7);          // 100 токенов
        const std::int32_t eos_big = 1000000;

        // (1) Эталон БЕЗ бюджета (default): resident-декодер шагает.
        Scheduler ref(runnerB, poolB);
        SessionId ra = ref.submit([&]{ SubmitRequest r; r.prompt_tokens = small_prompt;
            r.sampling.max_tokens = 4; r.eos_token = eos_big; return r; }());
        ref.step();                                    // A стала resident (префилл+1 декод)
        (void)ref.poll(ra);
        SessionId rb = ref.submit([&]{ SubmitRequest r; r.prompt_tokens = big_prompt;
            r.sampling.max_tokens = 2; r.eos_token = eos_big; return r; }());
        for (int i = 0; i < 4; ++i) ref.step();
        const std::uint32_t ref_a_completion = [&]{
            for (const Event& e : ref.poll(ra))
                if (e.type == Event::Type::Finish) return e.completion_tokens;
            return 0u; }();
        STELL_CHECK(ref_a_completion == 4);            // эталон: 4 декод-шага A
        STELL_CHECK(ref.prefill_deferrals() == 0);     // без бюджета — 0

        // (2) С бюджетом 4: B (100 токенов > бюджета) откладывается, A — шагает.
        Scheduler::Config cfgB{};
        cfgB.prefill_token_budget = 4;
        Scheduler bud(runnerB, poolB, nullptr, cfgB);
        SessionId ba = bud.submit([&]{ SubmitRequest r; r.prompt_tokens = small_prompt;
            r.sampling.max_tokens = 4; r.eos_token = eos_big; return r; }());
        bud.step();                                    // префилл A (2≤4) + 1 декод
        (void)bud.poll(ba);
        SessionId bb = bud.submit([&]{ SubmitRequest r; r.prompt_tokens = big_prompt;
            r.sampling.max_tokens = 2; r.eos_token = eos_big; return r; }());
        for (int i = 0; i < 4; ++i) bud.step();
        const std::uint32_t bud_a_completion = [&]{
            for (const Event& e : bud.poll(ba))
                if (e.type == Event::Type::Finish) return e.completion_tokens;
            return 0u; }();
        STELL_CHECK_MSG(bud_a_completion == ref_a_completion,
                        "бюджет префиллов не должен снижать число декод-шагов "
                        "resident-сессии за те же итерации планировщика");
        // B так и не начала декодировать (префилл отложен — больших шагов нет).
        STELL_CHECK(bud.poll(bb).empty());
        STELL_CHECK_MSG(bud.prefill_deferrals() == 4,
                        "каждый из 4 шагов прижат бюджетом (B не пущена)");

        // (3) Телеметрия: default (без лимита) — счётчик 0 (уже проверено на
        //     ref); при включённом бюджете и отложенном префилле — >0 (выше).
        //     Дополнительно: префилл, влезающий в бюджет, планируется.
        Scheduler::Config cfgB2{};
        cfgB2.prefill_token_budget = 8;
        Scheduler bud2(runnerB, poolB, nullptr, cfgB2);
        SessionId c = bud2.submit([&]{ SubmitRequest r; r.prompt_tokens = small_prompt;
            r.sampling.max_tokens = 2; r.eos_token = eos_big; return r; }());
        bud2.step();
        STELL_CHECK(bud2.poll(c).size() == 1);        // префилл 2≤8 — прошёл
        STELL_CHECK(bud2.prefill_deferrals() == 0);   // ничего не прижато
    }
}

STELL_TEST_MAIN(run)

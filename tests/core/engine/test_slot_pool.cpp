// Пул слотов фиксированного размера: ёмкость, отказ при исчерпании, LIFO reuse,
// потокобезопасность (ADR-05, ADR-11 §2: ≥8 одновременных сессий).
#include "test_util.hpp"

#include <atomic>
#include <thread>
#include <vector>

#include "stell/engine/slot_pool.hpp"

using stell::engine::FixedSlotPool;

void run() {
    FixedSlotPool pool(8, 4096);              // ≥8 сессий — целевой минимум
    STELL_CHECK(pool.capacity() == 8);
    STELL_CHECK(pool.slot_bytes() == 4096);

    // исчерпание: 8 acquire успешны, 9-й — отказ
    std::vector<stell::engine::SlotId> taken;
    stell::engine::SlotId s = 0;
    for (int i = 0; i < 8; ++i) {
        STELL_CHECK(pool.acquire(s));
        taken.push_back(s);
        STELL_CHECK(pool.occupied(s));
    }
    STELL_CHECK(pool.busy() == 8);
    STELL_CHECK(!pool.acquire(s));

    // release + reuse (LIFO)
    stell::engine::SlotId freed = taken.back();
    pool.release(freed);
    STELL_CHECK(!pool.occupied(freed));
    stell::engine::SlotId again = 999;
    STELL_CHECK(pool.acquire(again));
    STELL_CHECK(again == freed);              // LIFO free-list
    pool.release(again);

    // release уже свободного — идемпотентно
    pool.release(freed);
    STELL_CHECK(pool.busy() == 7);

    // конкурентный acquire/release: суммарно занятых слотов не растёт сверх пула
    FixedSlotPool pool2(16, 64);
    std::atomic<int> failures{0};
    {
        std::vector<std::thread> ths;
        for (int t = 0; t < 4; ++t)
            ths.emplace_back([&pool2, &failures] {
                for (int i = 0; i < 1000; ++i) {
                    stell::engine::SlotId x = 0;
                    if (!pool2.acquire(x)) { ++failures; continue; }
                    pool2.release(x);
                }
            });
        for (auto& th : ths) th.join();
    }
    STELL_CHECK_MSG(failures.load() == 0, "конкурентный пул не должен отказывать при release-цикле");
    STELL_CHECK(pool2.busy() == 0);
}

STELL_TEST_MAIN(run)

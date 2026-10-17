// Потокобезопасность ITierManager (ADR-08 §4): pin/unpin/prefetch/stats
// из нескольких потоков одновременно + идемпотентность prefetch → pin.
#include <stell/tiers/tier_manager.hpp>

#include <atomic>
#include <cmath>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

static const std::size_t kSlot = 4096;

T_CASE(stats_thread_safety) {
    std::vector<TensorGroup> groups;
    for (std::uint32_t i = 0; i < 6; ++i)
        groups.push_back(make_group(i, "g" + std::to_string(i), Role::Expert,
                                    DType::F16, kSlot));
    const std::string path = "/tmp/stell_tier_test_stats.stelltier";
    TierLayoutHeader h;
    h.n_groups = groups.size();
    TierLayoutWriter w(path, h);
    for (const auto& g : groups) {
        std::vector<std::uint8_t> payload(kSlot, std::uint8_t(g.id.value));
        w.add_group(g, 1, payload.data(), payload.size());
    }
    w.finalize();

    TierConfig cfg;
    cfg.vram_slots = 3; // eviction неизбежен → гонки pin/evict/IO
    cfg.vram_slot_size = kSlot;
    cfg.ram_capacity = 6 * kSlot;
    cfg.nvme_layout = path;
    cfg.lookahead = 1;
    TierManager mgr(cfg);
    mgr.set_plan(groups);

    std::atomic<bool> stop{false};
    std::atomic<int> errors{0};
    std::atomic<int> ok_pins{0};

    // 3 worker-потока: pin/unpin/prefetch по кругу (ADR-08 §4: одновременный
    // prefetch → pin той же группы обязаны корректно работать).
    std::vector<std::thread> ts;
    for (int t = 0; t < 3; ++t) {
        ts.emplace_back([&] {
            for (int i = 0; i < 200; ++i) {
                try {
                    const std::uint32_t id = std::uint32_t((i + t) % 6);
                    const GroupKey k{GroupId{id}, Role::Expert};
                    mgr.prefetch({k});
                    mgr.prefetch({k}); // идемпотентность под нагрузкой
                    auto a = mgr.pin(k);
                    a->wait();          // SlotExhausted при исчерпании слотов — ожидаемо
                    mgr.unpin(k);
                    ok_pins.fetch_add(1);
                } catch (const TierError& e) {
                    if (e.code() == TierError::Code::SlotExhausted)
                        continue; // допустимая деградация: слоты все pinned
                    errors.fetch_add(1);
                } catch (...) {
                    errors.fetch_add(1);
                }
            }
        });
    }
    // 2 потока stats(): неблокирующий снапшот (ADR-08 §4) — не бросает и не висит.
    for (int t = 0; t < 2; ++t) {
        ts.emplace_back([&] {
            while (!stop.load()) {
                try {
                    const TierStats st = mgr.stats();
                    T_REQUIRE(st.prefetch.hit_rate >= 0.0 && st.prefetch.hit_rate <= 1.0,
                              "hit_rate в [0,1]");
                    T_REQUIRE(std::isfinite(st.layer_pause.p50_ms), "p50 конечен");
                    std::this_thread::yield();
                } catch (...) {
                    errors.fetch_add(1);
                }
            }
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    stop.store(true);
    for (auto& th : ts) th.join();

    T_REQUIRE(errors.load() == 0, "нет ошибок кроме ожидаемого SlotExhausted");
    T_REQUIRE(ok_pins.load() > 0, "есть успешные pin/unpin");
    const TierStats st = mgr.stats();
    T_REQUIRE(st.prefetch.hit_rate >= 0.0 && st.prefetch.hit_rate <= 1.0, "итоговый снапшот валиден");
    std::printf("    ok_pins=%d hit_rate=%.3f pauses=%llu\n", ok_pins.load(),
                st.prefetch.hit_rate, (unsigned long long)st.io_ops);
}

T_MAIN

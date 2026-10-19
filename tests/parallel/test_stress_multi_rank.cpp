// tests/parallel: многоранговые стресс-тесты CPU-эмулятора (ADR-10 §4) —
// задача раунда 2. 2/4 ранка, все коллективы под нагрузкой, детерминированная
// редукция, тайм-ауты/deadlock-свободность.
//
// Правила тестов (из контракта эмулятора): после ошибки Timeout состояние
// группы не гарантируется — каждый такой кейс использует НОВЫЙ group_id.
#include "test_harness.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/emulated.hpp"
#include "stell/parallel/plan.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

using namespace stell::parallel;

namespace {

ParallelConfig make_cfg(rank_t world, int group_id, rank_t rank, int timeout_ms = 3000) {
    ParallelConfig c;
    c.mode = CommMode::CpuEmulation;
    c.pp = world;   // world == pp (топология безразлична для коллективов)
    c.group_id = group_id;
    c.rank = rank;
    c.timeout_ms = timeout_ms;
    return c;
}

template <class Fn>
std::vector<std::thread> run_world(rank_t world, int group_id, Fn&& fn) {
    std::vector<std::thread> th;
    for (rank_t r = 0; r < world; ++r)
        th.emplace_back(fn, make_cfg(world, group_id, r));
    return th;
}

// Чистая EP-топология: pp=1, tp=1, ep=world — маршрутизация expert_alltoall
// определяется cfg.ep (plan.cpp::expert_alltoall_route).
template <class Fn>
std::vector<std::thread> run_ep(rank_t world_ep, int group_id, Fn&& fn) {
    std::vector<std::thread> th;
    for (rank_t r = 0; r < world_ep; ++r) {
        ParallelConfig rc = make_cfg(1, group_id, r);
        rc.ep = world_ep;
        th.emplace_back(fn, rc);
    }
    return th;
}

// Детерминированный «рандом»: значения фиксированы по (ранг, индекс, итерация).
float stress_value(rank_t rank, int i, int iter) {
    return 0.5f * float(int(rank) + 1) + 0.125f * float((i + iter) % 11) + 1.0f;
}

} // namespace

TEST(stress_allreduce_tp4_heavy_iterations) {
    // 4 ранка × 30 итераций × буфер 4096 F32: барьерная семантика + точное
    // соответствие ранговому порядку редукции (ADR-10 §2: порядок ранков).
    constexpr int N = 4096, ITER = 30;
    SafeFlag ok;
    auto th = run_world(4, 401, [&](ParallelConfig rc) {
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        std::vector<float> buf(N);
        for (int it = 0; it < ITER; ++it) {
            for (int i = 0; i < N; ++i) buf[i] = stress_value(rc.rank, i, it);
            comm->allreduce(buf.data(), N, CommDType::F32);
            // Эталон: ранговый порядок, F32-аккумулятор, по элементу.
            for (int i = 0; i < N; ++i) {
                float acc = stress_value(0, i, it);
                for (rank_t r = 1; r < rc.world(); ++r)
                    acc += stress_value(r, i, it);
                if (!(buf[i] == acc)) ok.fail();
            }
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_allreduce_f16_deterministic_rank_order) {
    // F16: редукция в F32-аккумуляторе (ADR-10 §2) — проверяем битовое
    // соответствие ручному ранговому суммированию на неассоциативных данных.
    constexpr int N = 256;
    SafeFlag ok;
    auto th = run_world(4, 402, [&](ParallelConfig rc) {
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        std::vector<std::uint16_t> buf(N);
        // 1e7 / 1e-4-подобные величины: порядок сложения влияет на биты.
        const float big[4] = {1e7f, 1.0f, 1e-4f, 0.5f};
        for (int i = 0; i < N; ++i) {
            const float v = big[(int(rc.rank) + i) % 4] + 0.125f * float(i % 3);
            buf[i] = float_to_half(v);
        }
        comm->allreduce(buf.data(), N, CommDType::F16);
        for (int i = 0; i < N; ++i) {
            float acc = half_to_float(buf[std::size_t(0)]);   // заглушка ниже
            (void)acc;
        }
        // Эталон считаем рангово только на ранке 0 (детерминирован для всех).
        if (rc.rank == 0) {
            for (int i = 0; i < N; ++i) {
                float acc = 0.f;
                for (rank_t r = 0; r < rc.world(); ++r) {
                    const float v = big[(int(r) + i) % 4] + 0.125f * float(i % 3);
                    acc += v;   // порядок ранков, F32-аккумулятор
                }
                if (!(buf[std::size_t(i)] == float_to_half(acc))) ok.fail();
            }
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_allgather_4ranks_many_iterations) {
    constexpr int N = 512, ITER = 20;
    SafeFlag ok;
    auto th = run_world(4, 403, [&](ParallelConfig rc) {
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        std::vector<float> send(N), recv(std::size_t(N) * rc.world());
        for (int it = 0; it < ITER; ++it) {
            for (int i = 0; i < N; ++i) send[i] = stress_value(rc.rank, i, it) + 7.0f;
            std::memset(recv.data(), 0, recv.size() * sizeof(float));
            comm->allgather(send.data(), recv.data(), N, CommDType::F32);
            for (rank_t s = 0; s < rc.world(); ++s)
                for (int i = 0; i < N; ++i)
                    if (!(recv[std::size_t(s) * N + i] ==
                          stress_value(s, i, it) + 7.0f))
                        ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_alltoall_ep4_many_tokens_order) {
    // EP4, tp=1: 32 слота/ранк; проверяем количество входящих к владельцу и
    // детерминированный порядок (источник ↑, индекс ↑), ADR-10 §2.
    constexpr int SLOTS = 32;
    SafeFlag ok;
    auto th = run_ep(4, 404, [&](ParallelConfig rc) {
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        std::vector<TokenMeta> meta(SLOTS);
        std::vector<float> toks(SLOTS);
        for (int i = 0; i < SLOTS; ++i) {
            // Все ранки шлют токены ко ВСЕМ экспертам (0..31): inflow на ранок
            // = 4 ранка * 8 слотов на владельца = 32.
            meta[i] = TokenMeta{std::int32_t((i * 5 + int(rc.rank)) % 32),
                                std::int32_t(rc.rank * 100 + i)};
            toks[i] = stress_value(rc.rank, i, 0);
        }
        const std::size_t cap = std::size_t(SLOTS) * std::size_t(rc.world());
        std::vector<float> recv(cap);
        std::vector<TokenMeta> rm(cap);
        comm->expert_alltoall(toks.data(), meta.data(), recv.data(), rm.data(),
                              SLOTS, 1, CommDType::F32);
        // Порядок входящих: источник ↑ внутри индекс ↑; проверяем, что входящие
        // на ранок d = все (s,i) с expert_id%4==d, отсортированные по (s,i).
        std::size_t idx = 0;
        for (rank_t s = 0; s < rc.world(); ++s)
            for (int i = 0; i < SLOTS; ++i) {
                const std::int32_t eid = std::int32_t((i * 5 + int(s)) % 32);
                if (eid % 4 != int(rc.rank)) continue;
                if (!(rm[idx].expert_id == eid && rm[idx].slot_id == s * 100 + i))
                    ok.fail();
                if (!(recv[idx] == stress_value(s, i, 0))) ok.fail();
                ++idx;
            }
        for (std::size_t j = idx; j < cap; ++j)
            if (!(rm[j].expert_id == -1 && rm[j].slot_id == -1)) ok.fail();  // filler
        CHECK(idx == 32);
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_alltoall_recv_capacity_worst_case_invariant) {
    // Контракт (comm.hpp): ёмкость recv = max_tokens * world ТОКЕНОВ. При
    // fan-out TP inflow на ранок ≤ world*max_tokens = ёмкость, причём worst-case
    // (все токены в одну EP-группу) даёт inflow == ёмкость РОВНО: при контрактной
    // ёмкости полной группы CommError(BufferTooSmall) НЕДОСТИЖИМ (инвариант:
    // inflow(d) ≤ Σ_s max_tokens). Достижимая ветвь ошибки — подгруппа: маршруты
    // за пределы её состава (см. test_subgroup_token_width.cpp) — раунд 2. Здесь
    // фиксируем worst-case: успех, точные количества.
    constexpr int SLOTS = 16;
    constexpr std::size_t CAP = std::size_t(SLOTS) * 4;   // max_tokens*world
    SafeFlag ok;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < 4; ++r) {
        ParallelConfig rc = make_cfg(1, 405, r);
        rc.tp = 2;
        rc.ep = 2;                                          // world = 1*2*2
        th.emplace_back([rc, &ok] {
            auto comm = std::make_unique<EmulatedCommunicator>(rc);
            std::vector<TokenMeta> meta(SLOTS);
            std::vector<float> toks(SLOTS), recv(CAP);
            std::vector<TokenMeta> rm(CAP);
            for (int i = 0; i < SLOTS; ++i) {
                meta[i] = TokenMeta{0, std::int32_t(rc.rank * 100 + i)}; // fan-out на EP0
                toks[i] = stress_value(rc.rank, i, 0);
            }
            comm->expert_alltoall(toks.data(), meta.data(), recv.data(),
                                  rm.data(), SLOTS, 1, CommDType::F32);
            // Ранки 0/1 (EP-группа 0): ровно 4 источника × 16 токенов = 64.
            if (rc.rank <= 1) {
                int inflow = 0;
                for (std::size_t j = 0; j < CAP; ++j)
                    if (rm[j].expert_id >= 0) ++inflow;
                if (inflow != 64) ok.fail();
                // Порядок: источник ↑, затем индекс ↑ (ADR-10 §2).
                std::size_t idx = 0;
                for (rank_t s = 0; s < rc.world(); ++s)
                    for (int i = 0; i < SLOTS; ++i, ++idx) {
                        if (!(rm[idx].expert_id == 0 &&
                              rm[idx].slot_id == std::int32_t(s) * 100 + i))
                            ok.fail();
                        if (!(recv[idx] == stress_value(s, i, 0))) ok.fail();
                    }
            } else {
                // Ранки 2/3 (EP-группа 1): входящих нет — все слоты filler(-1).
                for (std::size_t j = 0; j < CAP; ++j)
                    if (!(rm[j].expert_id == -1 && rm[j].slot_id == -1)) ok.fail();
            }
        });
    }
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_chain_pingpong_ring_4ranks) {
    // Кольцевая цепочка 0→1→2→3→0, 200 обходов: значение растёт на 1 на хоп;
    // после полного цикла старт + 4*200 — FIFO-порядок без потерь.
    constexpr int LAPS = 200;
    const ParallelConfig cfg = make_cfg(4, 406, 0);
    SafeFlag ok;
    auto th = run_world(4, 406, [&](ParallelConfig rc) {
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        const rank_t next = (rc.rank + 1) % rc.world();
        const rank_t prev = (rc.rank + rc.world() - 1) % rc.world();
        for (int lap = 0; lap < LAPS; ++lap) {
            int v = 0;
            if (rc.rank == 0) {
                v = lap;
                comm->chain_send(&v, 1, CommDType::F32, next);
                comm->chain_recv(&v, 1, CommDType::F32, prev);
                // После 4 хопов значение = lap + 3 (инкременты на ранках 1,2,3;
                // ранг 0 сам не инкрементирует — старт и финиш кольца).
                if (!(v == lap + 3)) ok.fail();
            } else {
                comm->chain_recv(&v, 1, CommDType::F32, prev);
                ++v;
                comm->chain_send(&v, 1, CommDType::F32, next);
            }
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_kv_stream_flood_timeout_code) {
    // Переполнение кольца (capacity 2) → Timeout на продюсере; код ошибки
    // Timeout, не NcclFailure (ADR-10 §4: семантика тайм-аутов). Сценарий:
    // продюсер шлёт 5 блоков, потребитель снимает ровно 1 — кольцо остаётся
    // заполненным, очередной push блокирует до тайм-аута.
    const ParallelConfig cfg = make_cfg(2, 407, 0, /*timeout_ms=*/250);
    SafeFlag ok;
    std::atomic<bool> consumer_done{false};
    auto th = run_world(2, 407, [&](ParallelConfig rc) {
        rc.kv_ring_capacity = 2;                     // кольцо переполнится
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        float block[8] = {};
        if (rc.rank == 0) {
            bool threw_timeout = false;
            CommError::Code code = CommError::Code::NcclFailure;
            try {
                for (int b = 0; b < 5; ++b) {
                    for (std::size_t i = 0; i < 8; ++i) block[i] = float(b);
                    comm->kv_stream_push(block, sizeof(block));
                }
            } catch (const CommError& e) {
                threw_timeout = (e.code() == CommError::Code::Timeout);
                code = e.code();
            }
            while (!consumer_done.load()) { std::this_thread::yield(); }
            if (!(threw_timeout && code == CommError::Code::Timeout)) ok.fail();
        } else {
            std::size_t got = 0;
            // Тайм-базированный spin (не фиксированное число итераций — иначе
            // гонка с медленным стартом продюсера): до 5 с, шаг 1 мс.
            for (int spin = 0; spin < 5000 && got < 1; ++spin) {
                if (comm->kv_stream_pop(block, sizeof(block))) ++got;
                else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            consumer_done.store(true);
            if (got != 1) ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_timeout_partial_arrival_deadlock_free) {
    // 4-ранковый allreduce, участвуют только 3: все три получают Timeout
    // (не виснут), код Timeout; после — новая группа с полным составом работает.
    {
        SafeFlag ok;
        auto th = run_world(4, 408, [&](ParallelConfig rc) {
            auto comm = std::make_unique<EmulatedCommunicator>(rc);
            if (rc.rank == 3) return;                  // «отсутствующий» ранок
            float buf[4] = {1, 2, 3, 4};
            bool threw = false;
            CommError::Code code = CommError::Code::NcclFailure;
            try {
                comm->allreduce(buf, 4, CommDType::F32);
            } catch (const CommError& e) {
                threw = true;
                code = e.code();
            }
            if (!(threw && code == CommError::Code::Timeout)) ok.fail();
        });
        for (auto& t : th) t.join();
        CHECK(ok.get());
    }
    // Новая группа (тот же состав) — deadlock-свободность: коллективы работают.
    {
        SafeFlag ok;
        auto th = run_world(4, 409, [&](ParallelConfig rc) {
            auto comm = std::make_unique<EmulatedCommunicator>(rc);
            float buf[4] = {float(rc.rank) + 1, 0, 0, 0};
            comm->allreduce(buf, 4, CommDType::F32);
            if (!(buf[0] == float(rc.world()) * (rc.world() + 1) / 2)) ok.fail();
        });
        for (auto& t : th) t.join();
        CHECK(ok.get());
    }
}

TEST(stress_allreduce_and_allgather_interleaved) {
    // Чередование разных коллективов на одной группе (4 ранка × 20 пар):
    // рандеву коллективов независимы; данные не теряются.
    constexpr int N = 128, PAIRS = 20;
    SafeFlag ok;
    auto th = run_world(4, 410, [&](ParallelConfig rc) {
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        std::vector<float> ar(N), ag(N), recv(std::size_t(N) * rc.world());
        for (int p = 0; p < PAIRS; ++p) {
            for (int i = 0; i < N; ++i) {
                ar[i] = stress_value(rc.rank, i, p);
                ag[i] = stress_value(rc.rank, i, p) + 100.0f;
            }
            comm->allreduce(ar.data(), N, CommDType::F32);
            comm->allgather(ag.data(), recv.data(), N, CommDType::F32);
            for (int i = 0; i < N; ++i) {
                float acc = stress_value(0, i, p);
                for (rank_t r = 1; r < rc.world(); ++r) acc += stress_value(r, i, p);
                if (!(ar[i] == acc)) ok.fail();
            }
            for (rank_t s = 0; s < rc.world(); ++s)
                if (!(recv[std::size_t(s) * N] == stress_value(s, 0, p) + 100.0f))
                    ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(stress_world_2_all_collectives_smoke) {
    // world=2: каждый коллектив хотя бы раз под нагрузкой (smoke).
    SafeFlag ok;
    const ParallelConfig cfg = make_cfg(2, 411, 0);
    auto th = run_world(2, 411, [&](ParallelConfig rc) {
        auto comm = std::make_unique<EmulatedCommunicator>(rc);
        float ar[64], ag[32], recv[64];
        for (int i = 0; i < 64; ++i) ar[i] = float(i);
        for (int i = 0; i < 32; ++i) ag[i] = float(rc.rank) + float(i);
        comm->allreduce(ar, 64, CommDType::F32);
        comm->allgather(ag, recv, 32, CommDType::F32);
        TokenMeta m[8], rm[16];
        float t8[8], r8[16];
        for (int i = 0; i < 8; ++i) m[i] = TokenMeta{std::int32_t(i % 2), i};
        comm->expert_alltoall(t8, m, r8, rm, 8, 1, CommDType::F32);
        if (rc.rank == 0) {
            comm->chain_send(ar, 8, CommDType::F32, 1);   // ar после allreduce = 2*i
        } else {
            float got[8];
            comm->chain_recv(got, 8, CommDType::F32, 0);
            for (int i = 0; i < 8; ++i) if (!(got[i] == 2.0f * float(i))) ok.fail();
        }
        std::uint8_t kv[16] = {};
        if (rc.rank == 0) {
            for (int i = 0; i < 16; ++i) kv[i] = std::uint8_t(i);
            comm->kv_stream_push(kv, sizeof(kv));
        } else if (!comm->kv_stream_pop(kv, sizeof(kv))) {
            ok.fail();
        } else {
            for (int i = 0; i < 16; ++i) if (kv[i] != std::uint8_t(i)) ok.fail();
        }
        // Проверки after-collectives:
        for (int i = 0; i < 64; ++i)
            if (!(ar[i] == float(2 * i))) ok.fail();   // сумма рангов 0+1 == i+i
        for (int i = 0; i < 32; ++i)
            if (!(recv[std::size_t(rc.rank) * 32 + i] == float(rc.rank) + float(i)))
                ok.fail();
        // expert_alltoall: у владельца ровно 8 входящих (по одному с ранга-пары).
        int inflow = 0;
        for (int i = 0; i < 16; ++i) if (rm[i].expert_id >= 0) ++inflow;
        if (inflow != 8) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

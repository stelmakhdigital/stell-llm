// tests/parallel: CPU-эмулятор комм-паттернов (ADR-10 §4) — allreduce,
// allgather, expert_alltoall, chain send/recv, kv-stream, ошибки.
#include "test_harness.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/emulated.hpp"

#include <array>
#include <cmath>
#include <thread>
#include <vector>

using namespace stell::parallel;

namespace {

// Ранок-раннер: fn(comm) исполняется в потоке ранка rank. ВНУТРИ потоков
// CHECK не используется (счётчики харнесса не потокобезопасны) — ошибки
// собираются в локальные флаги и проверяются после join().
template <class Fn>
std::vector<std::thread> run_group(rank_t pp, rank_t tp, rank_t ep, int group_id, Fn&& fn) {
    ParallelConfig c;
    c.mode = CommMode::CpuEmulation;
    c.pp = pp; c.tp = tp; c.ep = ep;
    c.group_id = group_id;
    c.timeout_ms = 3000;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < c.world(); ++r) {
        ParallelConfig rc = c;
        rc.rank = r;
        th.emplace_back(fn, std::make_unique<EmulatedCommunicator>(rc));
    }
    return th;
}

} // namespace

TEST(em_allreduce_f32_tp4) {
    constexpr int N = 8;
    std::vector<std::array<float, N>> inputs(4);
    for (int r = 0; r < 4; ++r)
        for (int i = 0; i < N; ++i) inputs[r][i] = float(r + 1) * float(i + 1);
    SafeFlag ok;
    auto th = run_group(1, 4, 1, 101, [&](std::unique_ptr<ICommunicator> c) {
        float buf[N];
        for (int i = 0; i < N; ++i) buf[i] = inputs[c->rank()][i];
        c->allreduce(buf, N, CommDType::F32);
        // Детерминированная семантика: сумма по ранкам 0..3 по порядку.
        for (int i = 0; i < N; ++i) {
            float expect = 0.f;
            for (int r = 0; r < 4; ++r) expect += float(r + 1) * float(i + 1);
            if (!((buf[i] == expect))) ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(em_allreduce_f16_deterministic) {
    // F16: редукция в аккумуляторе F32, результат обратно в F16; порядок
    // ранков зафиксирован (ADR-10 §2 — неассоциативность float).
    constexpr int N = 4;
    const std::uint16_t inputs[2][N] = {
        {float_to_half(1.0f), float_to_half(0.5f), float_to_half(0.25f), float_to_half(2.0f)},
        {float_to_half(0.75f), float_to_half(0.125f), float_to_half(0.125f), float_to_half(1.5f)},
    };
    SafeFlag ok;
    auto th = run_group(1, 2, 1, 102, [&](std::unique_ptr<ICommunicator> c) {
        std::uint16_t buf[N];
        for (int i = 0; i < N; ++i) buf[i] = inputs[c->rank()][i];
        c->allreduce(buf, N, CommDType::F16);
        for (int i = 0; i < N; ++i) {
            const float acc = half_to_float(inputs[0][i]) + half_to_float(inputs[1][i]);
            if (!((buf[i] == float_to_half(acc)))) ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(em_allreduce_e4m3_and_count_mismatch) {
    // E4M3: редукция через F32-аккумулятор, запись обратно в E4M3 (погрешность
    // кванта E4M3 допустима, ADR-04).
    {
        constexpr int N = 4;
        const std::uint8_t inputs[2][N] = {
            {float_to_e4m3(1.f), float_to_e4m3(0.5f), float_to_e4m3(2.f), float_to_e4m3(4.f)},
            {float_to_e4m3(0.25f), float_to_e4m3(0.25f), float_to_e4m3(0.25f), float_to_e4m3(0.25f)},
        };
        SafeFlag ok;
        auto th = run_group(1, 2, 1, 103, [&](std::unique_ptr<ICommunicator> c) {
            std::uint8_t buf[N];
            for (int i = 0; i < N; ++i) buf[i] = inputs[c->rank()][i];
            c->allreduce(buf, N, CommDType::E4M3);
            for (int i = 0; i < N; ++i) {
                const float acc = e4m3_to_float(inputs[0][i]) + e4m3_to_float(inputs[1][i]);
                if (!((std::fabs(e4m3_to_float(buf[i]) - acc) <= 0.1f * std::fabs(acc)))) ok.fail();
            }
        });
        for (auto& t : th) t.join();
        CHECK(ok.get());
    }
    // count не одинаков по ранкам → RankMismatch у всех.
    {
        SafeFlag ok;
        auto th = run_group(1, 2, 1, 104, [&](std::unique_ptr<ICommunicator> c) {
            float buf[8] = {0};
            try {
                c->allreduce(buf, c->rank() == 0 ? 8 : 4, CommDType::F32);
            } catch (const CommError& e) {
                if (!((e.code() == CommError::Code::RankMismatch))) ok.fail();
            }
        });
        for (auto& t : th) t.join();
        CHECK(ok.get());
    }
}

TEST(em_allgather_tp2) {
    constexpr int N = 4, world = 2;
    float inputs[world][N];
    for (int r = 0; r < world; ++r)
        for (int i = 0; i < N; ++i) inputs[r][i] = float(r * 10 + i);
    SafeFlag ok;
    auto th = run_group(1, world, 1, 105, [&](std::unique_ptr<ICommunicator> c) {
        float send[N], recv[N * world];
        for (int i = 0; i < N; ++i) send[i] = inputs[c->rank()][i];
        c->allgather(send, recv, N, CommDType::F32);
        // Сбор по порядку ранков: [rank0 | rank1].
        for (int r = 0; r < world; ++r)
            for (int i = 0; i < N; ++i) if (!((recv[r * N + i] == inputs[r][i]))) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(em_expert_alltoall_pure_ep4) {
    // Чистый EP (tp=1): токен уезжает к owner(expert_id) = expert_id % ep
    // (статическое закрепление, ADR-10 §3).
    constexpr rank_t ep = 4;
    constexpr int slots = 6;
    SafeFlag ok;
    int got[4] = {0, 0, 0, 0};
    auto th = run_group(1, 1, ep, 106, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        float send_t[slots];
        TokenMeta send_m[slots];
        for (int i = 0; i < slots; ++i) {
            const std::int32_t eid = std::int32_t((my + 3 * i) % 16);
            send_t[i] = float(eid);                  // «эмбеддинг» — сам expert_id
            send_m[i] = TokenMeta{eid, std::int32_t(i)};
        }
        const std::size_t cap = std::size_t(slots) * std::size_t(ep);
        std::vector<float> recv_t(cap);
        std::vector<TokenMeta> recv_m(cap);
        c->expert_alltoall(send_t, send_m, recv_t.data(), recv_m.data(), slots, 1,
                           CommDType::F32);
        int count = 0;
        for (std::size_t i = 0; i < cap; ++i) {
            if (recv_m[i].expert_id < 0) continue;
            ++count;
            const std::int32_t eid = recv_m[i].expert_id;
            if (rank_t(eid) % ep != my) ok.fail();   // чужой токен!
            if (!((recv_t[i] == float(eid)))) ok.fail();
        }
        got[my] = count;
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
    // Полное покрытие: токенов 4*6=24, владелец = expert_id % 4.
    int per_owner[4] = {0, 0, 0, 0};
    for (rank_t r = 0; r < 4; ++r)
        for (int i = 0; i < slots; ++i)
            ++per_owner[((r + 3 * i) % 16) % 4];
    CHECK(per_owner[0] == got[0] && per_owner[1] == got[1] &&
          per_owner[2] == got[2] && per_owner[3] == got[3]);
}

TEST(em_expert_alltoall_with_tp_fanout) {
    // Комбинация TP×EP (tp=2, ep=2): токен уезжает на ОБА tp-ранка группы-
    // владельца эксперта (fan-out, см. comm.hpp/plan.hpp). На ранок с ep_r
    // приходит slots*world/ep входящих токенов.
    constexpr rank_t tp = 2, ep = 2;
    constexpr int slots = 4;
    const rank_t world = tp * ep;
    SafeFlag ok;
    auto th = run_group(1, tp, ep, 107, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        float send_t[slots] = {0};
        TokenMeta send_m[slots];
        for (int i = 0; i < slots; ++i)
            send_m[i] = TokenMeta{std::int32_t(i), std::int32_t(i)};
        const std::size_t cap = std::size_t(slots) * std::size_t(world);
        std::vector<float> recv_t(cap);
        std::vector<TokenMeta> recv_m(cap);
        c->expert_alltoall(send_t, send_m, recv_t.data(), recv_m.data(), slots, 1,
                           CommDType::F32);
        const rank_t ep_r = my / tp;
        std::size_t total = 0;
        for (std::size_t i = 0; i < cap; ++i) {
            if (recv_m[i].expert_id < 0) continue;
            ++total;
            if (rank_t(recv_m[i].expert_id) % ep != ep_r) ok.fail();
        }
        if (!((total == std::size_t(slots) * std::size_t(world) / std::size_t(ep)))) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(em_chain_pp_ring) {
    // PP-цепочка (world=3): ring-обмен активаций r0→r1→r2→r0.
    constexpr rank_t pp = 3;
    constexpr int N = 4;
    SafeFlag ok;
    auto th = run_group(pp, 1, 1, 108, [&](std::unique_ptr<ICommunicator> c) {
        const rank_t my = c->rank();
        const rank_t next = rank_t((my + 1) % pp);
        const rank_t prev = rank_t((my + pp - 1) % pp);
        float msg[N];
        for (int i = 0; i < N; ++i) msg[i] = float(my) + 100.f + float(i);
        c->chain_send(msg, N, CommDType::F32, next);
        float recv[N];
        c->chain_recv(recv, N, CommDType::F32, prev);
        for (int i = 0; i < N; ++i)
            if (!((recv[i] == float(prev) + 100.f + float(i)))) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(em_chain_fifo_order) {
    // FIFO на паре ранков: два блока приходят в порядке отправки.
    SafeFlag ok;
    auto th = run_group(1, 2, 1, 109, [&](std::unique_ptr<ICommunicator> c) {
        if (c->rank() == 0) {
            float a[1] = {1.f}, b[1] = {2.f};
            c->chain_send(a, 1, CommDType::F32, 1);
            c->chain_send(b, 1, CommDType::F32, 1);
        } else {
            float x[1] = {0};
            c->chain_recv(x, 1, CommDType::F32, 0);
            if (!(x[0] == 1.f)) ok.fail();
            c->chain_recv(x, 1, CommDType::F32, 0);
            if (!(x[0] == 2.f)) ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(em_kv_stream_ring) {
    // KvStream: push уходит следующему ранку, pop читает от предыдущего;
    // pop на пустом кольце возвращает false (неблокирующая семантика).
    constexpr std::size_t BLOCK = 16;
    SafeFlag ok;
    auto th = run_group(1, 2, 1, 110, [&](std::unique_ptr<ICommunicator> c) {
        std::uint8_t block[BLOCK] = {};
        if (c->rank() == 0) {
            // Кольцо ранка 0 гарантированно пусто (ранг 1 ничего не шлёт):
            // неблокирующая семантика pop на пустом кольце → false.
            if (!(c->kv_stream_pop(block, BLOCK) == false)) ok.fail();
            for (std::size_t i = 0; i < BLOCK; ++i) block[i] = std::uint8_t(i);
            c->kv_stream_push(block, BLOCK);
        } else {
            for (int spin = 0; spin < 500; ++spin)
                if (c->kv_stream_pop(block, BLOCK)) break;
        }
        if (c->rank() == 1)
            for (std::size_t i = 0; i < BLOCK; ++i)
                if (!((block[i] == std::uint8_t(i)))) ok.fail();
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
    // Размер блока не совпал → BufferTooSmall (в очереди есть блок BLOCK,
    // читаем меньшим размером).
    bool threw = false;
    auto th2 = run_group(1, 2, 1, 111, [&](std::unique_ptr<ICommunicator> c) {
        std::uint8_t block[BLOCK] = {};
        if (c->rank() == 0) {
            c->kv_stream_push(block, BLOCK);
            c->kv_stream_push(block, BLOCK);
        } else {
            // Один блок забираем, второй остаётся в кольце.
            for (int spin = 0; spin < 500; ++spin)
                if (c->kv_stream_pop(block, BLOCK)) break;
            try {
                std::uint8_t small[4] = {};
                c->kv_stream_pop(small, 4);
            } catch (const CommError& e) {
                threw = threw || (e.code() == CommError::Code::BufferTooSmall);
            }
        }
    });
    for (auto& t : th2) t.join();
    CHECK(threw);
}

TEST(em_timeout_deadlock_free) {
    // Один ранок не участвует в коллективе → остальные получают Timeout
    // (ADR-10 §4: семантика тайм-аутов, deadlock-свободность).
    bool timed_out = false;
    ParallelConfig c;
    c.mode = CommMode::CpuEmulation;
    c.tp = 2;
    c.group_id = 112;
    c.timeout_ms = 300;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < 2; ++r) {
        ParallelConfig rc = c;
        rc.rank = r;
        th.emplace_back([rc, &timed_out] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20 * rc.rank));
            if (rc.rank == 1) return;                  // ранок 1 не приходит
            auto comm = std::make_unique<EmulatedCommunicator>(rc);
            float buf[4] = {0};
            try {
                comm->allreduce(buf, 4, CommDType::F32);
            } catch (const CommError& e) {
                timed_out = timed_out || (e.code() == CommError::Code::Timeout);
            }
        });
    }
    for (auto& t : th) t.join();
    CHECK(timed_out);
    // chain_recv без отправки → Timeout.
    bool chain_timeout = false;
    ParallelConfig rc2 = c;
    rc2.rank = 0;
    rc2.group_id = 113;
    try {
        auto comm = std::make_unique<EmulatedCommunicator>(rc2);
        float buf[4] = {0};
        comm->chain_recv(buf, 4, CommDType::F32, 1);
    } catch (const CommError& e) {
        chain_timeout = e.code() == CommError::Code::Timeout;
    }
    CHECK(chain_timeout);
    // kv_stream_push при переполненном кольце (ёмкость 2, попов нет) → Timeout.
    bool kv_timeout = false;
    ParallelConfig rc3 = c;
    rc3.rank = 0;
    rc3.group_id = 114;
    rc3.kv_ring_capacity = 2;
    try {
        auto comm = std::make_unique<EmulatedCommunicator>(rc3);
        std::uint8_t block[8] = {};
        for (int i = 0; i < 4; ++i) comm->kv_stream_push(block, 8);
    } catch (const CommError& e) {
        kv_timeout = e.code() == CommError::Code::Timeout;
    }
    CHECK(kv_timeout);
}

TEST(em_nccl_mode_fails_locally) {
    // NCCL-режим без собранного бэкенда → NcclFailure (CPU-конфиг, ADR-07).
    ParallelConfig c;
    c.mode = CommMode::Nccl;
    CHECK_THROWS_AS(make_communicator(c), CommError);
}

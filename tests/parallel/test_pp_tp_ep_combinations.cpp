// tests/parallel: комбинации PP×TP / PP×EP (ADR-10 §3) — маппинг ранков +
// юнит-прогоны эмулятора: PP-цепочка активаций по стадиям, EP-закрепление
// внутри стадии, KV-streaming между узлами.
#include "test_harness.hpp"
#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/emulated.hpp"
#include "stell/parallel/plan.hpp"

#include <cmath>
#include <thread>
#include <vector>

using namespace stell::parallel;

namespace {

ParallelConfig make_cfg(rank_t pp, rank_t tp, rank_t ep) {
    ParallelConfig c;
    c.pp = pp;
    c.tp = tp;
    c.ep = ep;
    return c;
}

template <class Fn>
std::vector<std::thread> run_world(rank_t pp, rank_t tp, rank_t ep, int group_id, Fn&& fn) {
    ParallelConfig c = make_cfg(pp, tp, ep);
    c.mode = CommMode::CpuEmulation;
    c.group_id = group_id;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < c.world(); ++r) {
        ParallelConfig rc = c;
        rc.rank = r;
        th.emplace_back(fn, std::make_unique<EmulatedCommunicator>(rc));
    }
    return th;
}

} // namespace

TEST(combo_pp_tp_rank_mapping) {
    // PP×TP (ep=1): ранок = pp_rank*tp + tp_rank; слои — по стадиям,
    // разбиение полное: {0..3} ∪ {4..7} для L=8, pp=2.
    const rank_t pp = 2, tp = 2, L = 8;
    const ParallelConfig c = make_cfg(pp, tp, 1);
    CHECK(c.world() == 4);
    for (rank_t r = 0; r < c.world(); ++r) {
        const RankDecomp d = decompose_rank(r, c);
        CHECK(r == d.pp_rank * tp + d.tp_rank);
        const ShardRange lr = layer_range(int(L), pp, d.pp_rank);
        CHECK(lr.end - lr.begin == int(L) / int(pp));
    }
    const ShardRange s0 = layer_range(int(L), pp, 0), s1 = layer_range(int(L), pp, 1);
    CHECK(s0.begin == 0 && s0.end == 4 && s1.begin == 4 && s1.end == 8);
}

TEST(combo_pp_tp_activations_chain_e2e) {
    // E2E на эмуляторе: world=4 (pp=2,tp=2). Активации идут по pp-границе:
    // pp0.tp_r → pp1.tp_r (chain_send/chain_recv на одноимённый tp-ранок
    // следующей стадии). Golden TP2==TP1 внутри стадии покрыт тестами
    // test_golden_tp.cpp (allreduce эмулятора — над всем world; per-stage
    // TP потребует sub-communicators — открытый вопрос контракта, см. отчёт).
    constexpr int N = 4;
    const float stage_bias[2] = {1.f, 2.f};   // «вклад стадии» — детерминировано
    const ParallelConfig cfg = make_cfg(2, 2, 1);
    SafeFlag ok;
    auto th = run_world(2, 2, 1, 301, [&](std::unique_ptr<ICommunicator> c) {
        const RankDecomp d = decompose_rank(c->rank(), cfg);
        if (d.pp_rank == 0) {
            float act[N];
            for (int i = 0; i < N; ++i)
                act[i] = stage_bias[0] * float(i + 1) * float(d.tp_rank + 1);
            c->chain_send(act, N, CommDType::F32,
                          compose_rank(RankDecomp{1, 0, d.tp_rank}, cfg));
        } else {
            float act[N] = {0};
            c->chain_recv(act, N, CommDType::F32,
                          compose_rank(RankDecomp{0, 0, d.tp_rank}, cfg));
            for (int i = 0; i < N; ++i)
                if (!(act[i] == stage_bias[0] * float(i + 1) * float(d.tp_rank + 1)))
                    ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(combo_pp_ep_rank_mapping) {
    // PP×EP (tp=1): ранок = pp_rank*ep + ep_rank; эксперты закреплены статически
    // внутри стадии: expert_id % ep == ep_rank.
    const rank_t pp = 2, ep = 3;
    const ParallelConfig c = make_cfg(pp, 1, ep);
    for (rank_t r = 0; r < c.world(); ++r) {
        const RankDecomp d = decompose_rank(r, c);
        CHECK(r == d.pp_rank * ep + d.ep_rank);
        for (rank_t e = 0; e < 12; ++e)
            if (expert_owner(e, ep) == d.ep_rank)
                CHECK(expert_owner_global(e, d.pp_rank, c) == r);
    }
}

TEST(combo_pp_ep_tokens_e2e) {
    // E2E: world=6 (pp=2,ep=3); токены идут к владельцам экспертов своей
    // стадии (маршрут по глобальным ранкам).
    constexpr int slots = 4;
    const ParallelConfig cfg = make_cfg(2, 1, 3);
    SafeFlag ok;
    auto th = run_world(2, 1, 3, 303, [&](std::unique_ptr<ICommunicator> c) {
        const RankDecomp d = decompose_rank(c->rank(), cfg);
        TokenMeta send_m[slots];
        float send_t[slots] = {0};
        for (int i = 0; i < slots; ++i)
            send_m[i] = TokenMeta{std::int32_t((i + d.pp_rank * 5) % 12), std::int32_t(i)};
        const std::size_t cap = std::size_t(slots) * std::size_t(c->world_size());
        std::vector<float> recv_t(cap);
        std::vector<TokenMeta> recv_m(cap);
        c->expert_alltoall(send_t, send_m, recv_t.data(), recv_m.data(), slots, 1,
                           CommDType::F32);
        for (std::size_t i = 0; i < cap; ++i) {
            if (recv_m[i].expert_id < 0) continue;
            if (!((rank_t(recv_m[i].expert_id) % 3 == d.ep_rank))) ok.fail();
        }
    });
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(combo_pp_kv_stream_between_nodes) {
    // KV-streaming между узлами PP-цепочки: стадия 0 стримит KV-блоки
    // в стадию 1 (кольцевой канал; pop на пустом кольце = false).
    constexpr std::size_t BLOCK = 32;
    SafeFlag ok;
    ParallelConfig c = make_cfg(2, 1, 1);
    c.mode = CommMode::CpuEmulation;
    c.group_id = 304;
    c.kv_ring_capacity = 4;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < 2; ++r) {
        ParallelConfig rc = c;
        rc.rank = r;
        th.emplace_back([rc, &ok] {
            auto comm = std::make_unique<EmulatedCommunicator>(rc);
            std::uint8_t block[BLOCK] = {};
            if (rc.rank == 0) {
                if (!((comm->kv_stream_pop(block, BLOCK) == false))) ok.fail();  // пусто
                for (int b = 0; b < 3; ++b) {
                    for (std::size_t i = 0; i < BLOCK; ++i)
                        block[i] = std::uint8_t(b + i);
                    comm->kv_stream_push(block, BLOCK);
                }
            } else {
                std::size_t got = 0;
                for (int spin = 0; spin < 2000 && got < 3; ++spin)
                    if (comm->kv_stream_pop(block, BLOCK)) {
                        ++got;
                        for (std::size_t i = 0; i < BLOCK; ++i)
                            if (!((block[i] == std::uint8_t(int(got) - 1 + i)))) ok.fail();
                    }
                if (!((got == 3))) ok.fail();
            }
        });
    }
    for (auto& t : th) t.join();
    CHECK(ok.get());
}

TEST(combo_full_world_decomposition_pp_tp_ep) {
    // Полная декомпозиция pp=2,tp=2,ep=2 (world=8): ранок =
    // pp*(ep*tp) + ep*tp + tp; слоты одинаковы по ранкам (ADR-10 §3).
    const ParallelConfig c = make_cfg(2, 2, 2);
    std::vector<std::int64_t> slot_ids;
    for (rank_t r = 0; r < c.world(); ++r) {
        const RankDecomp d = decompose_rank(r, c);
        CHECK(r == d.pp_rank * (c.tp * c.ep) + d.ep_rank * c.tp + d.tp_rank);
        slot_ids.push_back(std::int64_t(slot_bytes_per_rank(256, c.tp)));
    }
    for (std::size_t i = 1; i < slot_ids.size(); ++i)
        CHECK(slot_ids[i] == slot_ids[0]);
}

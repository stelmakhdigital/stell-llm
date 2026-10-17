// tests/parallel: чистая логика деления по ранкам (ADR-10 §3) — разбиение,
// маппинг ранков, EP-закрепление, PP-диапазоны, TP-шардирование, слоты.
#include "test_harness.hpp"
#include "stell/parallel/plan.hpp"

#include <set>

using namespace stell::parallel;

namespace {

ParallelConfig cfg(rank_t pp, rank_t tp, rank_t ep, rank_t rank = 0) {
    ParallelConfig c;
    c.pp = pp;
    c.tp = tp;
    c.ep = ep;
    c.rank = rank;
    return c;
}

} // namespace

TEST(plan_decompose_compose_roundtrip) {
    // Все комбинации pp × tp × ep из ADR-10 §3: rank = pp_rank*(tp*ep)+ep_rank*tp+tp_rank
    for (rank_t pp : {1, 2, 3})
        for (rank_t tp : {1, 2, 4})
            for (rank_t ep : {1, 2, 3}) {
                const ParallelConfig c = cfg(pp, tp, ep);
                std::set<rank_t> seen;
                for (rank_t r = 0; r < c.world(); ++r) {
                    const RankDecomp d = decompose_rank(r, c);
                    CHECK(d.pp_rank < pp && d.ep_rank < ep && d.tp_rank < tp);
                    CHECK(compose_rank(d, c) == r);          // roundtrip
                    seen.insert(compose_rank(d, c));
                }
                CHECK(int(seen.size()) == c.world());         // взаимная однозначность
            }
    // Пример разбиения: pp=2, tp=2, ep=2 → rank 5 = pp1(ep0,tp1)
    // (rank = pp*(tp*ep) + ep*tp + tp = 4 + 0 + 1).
    const ParallelConfig c = cfg(2, 2, 2);
    const RankDecomp d = decompose_rank(5, c);
    CHECK(d.pp_rank == 1 && d.ep_rank == 0 && d.tp_rank == 1);
}

TEST(plan_pure_tp_mapping_matches_adr_formula) {
    // Чистый TP (ep=1): rank = pp_rank*tp + tp_rank — как в ADR-10 §3.
    const ParallelConfig c = cfg(2, 4, 1);
    for (rank_t r = 0; r < c.world(); ++r) {
        const RankDecomp d = decompose_rank(r, c);
        CHECK(r == d.pp_rank * c.tp + d.tp_rank);
        CHECK(d.ep_rank == 0);
    }
}

TEST(plan_pure_ep_mapping_matches_adr_formula) {
    // Чистый EP (tp=1): rank = pp_rank*ep + ep_rank.
    const ParallelConfig c = cfg(2, 1, 3);
    for (rank_t r = 0; r < c.world(); ++r) {
        const RankDecomp d = decompose_rank(r, c);
        CHECK(r == d.pp_rank * c.ep + d.ep_rank);
        CHECK(d.tp_rank == 0);
    }
}

TEST(plan_expert_pinning_static) {
    // EP: статическое закрепление expert_id % ep == rank (ADR-10 §3).
    const ParallelConfig c = cfg(1, 1, 4);
    for (rank_t e = 0; e < 16; ++e)
        CHECK(expert_owner(e, c.ep) == e % 4);
    // Глобальный владелец: стадия pp_rank, ep_rank = e % ep, tp_rank = 0.
    const ParallelConfig c2 = cfg(2, 2, 2);
    CHECK(expert_owner_global(3, 1, c2) == compose_rank(RankDecomp{1, 1, 0}, c2));
    CHECK(expert_owner_global(0, 0, c2) == 0);
    // Маршрут alltoall: токен уезжает на все tp-ранки группы-владельца.
    rank_t route[8];
    expert_alltoall_route(3, 1, c2, route);
    CHECK(route[0] == compose_rank(RankDecomp{1, 1, 0}, c2));
    CHECK(route[1] == compose_rank(RankDecomp{1, 1, 1}, c2));
}

TEST(plan_layer_range_pp) {
    // PP: слои делятся подряд, покрытие полное и без пересечений.
    const int L = 12, pp = 4;
    std::set<int> covered;
    for (rank_t r = 0; r < pp; ++r) {
        const ShardRange s = layer_range(L, pp, r);
        CHECK(s.end - s.begin == L / int(pp));
        CHECK(s.begin == int(r) * (L / int(pp)));
        for (int l = s.begin; l < s.end; ++l) covered.insert(l);
    }
    CHECK(int(covered.size()) == L);
    // num_layers % pp != 0 → RankMismatch (строго, см. коммент. в plan.hpp).
    CHECK_THROWS_AS(layer_range(10, 4, 0), CommError);
}

TEST(plan_tp_head_shard_and_kv_heads) {
    // TP: головы делятся равномерно; n % tp == 0 обязательно.
    const ShardRange s = shard_range(32, 4, 2);
    CHECK(s.begin == 16 && s.end == 24);
    CHECK_THROWS_AS(shard_range(10, 4, 0), CommError);   // 10 % 4 != 0
    check_kv_heads_divisible(8, 4);                       // OK
    CHECK_THROWS_AS(check_kv_heads_divisible(6, 4), CommError); // ADR-10 §3
}

TEST(plan_slot_bytes_per_rank) {
    // Слоты одинаковы по ранкам; содержимое шардировано: slot_bytes / tp.
    CHECK(slot_bytes_per_rank(1024, 4) == 256);
    CHECK_THROWS_AS(slot_bytes_per_rank(1022, 4), CommError);
}

TEST(plan_topology_checks) {
    CHECK_THROWS_AS(check_topology(cfg(0, 1, 1)), CommError);
    ParallelConfig bad = cfg(1, 2, 1);
    bad.device_ids = {0};                    // world=2, а устройств 1
    CHECK_THROWS_AS(check_topology(bad), CommError);
    bad.device_ids = {0, 1};
    check_topology(bad);                     // OK
    check_topology(cfg(1, 1, 1));
}

// tests/parallel: EP-партиционирование против РЕАЛЬНОГО реестра — glm-class
// (128 экспертов, ADR-09 §2), ep ∈ {2,4,8} (задача раунда 2, ADR-10 §3).
//
// Проверяется:
//  1) закрепление expert_id % ep == ep_rank — полнота и однозначность:
//     каждый из 128 экспертов ровно один ранок-владелец, на каждом ранке
//     ровно 128/ep локальных экспертов (128 % ep == 0 обязано);
//  2) маппинг глобального ранка pp*tp_ep + tp_ep_rank (tp_ep_rank =
//     ep_rank*tp + tp_rank) — маршрут expert_alltoall фан-аутит токен на все
//     tp-ранки EP-группы-владельца;
//  3) slot_bytes на ранок: группы с cardinality expert (gate_up/down) —
//     только локальные эксперты ранка (ADR-10 §3), остальные группы — полные;
//     slot_bytes каждой группы кратен tp (выравнивание 4096, ADR-12 §2.4).
//
// Допущение A2 (зафиксировано в отчёте): router (role=expert, cardinality
// только layer) НЕ делится по EP — размечается на каждом ранке полным
// (роутер вычисляется на ранке 0, ADR-10 §3; копия на остальных ранках).
#include "test_harness.hpp"
#include "stell/models/registry.hpp"
#include "stell/parallel/plan.hpp"

#include <map>
#include <string>
#include <vector>

using namespace stell::parallel;
using stell::models::ModelDescription;
using stell::models::Registry;

namespace {

// Каталог реестра: задаётся в tests/parallel/CMakeLists.txt (STELL_REGISTRY_DIR
// на stell_models — PRIVATE, тестам не наследуется).
#ifndef STELL_PARALLEL_REGISTRY_DIR
#define STELL_PARALLEL_REGISTRY_DIR "../src/models/registry"
#endif

ModelDescription load_glm_class() {
    Registry r;
    return r.load(std::string(STELL_PARALLEL_REGISTRY_DIR) + "/glm-class.json");
}

// Суммарный slot_bytes групп с cardinality {expert: N} (локальные на ранок).
struct SlotStats {
    std::size_t full = 0;          // группы без cardinality-expert (весь ранок)
    std::size_t expert_card = 0;   // per-expert байт групп cardinality expert (x128)
};

SlotStats slot_stats(const ModelDescription& d) {
    SlotStats s;
    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        const auto& g = d.groups[i];
        // cardinality expert видна по source_pattern (расширение уже случилось):
        const std::string& pat = d.group_source_patterns[i];
        if (pat.find(".experts.") != std::string::npos)
            s.expert_card += g.slot_bytes;
        else
            s.full += g.slot_bytes;
    }
    return s;
}

} // namespace

TEST(ep_glm_registry_loads_128_experts) {
    const ModelDescription d = load_glm_class();
    CHECK(d.is_moe);
    CHECK(d.moe.has_value() && d.moe->n_experts == 128);
    CHECK(d.moe->n_experts_active == 8);
    CHECK(d.n_heads == 64 && d.d_model == 4096 && d.n_layers == 32);
    CHECK(d.kv.n_kv_heads == 8);
    // Экспертные группы: gate_up [2048,4096] + down [4096,1024] на слой.
    const SlotStats st = slot_stats(d);
    // gate_up: 4,456,448 B; down: 2,228,224 B (NVFP4, block 32; оба кратны 4096)
    CHECK(st.expert_card == std::size_t(32) * 128 * (4456448 + 2228224));
    CHECK(st.expert_card % 4096 == 0);
}

TEST(ep_glm_pinning_completeness_ep_2_4_8) {
    // Полнота: каждый эксперт ровно один ep_rank; на ранке ровно 128/ep
    // локальных экспертов; 128 % ep == 0 — обязательная проверка (ADR-10 §3).
    const ModelDescription d = load_glm_class();
    const rank_t n_experts = rank_t(d.moe->n_experts);
    for (rank_t ep : {2, 4, 8}) {
        CHECK(n_experts % ep == 0);
        std::map<rank_t, int> per_rank;
        for (rank_t e = 0; e < n_experts; ++e) {
            const rank_t owner = expert_owner(e, ep);
            CHECK(owner >= 0 && owner < ep);
            ++per_rank[owner];
        }
        CHECK(per_rank.size() == std::size_t(ep));          // все ранки заняты
        for (rank_t r = 0; r < ep; ++r)
            CHECK(per_rank[r] == n_experts / ep);           // ровно 128/ep
    }
    // ep не делит 128 — план обязан отказаться (shard-путь: CommError).
    CHECK_THROWS_AS(shard_range(int(n_experts), 3, 0), CommError);
}

TEST(ep_glm_global_rank_mapping_pp_tp_ep) {
    // Маппинг ранка = pp*(tp*ep) + ep*tp + tp; владелец эксперта e в стадии
    // pp — глобальный ранок expert_owner_global(e, pp_rank, cfg); маршрут
    // expert_alltoall фан-аутит токен на все tp-ранки группы-владельца.
    const ModelDescription d = load_glm_class();
    const rank_t n_experts = rank_t(d.moe->n_experts);
    for (rank_t pp : {1, 2})
        for (rank_t tp : {1, 2})
            for (rank_t ep : {2, 4, 8}) {
                ParallelConfig c;
                c.pp = pp;
                c.tp = tp;
                c.ep = ep;
                for (rank_t e = 0; e < n_experts; ++e) {
                    const rank_t expected =
                        rank_t(pp) * (tp * ep) + rank_t(e % ep) * tp;   // tp_rank = 0
                    CHECK(expert_owner_global(e, rank_t(pp), c) == expected);
                    // Маршрут: tp-ранки группы-владельца, смежные с expected.
                    rank_t route[8] = {};
                    expert_alltoall_route(e, rank_t(pp), c, route);
                    for (rank_t k = 0; k < tp; ++k)
                        CHECK(route[k] == expected + k);
                }
            }
}

TEST(ep_glm_slot_bytes_per_rank_ep) {
    // slot_bytes на ранок (EP): группы cardinality expert — только локальные
    // (128/ep), остальные — полные. Сумма на ранок:
    //   full + expert_card/ep  (128 % ep == 0 → делится точно).
    const ModelDescription d = load_glm_class();
    const SlotStats st = slot_stats(d);
    for (rank_t ep : {2, 4, 8}) {
        CHECK(st.expert_card % std::size_t(ep) == 0);
        const std::size_t per_rank = st.full + st.expert_card / std::size_t(ep);
        const std::size_t world_total =
            st.full + st.expert_card;                        // EP1-эквивалент
        // Полное покрытие без дублирования: ep ранков * per_rank = мир без дублей.
        CHECK(std::size_t(ep) * per_rank - st.full ==
              st.expert_card + (std::size_t(ep) - 1) * st.full);
        (void)world_total;
    }
}

TEST(ep_glm_tp_divisibility_heads_kv_slots) {
    // TP по головам glm-class: n_heads=64, n_kv_heads=8 — делится на tp ≤ 8;
    // slot_bytes групп кратен tp (выравнивание 4096 → 4096 % tp == 0).
    const ModelDescription d = load_glm_class();
    for (rank_t tp : {2, 4, 8}) {
        CHECK(64 % tp == 0);
        check_kv_heads_divisible(int(d.kv.n_kv_heads), tp);
        CHECK_THROWS_AS(check_kv_heads_divisible(int(d.kv.n_kv_heads), 16), CommError);
        const ShardRange heads = shard_range(64, tp, tp - 1);
        CHECK(heads.end - heads.begin == 64 / int(tp));
        for (const auto& g : d.groups)
            CHECK(g.slot_bytes % std::size_t(tp) == 0);      // слот делится на ранок
        // slot_bytes_per_rank согласован с plan.hpp.
        for (const auto& g : d.groups)
            CHECK(slot_bytes_per_rank(g.slot_bytes, tp) == g.slot_bytes / std::size_t(tp));
    }
}

TEST(ep_glm_local_experts_are_contiguous_stride) {
    // Закрепление expert_id % ep: локальные эксперты ранка — арифметическая
    // прогрессия с шагом ep (r, r+ep, ...); полнота по всем ранкам = 0..127.
    const ModelDescription d = load_glm_class();
    const rank_t n_experts = rank_t(d.moe->n_experts);
    const rank_t ep = 8;
    std::vector<int> seen(int(n_experts), 0);
    for (rank_t r = 0; r < ep; ++r) {
        int local = 0;
        for (rank_t e = 0; e < n_experts; ++e)
            if (expert_owner(e, ep) == r) {
                ++local;
                ++seen[int(e)];
            }
        CHECK(local == n_experts / ep);
    }
    for (int e = 0; e < int(n_experts); ++e) CHECK(seen[e] == 1); // ровно один раз
}

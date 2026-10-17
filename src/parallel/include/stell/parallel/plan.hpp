// stell-llm — parallel: чистая логика деления по ранкам (ADR-10 §3).
//
// Контракт распределения, одинаково на всех ранках, без переговоров:
//  - планировщик continuous batching живёт на ранке 0; слоты сессий (ADR-08)
//    ОДИНАКОВЫ по ранкам (slot_id один и тот же, содержимое шардировано);
//  - TP: головы attention / KV-головы / колонки FFN делятся равномерно
//    (n % tp == 0 — обязательная проверка; KV-головы — тоже);
//  - EP: статическое закрепление экспертов expert_id % ep == ep_rank;
//  - PP: слои делятся на цепочку ранков (num_layers % pp == 0);
//  - комбинации: порядок PP, затем TP/EP внутри слоёв;
//    ранок = pp_rank * tp_ep + tp_ep_rank, где tp_ep = tp*ep и
//    tp_ep_rank = ep_rank * tp + tp_rank (TP-группа — контiguous-подмножество,
//    что даёт дешёвые NCCL sub-communicators по NVLink).
#pragma once

#include <cstddef>
#include "stell/parallel/comm.hpp"

namespace stell::parallel {

// --- топология и декомпозиция ранка ------------------------------------------

// Размер группы TP×EP внутри одной PP-стадии.
inline rank_t tp_ep_group_size(const ParallelConfig& cfg) noexcept { return rank_t(cfg.tp * cfg.ep); }

struct RankDecomp {
    rank_t pp_rank;    // стадия PP-цепочки (слои)
    rank_t ep_rank;    // EP-группа (эксперты) внутри стадии
    rank_t tp_rank;    // TP-группа (головы/колонки) внутри EP-группы
};

inline rank_t compose_rank(const RankDecomp& d, const ParallelConfig& cfg) {
    return d.pp_rank * tp_ep_group_size(cfg) + d.ep_rank * cfg.tp + d.tp_rank;
}

inline RankDecomp decompose_rank(rank_t world_rank, const ParallelConfig& cfg) {
    const rank_t g = tp_ep_group_size(cfg);
    RankDecomp d;
    d.pp_rank = world_rank / g;
    const rank_t in_stage = world_rank % g;
    d.ep_rank = in_stage / cfg.tp;
    d.tp_rank = in_stage % cfg.tp;
    return d;
}

// Ранг планировщика: ранок 0 (ADR-10 §3), рассылает разметку шага батча.
inline rank_t planner_rank(const ParallelConfig&) noexcept { return 0; }

// Валидация топологии (бросает CommError(RankMismatch)):
//  - world == pp * tp * ep и все компоненты > 0;
//  - device_ids пуст либо world_size == device_ids.size().
void check_topology(const ParallelConfig& cfg);

// --- TP: шардирование голов / KV-голов / колонок FFN --------------------------

struct ShardRange {       // полуинтервал [begin, end)
    int begin = 0;
    int end = 0;
};

// Равномерное шардирование n элементов на tp частей (n % tp == 0, иначе CommError).
ShardRange shard_range(int n, rank_t parts, rank_t part_index);

// KV-головы: n_kv_heads % tp == 0 — обязательная проверка при разметке (ADR-10 §3).
void check_kv_heads_divisible(int n_kv_heads, rank_t tp);

// --- EP: статическое закрепление экспертов -----------------------------------

// Локальный в EP-группе владелец эксперта: expert_id % ep == ep_rank (ADR-10 §3).
inline rank_t expert_owner(rank_t expert_id, rank_t ep) noexcept {
    return expert_id % ep;   // ep > 0 гарантируется check_topology
}

// Глобальный ранок «координатора» TP-группы-владельца эксперта: стадия pp_rank,
// ep_rank = expert_id % ep, tp_rank = 0. При tp > 1 токен фан-аутится на все
// tp-ранки этой EP-группы (см. коммент. к expert_alltoall в comm.hpp).
inline rank_t expert_owner_global(rank_t expert_id, rank_t pp_rank, const ParallelConfig& cfg) {
    return compose_rank(RankDecomp{pp_rank, expert_owner(expert_id, cfg.ep), 0}, cfg);
}

// Маршрут expert_alltoall для одного токена: пары (ранок, надо_ли) по всем
// tp-ранкам группы-владельца. Возвращается массив фиксированного размера tp.
void expert_alltoall_route(rank_t expert_id, rank_t src_pp_rank, const ParallelConfig& cfg,
                           rank_t* out /* размер cfg.tp */);

// --- PP: диапазоны слоёв ------------------------------------------------------

// Слои делятся на цепочку ранков подряд: [pp_rank * chunk, ...);
// num_layers % pp == 0 обязательна (иначе CommError(RankMismatch)).
ShardRange layer_range(int num_layers, rank_t pp, rank_t pp_rank);

// --- слоты сессий (ADR-08 ↔ ADR-10 §3) ---------------------------------------

// slot_id одинаков на всех ранках; содержимое слота шардировано по ранкам:
// байты слота на ранок = slot_bytes / tp (делимость обязательна).
std::size_t slot_bytes_per_rank(std::size_t slot_bytes_total, rank_t tp);

} // namespace stell::parallel

// stell-llm — parallel: чистая логика деления по ранкам (ADR-10 §3). Реализация.
#include "stell/parallel/plan.hpp"

namespace stell::parallel {

void check_topology(const ParallelConfig& cfg) {
    if (cfg.tp <= 0 || cfg.pp <= 0 || cfg.ep <= 0)
        throw CommError(CommError::Code::RankMismatch, "tp/pp/ep must be > 0");
    if (!cfg.device_ids.empty() &&
        rank_t(cfg.device_ids.size()) != cfg.world())
        throw CommError(CommError::Code::RankMismatch,
                        "device_ids.size() != world (pp*tp*ep)");
}

ShardRange shard_range(int n, rank_t parts, rank_t part_index) {
    if (n <= 0 || parts <= 0 || part_index >= parts || n % parts != 0)
        throw CommError(CommError::Code::RankMismatch,
                        "shard: n % parts != 0 (uniform TP/EP/PP split, ADR-10 §3)");
    const int chunk = n / int(parts);
    return ShardRange{int(part_index) * chunk, int(part_index) * chunk + chunk};
}

void check_kv_heads_divisible(int n_kv_heads, rank_t tp) {
    if (tp <= 0 || n_kv_heads % tp != 0)
        throw CommError(CommError::Code::RankMismatch,
                        "n_kv_heads % tp != 0 — KV-кеш не делится по KV-головам (ADR-10 §3)");
}

void expert_alltoall_route(rank_t expert_id, rank_t src_pp_rank, const ParallelConfig& cfg,
                           rank_t* out) {
    const rank_t ep_rank = expert_owner(expert_id, cfg.ep);
    for (rank_t tp_r = 0; tp_r < cfg.tp; ++tp_r)
        out[tp_r] = compose_rank(RankDecomp{src_pp_rank, ep_rank, tp_r}, cfg);
}

ShardRange layer_range(int num_layers, rank_t pp, rank_t pp_rank) {
    if (pp <= 0 || pp_rank >= pp || num_layers % pp != 0)
        throw CommError(CommError::Code::RankMismatch,
                        "num_layers % pp != 0 — слои делятся подряд по цепочке (ADR-10 §3)");
    const int chunk = num_layers / int(pp);
    return ShardRange{int(pp_rank) * chunk, int(pp_rank) * chunk + chunk};
}

std::size_t slot_bytes_per_rank(std::size_t slot_bytes_total, rank_t tp) {
    if (tp <= 0 || slot_bytes_total % std::size_t(tp) != 0)
        throw CommError(CommError::Code::RankMismatch,
                        "slot_bytes не делится по ранкам TP (ADR-10 §3)");
    return slot_bytes_total / std::size_t(tp);
}

} // namespace stell::parallel

#include "stell/engine/execution_plan.hpp"

#include <stdexcept>
#include <unordered_set>

namespace stell::engine {

const PlannedGroup* ExecutionPlan::find(const std::string& name) const {
    for (const PlannedGroup& g : groups)
        if (g.name == name) return &g;
    return nullptr;
}

std::unique_ptr<stell::tiers::Awaiter>
ExecutionPlan::pin_layer(stell::tiers::ITierManager& tm, std::uint32_t layer) const {
    // Массовая операция для слоя — интерфейс tiers (ADR-08 §2: pin_layer).
    // Слои нумеруются 0..n_layers-1; за пределами — TierError по контракту tiers.
    return tm.pin_layer(layer);
}

ExecutionPlan build_execution_plan(const stell::models::ModelDescription& desc) {
    if (desc.n_layers == 0 || desc.vocab_size == 0)
        throw std::invalid_argument("execution plan: model description has no layers/vocab");
    if (desc.groups.empty())
        throw std::invalid_argument("execution plan: empty tensor_groups (ADR-09 §1)");

    ExecutionPlan plan;
    plan.vocab_size = desc.vocab_size;
    plan.n_layers = desc.n_layers;
    plan.n_heads = desc.n_heads;
    plan.d_head = desc.d_head;
    plan.n_kv_heads = desc.n_kv_heads;
    plan.d_kv = desc.d_kv;
    plan.is_moe = desc.is_moe;
    plan.n_experts = desc.n_experts;
    plan.n_experts_active = desc.n_experts_active;

    bool has_kv = false;
    std::unordered_set<std::uint32_t> ids;

    std::size_t offset = 0;
    for (const stell::tiers::TensorGroup& g : desc.groups) {
        if (g.slot_bytes == 0)
            throw std::invalid_argument("execution plan: group slot_bytes==0 (ADR-08 §1: "
                                        "фиксируется ДО графа)");
        if (!ids.insert(g.id.value).second)
            throw std::invalid_argument("execution plan: duplicate GroupId");
        if (g.role == stell::tiers::Role::Kv) has_kv = true;

        PlannedGroup pg;
        pg.key = stell::tiers::GroupKey{g.id, g.role};
        pg.name = g.name;
        pg.slot_bytes = g.slot_bytes;
        pg.offset = offset;
        offset += g.slot_bytes;
        plan.groups.push_back(std::move(pg));
    }

    if (!has_kv)
        throw std::invalid_argument("execution plan: нет группы роли Kv (ADR-08 §1)");

    plan.slot_bytes_total = offset;      // фиксированный слот сессии (ADR-05)
    // Версия разметки: детерминировано из состава плана (переснятие графа при
    // любой смене состава/размеров групп — ADR-05/ADR-08).
    std::uint64_t h = 1469598103934665603ULL;
    for (const PlannedGroup& g : plan.groups) {
        h ^= g.key.id.value;
        h *= 1099511628211ULL;
        h ^= static_cast<std::uint64_t>(g.slot_bytes);
        h *= 1099511628211ULL;
    }
    plan.layout_version = static_cast<std::uint32_t>(h & 0xFFFFFFFFu);
    return plan;
}

} // namespace stell::engine

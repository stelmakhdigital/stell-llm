#include "stell/engine/execution_plan.hpp"

#include <cstdio>
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

std::vector<stell::tiers::TensorGroup> ExecutionPlan::tensor_groups() const {
    std::vector<stell::tiers::TensorGroup> out;
    out.reserve(groups.size());
    for (const PlannedGroup& g : groups) {
        stell::tiers::TensorGroup tg{};
        tg.id = g.key.id;
        tg.name = g.name;
        tg.role = g.key.role;
        tg.dtype = g.dtype;
        tg.shape = g.shape;
        tg.slot_bytes = g.slot_bytes;
        out.push_back(std::move(tg));
    }
    return out;
}

std::vector<std::vector<stell::tiers::GroupId>> ExecutionPlan::layer_map() const {
    // Имя слоя-группы: "l%03u." (pad3-конвенция имён, description.cpp::expand_tokens).
    auto layer_index_of = [&](const std::string& name) -> int {
        if (name.size() < 5 || name[0] != 'l') return -1;
        unsigned v = 0;
        for (int i = 1; i <= 3; ++i) {
            const char c = name[static_cast<std::size_t>(i)];
            if (c < '0' || c > '9') return -1;
            v = v * 10u + static_cast<unsigned>(c - '0');
        }
        if (name[4] != '.') return -1;
        if (v >= n_layers) return -1;                    // за пределами слоёв
        return static_cast<int>(v);
    };

    std::vector<std::vector<stell::tiers::GroupId>> map(n_layers);
    for (const PlannedGroup& g : groups) {
        const int l = layer_index_of(g.name);
        if (l >= 0) map[static_cast<std::size_t>(l)].push_back(g.key.id);
    }
    return map;
}

ExecutionPlan build_execution_plan(const PlainSpec& spec) {
    if (spec.n_layers == 0 || spec.vocab_size == 0)
        throw std::invalid_argument("execution plan: model description has no layers/vocab");
    if (spec.groups.empty())
        throw std::invalid_argument("execution plan: empty tensor_groups (ADR-09 §1)");

    ExecutionPlan plan;
    plan.vocab_size = spec.vocab_size;
    plan.n_layers = spec.n_layers;
    plan.n_heads = spec.n_heads;
    plan.d_head = spec.d_head;
    plan.n_kv_heads = spec.n_kv_heads;
    plan.d_kv = spec.d_kv;
    plan.is_moe = spec.is_moe;
    plan.n_experts = spec.n_experts;
    plan.n_experts_active = spec.n_experts_active;

    bool has_kv = false;
    std::unordered_set<std::uint32_t> ids;

    std::size_t offset = 0;
    for (const PlainGroup& g : spec.groups) {
        if (g.slot_bytes == 0)
            throw std::invalid_argument("execution plan: group slot_bytes==0 (ADR-08 §1: "
                                        "фиксируется ДО графа)");
        if (!ids.insert(g.id).second)
            throw std::invalid_argument("execution plan: duplicate GroupId");
        if (g.role == PlainRoleKv) has_kv = true;

        PlannedGroup pg;
        pg.key = stell::tiers::GroupKey{stell::tiers::GroupId{g.id},
                                        static_cast<stell::tiers::Role>(g.role)};
        pg.name = g.name;
        pg.dtype = static_cast<stell::tiers::DType>(g.dtype);
        pg.shape = g.shape;
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

// ============================================================================
// Мост engine ↔ models (round 2, интеграция): единственный TU engine,
// включающий stell/models/* (реестр, лоадеры — ADR-09/03/12).
//
// Зачем TU-изоляция: заголовки models тянут ЛОКАЛЬНОЕ ЗЕРКАЛО типов ADR-08
// (stell/models/tiers_compat.hpp, поток D). Совместное включение зеркала и
// реального stell/tiers/types.hpp в один TU — ODR-нарушение (переопределение
// stell::tiers::Role/DType/...). Поэтому конвертация ModelDescription →
// PlainSpec происходит здесь (только compat-типы), а TU engine с реальными
// типами (execution_plan.cpp, ops_runner.cpp) получают уже PlainSpec и
// конвертируют в реальные TensorGroup (раскладка 1:1, канонические значения
// enum — см. plain_spec.hpp).
//
// Конвертация честная: поля 1:1; slot_bytes/dtype/role переносятся как есть
// (значения enum зеркала совпадают с ADR-08 — сверка по values в тестах).
// ============================================================================
#include "stell/engine/plain_spec.hpp"

#include "stell/models/description.hpp"
#include "stell/models/registry.hpp"
#include "stell/models/weights.hpp"

#include <stdexcept>

namespace stell::engine {
namespace {

using CompatGroup = stell::tiers::TensorGroup;   // локальное зеркало models

PlainOpParam plain_param(const std::string& key, const stell::models::JsonValue& v) {
    PlainOpParam p{};
    p.key = key;
    switch (v.type) {
    case stell::models::JsonValue::Type::Int:
    case stell::models::JsonValue::Type::Double:
        p.is_number = true;
        p.fval = (v.type == stell::models::JsonValue::Type::Int)
                     ? static_cast<double>(v.int_v) : v.double_v;
        break;
    case stell::models::JsonValue::Type::Bool:
        p.is_number = true;
        p.fval = v.bool_v ? 1.0 : 0.0;
        break;
    case stell::models::JsonValue::Type::String:
        p.sval = v.string_v;
        break;
    default:
        throw std::runtime_error("plain spec: unsupported op param type '" + key + "'");
    }
    return p;
}

PlainGroup plain_group(const CompatGroup& g, const std::string& pattern) {
    PlainGroup out{};
    out.id = g.id.value;
    out.name = g.name;
    out.role = static_cast<std::uint8_t>(g.role);      // канонические значения ADR-08
    out.dtype = static_cast<std::uint8_t>(g.dtype);
    out.shape = g.shape;
    out.slot_bytes = g.slot_bytes;
    out.source_pattern = pattern;
    out.kv_block = (g.role == stell::tiers::Role::Kv);
    return out;
}

// Восстановление описания models из PlainSpec (для лоадеров — ADR-09 §3:
// loader.open(uri, desc) требует ModelDescription; здесь — compat-типы).
stell::models::ModelDescription compat_description(const PlainSpec& spec) {
    stell::models::ModelDescription d{};
    d.name = spec.name;
    d.is_moe = spec.is_moe;
    d.dtype_weights = spec.dtype_weights;
    d.vocab_size = spec.vocab_size;
    d.d_model = spec.d_model;
    d.n_layers = spec.n_layers;
    d.n_heads = spec.n_heads;
    d.d_head = spec.d_head;
    d.kv.dtype = spec.kv_dtype;
    d.kv.quant = spec.kv_quant;
    d.moe = std::nullopt;
    if (spec.is_moe) {
        stell::models::MoeSpec m{};
        m.n_experts = spec.n_experts;
        m.n_experts_active = spec.n_experts_active;
        m.expert_d_ff = spec.expert_d_ff;
        m.router = spec.router;
        d.moe = m;
    }
    for (const PlainGroup& g : spec.groups) {
        CompatGroup cg{};
        cg.id = stell::tiers::GroupId{g.id};
        cg.name = g.name;
        cg.role = static_cast<stell::tiers::Role>(g.role);
        cg.dtype = static_cast<stell::tiers::DType>(g.dtype);
        cg.shape = g.shape;
        cg.slot_bytes = g.slot_bytes;
        d.groups.push_back(std::move(cg));
        d.group_source_patterns.push_back(g.source_pattern);
        // [R6] round 5 (planner Q7): срез стековного тензора доезжает до
        // лоадера — без него sliced-группы (fused-тензоры) недоступны движку.
        d.group_source_slices.push_back(g.source_slice);
    }
    return d;
}

} // namespace

PlainSpec plain_spec_from_description(const stell::models::ModelDescription& desc) {
    PlainSpec spec{};
    spec.name = desc.name;
    spec.is_moe = desc.is_moe;
    spec.dtype_weights = desc.dtype_weights;
    spec.vocab_size = static_cast<std::uint32_t>(desc.vocab_size);
    spec.d_model = static_cast<std::uint32_t>(desc.d_model);
    spec.n_layers = desc.n_layers;
    spec.n_heads = desc.n_heads;
    spec.d_head = desc.d_head;
    spec.n_kv_heads = desc.kv.n_kv_heads;
    spec.d_kv = desc.kv.d_kv;
    spec.kv_dtype = desc.kv.dtype;
    spec.kv_quant = desc.kv.quant;
    if (desc.moe) {
        spec.n_experts = desc.moe->n_experts;
        spec.n_experts_active = desc.moe->n_experts_active;
        spec.expert_d_ff = desc.moe->expert_d_ff;
        spec.router = desc.moe->router;
    }
    for (const stell::models::OpSpec& op : desc.ops) {
        PlainOp po{};
        po.op = op.op;
        po.io = op.io;
        for (const auto& [key, v] : op.params) po.params.push_back(plain_param(key, v));
        spec.ops.push_back(std::move(po));
    }
    for (std::size_t i = 0; i < desc.groups.size(); ++i) {
        PlainGroup pg = plain_group(desc.groups[i], desc.group_source_patterns[i]);
        // [R6] round 5: PlainGroup несёт source_slice (−1 = тензор целиком)
        pg.source_slice = i < desc.group_source_slices.size()
                              ? desc.group_source_slices[i] : -1;
        spec.groups.push_back(std::move(pg));
    }
    return spec;
}

// Доступно тестам/отчётам: конвертация уже загруженного описания (round 2).

PlainSpec load_plain_spec(const std::string& json_path) {
    stell::models::Registry registry;
    return plain_spec_from_description(registry.load(json_path));
}

LoadedWeights load_plain_weights(const std::string& uri, const PlainSpec& spec,
                                 PlainWeightsFormat fmt) {
    const stell::models::ModelDescription desc = compat_description(spec);

    // [R6] round 5 (planner Q7, ADR-08/09): открытие источника — РОВНО ОДНО.
    // Ранее мост открывал ДВА источника (round 2: plain-лоадер для F16/F32
    // групп + квантующий для NVFP4/W4A16 — обход, пока профильный лоадер не
    // обслуживал описание целиком); с round 2-фикса NVFP4W4A16-лоадер
    // обслуживает ВЕСЬ план одним источником (квантование на read; плоские
    // группы — тот же fill_plain, побитовая идентичность путей с конвертером).
    // Своего открытия лоадера движком по стороннему описанию нет: sliced-
    // группы (source_slice) читаются общим IWeightsSource — контракты
    // ADR-08/ADR-09 не дублируются.
    bool has_quant = false;
    for (const PlainGroup& g : spec.groups)
        if (!g.kv_block && (g.dtype == PlainDTypeNVFP4 || g.dtype == PlainDTypeW4A16))
            has_quant = true;

    PlainWeightsFormat effective = fmt;
    if (fmt != PlainWeightsStellTier &&
        (has_quant || fmt == PlainWeightsNVFP4W4A16))
        effective = PlainWeightsNVFP4W4A16;   // весь план одним источником

    auto loader = stell::models::make_loader(
        static_cast<stell::models::WeightsFormat>(effective));
    std::unique_ptr<stell::models::IWeightsSource> src = loader->open(uri, desc);

    LoadedWeights out{};
    out.source_opens = 1;
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block) continue;                    // KV-группы весов не имеют (ADR-09 §3)
        std::vector<std::uint8_t> buf(g.slot_bytes, 0);
        src->read(stell::tiers::GroupKey{stell::tiers::GroupId{g.id},
                                         static_cast<stell::tiers::Role>(g.role)},
                  buf.data(), buf.size());
        out.slots[g.id] = std::move(buf);
    }
    src->close();
    return out;
}

const char* plain_weights_format_name(PlainWeightsFormat fmt) {
    return stell::models::weights_format_name(
        static_cast<stell::models::WeightsFormat>(fmt));
}

} // namespace stell::engine

// План исполнения: разметка модели (ADR-09) → слоты фиксированного размера
// (ADR-05/ADR-08). Engine — единственный, кто знает про слои; tiers оперирует
// обезличенными группами (PLAN.md §2.3).
//
// Round 2 (интеграция): вход — engine-side PlainSpec (см. plain_spec.hpp;
// почему не models::ModelDescription — комментарий там). Реальные типы
// stell::tiers (TensorGroup/GroupKey/Awaiter) используются здесь напрямую.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "plain_spec.hpp"
#include "stell/tiers/manager.hpp"

namespace stell::engine {

// Группа в плане: ключ tiers (ADR-08 §1) + фиксированный слот.
struct PlannedGroup {
    stell::tiers::GroupKey key{};
    std::string name;                 // человекочитаемое имя группы (ADR-08)
    stell::tiers::DType dtype = stell::tiers::DType::F16;
    std::vector<std::int64_t> shape;
    std::size_t slot_bytes = 0;       // фиксируется при разметке (ADR-08 §1)
    std::size_t offset = 0;           // смещение внутри слота сессии
};

struct ExecutionPlan {
    std::uint32_t vocab_size = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t n_heads = 0;
    std::uint32_t d_head = 0;
    std::uint32_t n_kv_heads = 0;
    std::uint32_t d_kv = 0;
    bool is_moe = false;
    std::uint32_t n_experts = 0;
    std::uint32_t n_experts_active = 0;

    std::size_t slot_bytes_total = 0;   // размер слота сессии (константа плана)
    std::uint32_t layout_version = 0;   // версия разметки (переснятие графа при смене)
    std::vector<PlannedGroup> groups;

    const PlannedGroup* find(const std::string& name) const;

    // ADR-08 §5: engine гарантирует готовность весов к началу слоя — pin через
    // tier-менеджера. Возвращает awaiter (блокировать поток планировщика можно
    // только через wait(), ADR-08 §2).
    std::unique_ptr<stell::tiers::Awaiter>
    pin_layer(stell::tiers::ITierManager& tm, std::uint32_t layer) const;

    // Разметка плана → реальные тензорные группы (ADR-08 §1) — единственный
    // вход ITierManager::set_plan (раскладка PlainGroup → TensorGroup 1:1,
    // канонические значения enum — plain_spec.hpp).
    std::vector<stell::tiers::TensorGroup> tensor_groups() const;

    // Соответствие слой → группы (ADR-08 «Уточнение»: set_layer_map на
    // конкретной реализации TierManager). Слой l = группы с именем "lNNN.*"
    // (pad3-конвенция names — description.cpp::expand_tokens), плюс kv-группы;
    // глобальные группы (embed/head/norm_final) в карту слоёв не входят.
    std::vector<std::vector<stell::tiers::GroupId>> layer_map() const;
};

// Разметка описания модели → план. Ошибки — std::invalid_argument
// (внутренняя политика engine; на стык с tiers/models не выходит).
// Бросает, если: пустые группы, slot_bytes==0 у группы, нет kv-роли,
// slot_bytes переполнение.
ExecutionPlan build_execution_plan(const PlainSpec& spec);

} // namespace stell::engine

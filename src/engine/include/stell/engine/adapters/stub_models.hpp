// FAKE/STUB header (engine-owned build aid) — ТИПЫ ВЫПИСАНЫ ДОСЛОВНО из ADR-09 §2.
// Единственный источник истины — docs/adr/ADR-09-engine-models.md.
// После вклейки реального модуля models (src/models/include/stell/models/registry.hpp)
// заменить на реальный заголовок. Только объявления — реализация контракта не здесь.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stell/engine/adapters/stub_tiers.hpp"

namespace stell::models {

// Результат парсинга JSON-описания (ADR-09 §1/§2), строго типизирован.
// Поля 1:1 из схемы §1 (для CPU-сборки engine до старта потока D).
struct ModelDescription {
    std::string name;
    bool is_moe;
    std::string dtype_weights;             // "F16" | "NVFP4" | "W4A16" (ADR-03)

    std::uint32_t vocab_size = 0;
    std::uint32_t d_model   = 0;
    std::uint32_t n_layers  = 0;
    std::uint32_t n_heads   = 0;
    std::uint32_t d_head    = 0;

    // kv: {"dtype": "F16"|"E4M3", "n_kv_heads": N, "d_kv": D, "quant": null|"E4M3"}
    std::string   kv_dtype = "F16";
    std::uint32_t n_kv_heads = 0;
    std::uint32_t d_kv       = 0;

    // moe: только для class=moe
    std::uint32_t n_experts        = 0;
    std::uint32_t n_experts_active = 0;
    std::uint32_t expert_d_ff      = 0;
    std::string   router           = "softmax-topk";

    // Разметка, размноженная по cardinality — единственный источник ролей (ADR-08),
    // готова к set_plan(); slot_bytes фиксируется при загрузке описания (ADR-09 §1).
    std::vector<stell::tiers::TensorGroup> groups;
};

class IRegistry {
public:
    virtual ~IRegistry() = default;
    virtual ModelDescription load(const std::string& json_path) = 0;
    virtual std::vector<std::string> available() const = 0;
};

} // namespace stell::models

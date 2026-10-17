// ============================================================================
// Декларативное описание модели — контракт ADR-09 §1 (схема JSON 1.0).
//
// ModelDescription — строго типизированный результат парсинга JSON:
//   ops            — операции engine (порядок = граф исполнения слоя);
//   group_templates— шаблоны tensor_groups из JSON;
//   groups         — РАСШИРЕННЫЕ тензорные группы (ADR-08), готовые к
//                    ITierManager::set_plan; GroupId монотонный 0,1,2,...;
//                    slot_bytes фиксирован при разметке (кратен 4096, ADR-12 §2.4).
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "stell/models/json.hpp"
#include "stell/models/tiers_compat.hpp"

namespace stell::models {

// Ошибка валидации/парсинга описания (реестр, ADR-09 §2).
class DescriptionError : public std::runtime_error {
public:
    explicit DescriptionError(const std::string& what) : std::runtime_error(what) {}
};

struct KvSpec {
    std::string dtype = "F16";
    std::uint32_t n_kv_heads = 0;
    std::uint32_t d_kv = 0;
    std::string quant;  // пусто = null (ADR-04: опционально "E4M3")
};

struct MoeSpec {
    std::uint32_t n_experts = 0;
    std::uint32_t n_experts_active = 0;
    std::uint32_t expert_d_ff = 0;
    std::string router;
};

struct OpSpec {
    std::string op;
    std::vector<std::string> io;
    std::vector<std::pair<std::string, JsonValue>> params;  // порядок сохранён
};

struct TensorGroupTemplate {
    std::string name;              // шаблон, напр. "l{layer}.e{expert}.ffn"
    std::string role;              // "attn" | "expert" | "kv"
    std::vector<std::int64_t> shape;
    std::string dtype;             // целевой dtype группы (профиль ADR-03)
    std::string source_pattern;    // имя тензора во входном формате (ADR-09 §1)
    bool kv_block = false;         // группа KV-кеша: весов в файле нет
    // cardinality в порядке документа JSON (итерация = внешний цикл → внутренний).
    std::vector<std::pair<std::string, std::uint32_t>> cardinality;
};

struct ModelDescription {
    std::string schema;        // "1.0" (мажор обязан быть 1)
    std::string name;
    bool is_moe = false;       // class: "dense" | "moe"
    std::string dtype_weights; // F16 | NVFP4 | W4A16 (ADR-03)
    std::uint64_t vocab_size = 0;
    std::uint64_t d_model = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t n_heads = 0;
    std::uint32_t d_head = 0;
    KvSpec kv;
    std::optional<MoeSpec> moe;  // обязана присутствовать при class == "moe"
    std::vector<OpSpec> ops;
    std::vector<TensorGroupTemplate> group_templates;
    // Расширенные группы (cardinality размножена), GroupId монотонный с 0.
    std::vector<stell::tiers::TensorGroup> groups;
    // Разрешённые source_pattern для каждой группы (тот же порядок, что groups);
    // для kv-групп паттерн указывает на KV-блок слоя, а не на весовой тензор.
    std::vector<std::string> group_source_patterns;
};

// Парсинг + валидация описания (ADR-09 §1: schema-версия, роли, dtype,
// cardinality-токены). Бросает DescriptionError (в т.ч. на JsonError внутри).
ModelDescription parse_model_description(const std::string& json_text,
                                         const std::string& source_name = "");

// --- утилиты разметки (единый источник истины для лоадеров и конвертера) ----

stell::tiers::Role parse_role(const std::string& role);       // "attn"→Attn, "expert"→Expert, "kv"→Kv
stell::tiers::DType parse_dtype(const std::string& dtype);    // F16|F32|E4M3|NVFP4|W4A16

// Размер полезной нагрузки группы (байты, до выравнивания), по профилям ADR-03:
//   F16=2B/эл, F32=4B/эл, E4M3=1B/эл,
//   NVFP4 = ceil(n/2) нибблов E2M1 + ceil(n/32) байт scale E4M3 (блок 32),
//   W4A16 = ceil(n/2) нибблов int4 + 2*ceil(n/32) байт scale F16 (блок 32).
std::size_t payload_bytes_for(const std::string& dtype, const std::vector<std::int64_t>& shape);

// slot_bytes группы (ADR-08): payload, выровненный вверх до 4096 — кратность
// slot_align stell-tier (ADR-12 §2.4) фиксируется при разметке.
std::size_t slot_bytes_for(const std::string& dtype, const std::vector<std::int64_t>& shape);

// Код dtype_profile заголовка stell-tier (ADR-12 §2): F16=1, NVFP4=2, W4A16=3.
std::uint32_t dtype_profile_code(const std::string& dtype_weights);

}  // namespace stell::models

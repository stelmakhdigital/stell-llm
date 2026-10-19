// ============================================================================
// Engine-side спецификация модели (round 2, интеграция engine ↔ models/tiers).
//
// Почему отдельный тип, а не stell::models::ModelDescription напрямую:
// публичные заголовки models (stell/models/description.hpp) тянут ЛОКАЛЬНОЕ
// ЗЕРКАЛО типов ADR-08 (stell/models/tiers_compat.hpp) — потоком D; включение
// его вместе с реальным stell/tiers/types.hpp в один TU = переопределение
// stell::tiers::Role/DType/GroupId/... (ODR-нарушение). Поэтому:
//   * plain_spec.hpp — POD-зеркало плана (без stell::tiers-типов);
//   * TU src/model_spec_bridge.cpp — единственный TU engine, включающий
//     stell/models/* (конвертация ModelDescription → PlainSpec, загрузка весов);
//   * TU engine с реальными типами (execution_plan.cpp, ops_runner.cpp)
//     конвертируют PlainGroup → stell::tiers::TensorGroup (раскладка поля
//     1:1, значения enum зафиксированы: Expert=0, Attn=1, Kv=2; F16=0..W4A16=4).
// Канонические значения продублированы из реализации контракта (ADR-08 §1,
// stell/tiers/types.hpp): Role::Expert=0, Attn=1, Kv=2; DType: F16=0..W4A16=4.
// Внимание: это коды ADR-08, а не stell-tier индекса (ADR-12 §2.2, там
// 0=attn/1=expert/2=kv) — конвертации между двумя кодировками делает models.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace stell::models { struct ModelDescription; }

namespace stell::engine {

// Канонические коды (ADR-08 §1: Role::Expert=0, Attn=1, Kv=2; DType F16=0..W4A16=4).
enum PlainRole : std::uint8_t {
    PlainRoleExpert = 0,
    PlainRoleAttn   = 1,
    PlainRoleKv     = 2,
};

enum PlainDType : std::uint8_t {
    PlainDTypeF16   = 0,
    PlainDTypeF32   = 1,
    PlainDTypeE4M3  = 2,
    PlainDTypeNVFP4 = 3,
    PlainDTypeW4A16 = 4,
};

// Формат весов (ADR-03; отображение 1:1 на stell::models::WeightsFormat).
enum PlainWeightsFormat : std::uint8_t {
    PlainWeightsSafetensors = 0,
    PlainWeightsGGUF        = 1,
    PlainWeightsNVFP4W4A16  = 2,
    PlainWeightsStellTier   = 3,
};

// Параметры операции: числа и строки раздельно (порядок сохранён).
struct PlainOpParam {
    std::string key;
    double fval = 0.0;        // если параметр — число
    std::string sval;         // если параметр — строка/bool-имя
    bool is_number = false;
};

// Операция графа исполнения слоя (ADR-09 §1: порядок = граф слоя).
struct PlainOp {
    std::string op;                       // "rmsnorm" | "attention" | "ffn" | "moe_gather_scatter"
    std::vector<std::string> io;
    std::vector<PlainOpParam> params;
};

// Тензорная группа плана: POD-копия stell::tiers::TensorGroup (раскладка 1:1).
struct PlainGroup {
    std::uint32_t id = 0;                 // GroupId.value (монотонный, ADR-08)
    std::string name;                     // расширенное имя (pad3: "l000.attn.wq")
    std::uint8_t role = PlainRoleAttn;
    std::uint8_t dtype = PlainDTypeF16;
    std::vector<std::int64_t> shape;
    std::size_t slot_bytes = 0;           // ADR-08 §1: фиксируется при разметке
    std::string source_pattern;           // ADR-09 §1: имя тензора во входном формате
    std::int64_t source_slice = -1;       // [R6] round 5: срез стековного тензора
                                          // по старшей размерности (−1 = тензор
                                          // целиком); мост обязан нести срез в
                                          // compat-описание для лоадера (ADR-09
                                          // «Уточнение round 5», planner Q7)
    bool kv_block = false;                // KV-группа: весов в файле нет
};

// Спецификация модели: поля 1:1 из схемы ADR-09 §1 (после расширения
// cardinality в models — см. plain_spec_bridge.cpp).
struct PlainSpec {
    std::string name;
    bool is_moe = false;
    std::string dtype_weights = "F16";    // F16 | NVFP4 | W4A16 (ADR-03)

    std::uint32_t vocab_size = 0;
    std::uint32_t d_model   = 0;
    std::uint32_t n_layers  = 0;
    std::uint32_t n_heads   = 0;
    std::uint32_t d_head    = 0;
    std::uint32_t n_kv_heads = 0;
    std::uint32_t d_kv      = 0;

    std::uint32_t n_experts        = 0;
    std::uint32_t n_experts_active = 0;
    std::uint32_t expert_d_ff      = 0;
    std::string router = "softmax-topk";

    std::string kv_dtype = "F16";
    std::string kv_quant;                 // пусто = null (ADR-04)

    std::vector<PlainOp> ops;             // граф исполнения слоя
    std::vector<PlainGroup> groups;       // расширенные группы (ADR-08)
};

// Материализованные веса: GroupId.value → payload группы (slot_bytes байт,
// целевой dtype группы; деquant — ядра engine/kernels, не лоадер — ADR-09 §3).
// KV-группы в карте отсутствуют.
struct LoadedWeights {
    std::unordered_map<std::uint32_t, std::vector<std::uint8_t>> slots;

    // [R6] round 5: сколько источников IWeightsSource открыл мост (честная
    // метрика контракта «открытие источника ровно одно», planner Q7).
    std::size_t source_opens = 0;
};

// --- мост к модулю models (реализация: src/model_spec_bridge.cpp) ------------
// Конвертация уже загруженного описания models (ADR-09 §2) → PlainSpec;
// TU-изолирована (см. комментарий о ODR выше).
PlainSpec plain_spec_from_description(const stell::models::ModelDescription& desc);

// Загрузка описания из JSON-файла через реестр (ADR-09 §2: Registry::load).
PlainSpec load_plain_spec(const std::string& json_path);

// Загрузка весов: формат → payload групп (ADR-09 §3; ошибки — LoaderError).
LoadedWeights load_plain_weights(const std::string& uri, const PlainSpec& spec,
                                 PlainWeightsFormat fmt);

// Синхронизация: имя формата → код (для честных отчётов/конфига C ABI).
const char* plain_weights_format_name(PlainWeightsFormat f);

} // namespace stell::engine

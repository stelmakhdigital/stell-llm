// ============================================================================
// ЛОКАЛЬНОЕ ЗЕРКАЛО типов ADR-08 (namespace stell::tiers).
//
// Контракт ADR-09 требует, чтобы публичные заголовки models ссылались на
// stell::tiers::TensorGroup / GroupKey. Поток B (src/tiers) ещё не опубликовал
// stell/tiers/types.hpp, поэтому типы продублированы 1:1 из
// docs/adr/ADR-08-engine-tiers.md (§1) ниже.
//
// ПРИ ИНТЕГРАЦИИ: заменить include этого файла на #include <stell/tiers/types.hpp>
// (поток B) и удалить зеркало. Отличие зеркала от контракта = ошибка интеграции;
// сверять по ADR-08 §1.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stell::tiers {

enum class Role { Expert, Attn, Kv };

enum class DType { F16, F32, E4M3, NVFP4, W4A16 };  // W4A16 = упакованный W4 + F16 scale

// Непрозрачный идентификатор тензорной группы (ADR-08 §1): монотонный,
// уникален в рамках одной модели, присваивается при разметке (ADR-09).
struct GroupId {
    std::uint32_t value;
    bool operator==(const GroupId& o) const { return value == o.value; }
    bool operator!=(const GroupId& o) const { return value != o.value; }
    bool operator<(const GroupId& o) const { return value < o.value; }
};

// Описание группы: выдаётся при разметке модели, далее только читается.
struct TensorGroup {
    GroupId id;
    std::string name;                     // напр. "l012.e003.wqkv"
    Role role;
    DType dtype;
    std::vector<std::int64_t> shape;      // логическая форма
    std::size_t slot_bytes;               // ВЫРОВНЕН, фиксируется ДО CUDA Graph (ADR-05/08)
};

struct GroupKey {  // минимальный запрос tiers со стороны engine (ADR-08 §1)
    GroupId id;
    Role role;
};

}  // namespace stell::tiers

// ADR-08 §1: базовые типы слоя tiers — «тензорные группы с ролями».
// Контракт зафиксирован; изменения — только через ADR и согласование с worker.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace stell::tiers {

enum class Role {
    Expert = 0,
    Attn   = 1,
    Kv     = 2,
};

enum class DType {
    F16   = 0,
    F32   = 1,
    E4M3  = 2,
    NVFP4 = 3,
    W4A16 = 4, // упакованный W4 + F16 scale
};

// Непрозрачный идентификатор тензорной группы.
// Присваивается при разметке модели (см. ADR-09), стабилен в рамках сессии модели.
struct GroupId {
    std::uint32_t value;
    bool operator==(const GroupId& o) const { return value == o.value; }
    bool operator!=(const GroupId& o) const { return value != o.value; }
    bool operator<(const GroupId& o) const { return value < o.value; }
};

struct GroupIdHash {
    std::size_t operator()(const GroupId& g) const noexcept { return std::hash<std::uint32_t>{}(g.value); }
};

// Описание группы: выдаётся engine/tiers при разметке, далее только читается.
struct TensorGroup {
    GroupId   id;
    std::string name;               // человекочитаемое имя, напр. "l012.e003.wqkv"
    Role      role;
    DType     dtype;
    std::vector<std::int64_t> shape; // логическая форма
    std::size_t   slot_bytes;        // РАЗМЕР СЛОТА В БАЙТАХ (выровнен, фиксируется
                                     // ДО старта CUDA Graph — см. ADR-05; гарантия tiers)
};

struct GroupKey {       // минимальный запрос tiers со стороны engine
    GroupId   id;
    Role      role;         // дублируется для читабельности метрик/логов
};

inline bool operator==(const GroupKey& a, const GroupKey& b) noexcept {
    return a.id == b.id && a.role == b.role;
}

} // namespace stell::tiers

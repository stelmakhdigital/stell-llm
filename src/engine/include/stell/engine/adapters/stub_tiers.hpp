// FAKE/STUB header (engine-owned build aid) — ТИПЫ ВЫПИСАНЫ ДОСЛОВНО из ADR-08 §1-§3.
// Единственный источник истины по контракту — docs/adr/ADR-08-engine-tiers.md.
// После вклейки реального модуля tiers (src/tiers/include/stell/tiers/types.hpp)
// этот заголовок должен быть заменён на реальный (одинаковые декларации не
// должны остаться в двух местах). Здесь НЕ реализуется контракт — только
// объявления, чтобы CPU-конфигурация engine собиралась до старта потоков.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace stell::tiers {

enum class Role { Expert, Attn, Kv };

enum class DType { F16, F32, E4M3, NVFP4, W4A16 };  // W4A16 = упакованный W4 + F16 scale

// Непрозрачный идентификатор тензорной группы (ADR-08 §1).
struct GroupId {
    std::uint32_t value;            // монотонный, уникален в рамках одной модели
    bool operator==(const GroupId& o) const { return value == o.value; }
};

// Описание группы: выдаётся engine/tiers при разметке, далее только читается.
struct TensorGroup {
    GroupId   id;
    std::string name;
    Role      role;
    DType     dtype;
    std::vector<std::int64_t> shape;
    std::size_t   slot_bytes;        // размер слота в байтах, фиксируется ДО CUDA Graph (ADR-05)
};

struct GroupKey {
    GroupId   id;
    Role      role;
};


class Awaiter {
public:
    virtual ~Awaiter() = default;
    virtual bool ready() const = 0;
    virtual void wait() = 0;
};

// --- ADR-08 §2: статистика (объявлены ДО ITierManager, как в ADR-08 §2) ------
struct PrefetchStats {
    std::uint64_t hits;
    std::uint64_t misses;
    double        hit_rate;
};

struct PauseStats {
    double p50_ms;
    double p95_ms;
    double p99_ms;
};

struct TierStats {
    PrefetchStats prefetch;
    PauseStats    layer_pause;
    std::array<std::uint64_t, 3> tier_bytes;   // [VRAM, pinned RAM, NVMe]
    std::uint64_t io_bytes;
    std::uint64_t io_ops;
};

class ITierManager {
public:
    virtual ~ITierManager() = default;
    virtual void set_plan(const std::vector<TensorGroup>& groups) = 0;
    virtual std::unique_ptr<Awaiter> pin(const GroupKey& key) = 0;
    virtual void unpin(const GroupKey& key) = 0;
    virtual std::unique_ptr<Awaiter> pin_layer(std::uint32_t layer) = 0;
    virtual void prefetch(const std::vector<GroupKey>& keys) = 0;
    virtual TierStats stats() const = 0;
};

// --- ADR-08 §3: ошибки — исключения ------------------------------------------
class TierError : public std::runtime_error {
public:
    enum class Code { PlanMismatch, SlotExhausted, IoFailure, NotPinned, Timeout };
    TierError(Code c, const std::string& what)
        : std::runtime_error(what), code_(c) {}
    Code code() const noexcept { return code_; }
private:
    Code code_;
};

} // namespace stell::tiers

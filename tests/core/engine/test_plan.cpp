// ExecutionPlan: разметка модели (ADR-09) → слоты фиксированного размера
// (ADR-08 §1); валидация (kv-роль, slot_bytes>0, дубликаты), pin_layer (ADR-08 §5).
#include "test_util.hpp"

#include <stdexcept>
#include <vector>

#include "stell/engine/execution_plan.hpp"

using namespace stell::engine;
using stell::tiers::DType;
using stell::tiers::GroupId;
using stell::tiers::Role;
using stell::tiers::TensorGroup;

namespace {

TensorGroup group(std::uint32_t id, const std::string& name, Role role,
                  DType dtype, std::size_t slot_bytes) {
    TensorGroup g{};
    g.id = GroupId{id};
    g.name = name;
    g.role = role;
    g.dtype = dtype;
    g.shape = {64, 64};
    g.slot_bytes = slot_bytes;
    return g;
}

class FakeTierManager : public stell::tiers::ITierManager {
public:
    void set_plan(const std::vector<TensorGroup>&) override {}
    std::unique_ptr<stell::tiers::Awaiter> pin(const stell::tiers::GroupKey&) override {
        return nullptr;
    }
    void unpin(const stell::tiers::GroupKey&) override {}
    std::unique_ptr<stell::tiers::Awaiter> pin_layer(std::uint32_t layer) override {
        pinned_.push_back(layer);
        return nullptr;
    }
    void prefetch(const std::vector<stell::tiers::GroupKey>&) override {}
    stell::tiers::TierStats stats() const override { return {}; }

    std::vector<std::uint32_t> pinned_;
};

stell::models::ModelDescription dense_model() {
    stell::models::ModelDescription d{};
    d.name = "dense-test";
    d.is_moe = false;
    d.dtype_weights = "F16";
    d.vocab_size = 1024;
    d.n_layers = 2;
    d.n_heads = 4;
    d.d_head = 32;
    d.n_kv_heads = 2;
    d.d_kv = 32;
    // ADR-09 §1: размеченные группы (слой-шаблон уже размножен)
    d.groups.push_back(group(1, "l0.attn.wq", Role::Attn, DType::F16, 1024));
    d.groups.push_back(group(2, "l1.attn.wq", Role::Attn, DType::F16, 1024));
    d.groups.push_back(group(3, "l0.kv", Role::Kv, DType::F16, 2048));
    d.groups.push_back(group(4, "l1.kv", Role::Kv, DType::F16, 2048));
    return d;
}

} // namespace

void run() {
    // --- happy path ---------------------------------------------------------
    ExecutionPlan plan = build_execution_plan(dense_model());
    STELL_CHECK(plan.n_layers == 2);
    STELL_CHECK(plan.is_moe == false);
    STELL_CHECK(plan.slot_bytes_total == 1024 + 1024 + 2048 + 2048);
    STELL_CHECK(plan.groups.size() == 4);

    // offsets монотонны и накапливаются
    std::size_t expect_off = 0;
    for (const PlannedGroup& g : plan.groups) {
        STELL_CHECK(g.offset == expect_off);
        expect_off += g.slot_bytes;
    }

    const PlannedGroup* kv = plan.find("l0.kv");
    STELL_CHECK(kv != nullptr && kv->key.role == Role::Kv);
    STELL_CHECK(plan.find("nope") == nullptr);

    // layout_version детерминирован: тот же план — та же версия
    ExecutionPlan again = build_execution_plan(dense_model());
    STELL_CHECK(again.layout_version == plan.layout_version);
    // другой план — другая версия (переснятие графа, ADR-05)
    stell::models::ModelDescription d2 = dense_model();
    d2.groups[0].slot_bytes = 1536;
    ExecutionPlan other = build_execution_plan(d2);
    STELL_CHECK(other.layout_version != plan.layout_version);

    // --- pin_layer проходит через контракт tiers (ADR-08 §5) ----------------
    FakeTierManager tm;
    STELL_CHECK(plan.pin_layer(tm, 0) == nullptr);   // awaiter: CPU-фейк без ожидания
    STELL_CHECK(tm.pinned_.size() == 1);
    STELL_CHECK(tm.pinned_[0] == 0);

    // --- валидация ----------------------------------------------------------
    bool threw = false;
    try {
        stell::models::ModelDescription bad = dense_model();
        bad.groups.clear();                          // пустые группы
        build_execution_plan(bad);
    } catch (const std::invalid_argument&) { threw = true; }
    STELL_CHECK(threw);

    threw = false;
    try {
        stell::models::ModelDescription bad = dense_model();
        bad.groups.clear();
        bad.groups.push_back(group(9, "x", Role::Attn, DType::F16, 0));  // slot_bytes==0
        bad.groups.push_back(group(10, "kv", Role::Kv, DType::F16, 16));
        build_execution_plan(bad);
    } catch (const std::invalid_argument&) { threw = true; }
    STELL_CHECK(threw);

    threw = false;
    try {
        stell::models::ModelDescription bad = dense_model();
        bad.groups.clear();
        bad.groups.push_back(group(1, "a", Role::Attn, DType::F16, 64));
        bad.groups.push_back(group(1, "b", Role::Kv, DType::F16, 64));   // дубликат id
        build_execution_plan(bad);
    } catch (const std::invalid_argument&) { threw = true; }
    STELL_CHECK(threw);

    threw = false;
    try {
        stell::models::ModelDescription bad = dense_model();
        bad.groups.clear();
        bad.groups.push_back(group(1, "attn", Role::Attn, DType::F16, 64));  // нет kv
        build_execution_plan(bad);
    } catch (const std::invalid_argument&) { threw = true; }
    STELL_CHECK(threw);
}

STELL_TEST_MAIN(run)

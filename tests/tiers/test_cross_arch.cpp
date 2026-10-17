// Фиксация границы ADR-12 §3: «tiers не знает про модели» — один кодовый путь
// для групп трёх разных архитектур (1 dense + 2 MoE, PLAN.md §4).
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

static const std::size_t kSlot = 4096;

// Три архитектуры реестра (модели их разбивают — здесь только «готовые группы»):
//   dense-llama: attn(qkv) + kv
//   moe-glm:     attn + experts×2
//   moe-deepseek: experts×2 + kv (другой порядок ролей)
static std::vector<TensorGroup> three_architectures() {
    return {
        // dense
        make_group(0, "dense.l00.attn.qkv", Role::Attn, DType::F16, kSlot, {3, 512, 128}),
        make_group(1, "dense.l00.kv", Role::Kv, DType::F32, kSlot, {512, 64}),
        // moe-glm
        make_group(2, "glm.l07.e00.wqkv", Role::Expert, DType::NVFP4, kSlot, {128, 256}),
        make_group(3, "glm.l07.e01.wqkv", Role::Expert, DType::NVFP4, kSlot, {128, 256}),
        make_group(4, "glm.l07.attn", Role::Attn, DType::F16, kSlot, {3, 512, 128}),
        // moe-deepseek
        make_group(5, "dsk.l03.e00.wk", Role::Expert, DType::W4A16, kSlot, {128, 256}),
        make_group(6, "dsk.l03.e01.wk", Role::Expert, DType::W4A16, kSlot, {128, 256}),
        make_group(7, "dsk.l03.kv", Role::Kv, DType::E4M3, kSlot, {512, 64}),
    };
}

T_CASE(cross_arch_single_code_path) {
    const auto groups = three_architectures();
    const std::string path = "/tmp/stell_tier_test_crossarch.stelltier";
    TierLayoutHeader h;
    h.n_groups = groups.size();
    TierLayoutWriter w(path, h);
    for (const auto& g : groups) {
        std::vector<std::uint8_t> payload(kSlot, std::uint8_t(g.id.value));
        w.add_group(g, 2, payload.data(), payload.size());
    }
    w.finalize();

    TierConfig cfg;
    cfg.vram_slots = 8; // все resident одновременно — eviction не блокирует pin
    cfg.vram_slot_size = kSlot;
    cfg.ram_capacity = 8 * kSlot;
    cfg.nvme_layout = path;
    cfg.lookahead = 2;
    TierManager mgr(cfg);
    mgr.set_plan(groups); // tiers не знает, из каких архитектур эти группы

    // Перекрёстный порядок pin: dense → moe-deepseek → moe-glm → dense …
    const std::vector<std::uint32_t> order = {0, 5, 2, 1, 6, 4, 7, 3};
    const std::uint64_t io_before = mgr.stats().io_bytes;
    std::vector<std::unique_ptr<Awaiter>> waiters;
    for (std::uint32_t id : order)
        waiters.push_back(mgr.pin(GroupKey{GroupId{id},
                                           groups[id].role}));
    for (auto& a : waiters) a->wait();
    for (std::uint32_t id : order) {
        T_REQUIRE(mgr.resident_in_vram(GroupId{id}),
                  "группа загружена тем же кодовым путём независимо от архитектуры");
    }
    const TierStats st = mgr.stats();
    T_REQUIRE(st.io_bytes > io_before, "NVMe→RAM→VRAM перенос есть");
    T_REQUIRE(st.prefetch.misses == groups.size(), "первое обращение к каждой группе — промах");

    // Повторный pin всего множества — поведение одинаково для всех архитектур:
    // предиктор/политика не ветвятся по role (метрики едины).
    mgr.prefetch({});
    for (std::uint32_t id : order) {
        auto a = mgr.pin(GroupKey{GroupId{id}, groups[id].role});
        a->wait();
    }
    const TierStats st2 = mgr.stats();
    T_REQUIRE(st2.prefetch.hit_rate >= 0.0 && st2.prefetch.hit_rate <= 1.0, "hit_rate валиден");

    // Слои трёх архитектур пинятся одной операцией pin_layer (единый fan-in).
    mgr.set_layer_map({{GroupId{0}, GroupId{1}},   // dense слой
                       {GroupId{2}, GroupId{3}},   // moe-glm слой (эксперты)
                       {GroupId{5}, GroupId{7}}}); // moe-deepseek слой
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        auto a = mgr.pin_layer(layer);
        a->wait();
    }
    T_REQUIRE(mgr.resident_in_vram(GroupId{1}), "dense слой resident");
    T_REQUIRE(mgr.resident_in_vram(GroupId{3}), "moe-glm слой resident");
    T_REQUIRE(mgr.resident_in_vram(GroupId{7}), "moe-deepseek слой resident");
}

T_MAIN

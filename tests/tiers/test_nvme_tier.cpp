// Тест NVMe-тира: stell-tier файл на tmp + io_uring roundtrip (ADR-12 + ADR-08).
#include <stell/tiers/tier_manager.hpp>

#include "nvme_file.hpp"

#include <algorithm>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

static const std::size_t kSlot = 4096;

static std::vector<TensorGroup> make_plan() {
    return {
        make_group(10, "l000.attn", Role::Attn, DType::F16, kSlot),
        make_group(11, "l000.exp0", Role::Expert, DType::F16, kSlot),
        make_group(12, "l000.exp1", Role::Expert, DType::F16, kSlot),
        make_group(13, "l000.kv", Role::Kv, DType::F16, kSlot),
    };
}

static std::string write_layout(const std::string& path, const std::vector<TensorGroup>& groups) {
    TierLayoutHeader h;
    h.dtype_profile = 1;
    h.n_groups = groups.size();
    TierLayoutWriter w(path, h);
    for (const auto& g : groups) {
        std::vector<std::uint8_t> payload(g.slot_bytes, std::uint8_t(g.id.value));
        w.add_group(g, 1, payload.data(), payload.size());
    }
    w.finalize();
    return path;
}

T_CASE(nvme_io_uring_roundtrip) {
    const auto groups = make_plan();
    const std::string path = write_layout("/tmp/stell_tier_test_nvme.stelltier", groups);

    TierNvmeFile file(path);
    // Полный путь io_uring (см. отчёт: локальное ядро WSL2 6.18 поддерживает).
    T_REQUIRE(nvme_io_backend() == "io_uring",
              "ожидался io_uring-бэкенд NVMe-тира (см. отчёт про fallback)");

    const std::uint64_t ops_before = file.io_ops();
    for (const auto& g : groups) {
        std::vector<std::uint8_t> buf(kSlot);
        file.read_group(g.id.value, buf.data());
        T_REQUIRE(buf[0] == g.id.value, "payload roundtrip");
        T_REQUIRE(std::all_of(buf.begin(), buf.end(),
                              [b = buf[0]](std::uint8_t x) { return x == b; }),
                  "слот заполнен payload'ом + нулевым padding");
    }
    T_REQUIRE(file.io_ops() == ops_before + groups.size(), "io_ops counter");
    T_REQUIRE(file.io_bytes() == groups.size() * kSlot, "io_bytes counter");

    const TierGroupIndexEntry* e = file.find(11);
    T_REQUIRE(e != nullptr, "find()");
    T_REQUIRE(e->group_id == 11 && e->slot_bytes == kSlot, "entry fields");
}

T_CASE(nvme_unknown_group_is_io_failure) {
    const auto groups = make_plan();
    const std::string path = write_layout("/tmp/stell_tier_test_nvme2.stelltier", groups);
    TierNvmeFile file(path);
    std::vector<std::uint8_t> buf(kSlot);
    bool thrown = false;
    try {
        file.read_group(999, buf.data());
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
    }
    T_REQUIRE(thrown, "неизвестная группа → IoFailure");
}

T_CASE(nvme_tier_through_manager) {
    // Полный контур: план → stell-tier файл → TierManager (io_uring) → VRAM.
    const auto groups = make_plan();
    const std::string path = write_layout("/tmp/stell_tier_test_nvme3.stelltier", groups);

    TierConfig cfg;
    cfg.vram_slots = 4; // все группы resident одновременно (pin без unpin)
    cfg.vram_slot_size = kSlot;
    cfg.ram_capacity = 4 * kSlot;
    cfg.nvme_layout = path;
    TierManager mgr(cfg);
    mgr.set_plan(groups);

    // pin всех групп слоя по одной: данные должны прийти из NVMe-тира.
    std::uint64_t io_bytes_before = mgr.stats().io_bytes;
    std::vector<std::unique_ptr<Awaiter>> waiters;
    for (const auto& g : groups)
        waiters.push_back(mgr.pin(GroupKey{g.id, g.role}));
    for (auto& a : waiters) a->wait();

    const TierStats st = mgr.stats();
    T_REQUIRE(st.io_bytes > io_bytes_before, "io_bytes растёт при загрузке из NVMe");
    T_REQUIRE(st.tier_bytes[0] == 4 * kSlot, "VRAM занят 4 слотами");
    T_REQUIRE(st.tier_bytes[1] > 0 && st.tier_bytes[1] <= cfg.ram_capacity, "RAM-тир в лимите");
    T_REQUIRE(st.tier_bytes[2] == groups.size() * kSlot, "NVMe занято = сумме слотов");
    T_REQUIRE(st.prefetch.misses == groups.size(), "первый pin каждой группы — промах");
}

T_MAIN

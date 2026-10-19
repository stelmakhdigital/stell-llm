// Тест NVMe-тира: stell-tier файл на tmp + io_uring roundtrip (ADR-12 + ADR-08).
#include <stell/tiers/tier_manager.hpp>

#include "nvme_file.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <cstring>
#include <stdlib.h>
#include <vector>

#include "harness.hpp"

#include <cstdlib>

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

T_CASE(nvme_odirect_bitwise_equals_buffered) {
    // [R2a] round 4: ОДИН layout читается buffered и O_DIRECT — содержимое
    // побитово равно. Оба пути O_DIRECT: (а) staging-чтение в НЕвыровненный
    // dst (сырой контракт read_group), (б) прямое чтение в выровненный на
    // 4096 буфер (прод-путь RAM-буферов менеджера).
    const auto groups = make_plan();
    const std::string path =
        write_layout("/tmp/stell_tier_test_nvme_od.stelltier", groups);

    // Эталон: buffered-чтение (дефолт/прецедент).
    std::vector<std::vector<std::uint8_t>> ref(groups.size());
    {
        TierNvmeFile file(path);
        for (std::size_t i = 0; i < groups.size(); ++i) {
            ref[i].resize(kSlot);
            file.read_group(groups[i].id.value, ref[i].data());
        }
    }

    TierNvmeFile file(path, /*verify_payload_crc=*/false, /*odirect=*/true);
    T_REQUIRE(file.odirect_active(),
              "O_DIRECT-тракт активен (FS/ядро поддерживают O_DIRECT; на tmpfs "
              "тест честно падает — odirect_active()==false)");

    // (а) staging-путь: гарантированно невыровненный dst (сдвиг +1).
    for (std::size_t i = 0; i < groups.size(); ++i) {
        std::vector<std::uint8_t> buf(kSlot + 1, 0xEE); // шум до чтения
        file.read_group(groups[i].id.value, buf.data() + 1);
        T_REQUIRE(std::memcmp(buf.data() + 1, ref[i].data(), kSlot) == 0,
                  "O_DIRECT (staging, невыровненный dst) == buffered побитово");
        T_REQUIRE(buf[0] == 0xEE, "staging-чтение не пишет за границей слота");
    }
    // (б) прямой путь: dst выровнен на 4096 (posix_memalign — как RAM-буферы
    // менеджера): чтение сразу в dst, без staging-копии.
    for (std::size_t i = 0; i < groups.size(); ++i) {
        void* raw = nullptr;
        T_REQUIRE(posix_memalign(&raw, 4096, kSlot) == 0, "posix_memalign 4096");
        std::unique_ptr<std::uint8_t, void (*)(void*)> up(
            static_cast<std::uint8_t*>(raw), std::free);
        std::memset(up.get(), 0xEE, kSlot);
        file.read_group(groups[i].id.value, up.get());
        T_REQUIRE(std::memcmp(up.get(), ref[i].data(), kSlot) == 0,
                  "O_DIRECT (прямо в выровненный буфер) == buffered побитово");
    }
    T_REQUIRE(file.io_ops() == 2 * groups.size(), "io_ops counter (O_DIRECT-чтения)");
    T_REQUIRE(file.io_bytes() == 2 * groups.size() * kSlot, "io_bytes counter");
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

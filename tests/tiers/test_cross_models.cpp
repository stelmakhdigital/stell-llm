// Кросс-модульный тест tiers↔models (ADR-12): конвертер models
// (safetensors → stell-tier, writer по ADR-12 §2) → читатель tiers.
//
// Проверяемый срез интеграции:
//   1. header_crc32 канон — [0x00..0x2C) (уточнение round 1, ADR-12):
//      файл конвертера models читается TierLayoutReader; CRC по первым
//      0x28 байт (старый ошибочный канон) — отклоняется.
//   2. План групп из описания models (реестр, ADR-09) принимается
//      TierManager::set_plan с NVMe-тиром (сверка slot_bytes/role/dtype).
//   3. Roundtrip данных: слот группы из tiers-пути (TierLayoutReader,
//      NVMe-тир через io_uring в TierManager) == слоту, прочитанному
//      читателем models.
//   4. pin_layer через set_layer_map (расширение, ADR-08 «Уточнение») —
//      мини-симуляция вызова engine→tiers по контракту.
//
// Из-за ODR-конфликта зеркала типов models (tiers_compat.hpp) models-сторона
// живёт в отдельной TU и наружу выставлен плоский C-ABI (cross_models_abi.h).
#include <stell/tiers/layout.hpp>
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cross_models_abi.h"
#include "harness.hpp"

using namespace stell::tiers;

#ifndef STELL_FIXTURES_DIR
#define STELL_FIXTURES_DIR "tests/models/fixtures"
#endif

static std::vector<TensorGroup> plan_from_shim(int which, int n) {
    std::vector<TensorGroup> plan;
    for (int i = 0; i < n; ++i) {
        CMGroupInfo info{};
        T_REQUIRE(cm_plan_group(which, i, &info) == 0, "cm_plan_group");
        TensorGroup g;
        g.id = GroupId{info.id};
        g.name = info.name;
        g.role = role_from_code(info.role);
        g.dtype = dtype_from_code(info.dtype);
        g.slot_bytes = (std::size_t)info.slot_bytes;
        for (std::uint32_t s = 0; s < info.shape_rank && s < 4; ++s)
            g.shape.push_back(info.shape[s]);
        plan.push_back(std::move(g));
    }
    return plan;
}

// Слой → группы: имена "l000.<rest>" → слой 0, "l001." → 1; группы без
// l-префикса (embed/norm_final/head) — слой 0 (как у engine dense-модели).
static std::vector<std::vector<GroupId>> layer_map_of(const std::vector<TensorGroup>& plan) {
    std::vector<std::vector<GroupId>> layers(2);
    for (const auto& g : plan) {
        unsigned layer = 0;
        if (g.name.rfind("l", 0) == 0 && g.name.size() > 4 &&
            g.name[1] >= '0' && g.name[1] <= '9') {
            unsigned v = 0;
            for (std::size_t i = 1; i < g.name.size() && g.name[i] >= '0' && g.name[i] <= '9'; ++i)
                v = v * 10 + unsigned(g.name[i] - '0');
            layer = v;
        }
        if (layer >= layers.size()) layers.resize(layer + 1);
        layers[layer].push_back(g.id);
    }
    return layers;
}

static Role role_of(const std::vector<TensorGroup>& plan, GroupId id) {
    for (const auto& g : plan)
        if (g.id == id) return g.role;
    throw std::runtime_error("role_of: unknown GroupId");
}

static void run_tiers_side(const std::string& path, const std::vector<TensorGroup>& plan,
                           std::uint32_t expected_dtype_profile) {
    const std::size_t slot = plan.empty() ? 0 : plan[0].slot_bytes;

    // (1) tiers reader принимает файл конвертера models (канон CRC [0x00..0x2C)).
    TierLayoutReader rd(path);
    T_REQUIRE(rd.header().version == 1, "version 1");
    T_REQUIRE(rd.header().dtype_profile == expected_dtype_profile,
              "dtype_profile (F16=1 / NVFP4=2, ADR-12 §2)");
    T_REQUIRE(rd.index().size() == plan.size(), "index == non-kv план");
    // Сверка индекса с планом (порядок/роли/dtype/slot_bytes/имена ≤16).
    for (std::size_t i = 0; i < rd.index().size(); ++i) {
        const auto& e = rd.index()[i];
        const auto& g = plan[i];
        T_REQUIRE(e.group_id == g.id.value, "group_id порядок");
        T_REQUIRE(e.role == role_code(g.role), "role код");
        T_REQUIRE(e.dtype == dtype_code(g.dtype), "dtype код");
        T_REQUIRE(e.slot_bytes == g.slot_bytes, "slot_bytes согласован (ADR-12 §2.2)");
        T_REQUIRE(e.data_offset % 4096 == 0, "выравнивание слота");
        T_REQUIRE(e.name == g.name.substr(0, 16), "имя обрезано до 16 байт");
    }

    // (2) Канон CRC-границы: пересчёт по [0x00..0x2C) == хранимое; по
    //     [0x00..0x28) (старый канон) — НЕ совпадает (байты n_tensors входят).
    {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        T_REQUIRE(f != nullptr, "open header");
        std::uint8_t hdr[512];
        T_REQUIRE(std::fread(hdr, 1, sizeof hdr, f) == sizeof hdr, "header read");
        std::fclose(f);
        const std::uint32_t stored = cm_stored_header_crc(path.c_str());
        T_REQUIRE(crc32(hdr, 0x2C) == stored,
                  "crc32(hdr, 0x2C) == header_crc32 файла models (канон ADR-12)");
        T_REQUIRE(crc32(hdr, 0x28) != stored,
                  "старый канон [0x00..0x28) не совпадает с хранимым CRC");
    }

    // (3) Negative: файл с header_crc32 по старому канону отклоняется reader'ом.
    {
        const std::string old_path = path + ".oldcanon";
        std::FILE* src = std::fopen(path.c_str(), "rb");
        T_REQUIRE(src != nullptr, "open for copy");
        std::vector<std::uint8_t> blob;
        int c;
        while ((c = std::fgetc(src)) != EOF) blob.push_back(std::uint8_t(c));
        std::fclose(src);
        T_REQUIRE(blob.size() >= 0x30, "file size");
        const std::uint32_t crc_old = crc32(blob.data(), 0x28);
        for (int b = 0; b < 4; ++b)
            blob[0x2C + b] = std::uint8_t((crc_old >> (8 * b)) & 0xFF);
        std::FILE* out = std::fopen(old_path.c_str(), "wb");
        std::fwrite(blob.data(), 1, blob.size(), out);
        std::fclose(out);
        bool thrown = false;
        try {
            TierLayoutReader r(old_path);
            (void)r;
        } catch (const TierError& e) {
            thrown = e.code() == TierError::Code::IoFailure;
        }
        std::remove(old_path.c_str());
        T_REQUIRE(thrown,
                  "CRC по [0x00..0x28) → header CRC mismatch → IoFailure "
                  "(файл models НЕ принимается старым reader'ом tiers)");
    }

    // (4) Полный контур: NVMe-тир (io_uring) + set_plan + pin всех групп.
    TierConfig cfg;
    cfg.vram_slots = plan.size();
    cfg.lookahead = 0; // предсказатель выкл.: детерминизм счётчиков io/hit в тесте
    cfg.ram_capacity = std::size_t(1) << 20; // 1 MiB pinned RAM — хватает слотов
    cfg.nvme_layout = path;
    TierManager mgr(cfg);
    mgr.set_plan(plan);

    const std::uint64_t io_before = mgr.stats().io_bytes;
    std::vector<std::unique_ptr<Awaiter>> waiters;
    for (const auto& g : plan)
        waiters.push_back(mgr.pin(GroupKey{g.id, g.role}));
    for (auto& a : waiters) a->wait();
    for (const auto& g : plan)
        T_REQUIRE(mgr.resident_in_vram(g.id), "resident после pin из NVMe");
    const TierStats st = mgr.stats();
    T_REQUIRE(st.io_bytes == io_before + 2 * plan.size() * slot,
              "io_bytes: NVMe→RAM + RAM→VRAM по слоту на группу");
    T_REQUIRE(st.tier_bytes[0] == plan.size() * slot, "VRAM занят всеми группами");

    // (5) Roundtrip данных: слот из tiers == слот, прочитанный models reader'ом;
    //     padding слота нулевой (ADR-12 §2.3: CRC по слоту целиком).
    for (const auto& g : plan) {
        std::vector<std::uint8_t> a(g.slot_bytes, 0xEE);
        std::vector<std::uint8_t> b(g.slot_bytes, 0x11);
        rd.read_group(g.id.value, a.data());
        T_REQUIRE(cm_read_slot(path.c_str(), g.id.value, b.data(), b.size()) > 0,
                  "models reader read slot");
        T_REQUIRE(std::memcmp(a.data(), b.data(), g.slot_bytes) == 0,
                  "payload roundtrip tiers↔models");
    }

    // (6) Мини-симуляция engine→tiers по контракту (ADR-08 «Уточнение»):
    //     set_plan → set_layer_map → pin_layer(l) → wait → unpin; prefetch
    //     следующего слоя → hit prefetch-счётчика при pin_layer.
    const auto layers = layer_map_of(plan);
    mgr.set_layer_map(layers);
    for (unsigned l = 0; l < layers.size(); ++l) {
        if (l + 1 < layers.size()) {
            std::vector<GroupKey> keys;
            for (GroupId gid : layers[l + 1]) {
                const TensorGroup* g = nullptr;
                for (const auto& pg : plan)
                    if (pg.id == gid) { g = &pg; break; }
                T_REQUIRE(g != nullptr, "layer map group in plan");
                keys.push_back(GroupKey{gid, g->role});
            }
            mgr.prefetch(keys); // router-lookahead: engine prefetch заранее
            // Дождаться resident-ности предсказанных групп — детерминизм hit.
            for (const auto& k : keys) {
                using clock = std::chrono::steady_clock;
                auto t0 = clock::now();
                while (!mgr.resident_in_vram(k.id)) {
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now() - t0).count() > 10000)
                        throw std::runtime_error("prefetch timeout");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        auto a = mgr.pin_layer(l);
        a->wait();
        for (GroupId gid : layers[l]) mgr.unpin(GroupKey{gid, role_of(plan, gid)});
    }
    const TierStats st2 = mgr.stats();
    T_REQUIRE(st2.prefetch.hits >= layers.size() - 1,
              "prefetch следующего слоя → hit при pin_layer");
    T_REQUIRE(st2.layer_pause.p95_ms >= 0.0, "паузы слоёв измеряются (ADR-08 §2)");

    bool thrown = false;
    try {
        mgr.pin_layer(unsigned(layers.size())); // слой без карты → PlanMismatch
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "pin_layer вне карты → PlanMismatch (ADR-08 «Уточнение»)");
}

static GroupId plan_role_unused_guard(const TensorGroup& g) { return g.id; }

T_CASE(cross_models_converter_to_tiers_reader) {
    char dense[1024] = {}, moe[1024] = {};
    const int rc = cm_build(STELL_FIXTURES_DIR, "/tmp/stell_cross_models",
                            dense, sizeof dense, moe, sizeof moe);
    if (rc != 0)
        throw std::runtime_error(std::string("cm_build failed: ") + cm_last_error());
    const int n_dense = cm_plan_size(0);
    const int n_moe = cm_plan_size(1);
    T_REQUIRE(n_dense > 0 && n_moe > 0, "планы построены");

    run_tiers_side(dense, plan_from_shim(0, n_dense), /*profile=*/1);
    run_tiers_side(moe, plan_from_shim(1, n_moe), /*profile=*/2);
}

T_MAIN

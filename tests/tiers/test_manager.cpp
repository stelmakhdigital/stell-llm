// Тесты TierManager по ADR-08: pin/unpin/pin_layer, prefetch-идемпотентность,
// коды ошибок (PlanMismatch/SlotExhausted/IoFailure/NotPinned), ResetPlan.
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stdlib.h>

#include "harness.hpp"

using namespace stell::tiers;

static const std::size_t kSlot = 4096;

static std::vector<TensorGroup> plan3() {
    return {
        make_group(0, "g0", Role::Attn, DType::F16, kSlot),
        make_group(1, "g1", Role::Expert, DType::F16, kSlot),
        make_group(2, "g2", Role::Kv, DType::F16, kSlot),
    };
}

static std::string write_layout(const std::string& path, const std::vector<TensorGroup>& groups) {
    TierLayoutHeader h;
    h.n_groups = groups.size();
    TierLayoutWriter w(path, h);
    for (const auto& g : groups) {
        std::vector<std::uint8_t> payload(kSlot, std::uint8_t(g.id.value));
        w.add_group(g, 1, payload.data(), payload.size());
    }
    w.finalize();
    return path;
}

static TierConfig ram_cfg(const std::string& path, std::size_t slots) {
    TierConfig cfg;
    cfg.vram_slots = slots;
    cfg.vram_slot_size = kSlot;
    cfg.ram_capacity = 8 * kSlot;
    cfg.nvme_layout = path;
    cfg.lookahead = 0; // без предсказаний в этих тестах
    return cfg;
}

T_CASE(manager_pin_unpin_roundtrip) {
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr1.stelltier", groups);
    TierManager mgr(ram_cfg(path, 3));
    mgr.set_plan(groups);

    auto a = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{0}), "group resident после pin");

    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
    // Повторный unpin → NotPinned (ADR-08 §3).
    bool thrown = false;
    try {
        mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::NotPinned;
    }
    T_REQUIRE(thrown, "повторный unpin → NotPinned");

    // Роль в GroupKey не соответствует плану → PlanMismatch.
    thrown = false;
    try {
        mgr.pin(GroupKey{GroupId{0}, Role::Kv});
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "role mismatch → PlanMismatch");

    // Неизвестная группа → PlanMismatch.
    thrown = false;
    try {
        mgr.pin(GroupKey{GroupId{99}, Role::Kv});
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "unknown group → PlanMismatch");

    // prefetch неизвестной группы → PlanMismatch.
    thrown = false;
    try {
        mgr.prefetch({GroupKey{GroupId{99}, Role::Kv}});
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "prefetch unknown → PlanMismatch");
}

T_CASE(manager_set_plan_cold_only) {
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr2.stelltier", groups);
    TierManager mgr(ram_cfg(path, 3));
    mgr.set_plan(groups);

    // pin до set_plan → PlanMismatch (ADR-08 §4: «холодная» операция).
    TierManager mgr2(ram_cfg(path, 3));
    bool thrown = false;
    try {
        mgr2.pin(GroupKey{GroupId{0}, Role::Attn});
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "pin до set_plan → PlanMismatch");

    // Повторный set_plan = ResetPlan: resident-ность сбрасывается.
    mgr.set_plan(groups);
    T_REQUIRE(!mgr.resident_in_vram(GroupId{0}), "ResetPlan сбрасывает resident");

    // set_plan при закреплённой группе → PlanMismatch (ADR-08 §4).
    {
        auto a = mgr.pin(GroupKey{GroupId{1}, Role::Expert});
        a->wait();
        thrown = false;
        try {
            mgr.set_plan(groups);
        } catch (const TierError& e) {
            thrown = e.code() == TierError::Code::PlanMismatch;
        }
        T_REQUIRE(thrown, "set_plan при pinned-группе → PlanMismatch");
        mgr.unpin(GroupKey{GroupId{1}, Role::Expert});
    }
}

T_CASE(manager_slot_exhausted_when_all_pinned) {
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr3.stelltier", groups);
    TierManager mgr(ram_cfg(path, 1)); // один VRAM-слот
    mgr.set_plan(groups);

    auto a0 = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a0->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{0}), "g0 resident");

    // g1: единственный слот занят закреплённой g0 → SlotExhausted в wait().
    auto a1 = mgr.pin(GroupKey{GroupId{1}, Role::Expert});
    bool thrown = false;
    try {
        a1->wait();
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::SlotExhausted;
    }
    T_REQUIRE(thrown, "нет свободных слотов, всё pinned → SlotExhausted");

    // После unpin(g0) — g1 загружается (eviction разрешён).
    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
    auto a1b = mgr.pin(GroupKey{GroupId{1}, Role::Expert});
    a1b->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{1}), "g1 resident после eviction");
}

T_CASE(manager_prefetch_idempotent) {
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr4.stelltier", groups);
    TierManager mgr(ram_cfg(path, 3));
    mgr.set_plan(groups);

    const GroupKey k{GroupId{1}, Role::Expert};
    mgr.prefetch({k});
    mgr.prefetch({k});
    mgr.prefetch({k}); // идемпотентно: не удваивает работу (ADR-08 §2/§4)
    wait_resident(mgr, GroupId{1});
    mgr.prefetch({k}); // уже resident — просто метка hit для следующего pin

    auto a = mgr.pin(k);
    a->wait();
    const TierStats st = mgr.stats();
    T_REQUIRE(st.prefetch.hits >= 1, "prefetch → pin считается hit");
    T_REQUIRE(mgr.resident_in_vram(GroupId{1}), "resident");
}

T_CASE(manager_pin_layer_composite) {
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr5.stelltier", groups);
    TierManager mgr(ram_cfg(path, 3));
    mgr.set_plan(groups);
    mgr.set_layer_map({{GroupId{0}, GroupId{1}}}); // слой 0: две группы

    // Без карты слоя → PlanMismatch (слои знает engine, см. ADR-08 §5).
    TierManager mgr2(ram_cfg(path, 3));
    mgr2.set_plan(groups);
    bool thrown = false;
    try {
        mgr2.pin_layer(0);
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "pin_layer без layer map → PlanMismatch");

    auto layer = mgr.pin_layer(0);
    layer->wait(); // fan-in: обе группы resident
    T_REQUIRE(mgr.resident_in_vram(GroupId{0}), "l0.g0 resident");
    T_REQUIRE(mgr.resident_in_vram(GroupId{1}), "l0.g1 resident");

    const TierStats st = mgr.stats();
    T_REQUIRE(st.layer_pause.p50_ms >= 0.0 && st.layer_pause.p95_ms >= 0.0 &&
              st.layer_pause.p99_ms >= 0.0,
              "паузы слоя p50/p95/p99 записаны (>= 0)");
}

T_CASE(manager_ram_tier_cache_and_eviction) {
    // RAM-тир: после eviction данные остаются в RAM, повторный pin без NVMe-IO.
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr6.stelltier", groups);
    TierManager mgr(ram_cfg(path, 1));
    mgr.set_plan(groups);

    auto a0 = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a0->wait();
    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
    const std::uint64_t io_ops_after_cold = mgr.stats().io_ops;

    auto a1 = mgr.pin(GroupKey{GroupId{1}, Role::Expert});
    a1->wait(); // вытесняет g0 (LRU), g0 остаётся в RAM-тире
    mgr.unpin(GroupKey{GroupId{1}, Role::Expert});

    auto a0b = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a0b->wait();
    T_REQUIRE(mgr.resident_in_vram(GroupId{0}), "g0 снова resident");
    const std::uint64_t io_ops_after_repin = mgr.stats().io_ops;
    // Между снапшотами: загрузка g1 из NVMe (read+copy = 2 ops) и повторная
    // загрузка g0 из RAM-тира (copy = 1 op). Если бы g0 снова читался из NVMe,
    // дельта была бы 4 — RAM-тир кеша не работает.
    T_REQUIRE(io_ops_after_repin - io_ops_after_cold == 3,
              "RAM-тир кеша повторную загрузку без NVMe-чтения");

    // Лимит RAM-тира (tier_bytes[1] ≤ capacity).
    const TierStats st = mgr.stats();
    T_REQUIRE(st.tier_bytes[1] <= 8 * kSlot, "RAM-тир в пределах capacity");
}

T_CASE(manager_read_ram_slot_roundtrip) {
    // Расширение R8: чтение содержимого слота из RAM-тира после pin→wait —
    // байты == payload, записанному writer'ом layout (канонический путь
    // «layout → tiers → данные для логитов»).
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr8.stelltier", groups);
    TierManager mgr(ram_cfg(path, 3));
    mgr.set_plan(groups);

    // Без загрузки RAM-копии нет → IoFailure (pin→wait обязателен до чтения).
    bool thrown = false;
    std::vector<std::uint8_t> buf(kSlot);
    try {
        mgr.read_ram_slot(GroupId{0}, buf.data(), buf.size());
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
    }
    T_REQUIRE(thrown, "read_ram_slot без pin→wait → IoFailure");

    auto a = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a->wait();
    mgr.read_ram_slot(GroupId{0}, buf.data(), buf.size());
    const std::vector<std::uint8_t> expected(kSlot, std::uint8_t(0));
    T_REQUIRE(std::memcmp(buf.data(), expected.data(), kSlot) == 0,
              "слот RAM-тира == payload writer'а (gid=0)");

    // Неизвестная группа → PlanMismatch; dst меньше слота → PlanMismatch.
    thrown = false;
    try {
        mgr.read_ram_slot(GroupId{99}, buf.data(), buf.size());
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "read_ram_slot unknown group → PlanMismatch");
    thrown = false;
    try {
        mgr.read_ram_slot(GroupId{0}, buf.data(), kSlot - 1);
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "read_ram_slot dst_bytes < slot_bytes → PlanMismatch");
    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
}

T_CASE(manager_read_ram_slot_after_ram_eviction) {
    // Семантика честно фиксируется: RAM-копия группы может быть вытеснена
    // (evict_ram_if_needed), данные восстанавливаются перечитыванием из NVMe
    // при следующем холодном pin (группа не в VRAM и без RAM-копии).
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr9.stelltier", groups);
    TierConfig cfg = ram_cfg(path, 1); // один VRAM-слот
    cfg.ram_capacity = 2 * kSlot;      // RAM-тир меньше плана → вытеснение RAM-копий
    TierManager mgr(cfg);
    mgr.set_plan(groups);

    auto a0 = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a0->wait();
    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});

    auto a1 = mgr.pin(GroupKey{GroupId{1}, Role::Expert});
    a1->wait(); // вытесняет g0 из VRAM; RAM-копия g0 остаётся
    mgr.unpin(GroupKey{GroupId{1}, Role::Expert});

    auto a2 = mgr.pin(GroupKey{GroupId{2}, Role::Kv});
    a2->wait(); // RAM-тир переполнен → RAM-копия g0 (не в VRAM, unpinned) вытесняется
    mgr.unpin(GroupKey{GroupId{2}, Role::Kv});

    bool thrown = false;
    std::vector<std::uint8_t> buf(kSlot);
    try {
        mgr.read_ram_slot(GroupId{0}, buf.data(), buf.size());
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
    }
    T_REQUIRE(thrown, "RAM-копия вытеснена → IoFailure (семантика расширения R8)");
    T_REQUIRE(!mgr.resident_in_vram(GroupId{0}), "g0 не в VRAM — данные восстановимы");

    // Повторный холодный pin g0: не в VRAM и без RAM-копии → NVMe→RAM→VRAM,
    // чтение RAM-копии снова работает.
    auto a0b = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a0b->wait();
    mgr.read_ram_slot(GroupId{0}, buf.data(), buf.size());
    const std::vector<std::uint8_t> expected(kSlot, std::uint8_t(0));
    T_REQUIRE(std::memcmp(buf.data(), expected.data(), kSlot) == 0,
              "после повторной загрузки чтение RAM-копии == payload writer'а");
    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
}

T_CASE(manager_ram_capacity_eviction) {
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr7.stelltier", groups);
    TierConfig cfg = ram_cfg(path, 3);
    cfg.ram_capacity = 2 * kSlot; // меньше, чем нужно всем трём
    TierManager mgr(cfg);
    mgr.set_plan(groups);
    for (const auto& g : groups) {
        auto a = mgr.pin(GroupKey{g.id, g.role});
        a->wait();
        mgr.unpin(GroupKey{g.id, g.role});
    }
    const TierStats st = mgr.stats();
    T_REQUIRE(st.tier_bytes[1] <= cfg.ram_capacity, "RAM-тир не превышает capacity");
}

// --- R1/R7 (раунд 3): CRC-политика и конфигурируемые границы тиров ----------

static std::string write_layout_payloads(const std::string& path,
                                         const std::vector<TensorGroup>& groups,
                                         std::size_t slot) {
    TierLayoutHeader h;
    h.n_groups = groups.size();
    TierLayoutWriter w(path, h);
    std::vector<std::uint8_t> payload(slot);
    for (const auto& g : groups) {
        for (std::size_t k = 0; k < slot; ++k)
            payload[k] = std::uint8_t((g.id.value * 131 + k * 7 + 5) & 0xFF);
        w.add_group(g, 1, payload.data(), payload.size());
    }
    w.finalize();
    return path;
}

static void corrupt_byte(const std::string& path, std::uint64_t offset) {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    T_REQUIRE(f.is_open(), "corrupt_byte: файл открыт");
    f.seekp(static_cast<std::streamoff>(offset));
    f.write("\x5a", 1);
    f.close();
}

T_CASE(manager_verify_crc_off_hot_path) {
    // R1-а (ADR-12 §2.3, round 3): pin при verify=off НЕ пересчитывает
    // payload-CRC — фиксация по метрике crc_checks() (0 = hot path без CRC;
    // header_crc32 reader проверяет всегда — отдельно в test_layout).
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr_r1a.stelltier", groups);
    TierConfig cfg = ram_cfg(path, 3);
    T_REQUIRE(cfg.verify_crc == false,
              "дефолт конфига — payload-CRC off (hot path без CRC)");
    TierManager mgr(cfg);
    mgr.set_plan(groups);
    for (const auto& g : groups) {
        auto a = mgr.pin(GroupKey{g.id, g.role});
        a->wait(); // холодная загрузка NVMe→RAM→VRAM
        mgr.unpin(GroupKey{g.id, g.role});
    }
    T_REQUIRE(mgr.stats().io_ops > 0, "загрузки были (контроль негатива)");
    T_REQUIRE(mgr.crc_checks() == 0,
              "verify=off: payload-CRC не пересчитывался ни разу за все pin");
}

T_CASE(manager_verify_crc_on_catches_corruption) {
    // R1-б: verify=on ловит испорченный байт payload при первом pin —
    // TierError(IoFailure, ровно "layout: payload CRC mismatch").
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr_r1b.stelltier", groups);
    {
        TierLayoutReader probe(path);
        const auto& e1 = probe.index()[1];
        corrupt_byte(path, e1.data_offset + e1.slot_bytes / 2); // середина слота
    }
    TierConfig cfg = ram_cfg(path, 3);
    cfg.verify_crc = true; // явное включение (эквивалент STELL_TIER_VERIFY_CRC=1)
    TierManager mgr(cfg);
    mgr.set_plan(groups); // индекс цел: сверка slot_bytes/dtype/role проходит
    bool thrown = false;
    std::string msg;
    try {
        auto a = mgr.pin(GroupKey{GroupId{1}, Role::Expert});
        a->wait();
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
        msg = e.what();
    }
    T_REQUIRE(thrown, "испорченный байт payload → IoFailure в wait()");
    T_REQUIRE(msg == "layout: payload CRC mismatch",
              "сообщение ровно \"layout: payload CRC mismatch\"");
}

T_CASE(manager_verify_slots_pass_marks_groups) {
    // R1: публичный verify-проход — внепутевая сверка всех слотов layout;
    // успех помечает группы: последующий горячий путь (verify=on) не
    // пересчитывает CRC повторно (CRC-once за жизнь файла в кеше tiers).
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr_r1v.stelltier", groups);
    TierConfig cfg = ram_cfg(path, 3);
    cfg.verify_crc = true;
    TierManager mgr(cfg);
    mgr.set_plan(groups);

    T_REQUIRE(mgr.verify_slots() == groups.size(),
              "verify-проход покрыл все слоты layout");
    T_REQUIRE(mgr.crc_checks() == groups.size(),
              "метрика: по одному пересчёту payload-CRC на слот");

    for (const auto& g : groups) {
        auto a = mgr.pin(GroupKey{g.id, g.role});
        a->wait();
        mgr.unpin(GroupKey{g.id, g.role});
    }
    T_REQUIRE(mgr.crc_checks() == groups.size(),
              "после verify-прохода hot path (verify=on) не перепроверяет");
    // NVMe-тира нет → verify-проход невозможен (PlanMismatch).
    TierConfig no_nvme = ram_cfg("", 2);
    no_nvme.nvme_layout.clear();
    TierManager mgr2(no_nvme);
    mgr2.set_plan({make_group(0, "g0", Role::Attn, DType::F16, kSlot)});
    bool thrown = false;
    try {
        mgr2.verify_slots();
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "verify_slots без NVMe-тира → PlanMismatch");
}

T_CASE(manager_verify_slots_catches_corruption) {
    // R1: verify-проход ловит испорченный байт payload вне пути декода.
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr_r1v2.stelltier", groups);
    {
        TierLayoutReader probe(path);
        const auto& e2 = probe.index()[2];
        corrupt_byte(path, e2.data_offset + e2.slot_bytes / 2); // середина слота группы
    }
    TierConfig cfg = ram_cfg(path, 3);
    TierManager mgr(cfg); // verify=off — порчу ловит именно verify-проход
    mgr.set_plan(groups);
    bool thrown = false;
    std::string msg;
    try {
        mgr.verify_slots();
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
        msg = e.what();
    }
    T_REQUIRE(thrown, "verify-проход → IoFailure при порче payload");
    T_REQUIRE(msg == "layout: payload CRC mismatch",
              "сообщение ровно \"layout: payload CRC mismatch\"");
}

T_CASE(manager_env_config_override) {
    // R7: параметризация границ тиров через окружение (поверх базового;
    // нечисловое значение → базовое, консервативно).
    ::setenv("STELL_TIER_VRAM_SLOTS", "7", 1);
    ::setenv("STELL_TIER_VRAM_SLOT_BYTES", "8192", 1);
    ::setenv("STELL_TIER_RAM_MB", "3", 1);
    ::setenv("STELL_TIER_LOOKAHEAD", "0", 1);
    ::setenv("STELL_TIER_VERIFY_CRC", "1", 1);
    ::setenv("STELL_TIER_POLICY", "lcu", 1);
    const TierConfig cfg = tier_config_from_env();
    ::unsetenv("STELL_TIER_VRAM_SLOTS");
    ::unsetenv("STELL_TIER_VRAM_SLOT_BYTES");
    ::unsetenv("STELL_TIER_RAM_MB");
    ::unsetenv("STELL_TIER_LOOKAHEAD");
    ::unsetenv("STELL_TIER_VERIFY_CRC");
    ::unsetenv("STELL_TIER_POLICY");
    T_REQUIRE(cfg.vram_slots == 7, "env: vram_slots");
    T_REQUIRE(cfg.vram_slot_size == 8192, "env: vram_slot_size");
    T_REQUIRE(cfg.ram_capacity == 3u << 20, "env: ram_capacity (МиБ → байты)");
    T_REQUIRE(cfg.lookahead == 0, "env: lookahead");
    T_REQUIRE(cfg.verify_crc == true, "env: STELL_TIER_VERIFY_CRC=1");
    T_REQUIRE(cfg.policy == EvictionPolicy::Lcu, "env: политика lcu");

    ::setenv("STELL_TIER_VRAM_SLOTS", "не-число", 1);
    const TierConfig cfg2 = tier_config_from_env(TierConfig{});
    ::unsetenv("STELL_TIER_VRAM_SLOTS");
    T_REQUIRE(cfg2.vram_slots == 4, "нечисловое env → базовое значение");
}

T_CASE(manager_vram_slot_accounting_under_churn) {
    // Регрессия bugfix раунда 3: acquire_slot при eviction забирает слот
    // жертвы себе (pop_back). Раньше возвращал free_slots_.back() без pop →
    // слот оставался в списке свободных → следующий acquire_slot выдавал ТОТ
    // ЖЕ слот другой группе → aliasing (resident > vram_slots, порча VRAM-
    // копий при churn — сценарий MoE > VRAM, срез 3).
    const auto groups = plan3();
    const auto path =
        write_layout("/tmp/stell_tier_test_mgr_slotacc.stelltier", groups);
    TierManager mgr(ram_cfg(path, 1)); // один VRAM-слот — максимальный churn
    mgr.set_plan(groups);
    for (int round = 0; round < 4; ++round)
        for (const auto& g : groups) {
            auto a = mgr.pin(GroupKey{g.id, g.role});
            a->wait();
            mgr.unpin(GroupKey{g.id, g.role});
        }
    int resident = 0;
    for (const auto& g : groups)
        resident += mgr.resident_in_vram(g.id) ? 1 : 0;
    T_REQUIRE(resident <= 1, "resident-группы ≤ vram_slots (нет aliasing слотов)");
    // Слот учётен корректно: VRAM-байты == resident × slot.
    const TierStats st = mgr.stats();
    T_REQUIRE(st.tier_bytes[0] == std::uint64_t(resident) * kSlot,
              "tier_bytes[VRAM] == resident × slot_bytes");
}

T_CASE(manager_co_tenancy_profile_full_cycle) {
    // R7: профиль «свободно 2–11 ГБ на V100 при co-tenancy» — RAM-тир
    // основной, VRAM-тир мал. Фикстура в масштабе 1:2048 (64 КиБ-слоты):
    // 16 групп (8 слоёв × 2 группы), план 1 МиБ; VRAM-бюджет 384 КиБ → 6
    // слотов (3 слоя — детерминизм victim-выбора при churn, см. run_pass).
    // Полный цикл pin/read + замер p95-пауз и hit-rate prefetch.
    const std::size_t slot = 64 * 1024; // кратен 4096 (ADR-12 §2.4)
    const std::size_t n_groups = 16, layers = 8;
    const std::size_t plan_bytes = n_groups * slot;

    std::vector<TensorGroup> plan;
    for (std::size_t i = 0; i < n_groups; ++i)
        plan.push_back(make_group(std::uint32_t(i),
            "l" + std::to_string(i / 2) + ".g" + std::to_string(i % 2),
            (i % 2 == 0) ? Role::Attn : Role::Expert, DType::F16, slot,
            {std::int64_t(slot / 2), 2}));
    const auto path =
        write_layout_payloads("/tmp/stell_tier_test_mgr_r7.stelltier", plan, slot);

    const TierConfig cfg = tier_config_co_tenancy(6 * 64 * 1024, plan_bytes, slot);
    // NVMe-тир задаёт вызывающий (профиль фиксирует только границы тиров):
    TierConfig mut = cfg;
    mut.nvme_layout = path;
    T_REQUIRE(cfg.vram_slot_size == slot, "профиль: размер слота — из плана");
    T_REQUIRE(cfg.vram_slots == 6, "профиль: VRAM-тир мал (384 КиБ / 64 КиБ = 6 слотов)");
    T_REQUIRE(cfg.ram_capacity == plan_bytes, "профиль: RAM-тир основной (план целиком)");

    // Кратность 4096 нарушена → PlanMismatch.
    bool thrown = false;
    try {
        tier_config_co_tenancy(256 * 1024, plan_bytes, 512);
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "профиль: slot_bytes не кратен 4096 → PlanMismatch");

    TierManager mgr(mut);
    mgr.set_plan(plan);
    std::vector<std::vector<GroupId>> lm(layers);
    for (std::size_t i = 0; i < n_groups; ++i)
        lm[i / 2].push_back(GroupId{std::uint32_t(i)});
    mgr.set_layer_map(lm);

    auto keys_of = [&](std::size_t l) {
        std::vector<GroupKey> ks;
        for (GroupId gid : lm[l]) ks.push_back(GroupKey{gid, plan[gid.value].role});
        return ks;
    };
    auto run_pass = [&](bool with_prefetch) {
        std::vector<double> pauses;
        if (with_prefetch) {
            // Priming: слой 0 префетчен до начала прохода (engine стартует с
            // уже подгруженным первым слоем), иначе первый pin — честный промах.
            mgr.prefetch(keys_of(0));
            for (GroupId gid : lm[0]) wait_resident(mgr, gid);
        }
        for (std::size_t l = 0; l < layers; ++l) {
            // Порядок engine-цикла: pin_layer(l) (текущий слой закреплён →
            // неприкасаем для eviction), затем router-lookahead prefetch(l+1)
            // — асинхронно; ожидание завершения prefetch ПОКА l закреплён
            // (в реальном engine на это уходит compute слоя; здесь — детерминизм
            // фикстуры: victim'ы нагрузки l+1 — непиновенные группы l−1/старые,
            // не слой l). После этого pin(l+1) гарантированно hit.
            const auto t0 = std::chrono::steady_clock::now();
            auto a = mgr.pin_layer(std::uint32_t(l));
            if (with_prefetch && l + 1 < layers) mgr.prefetch(keys_of(l + 1));
            a->wait();
            if (with_prefetch && l + 1 < layers)
                for (GroupId gid : lm[l + 1]) wait_resident(mgr, gid);
            pauses.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
            for (GroupId gid : lm[l]) mgr.unpin(GroupKey{gid, plan[gid.value].role});
        }
        return pauses;
    };
    auto p95 = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, std::size_t(0.95 * double(v.size() - 1)))];
    };

    const std::uint64_t hits0 = mgr.stats().prefetch.hits;
    const auto cold = run_pass(false); // холодный: NVMe→RAM→VRAM, без lookahead
    const std::uint64_t misses_cold = mgr.stats().prefetch.misses;
    const auto warm = run_pass(true);  // с router-lookahead prefetch
    const TierStats st = mgr.stats();
    const std::uint64_t hits = st.prefetch.hits - hits0;
    const std::uint64_t misses = st.prefetch.misses - misses_cold;

    T_REQUIRE(misses == 0,
              "warm-проход на профиле (после cold): промахов prefetch нет");
    T_REQUIRE(hits == n_groups, "hit-rate prefetch на профиле = 1.0 (16/16 в warm-проходе)");
    T_REQUIRE(p95(warm) < 50.0,
              "p95-пауза warm-прохода ≤ 50 мс (64 КиБ-слоты, CPU-контур)");
    // (сравнение cold/warm пауз — в печати ниже, без ассершна: scheduling-шум
    // не должен флакать тест; содержательно cold ≥ warm на любом носителе.)

    // Полный цикл pin/read: содержимое слота через NVMe→RAM→VRAM == payload.
    auto a = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
    a->wait();
    std::vector<std::uint8_t> buf(slot), expected(slot);
    mgr.read_ram_slot(GroupId{0}, buf.data(), buf.size());
    for (std::size_t k = 0; k < slot; ++k)
        expected[k] = std::uint8_t((0 * 131 + k * 7 + 5) & 0xFF);
    T_REQUIRE(std::memcmp(buf.data(), expected.data(), slot) == 0,
              "слот через полный цикл тиров == payload writer'а layout");
    mgr.unpin(GroupKey{GroupId{0}, Role::Attn});

    // Честная печать замеров профиля (фикстура; серверный NVMe-прогон —
    // bench_p95_nvme.sh, worker после merge).
    std::printf("  [r7-профиль co-tenancy] cold p95=%.3f ms, warm p95=%.3f ms, \
prefetch hit-rate=%.2f, vram_slots=%zu, ram_capacity=%zu B\n",
                p95(cold), p95(warm), double(hits) / double(hits + misses),
                cfg.vram_slots, cfg.ram_capacity);
}

T_MAIN

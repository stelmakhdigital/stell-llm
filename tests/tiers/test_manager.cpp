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

// --- [R2a]/[R2b]/[R5] round 4: O_DIRECT, deep-prefetch, сверка VRAM-слотов --

T_CASE(manager_odirect_full_cycle) {
    // [R2a] round 4: полный цикл тиров при odirect=true — O_DIRECT-чтение
    // идёт ПРЯМО в выровненные RAM-буферы менеджера (posix_memalign 4096);
    // содержимое после pin→wait побитово == payload writer'а layout.
    // Дефолт buffered не тронут (fallback-прецедент; фиксация ниже).
    const auto groups = plan3();
    const auto path = write_layout("/tmp/stell_tier_test_mgr_od.stelltier", groups);
    TierConfig cfg = ram_cfg(path, 3);
    cfg.odirect = true;
    TierManager mgr(cfg);
    mgr.set_plan(groups);
    for (const auto& g : groups) {
        auto a = mgr.pin(GroupKey{g.id, g.role});
        a->wait();
        mgr.unpin(GroupKey{g.id, g.role});
    }
    std::vector<std::uint8_t> got(kSlot), expected(kSlot);
    for (const auto& g : groups) {
        mgr.read_ram_slot(g.id, got.data(), got.size());
        for (std::size_t k = 0; k < kSlot; ++k)
            expected[k] = std::uint8_t(g.id.value);
        T_REQUIRE(std::memcmp(got.data(), expected.data(), kSlot) == 0,
                  "O_DIRECT: RAM-слот == payload writer'а (побитово)");
    }
    // Env OR-семантика (та же, что у verify_crc): cfg=false + env "1" → on;
    // без env и в дефолте — buffered.
    ::setenv("STELL_TIER_ODIRECT", "1", 1);
    const TierConfig cfg_env = tier_config_from_env(cfg);
    ::unsetenv("STELL_TIER_ODIRECT");
    T_REQUIRE(cfg_env.odirect == true, "env STELL_TIER_ODIRECT=1 → odirect on");
    T_REQUIRE(TierConfig{}.odirect == false,
              "дефолт конфига — buffered (fallback-прецедент сохранён)");
    T_REQUIRE(tier_config_from_env(TierConfig{}).odirect == false,
              "без env — buffered");
}

T_CASE(manager_deep_prefetch_k_in_flight) {
    // [R2b] round 4: k групп в полёте (батч-глубина очереди загрузок).
    // Детерминизм через диагностический крюк io_delay_ms (модель латентности
    // ОДНОЙ группы; прод: 0): k=4 — группы материализуются в фоне
    // ПАРАЛЛЕЛЬНО (существенно быстрее суммы латентностей), k=1 (дефолт/
    // прецедент) — последовательно (≈ сумма). Декод префетченных групп —
    // без пауз загрузки (prefetch-hit). Честно: порог параллельности
    // 3.5×delay допускает разбиение очереди на ≤3 батча гонкой worker'а —
    // любая конфигурация строго быстрее полностью последовательной (4×).
    const std::size_t slot = 256 * 1024; // кратен 4096 (ADR-12 §2.4)
    const std::size_t n_groups = 16, layers = 8;
    const std::size_t delay_ms = 60;

    std::vector<TensorGroup> plan;
    for (std::size_t i = 0; i < n_groups; ++i)
        plan.push_back(make_group(
            std::uint32_t(i),
            "l" + std::to_string(i / 2) + ".g" + std::to_string(i % 2),
            (i % 2 == 0) ? Role::Attn : Role::Expert, DType::F16, slot,
            {std::int64_t(slot / 2), 2}));
    const auto path =
        write_layout_payloads("/tmp/stell_tier_test_mgr_r2b.stelltier", plan, slot);

    auto make_cfg = [&](std::size_t k) {
        TierConfig c;
        c.vram_slots = 8;          // 8 групп (4 слоя) resident — без eviction
        c.vram_slot_size = slot;
        c.ram_capacity = n_groups * slot;
        c.nvme_layout = path;
        c.lookahead = 0;           // prefetch — явно (как от engine)
        c.deep_prefetch_depth = k;
        c.io_delay_ms = delay_ms;  // диагностический крюк (тесты; прод: 0)
        return c;
    };
    std::vector<std::vector<GroupId>> lm(layers);
    for (std::size_t i = 0; i < n_groups; ++i)
        lm[i / 2].push_back(GroupId{std::uint32_t(i)});
    auto keys_of = [&](std::size_t l) {
        std::vector<GroupKey> ks;
        for (GroupId gid : lm[l])
            ks.push_back(GroupKey{gid, plan[gid.value].role});
        return ks;
    };

    // --- k=4: 4 группы (слои 1–2) материализуются в фоне ---
    TierManager mgr4(make_cfg(4));
    mgr4.set_plan(plan);
    mgr4.set_layer_map(lm);
    mgr4.prefetch(keys_of(0)); // priming: слой 0 к «началу декода»
    for (GroupId gid : lm[0]) wait_resident(mgr4, gid);

    const auto t4 = std::chrono::steady_clock::now();
    mgr4.prefetch(keys_of(1));
    mgr4.prefetch(keys_of(2)); // 4 job'а → батч-глубина 4 (k в полёте)
    for (std::size_t l : {std::size_t(1), std::size_t(2)})
        for (GroupId gid : lm[l]) wait_resident(mgr4, gid);
    const double ms4 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t4).count();
    T_REQUIRE(ms4 < 3.5 * double(delay_ms),
              "k=4: 4 группы материализованы в фоне параллельно (строго быстрее 4× латентности)");

    // --- k=1 (дефолт/прецедент): те же 4 группы — последовательно ---
    TierManager mgr1(make_cfg(1));
    mgr1.set_plan(plan);
    mgr1.set_layer_map(lm);
    mgr1.prefetch(keys_of(0));
    for (GroupId gid : lm[0]) wait_resident(mgr1, gid);

    const auto t1 = std::chrono::steady_clock::now();
    mgr1.prefetch(keys_of(1));
    mgr1.prefetch(keys_of(2));
    for (std::size_t l : {std::size_t(1), std::size_t(2)})
        for (GroupId gid : lm[l]) wait_resident(mgr1, gid);
    const double ms1 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count();
    T_REQUIRE(ms1 > 3.5 * double(delay_ms),
              "k=1: те же 4 группы — последовательно (≈ 4× латентность)");
    T_REQUIRE(ms1 > ms4, "k=4 быстрее k=1 на том же layout/задержке");

    // --- декод без пауз на первых k группах (после материализации) ---
    const std::uint64_t hits4_before = mgr4.stats().prefetch.hits;
    for (std::size_t l : {std::size_t(1), std::size_t(2)}) { // «декод» слоёв
        const auto tw = std::chrono::steady_clock::now();
        auto a = mgr4.pin_layer(std::uint32_t(l));
        a->wait();
        const double pause = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tw).count();
        T_REQUIRE(pause < double(delay_ms),
                  "декод префетченной группы без паузы загрузки (< латентности)");
        for (GroupId gid : lm[l])
            mgr4.unpin(GroupKey{gid, plan[gid.value].role});
    }
    const std::uint64_t hits4 = mgr4.stats().prefetch.hits - hits4_before;
    T_REQUIRE(hits4 == 4,
              "pin 4 префетченных групп = 4 prefetch-hit (материализованы в фоне)");

    std::printf(
        "  [r2b deep-prefetch] k=4: %.1f ms (4 группы в полёте), k=1: %.1f ms"
        " (последовательно), delay=%zu ms — prefetch-hit 4/4, паузы декода < %zu ms\n",
        ms4, ms1, delay_ms, delay_ms);
}

T_CASE(manager_vram_slots_bitwise_direct_loader) {
    // [R5] round 4: интеграционный тест resident > slots — после КАЖДОГО
    // pin/DMA содержимое VRAM-слота побитово == эталону прямого лоадера
    // (TierLayoutReader — чтение layout ВНЕ тиров). Сценарий жертвы =
    // префетченная группа (регрессия bugfix ba42267: aliasing VRAM-слотов —
    // resident > vram_slots, две группы в одном слоте — не ловился тестами
    // логитов; прямая сверка слотов — единственный прямой тест инварианта
    // «одна группа = один слот»). CPU-конфигурация: VRAM-слот эмулируется
    // host-памятью (vram_storage_) с тем же кодовым путём записи слота
    // (memcpy RAM→VRAM в process_batch); DMA-путь cudaMemcpy при STELL_CUDA
    // покрыт серверной сборкой (пометка для среза 3 — co-tenancy bench).
    const std::size_t slot = 4096;
    const std::size_t n_groups = 6; // 6 групп > 2 слотов — churn гарантирован
    std::vector<TensorGroup> plan;
    for (std::size_t i = 0; i < n_groups; ++i)
        plan.push_back(make_group(
            std::uint32_t(i), "g" + std::to_string(i),
            (i % 3 == 0) ? Role::Attn
                         : ((i % 3 == 1) ? Role::Expert : Role::Kv),
            DType::F16, slot, {std::int64_t(slot / 2), 2}));
    const auto path =
        write_layout_payloads("/tmp/stell_tier_test_mgr_r5.stelltier", plan, slot);

    // Эталон: прямой лоадер (TierLayoutReader, минуя TierManager).
    std::vector<std::vector<std::uint8_t>> ref(n_groups);
    {
        TierLayoutReader r(path);
        for (std::size_t i = 0; i < n_groups; ++i) {
            ref[i].resize(slot);
            r.read_group(std::uint32_t(i), ref[i].data());
        }
    }

    TierConfig cfg = ram_cfg(path, 2); // vram_slots = 2 << план (6 групп)
    TierManager mgr(cfg);
    mgr.set_plan(plan);

    int checks = 0;
    auto verify_resident = [&](const char* what) {
        std::size_t resident = 0;
        for (const auto& g : plan) {
            if (!mgr.resident_in_vram(g.id)) continue;
            ++resident;
            std::vector<std::uint8_t> got(slot);
            mgr.read_vram_slot(g.id, got.data(), got.size());
            T_REQUIRE(std::memcmp(got.data(), ref[g.id.value].data(), slot) == 0,
                      what);
            ++checks;
        }
        T_REQUIRE(resident <= cfg.vram_slots,
                  "resident-группы ≤ vram_slots (нет aliasing слотов)");
        const TierStats st = mgr.stats();
        T_REQUIRE(st.tier_bytes[0] == resident * slot,
                  "tier_bytes[VRAM] == resident × slot_bytes");
    };

    // (1) Сценарий жертвы = префетченная группа: g0/g1 префетчены и
    // материализованы в фоне; затем префетч g2/g3 — жертвы эвиции именно
    // префетченные (unpinned). После каждой материализации — побитовая
    // сверка ВСЕХ resident-слотов (aliasing ba42267 дал бы resident > slots
    // и/или несовпадение байтов).
    mgr.prefetch({GroupKey{plan[0].id, plan[0].role},
                  GroupKey{plan[1].id, plan[1].role}});
    wait_resident(mgr, plan[0].id);
    wait_resident(mgr, plan[1].id);
    verify_resident("слоты префетченных групп == эталону прямого лоадера");

    mgr.prefetch({GroupKey{plan[2].id, plan[2].role},
                  GroupKey{plan[3].id, plan[3].role}});
    wait_resident(mgr, plan[2].id);
    wait_resident(mgr, plan[3].id);
    verify_resident("после эвиции префетченных жертв: новые слоты == эталону");

    // (2) Полный churn-цикл: pin/unpin всех групп — сверка ПОСЛЕ каждого pin
    // (после каждого pin/DMA содержимое слота побитово == эталону).
    for (int round = 0; round < 3; ++round)
        for (const auto& g : plan) {
            auto a = mgr.pin(GroupKey{g.id, g.role});
            a->wait();
            verify_resident("после каждого pin/DMA слот == эталону прямого лоадера");
            mgr.unpin(GroupKey{g.id, g.role});
        }
    T_REQUIRE(checks >= 12, "покрытие сверок достаточно (≥ 12 проверок слотов)");

    // (3) Ошибки расширения на детерминированном менеджере (1 слот, всё
    // закреплено): группа не resident → IoFailure; unknown → PlanMismatch;
    // dst меньше слота → PlanMismatch.
    TierManager mgr2(ram_cfg(path, 1));
    mgr2.set_plan(plan);
    {
        auto a = mgr2.pin(GroupKey{plan[0].id, plan[0].role});
        a->wait(); // единственный слот занят закреплённой g0
    }
    bool thrown = false;
    std::vector<std::uint8_t> buf(slot);
    try {
        mgr2.read_vram_slot(GroupId{1}, buf.data(), buf.size());
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
    }
    T_REQUIRE(thrown, "read_vram_slot не-resident группы → IoFailure");
    thrown = false;
    try {
        mgr2.read_vram_slot(GroupId{99}, buf.data(), buf.size());
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "read_vram_slot unknown → PlanMismatch");
    thrown = false;
    try {
        mgr2.read_vram_slot(GroupId{0}, buf.data(), slot - 1);
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "read_vram_slot dst_bytes < slot_bytes → PlanMismatch");

    std::printf("  [r5 vram-slot bitwise] сверок слотов: %d (vram_slots=2, план=%zu групп)\n",
                checks, n_groups);
}

// --- [R2] round 5: per-group layout (слоты групп переменной ёмкости) -------

T_CASE(manager_per_group_variable_slot_sizes) {
    // Реальный stell-tier layout v1: у групп РАЗНЫЕ slot_bytes (3× разницы в
    // фикстуре). Менеджер обязан:
    //   - принимать план (vram_slot_size = 0 → авто = max slot группы);
    //   - читать/копировать ровно slot_bytes ГРУППЫ (никакого мусорного
    //     хвоста max-слота: memcpy фазы C — gb байтов, read_ram_slot/
    //     read_vram_slot — gb байтов) — побитовая сверка с payload writer'а;
    //   - честно учитывать байты RAM-тира и io_bytes по группам;
    //   - вести per-group счётчики prefetch (hit-rate по группам, bench).
    struct Spec { std::uint32_t gid; const char* name; Role role; std::size_t slot; };
    const Spec specs[] = {
        {0, "embed",     Role::Attn,   8192},
        {1, "l0.attn",   Role::Attn,   4096},
        {2, "l0.router", Role::Expert, 8192},
        {3, "l0.kv",     Role::Kv,     12288},
        {4, "l1.attn",   Role::Attn,   4096},
        {5, "l1.router", Role::Expert, 4096},
    };
    const std::size_t n = sizeof(specs) / sizeof(specs[0]);
    std::vector<TensorGroup> plan;
    std::uint64_t plan_bytes = 0;
    for (const auto& s : specs) {
        plan.push_back(make_group(s.gid, s.name, s.role, DType::F16, s.slot,
                                  {std::int64_t(s.slot / 2), 2}));
        plan_bytes += s.slot;
    }
    const std::string path = "/tmp/stell_tier_test_mgr_pergroup.stelltier";
    {   // writer с per-group payload (детерминированный паттерн, как в R7-тесте)
        TierLayoutHeader h;
        h.dtype_profile = 1;
        h.n_groups = n;
        TierLayoutWriter w(path, h);
        for (const auto& s : specs) {
            std::vector<std::uint8_t> payload(s.slot);
            for (std::size_t k = 0; k < s.slot; ++k)
                payload[k] = std::uint8_t((s.gid * 131 + k * 7 + 5) & 0xFF);
            w.add_group(plan[s.gid], 1, payload.data(), payload.size());
        }
        w.finalize();
    }

    TierConfig cfg;
    cfg.vram_slots = 2;                          // < плана — churn
    cfg.vram_slot_size = 0;                      // авто: max slot плана (12288)
    cfg.ram_capacity = (std::size_t)plan_bytes;  // план целиком (Σ gb)
    cfg.nvme_layout = path;
    cfg.lookahead = 0;
    TierManager mgr(cfg);
    mgr.set_plan(plan);

    // Полный цикл: холодный pin каждой группы → побитовая сверка RAM- и
    // VRAM-копий с payload writer'а (ровно gb байтов группы).
    std::vector<std::uint8_t> buf(12288), expected(12288);
    for (const auto& s : specs) {
        auto a = mgr.pin(GroupKey{GroupId{s.gid}, s.role});
        a->wait();
        mgr.read_ram_slot(GroupId{s.gid}, buf.data(), buf.size());
        for (std::size_t k = 0; k < s.slot; ++k)
            expected[k] = std::uint8_t((s.gid * 131 + k * 7 + 5) & 0xFF);
        T_REQUIRE(std::memcmp(buf.data(), expected.data(), s.slot) == 0,
                  "RAM-копия группы побитово == payload (ровно gb байтов)");
        mgr.read_vram_slot(GroupId{s.gid}, buf.data(), buf.size());
        T_REQUIRE(std::memcmp(buf.data(), expected.data(), s.slot) == 0,
                  "VRAM-слот группы побитово == payload (ровно gb байтов)");
        mgr.unpin(GroupKey{GroupId{s.gid}, s.role});
    }

    // Учёт per-group: холодная загрузка каждой группы — 2 переноса (NVMe→RAM
    // + RAM→VRAM) по gb байтов: io_bytes == 2×Σ gb; io_ops == 2×n.
    const TierStats st = mgr.stats();
    T_REQUIRE(st.io_bytes == 2 * plan_bytes,
              "io_bytes == 2×Σ slot_bytes групп (пер-group учёт, не max-слоты)");
    T_REQUIRE(st.io_ops == 2 * n, "io_ops == 2 переноса на группу");
    T_REQUIRE(st.tier_bytes[1] == plan_bytes,
              "RAM-тир занят ровно Σ slot_bytes групп (per-group учёт)");
    T_REQUIRE(st.tier_bytes[1] <= cfg.ram_capacity, "RAM-тир в лимите");
    T_REQUIRE(st.tier_bytes[2] == plan_bytes, "NVMe-тир = сумме слотов layout");

    // Per-group счётчики prefetch: холодный pin → pf_misses=1, pf_hits=0;
    // prefetch + материализация + pin → pf_hits=1.
    {
        const auto stats0 = mgr.group_prefetch_stats();
        T_REQUIRE(stats0.size() == n, "снапшот покрывает все группы плана");
        T_REQUIRE(stats0[0].gid == GroupId{0} && stats0[0].misses == 1 &&
                  stats0[0].hits == 0,
                  "холодный pin → per-group misses=1, hits=0");
        mgr.prefetch({GroupKey{GroupId{0}, Role::Attn}});
        wait_resident(mgr, GroupId{0}); // детерминизм: материализация до pin
        auto a = mgr.pin(GroupKey{GroupId{0}, Role::Attn});
        a->wait();
        mgr.unpin(GroupKey{GroupId{0}, Role::Attn});
        const auto stats1 = mgr.group_prefetch_stats();
        T_REQUIRE(stats1[0].hits == 1 && stats1[0].misses == 1,
                  "pin после префетча → per-group hits=1 (hit-rate по группам)");
    }

    // read_ram_slot с dst меньше slot_bytes ГРУППЫ → PlanMismatch (per-group
    // граница); dst ≥ gb — работает (проверено выше буфером 12288).
    bool thrown = false;
    try {
        mgr.read_ram_slot(GroupId{3}, buf.data(), 4096); // gb(gid3)=12288 > 4096
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::PlanMismatch;
    }
    T_REQUIRE(thrown, "read_ram_slot: dst < slot_bytes группы → PlanMismatch");
}

T_MAIN

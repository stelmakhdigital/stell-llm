// [R2] round 5 — тесты bench per-group layout (bench_tiers_pause, режим
// "pergroup"): реальный stell-tier layout v1 (слоты групп переменной
// ёмкости), пауза слоя = Σ пауз групп слоя, hit-rate по группам.
//
// Честные ассершны:
//   1. Фикстура-layout (переменные слоты, роли attn/expert/kv, имена с
//      префиксом «l<N>.») — побитово == ожидаемым паттернам через прямой
//      лоадер (TierLayoutReader: header/payload CRC).
//   2. Бинарник bench в режиме pergroup: p50/p95 считаются (bench-side
//      строки присутствуют), hit-rate ПО ГРУППАМ побитово == ожидаемому
//      паттерну (детерминизм: префетч слоя материализуется до паузы —
//      каждый pin после префетча = hit; cold — каждый pin = miss).
//   3. Обратная совместимость uniform-режима: старая форма аргументов
//      (6 аргументов) работает, строит layout, печатает bench-строки.
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

#ifndef STELL_BENCH_PAUSE_BIN
#define STELL_BENCH_PAUSE_BIN "./bench_tiers_pause"
#endif

static const char* kLayoutPath = "/tmp/stell_tier_test_pergroup.stelltier";
static const char* kUniformPath = "/tmp/stell_tier_test_bench_uniform.stelltier";

// Фикстура: 8 групп, 3 «слоя» (слой 0 — глобальные embed/norm; слои 1–2 —
// «l0.*»/«l1.*»), слоты ПЕРЕМЕННОЙ ёмкости (4/8/12 КиБ), payload — детерми-
// нированный паттерн (gid*131 + k*7 + 5) — как в тестах round 3/4.
struct FixtureGroup { std::uint32_t gid; const char* name; Role role; std::size_t slot; };

static const FixtureGroup kFixture[] = {
    {0, "embed",      Role::Attn,   8192},
    {1, "norm_final", Role::Attn,   4096},
    {2, "l0.attn",    Role::Attn,   4096},
    {3, "l0.router",  Role::Expert, 8192},
    {4, "l0.kv",      Role::Kv,     12288},
    {5, "l1.attn",    Role::Attn,   4096},
    {6, "l1.router",  Role::Expert, 8192},
    {7, "l1.kv",      Role::Kv,     4096},
};
static const std::size_t kNFixture = sizeof(kFixture) / sizeof(kFixture[0]);

static std::uint8_t pattern_byte(std::uint32_t gid, std::size_t k) {
    return std::uint8_t((gid * 131 + k * 7 + 5) & 0xFF);
}

static void build_fixture() {
    TierLayoutHeader h;
    h.dtype_profile = 1;
    h.n_groups = kNFixture;
    h.model_name = "bench-pergroup";
    TierLayoutWriter w(kLayoutPath, h);
    for (const auto& f : kFixture) {
        std::vector<std::uint8_t> payload(f.slot);
        for (std::size_t k = 0; k < f.slot; ++k) payload[k] = pattern_byte(f.gid, k);
        w.add_group(make_group(f.gid, f.name, f.role, DType::F16, f.slot,
                               {std::int64_t(f.slot / 2), 2}),
                    1, payload.data(), payload.size());
    }
    w.finalize();
}

static std::string run_bench(const std::string& args_csv, int* rc) {
    // args_csv: аргументы через '|'; popen наследует env (для O_DIRECT-прогона).
    std::string cmd = std::string(STELL_BENCH_PAUSE_BIN) + " " + args_csv + " 2>&1";
    FILE* p = popen(cmd.c_str(), "r");
    T_REQUIRE(p != nullptr, "popen(bench_tiers_pause)");
    std::string out;
    char buf[4096];
    while (std::size_t n = std::fread(buf, 1, sizeof(buf), p))
        out.append(buf, n);
    const int code = pclose(p);
    if (rc) *rc = code;
    return out;
}

static bool contains(const std::string& s, const std::string& sub) {
    return s.find(sub) != std::string::npos;
}

T_CASE(bench_per_group_fixture_bitwise_layout) {
    // Фикстура-layout побитово == паттернам: прямой лоадер (TierLayoutReader,
    // header-CRC всегда + payload-CRC) читает каждую группу → memcmp паттерна.
    build_fixture();
    TierLayoutReader r(kLayoutPath);
    T_REQUIRE(r.index().size() == kNFixture, "в индексе 8 групп");
    for (const auto& f : kFixture) {
        const TierGroupIndexEntry* e = r.find(f.gid);
        T_REQUIRE(e != nullptr, "группа есть в индексе");
        T_REQUIRE(e->slot_bytes == f.slot, "ёмкость слота группы == фикс­туре");
        T_REQUIRE(e->name == f.name, "имя группы (усечение до 16 байт не тут)");
        std::vector<std::uint8_t> buf(f.slot);
        r.read_group(f.gid, buf.data());
        std::vector<std::uint8_t> expected(f.slot);
        for (std::size_t k = 0; k < f.slot; ++k) expected[k] = pattern_byte(f.gid, k);
        T_REQUIRE(std::memcmp(buf.data(), expected.data(), f.slot) == 0,
                  "payload группы побитово == паттерну (layout roundtrip)");
    }
}

T_CASE(bench_per_group_pause_and_hitrate_pattern) {
    // Прогон bench в pergroup-режиме (passes=3, prefetch): детерминизм —
    // префетч слоя материализуется до паузы → каждый pin = prefetch-hit.
    // Ожидаемый паттерн ПО ГРУППАМ (побитово): каждая группа — hits=3, misses=0;
    // hit-rate overall/expert/attn/kv = 1.0000; p50/p95/p99 строки присутствуют.
    build_fixture();
    int rc = 0;
    const std::string out = run_bench(
        std::string(kLayoutPath) + " 0 0 0 3 prefetch pergroup", &rc);
    T_REQUIRE(rc == 0, "bench pergroup: код возврата 0");
    T_REQUIRE(contains(out, "mode=pergroup"), "в строке прогона mode=pergroup");
    T_REQUIRE(contains(out, "8 групп, слоёв 3, max slot 12288 B"),
              "pergroup layout: 8 групп / 3 слоя (префикс l<N>.+глобальные) / max slot 12288");
    T_REQUIRE(contains(out, "vram_slots=3"),
              "VRAM-слоты = max групп на слой («вмещает слой»)");
    T_REQUIRE(contains(out, "bench-side pause p50=") &&
              contains(out, "p95=") && contains(out, "p99="),
              "p50/p95/p99 Σ-пауз слоёв считаются (bench-side двойной замер)");
    T_REQUIRE(contains(out, "bench io mode: odirect_requested=0 odirect_active="),
              "режим I/O зафиксирован строкой bench io mode (uniform-протокол)");
    T_REQUIRE(contains(out, "pergroup hit-rate: overall=1.0000 expert=1.0000 (6/6) "
                             "attn=1.0000 (12/12) kv=1.0000 (6/6)"),
              "hit-rate по группам/ролям == ожидаемому паттерну (все префетч-хиты)");
    // Per-group таблица (BENCH_DEBUG): gid:hits/misses == 3/0 для всех 8 групп.
    ::setenv("BENCH_DEBUG", "1", 1);
    int rc2 = 0;
    const std::string dbg = run_bench(
        std::string(kLayoutPath) + " 0 0 0 3 prefetch pergroup", &rc2);
    ::unsetenv("BENCH_DEBUG");
    T_REQUIRE(rc2 == 0, "bench pergroup (BENCH_DEBUG): код возврата 0");
    T_REQUIRE(contains(dbg, "raw layer sums (ms):"), "сырые Σ-паузы слоёв в логе (профиль остатка)");
    for (const auto& f : kFixture) {
        const std::string tok = " " + std::to_string(f.gid) + ":3/0";
        T_REQUIRE(contains(dbg, tok),
                  "per-group паттерн: gid " + std::to_string(f.gid) + " — hits=3/misses=0");
    }
}

T_CASE(bench_per_group_cold_miss_pattern) {
    // Cold-режим (без префетча): каждый pin — честный холодный промах;
    // паттерн: все группы misses=3/hits=0, hit-rate overall=0.0000.
    build_fixture();
    int rc = 0;
    const std::string out = run_bench(
        std::string(kLayoutPath) + " 0 0 0 3 cold pergroup", &rc);
    T_REQUIRE(rc == 0, "bench pergroup cold: код возврата 0");
    T_REQUIRE(contains(out, "mode=pergroup") && contains(out, "mode=cold"),
              "pergroup cold: режим зафиксирован");
    T_REQUIRE(contains(out, "pergroup hit-rate: overall=0.0000"),
              "cold: hit-rate overall == 0 (нет префетча — честные промахи)");
    T_REQUIRE(contains(out, "bench-side pause p50="), "p50 считается и в cold");
}

T_CASE(bench_per_group_odirect_requested) {
    // Режим I/O — те же env, что в uniform ([Q8]): STELL_TIER_ODIRECT=1 →
    // odirect_requested=1 в строке bench io mode (odirect_active зависит от
    // FS — на tmpfs честный откат, ассершн только на requested).
    build_fixture();
    ::setenv("STELL_TIER_ODIRECT", "1", 1);
    ::setenv("STELL_TIER_PREFETCH_DEPTH", "4", 1);
    int rc = 0;
    const std::string out = run_bench(
        std::string(kLayoutPath) + " 0 0 0 2 prefetch pergroup", &rc);
    ::unsetenv("STELL_TIER_ODIRECT");
    ::unsetenv("STELL_TIER_PREFETCH_DEPTH");
    T_REQUIRE(rc == 0, "bench pergroup (O_DIRECT+k4): код возврата 0");
    T_REQUIRE(contains(out, "odirect_requested=1"),
              "env STELL_TIER_ODIRECT=1 → odirect_requested=1 (pergroup — те же env)");
    T_REQUIRE(contains(out, "k_in_flight=4"),
              "env STELL_TIER_PREFETCH_DEPTH=4 → k_in_flight=4");
}

T_CASE(bench_uniform_mode_backward_compatible) {
    // Обратная совместимость uniform-режима: старая форма аргументов
    // (6 аргументов, без 8-го) — строит синтетический uniform-layout,
    // печатает bench-строки; поведение прежнее (pin_layer fan-in).
    ::remove(kUniformPath);
    int rc = 0;
    const std::string out = run_bench(
        std::string(kUniformPath) + " 4 4096 2 2 prefetch", &rc);
    T_REQUIRE(rc == 0, "bench uniform (старая форма аргументов): код возврата 0");
    T_REQUIRE(contains(out, "layout: создан"), "uniform: синтетический layout создан");
    T_REQUIRE(contains(out, "bench io mode:"), "uniform: строка bench io mode");
    T_REQUIRE(contains(out, "bench-side pause p50="), "uniform: p50/p95/p99");
    T_REQUIRE(contains(out, "mode=prefetch"), "uniform: режим в строке прогона");
    T_REQUIRE(!contains(out, "mode=pergroup") && !contains(out, "pergroup hit-rate"),
              "uniform: pergroup-строк нет (старый вывод не изменён)");
    // И перgroup-аргументы в uniform-валидации: 8-й аргумент «uniform» явно.
    rc = -1;
    run_bench(std::string(kUniformPath) + " 4 4096 2 2 prefetch uniform", &rc);
    T_REQUIRE(rc == 0, "uniform: явный 8-й аргумент uniform работает");
}

T_MAIN

// bench_pause — измерение пауз между слоями (ADR-08 §2 PauseStats) на
// CPU-пути tier-контур: TierManager (VRAM-slots эмуляция + pinned RAM +
// NVMe io_uring, stell-tier layout ADR-12). Не add_test — измерение.
//
// Использование:
//   bench_tiers_pause <layout.stelltier> <groups> <slot_bytes> <layers> <passes> [prefetch|cold] [uniform|pergroup]
//
// Режимы layout ([R2] round 5):
//   uniform (дефолт, обратная совместимость) — синтетический uniform-слоты
//     профиль: layout создаётся, если отсутствует (writer tiers, CRC по
//     канону ADR-12); groups == 2*layers, слот один у всех групп;
//     пауза слоя = pin_layer fan-in (composite).
//   pergroup — РЕАЛЬНЫЙ stell-tier layout v1: слоты групп ПЕРЕМЕННОЙ ёмкости
//     (layout читается TierLayoutReader — группы/роли/размеры из индекса);
//     слой = множество групп (карта слоёв выводится из имён групп: префикс
//     «l<N>.» → слой N+1; группы без префикса слоя — глобальные (embed/norm/
//     head) → слой 0; допущение зафиксировано — в проде карту слоёв знает
//     engine, ADR-08 §5). Пауза слоя = Σ пауз групп слоя (pin каждой группы,
//     ожидание готовности; загрузки групп слоя — в очереди, батчи до k в
//     полёте — deep-prefetch [R2b]); wall-пауза fan-in — справочно.
//     hit-rate prefetch считается ПО ГРУППАМ (group_prefetch_stats, разрез
//     по ролям expert/attn/kv). Режим I/O (STELL_TIER_ODIRECT) и глубина k
//     (STELL_TIER_PREFETCH_DEPTH) — те же env, что в uniform-режиме.
//
// Печать: бэкенд I/O, p50/p95/p99 пауз (мс), io_bytes/io_ops, GB/s;
// в pergroup — hit-rate по группам/ролям, max slot, число слотов VRAM.
#include <stell/tiers/tier_manager.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

using namespace stell::tiers;

namespace {

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

void build_layout(const std::string& path, int groups, std::size_t slot) {
    std::vector<TensorGroup> plan;
    char name[32];
    for (int i = 0; i < groups; ++i) {
        std::snprintf(name, sizeof name, "l%04d.g%03d", i / 2, i % 2);
        TensorGroup g;
        g.id = GroupId{std::uint32_t(i)};
        g.name = name;
        g.role = (i % 2 == 0) ? Role::Attn : Role::Expert;
        g.dtype = DType::F16;
        g.shape = {static_cast<std::int64_t>(slot / 2), 2};
        g.slot_bytes = slot;
        plan.push_back(g);
    }
    TierLayoutHeader h;
    h.dtype_profile = 1;
    h.n_groups = plan.size();
    h.model_name = "bench-pause";
    TierLayoutWriter w(path, h);
    std::vector<std::uint8_t> payload(slot);
    for (int i = 0; i < groups; ++i) {
        for (std::size_t k = 0; k < slot; ++k)
            payload[k] = std::uint8_t((i * 31 + k * 7 + 13) & 0xFF);
        w.add_group(plan[(size_t)i], 1, payload.data(), payload.size());
    }
    w.finalize();
    std::printf("layout: создан %s (%d групп × %zu B)\n", path.c_str(), groups, slot);
}

// --- [R2] round 5: per-group layout -----------------------------------------

// Слой из имени группы: префикс «l<N>.» → слой N+1 (сдвиг: слой 0 —
// глобальные группы без префикса слоя: embed/norm/head — читаются один раз
// в начале декода). Имена в индексе layout обрезаны до 16 байт (ADR-12 §2.2)
// — префикс парсится и в усечённом виде. false → группа глобальная (слой 0).
bool layer_from_name(const std::string& name, std::uint32_t& layer) {
    if (name.size() < 3 || name[0] != 'l') return false;
    std::uint64_t n = 0;
    std::size_t i = 1;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        n = n * 10 + (name[i] - '0');
        ++i;
    }
    if (i == 1 || i >= name.size() || name[i] != '.') return false;
    layer = static_cast<std::uint32_t>(n) + 1;
    return true;
}

// Ожидание resident-ности (детерминизм per-group bench: префетч слоя
// материализуется до паузы; в прод-контуре его перекрывает compute слоя).
bool wait_resident(const TierManager& mgr, GroupId id, int timeout_ms) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!mgr.resident_in_vram(id)) {
        if (std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count() > timeout_ms)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

double pct(std::vector<double> v, double f) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1,
        static_cast<std::size_t>(f * double(v.size() - 1)))];
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <layout.stelltier> <groups> <slot_bytes> <layers> <passes> [prefetch|cold] [uniform|pergroup]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int groups = std::atoi(argv[2]);
    const std::size_t slot = std::strtoull(argv[3], nullptr, 10);
    const int layers = std::atoi(argv[4]);
    const int passes = std::atoi(argv[5]);
    const std::string mode = argc > 6 ? argv[6] : "prefetch";
    const std::string kind = argc > 7 ? argv[7] : "uniform";
    if (kind != "uniform" && kind != "pergroup") {
        std::fprintf(stderr, "bench: 8-й аргумент — uniform|pergroup\n");
        return 2;
    }
    if (mode != "prefetch" && mode != "cold") {
        std::fprintf(stderr, "bench: 7-й аргумент — prefetch|cold\n");
        return 2;
    }
    if (kind == "uniform") {
        if (groups <= 0 || layers <= 0 || layers * 2 != groups || slot % 4096 != 0 ||
            passes <= 0) {
            std::fprintf(stderr, "bench: groups==2*layers, slot кратно 4096\n");
            return 2;
        }
    } else if (passes <= 0) {
        std::fprintf(stderr, "bench: passes > 0\n");
        return 2;
    }

    // ================================================================ pergroup
    if (kind == "pergroup") {
        // Реальный stell-tier layout v1: группы/роли/ёмкости — из индекса
        // (TierLayoutReader; header-CRC всегда, payload-CRC here не читается —
        // данные идут через менеджер с verify-политикой конфига).
        std::FILE* probe = std::fopen(path.c_str(), "rb");
        if (!probe) {
            std::fprintf(stderr, "pergroup: layout '%s' не найден — постройте его конвертером models\n",
                         path.c_str());
            return 2;
        }
        std::fclose(probe);
        TierLayoutReader rdr(path);
        std::vector<TensorGroup> plan;
        std::uint64_t plan_bytes = 0;
        std::size_t max_slot = 0;
        for (const auto& e : rdr.index()) {
            TensorGroup g;
            g.id = GroupId{e.group_id};
            g.name = e.name;
            g.role = role_from_code(e.role);
            g.dtype = dtype_from_code(e.dtype);
            g.slot_bytes = e.slot_bytes;
            for (int d = 0; d < static_cast<int>(e.shape_rank) && d < 4; ++d)
                g.shape.push_back(static_cast<std::int64_t>(e.shape[(size_t)d]));
            plan_bytes += e.slot_bytes;
            max_slot = std::max(max_slot, (size_t)e.slot_bytes);
            plan.push_back(g);
        }
        if (plan.empty()) {
            std::fprintf(stderr, "pergroup: layout пуст\n");
            return 2;
        }
        // Карта слоёв: «l<N>.» → слой N+1; глобальные (embed/norm/head) → слой 0.
        std::uint32_t n_layers = 0;
        std::vector<std::vector<GroupId>> layers_map;
        for (const auto& g : plan) {
            std::uint32_t l = 0;
            layer_from_name(g.name, l);
            n_layers = std::max(n_layers, l + 1);
            if (layers_map.size() < (size_t)l + 1) layers_map.resize((size_t)l + 1);
            layers_map[l].push_back(g.id);
        }
        std::size_t max_layer_groups = 0;
        for (const auto& lv : layers_map) max_layer_groups = std::max(max_layer_groups, lv.size());

        // [R2] round 4: env-профиль (STELL_TIER_ODIRECT, STELL_TIER_PREFETCH_DEPTH
        // и др.). Per-group дефолты — только если соответствующий env НЕ задан:
        //   VRAM-слоты = max групп на слой («VRAM-тир вмещает слой» —
        //     steady-state hit-путь; env STELL_TIER_VRAM_SLOTS переопределяет);
        //   VRAM-слот = 0 → авто (max slot_bytes плана — слоты групп переменной
        //     ёмкости, VRAM-слот фиксированного размера = max ёмкости группы);
        //   RAM-тир = план целиком (Σ slot_bytes; pinned).
        // Честная пометка: целевой профиль uniform-матрицы round 4 «4 МиБ-слоты»
        // к per-group layout неприменим напрямую (embed > 4 МиБ на реальных
        // чекпоинтах) — фактические max slot/слоты печатаются ниже.
        TierConfig cfg = tier_config_from_env();
        if (!std::getenv("STELL_TIER_VRAM_SLOTS")) cfg.vram_slots = max_layer_groups;
        if (!std::getenv("STELL_TIER_RAM_MB")) cfg.ram_capacity = (std::size_t)plan_bytes;
        cfg.nvme_layout = path;
        cfg.lookahead = 0; // prefetch — явно от bench (как от engine)
        TierManager mgr(cfg);
        mgr.set_plan(plan);
        mgr.set_layer_map(layers_map);

        std::printf("pergroup layout: %s (%zu групп, слоёв %u, max slot %zu B, план %llu B; "
                    "vram_slots=%zu, ram_capacity=%zu B)\n",
                    path.c_str(), plan.size(), (unsigned)n_layers, max_slot,
                    (unsigned long long)plan_bytes, cfg.vram_slots, cfg.ram_capacity);

        auto role_of = [&](GroupId gid) -> Role {
            for (const auto& g : plan)
                if (g.id == gid) return g.role;
            return Role::Expert;
        };
        auto keys_of = [&](std::size_t l) {
            std::vector<GroupKey> ks;
            for (GroupId gid : layers_map[l])
                ks.push_back(GroupKey{gid, role_of(gid)});
            return ks;
        };

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<double> layer_ms, wall_ms; // Σ пауз групп слоя / wall fan-in
        for (int p = 0; p < passes; ++p) {
            for (std::uint32_t l = 0; l < n_layers; ++l) {
                if (mode == "prefetch") {
                    // Router-lookahead: prefetch ВСЕХ групп слоя (Σ групп —
                    // deep-prefetch k в полёте), материализация — до паузы
                    // (в проде перекрывается compute; bench — steady-state).
                    mgr.prefetch(keys_of(l));
                    for (GroupId gid : layers_map[l])
                        if (!wait_resident(mgr, gid, 30000)) {
                            std::fprintf(stderr, "pergroup: тайм-аут материализации gid=%u\n",
                                         gid.value);
                            return 3;
                        }
                }
                const auto tw_l = std::chrono::steady_clock::now();
                std::vector<std::unique_ptr<Awaiter>> waits;
                for (GroupId gid : layers_map[l])
                    waits.push_back(mgr.pin(GroupKey{gid, role_of(gid)}));
                double layer_sum = 0.0; // пауза слоя = Σ пауз групп слоя
                for (std::size_t i = 0; i < waits.size(); ++i) {
                    const auto tw = std::chrono::steady_clock::now();
                    waits[i]->wait();
                    layer_sum += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tw).count();
                }
                const double wall = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tw_l).count();
                layer_ms.push_back(layer_sum);
                wall_ms.push_back(wall);
                for (GroupId gid : layers_map[l])
                    mgr.unpin(GroupKey{gid, role_of(gid)});
            }
        }
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        const TierStats st = mgr.stats();
        if (std::getenv("BENCH_DEBUG")) {
            std::printf("raw layer sums (ms): ");
            for (double v : layer_ms) std::printf("%.1f ", v);
            std::printf("\npergroup groups (gid:hits/misses):");
            for (const auto& s : mgr.group_prefetch_stats())
                std::printf(" %u:%llu/%llu", s.gid.value,
                            (unsigned long long)s.hits, (unsigned long long)s.misses);
            std::printf("\n");
        }
        std::printf("bench io mode: odirect_requested=%d odirect_active=%s k_in_flight=%zu io_delay_ms=%zu\n",
                    (int)cfg.odirect,
                    (mgr.nvme_odirect_active() ? "true" : "false"),
                    cfg.deep_prefetch_depth, cfg.io_delay_ms);
        // Hit-rate ПО ГРУППАМ ([R2] round 5): общий + разрез по ролям плана.
        double hits = 0, misses = 0, h_exp = 0, m_exp = 0, h_attn = 0, m_attn = 0,
               h_kv = 0, m_kv = 0;
        for (const auto& s : mgr.group_prefetch_stats()) {
            const Role r = role_of(s.gid);
            hits += s.hits; misses += s.misses;
            if (r == Role::Expert) { h_exp += s.hits; m_exp += s.misses; }
            else if (r == Role::Attn) { h_attn += s.hits; m_attn += s.misses; }
            else { h_kv += s.hits; m_kv += s.misses; }
        }
        auto rate = [](double h, double m) {
            return (h + m) > 0 ? h / (h + m) : 0.0;
        };
        std::printf("pergroup hit-rate: overall=%.4f expert=%.4f (%.0f/%.0f) attn=%.4f (%.0f/%.0f) kv=%.4f (%.0f/%.0f)\n",
                    rate(hits, misses),
                    rate(h_exp, m_exp), h_exp, h_exp + m_exp,
                    rate(h_attn, m_attn), h_attn, h_attn + m_attn,
                    rate(h_kv, m_kv), h_kv, h_kv + m_kv);
        const double io_gb = double(st.io_bytes) / double(1 << 30);
        std::printf(
            "  bench-side pause p50=%.3f ms p95=%.3f ms p99=%.3f ms (Σ пауз групп слоя)\n"
            "  bench-side wall p50=%.3f ms p95=%.3f ms p99=%.3f ms (fan-in, справочно)\n",
            pct(layer_ms, 0.50), pct(layer_ms, 0.95), pct(layer_ms, 0.99),
            pct(wall_ms, 0.50), pct(wall_ms, 0.95), pct(wall_ms, 0.99));
        std::printf(
            "bench_pause: backend=%s path=%s mode=pergroup groups=%zu slot=%zu(max) layers=%u passes=%d mode=%s\n"
            "  pause p50=%.3f ms p95=%.3f ms p99=%.3f ms (цель PLAN.md §4: p95 ≤ 5 мс)\n"
            "  io_bytes=%llu (%.2f GiB) io_ops=%llu throughput=%.2f GiB/s total=%.0f ms\n"
            "  payload-CRC пересчётов=%llu (verify-политика конфига/env, ADR-12 §2.3)\n",
            nvme_io_backend().c_str(), path.c_str(), plan.size(), max_slot,
            (unsigned)n_layers, passes, mode.c_str(),
            st.layer_pause.p50_ms, st.layer_pause.p95_ms, st.layer_pause.p99_ms,
            (unsigned long long)st.io_bytes, io_gb, (unsigned long long)st.io_ops,
            io_gb / (total_ms / 1000.0), total_ms,
            (unsigned long long)mgr.crc_checks());
        return 0;
    }

    // ================================================================ uniform
    std::FILE* probe = std::fopen(path.c_str(), "rb");
    const bool exists = probe != nullptr;
    if (probe) std::fclose(probe);
    if (!exists) build_layout(path, groups, slot);

    // [R2] round 4: env-профиль поверх bench-базы (STELL_TIER_ODIRECT,
    // STELL_TIER_PREFETCH_DEPTH, STELL_TIER_IO_DELAY_MS и др. — матрица
    // [R2] прогоняется bench_p95_nvme.sh через эти env).
    TierConfig cfg = tier_config_from_env();
    cfg.vram_slots = 2;                    // churn: слои > VRAM-слотов
    cfg.vram_slot_size = slot;
    cfg.ram_capacity = 4 * slot;
    cfg.nvme_layout = path;
    cfg.lookahead = 0;                     // prefetch — явно от engine (lookahead)
    TierManager mgr(cfg);

    std::vector<TensorGroup> plan;
    char name[32];
    for (int i = 0; i < groups; ++i) {
        std::snprintf(name, sizeof name, "l%04d.g%03d", i / 2, i % 2);
        TensorGroup g;
        g.id = GroupId{std::uint32_t(i)};
        g.name = name;
        g.role = (i % 2 == 0) ? Role::Attn : Role::Expert;
        g.dtype = DType::F16;
        g.shape = {static_cast<std::int64_t>(slot / 2), 2};
        g.slot_bytes = slot;
        plan.push_back(g);
    }
    mgr.set_plan(plan);

    auto layers_map = std::vector<std::vector<GroupId>>(unsigned(layers));
    for (int i = 0; i < groups; ++i)
        layers_map[unsigned(i / 2)].push_back(GroupId{std::uint32_t(i)});
    mgr.set_layer_map(layers_map);

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> raw_ms;
    for (int p = 0; p < passes; ++p) {
        for (int l = 0; l < layers; ++l) {
            // router-lookahead: prefetch следующего слоя ДО паузы слоя.
            // (mode cold — без prefetch: честная пауза холодной загрузки
            // слоя NVMe→RAM→VRAM, когда lookahead не угадал.)
            if (mode == "prefetch" && l + 1 < layers) {
                std::vector<GroupKey> keys;
                for (GroupId gid : layers_map[unsigned(l + 1)])
                    keys.push_back(GroupKey{gid, plan[(size_t)gid.value].role});
                mgr.prefetch(keys);
            }
            const auto tw = std::chrono::steady_clock::now();
            auto a = mgr.pin_layer(unsigned(l)); // пауза слоя меряется sink'ом pin_layer
            a->wait();
            raw_ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tw).count());
            for (GroupId gid : layers_map[unsigned(l)])
                mgr.unpin(GroupKey{gid, plan[(size_t)gid.value].role});
        }
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    const TierStats st = mgr.stats();
    if (std::getenv("BENCH_DEBUG")) {
        // Честный дамп сырых пауз (двойной замер в бенче, независимый от
        // stats-снапшота tiers).
        std::printf("raw pauses (ms): ");
        for (double p : raw_ms) std::printf("%.1f ", p);
        std::printf("\n");
    }
    // [R2] round 4: честная фиксация режима I/O и глубины k в строке прогона
    // (odirect_active() — O_DIRECT-тракт действительно активен; иначе отката
    // на buffered — прогон обязан быть помечен).
    std::printf("bench io mode: odirect_requested=%d odirect_active=%s k_in_flight=%zu io_delay_ms=%zu\n",
                (int)cfg.odirect,
                (mgr.nvme_odirect_active() ? "true" : "false"),
                cfg.deep_prefetch_depth, cfg.io_delay_ms);
    const double io_gb = double(st.io_bytes) / double(1 << 30);
    // Двойной замер: bench-side percentiles по raw_ms (независимы от tiers).
    std::vector<double> sorted(raw_ms);
    std::sort(sorted.begin(), sorted.end());
    auto q = [&](double f) {
        const std::size_t i = std::min(sorted.size() - 1,
            static_cast<std::size_t>(f * double(sorted.size() - 1)));
        return sorted.empty() ? 0.0 : sorted[i];
    };
    std::printf(
        "  bench-side pause p50=%.3f ms p95=%.3f ms p99=%.3f ms\n",
        q(0.50), q(0.95), q(0.99));
    std::printf(
        "bench_pause: backend=%s path=%s groups=%d slot=%zu layers=%d passes=%d mode=%s\n"
        "  pause p50=%.3f ms p95=%.3f ms p99=%.3f ms (цель PLAN.md §4: p95 ≤ 5 мс)\n"
        "  io_bytes=%llu (%.2f GiB) io_ops=%llu throughput=%.2f GiB/s total=%.0f ms\n"
        "  payload-CRC пересчётов=%llu (verify=off по умолчанию — hot path без CRC, [R1];\n"
        "    STELL_TIER_VERIFY_CRC=1 включает; verify-проход — TierManager::verify_slots)\n",
        nvme_io_backend().c_str(), path.c_str(), groups, slot, layers, passes,
        mode.c_str(),
        st.layer_pause.p50_ms, st.layer_pause.p95_ms, st.layer_pause.p99_ms,
        (unsigned long long)st.io_bytes, io_gb, (unsigned long long)st.io_ops,
        io_gb / (total_ms / 1000.0), total_ms,
        (unsigned long long)mgr.crc_checks());
    return 0;
}

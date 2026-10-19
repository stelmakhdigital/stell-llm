// bench_pause — измерение пауз между слоями (ADR-08 §2 PauseStats) на
// CPU-пути tier-контур: TierManager (VRAM-slots эмуляция + pinned RAM +
// NVMe io_uring, stell-tier layout ADR-12). Не add_test — измерение.
//
// Использование:
//   bench_tiers_pause <layout.stelltier> <groups> <slot_bytes> <layers> <passes>
// layout создаётся, если отсутствует (writer tiers, CRC по канону ADR-12).
// Печать: бэкенд I/O, p50/p95/p99 пауз (мс), io_bytes/io_ops, GB/s.
#include <stell/tiers/tier_manager.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
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

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <layout.stelltier> <groups> <slot_bytes> <layers> <passes> [prefetch|cold]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int groups = std::atoi(argv[2]);
    const std::size_t slot = std::strtoull(argv[3], nullptr, 10);
    const int layers = std::atoi(argv[4]);
    const int passes = std::atoi(argv[5]);
    const std::string mode = argc > 6 ? argv[6] : "prefetch";
    if (groups <= 0 || layers <= 0 || layers * 2 != groups || slot % 4096 != 0 ||
        passes <= 0) {
        std::fprintf(stderr, "bench: groups==2*layers, slot кратно 4096\n");
        return 2;
    }

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

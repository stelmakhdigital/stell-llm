// [R2] round 10 (ADR-13 §[R3], planner T-20261022183913 R2): trace-driven
// замер hit-rate prefetch для матрицы {LRU,LCU}×{k} на ЭКСПЕРТНОЙ
// гранулярности (после [R1] матрица перестала быть вырожденной).
//
// Модель: MoE > VRAM — L слоёв × (attn + kv + router + E экспертных групп),
// VRAM-слотов меньше суммарного плана. Трейс выбора экспертов — синтетический,
// детерминированный (LCG, фиксированный seed 20261022, зафиксирован в тесте).
// Каждая ячейка матрицы {policy}×{k}: k экспертов на токен, политика
// выгрузки LRU/LCU, глубина батча тира = k (STELL_TIER deep-prefetch).
//
// Измеряется на ячейку:
//   * hit_rate_expert — доля пинов экспертных групп, обслуженных prefetch
//     (router-lookahead по трейсу следующего шага), per-group учёт;
//   * p95_groups_ms / p95_layerpin_ms — p95 паузы слоя: гранулярный поток
//     (Σ пауз групп слоя: base + выбранные эксперты) против пиннинга слоя
//     целиком (pin_layer) на ТОМ ЖЕ трейсе.
//
// Детерминизм: io_delay_ms — диагностический крюк тира (искусственная
// латентность загрузки батча): абсолютные мс НЕ переносятся на GPU, отчёт
// честно помечен средой CPU + co-tenancy; сравнение p95 — относительное,
// на одном трейсе и одном крюке.
//
// Отчёт: bench/results/<date>-tiers-hitrate-matrix/report.json
// (report.json + матрица ≥ 4 ячеек с числовыми hit-rate).
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

namespace {

struct Cell {
    const char* policy_name;
    EvictionPolicy policy;
    std::size_t k;
    double hit_rate_expert = 0.0;
    double p95_groups_ms = 0.0;
    double p95_layerpin_ms = 0.0;
};

constexpr std::uint32_t kSeed = 20261022;   // зафиксирован в тесте (детерминизм)
constexpr std::size_t kLayers = 4;
constexpr std::size_t kExperts = 8;
constexpr std::size_t kTokens = 32;         // токенов в трейсе на проход
constexpr std::size_t kPasses = 3;
constexpr std::size_t kSlotBytes = 4096;   // кратно 4096 (ADR-12 §2.4)
constexpr std::size_t kIoDelayMs = 2;       // честный крюк (см. шапку)

// Детерминированный трейс: LCG (seed kSeed) + ЛОКАЛЬНОСТЬ роутера —
// реальный MoE-роутер имеет устойчивые «горячие» эксперты: выбор = базовый
// горячий набор (медленно дрейфует) + редкие замены. На таком трейсе
// повторные пины и prefetch-попадания наблюдаемы честно (равномерный
// случайный трейс вымел бы ВСЁ между токенами — hit-rate 0 артефактно).
std::vector<std::vector<std::vector<std::uint32_t>>> make_trace(std::size_t k) {
    std::uint32_t s = kSeed;
    auto next = [&s]() { s = s * 1664525u + 1013904223u; return s; };
    std::vector<std::vector<std::vector<std::uint32_t>>> trace(
        kLayers, std::vector<std::vector<std::uint32_t>>(
                     kTokens, std::vector<std::uint32_t>(k)));
    for (std::size_t l = 0; l < kLayers; ++l) {
        std::uint32_t hot = next() % kExperts;
        for (std::size_t t = 0; t < kTokens; ++t) {
            if (next() % 8 == 0) hot = next() % kExperts;   // медленный дрейф
            for (std::size_t j = 0; j < k; ++j)
                trace[l][t][j] = static_cast<std::uint32_t>(
                    (hot + j + (next() % 2)) % kExperts);
        }
    }
    return trace;
}

std::vector<TensorGroup> make_plan() {
    std::vector<TensorGroup> groups;
    std::uint32_t gid = 0;
    for (std::size_t l = 0; l < kLayers; ++l) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "l%03zu.attn", l);
        groups.push_back(make_group(gid++, buf, Role::Attn, DType::F16, kSlotBytes));
        std::snprintf(buf, sizeof buf, "l%03zu.kv", l);
        groups.push_back(make_group(gid++, buf, Role::Kv, DType::F16, kSlotBytes));
        std::snprintf(buf, sizeof buf, "l%03zu.router", l);
        groups.push_back(make_group(gid++, buf, Role::Expert, DType::F32, kSlotBytes));
        for (std::size_t e = 0; e < kExperts; ++e) {
            std::snprintf(buf, sizeof buf, "l%03zu.e%03zu.up", l, e);
            groups.push_back(make_group(gid++, buf, Role::Expert, DType::F16, kSlotBytes));
        }
    }
    return groups;
}

// Базовые (не-экспертные) группы слоя: attn + kv + router (роутер нужен
// всегда; роль expert, но пинуется всегда — прецедент [R1]).
std::vector<GroupKey> base_keys(const std::vector<TensorGroup>& plan,
                                std::size_t layer) {
    std::vector<GroupKey> out;
    const std::size_t per_layer = 3 + kExperts;
    for (std::size_t i = 0; i < 3; ++i) {
        const TensorGroup& g = plan[layer * per_layer + i];
        out.push_back(GroupKey{g.id, g.role});
    }
    return out;
}

std::vector<GroupKey> expert_keys(const std::vector<TensorGroup>& plan,
                                  std::size_t layer,
                                  const std::vector<std::uint32_t>& chosen) {
    std::vector<GroupKey> out;
    const std::size_t per_layer = 3 + kExperts;
    for (std::uint32_t e : chosen) {
        const TensorGroup& g = plan[layer * per_layer + 3 + e];
        out.push_back(GroupKey{g.id, g.role});
    }
    return out;
}

double wait_ms(std::unique_ptr<Awaiter>& a) {
    const auto t0 = std::chrono::steady_clock::now();
    a->wait();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

double p95_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double rank = 0.95 * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(rank));
    const auto hi = static_cast<std::size_t>(std::ceil(rank));
    return v[lo] + (v[hi] - v[lo]) * (rank - static_cast<double>(lo));
}

// Прогон трейса в гранулярном режиме: base + ВЫБРАННЫЕ эксперты; prefetch —
// router-lookahead по трейсу следующего шага. Возвращает p95 пауз слоёв
// (Σ пауз групп слоя — запись record_layer_pause) и агрегат per-group
// hit-rate экспертных групп.
struct GranularRun {
    double p95_ms = 0.0;
    double hit_rate = 0.0;
};

GranularRun run_granular(const std::string& layout,
                         const std::vector<TensorGroup>& plan,
                         EvictionPolicy policy, std::size_t k,
                         const std::vector<std::vector<std::vector<std::uint32_t>>>& trace) {
    TierConfig cfg;
    // Рабочий слой (3+k) + prefetch в полёте (k) + выжившие эксперты
    // предыдущего токена (4k); план (44×4KB=176KB) > VRAM при k ≤ 4 —
    // MoE > VRAM сохранён (максимум 27×4KB=108KB < 176KB).
    cfg.vram_slots = 3 + 6 * k;
    cfg.vram_slot_size = kSlotBytes;
    cfg.ram_capacity = 64 * kSlotBytes;
    cfg.nvme_layout = layout;
    cfg.policy = policy;
    cfg.deep_prefetch_depth = k;     // глубина батча тира = k (ячейка матрицы)
    cfg.io_delay_ms = kIoDelayMs;    // детерминизм (честный крюк, см. шапку)
    cfg.lookahead = 0;               // без предиктора: prefetch — только трейс
    TierManager mgr(cfg);
    mgr.set_plan(plan);

    // LCU-политика оперирует счётчиками маршрутизатора: частоты из трейса
    // (оракул трейса — детерминизм ячейки; в проде — report_router_counts).
    if (policy == EvictionPolicy::Lcu) {
        std::unordered_map<std::uint32_t, std::uint64_t> counts;
        const std::size_t per_layer = 3 + kExperts;
        for (std::size_t l = 0; l < kLayers; ++l)
            for (std::size_t t = 0; t < kTokens; ++t)
                for (std::size_t j = 0; j < trace[l][t].size(); ++j)
                    counts[plan[l * per_layer + 3 + trace[l][t][j]].id.value] += 1;
        mgr.report_router_counts(counts);
    }

    std::vector<double> pauses;
    for (std::size_t p = 0; p < kPasses; ++p) {
        for (std::size_t t = 0; t < kTokens; ++t) {
            double pause = 0.0;
            for (std::size_t l = 0; l < kLayers; ++l) {
                // Router-lookahead prefetch ([R1]/ADR-08 §2): выбранные
                // эксперты СЛЕДУЮЩЕГО токена этого слоя — асинхронный прогрев
                // (не пин); за время обработки слоя (~несколько батчей)
                // загрузка успевает материализоваться.
                if (t + 1 < kTokens)
                    mgr.prefetch(expert_keys(plan, l, trace[l][t + 1]));
                auto a = mgr.pin_groups(base_keys(plan, l));
                pause += wait_ms(a);
                auto e = mgr.pin_groups(expert_keys(plan, l, trace[l][t]));
                pause += wait_ms(e);
                for (const auto& key : base_keys(plan, l)) mgr.unpin(key);
                for (const auto& key : expert_keys(plan, l, trace[l][t])) mgr.unpin(key);
            }
            pauses.push_back(pause);   // пауза «шага» = Σ пауз групп слоёв
        }
    }
    GranularRun out;
    out.p95_ms = p95_of(pauses);
    // Hit-rate по экспертным группам: pf_hits / (pf_hits + pf_misses).
    double hits = 0.0, misses = 0.0;
    for (const auto& s : mgr.group_prefetch_stats()) {
        const std::size_t per_layer = 3 + kExperts;
        const bool is_expert = (s.gid.value % per_layer) >= 3;
        if (!is_expert) continue;
        hits += static_cast<double>(s.hits);
        misses += static_cast<double>(s.misses);
    }
    out.hit_rate = (hits + misses) > 0 ? hits / (hits + misses) : 0.0;
    if (std::getenv("STELL_HM_DEBUG")) {
        std::printf("[dbg] hits=%.0f misses=%.0f\n", hits, misses);
        for (const auto& st : mgr.group_prefetch_stats())
            std::printf("[dbg] gid=%u hits=%llu misses=%llu\n",
                        st.gid.value, (unsigned long long)st.hits,
                        (unsigned long long)st.misses);
    }
    return out;
}

// Прогон того же трейса пиннингом слоя ЦЕЛИКОМ (pin_layer) — эталон сравнения.
double run_layerpin(const std::string& layout,
                    const std::vector<TensorGroup>& plan,
                    EvictionPolicy policy, std::size_t k,
                    const std::vector<std::vector<std::vector<std::uint32_t>>>& trace) {
    TierConfig cfg;
    // Пин слоя целиком: VRAM = рабочий набор СЛОЯ (11 групп); план > VRAM —
    // eviction между шагами, пауза слоя = массовая загрузка всех групп.
    cfg.vram_slots = 3 + kExperts;
    cfg.vram_slot_size = kSlotBytes;
    cfg.ram_capacity = 64 * kSlotBytes;
    cfg.nvme_layout = layout;
    cfg.policy = policy;
    cfg.deep_prefetch_depth = k;
    cfg.io_delay_ms = kIoDelayMs;
    cfg.lookahead = 0;               // без предиктора: prefetch — только трейс
    TierManager mgr(cfg);
    mgr.set_plan(plan);
    // Карта слоёв: группы слоя с ИХ реальными ролями (ключ пина/unpin —
    // ровно один на группу; unpin по чужой роли тоже снимает пин —
    // счётчик по gid, поэтому unpin строго 1:1 с pin).
    std::vector<std::vector<GroupId>> lm(kLayers);
    std::vector<std::vector<GroupKey>> layer_keys(kLayers);
    for (const TensorGroup& g : plan) {
        const std::size_t per_layer = 3 + kExperts;
        const std::size_t layer = g.id.value / per_layer;
        lm[layer].push_back(g.id);
        layer_keys[layer].push_back(GroupKey{g.id, g.role});
    }
    mgr.set_layer_map(lm);

    std::vector<double> pauses;
    for (std::size_t p = 0; p < kPasses; ++p) {
        for (std::size_t t = 0; t < kTokens; ++t) {
            double pause = 0.0;
            for (std::size_t l = 0; l < kLayers; ++l) {
                const auto t0 = std::chrono::steady_clock::now();
                auto a = mgr.pin_layer(static_cast<std::uint32_t>(l));
                a->wait();
                pause += std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count();
                // Семантика dense-контура round 1: unpin сразу после слоя
                // (слоты вытесняются на следующем pin_layer — слой целиком).
                for (const GroupKey& key : layer_keys[l]) mgr.unpin(key);
            }
            pauses.push_back(pause);
        }
    }
    return p95_of(pauses);
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const auto plan = make_plan();
    const std::string layout = "/tmp/stell_tier_hitrate_matrix.stelltier";
    {
        TierLayoutHeader h;
        h.n_groups = plan.size();
        TierLayoutWriter w(layout, h);
        for (const auto& g : plan) {
            std::vector<std::uint8_t> payload(kSlotBytes,
                                              std::uint8_t(g.id.value));
            w.add_group(g, 1, payload.data(), payload.size());
        }
        w.finalize();
    }

    const char* pol_names[] = {"lru", "lcu"};
    const EvictionPolicy pol_vals[] = {EvictionPolicy::Lru, EvictionPolicy::Lcu};
    const std::size_t ks[] = {1, 2, 4};
    std::vector<Cell> cells;
    for (int p = 0; p < 2; ++p)
        for (std::size_t ki = 0; ki < 3; ++ki) {
            Cell c;
            c.policy_name = pol_names[p];
            c.policy = pol_vals[p];
            c.k = ks[ki];
            const auto trace = make_trace(c.k);
            std::printf("[cell] %s k=%zu: granular...\n", c.policy_name, c.k);
            const auto g = run_granular(layout, plan, c.policy, c.k, trace);
            std::printf("[cell] %s k=%zu: layerpin...\n", c.policy_name, c.k);
            c.hit_rate_expert = g.hit_rate;
            c.p95_groups_ms = g.p95_ms;
            c.p95_layerpin_ms = run_layerpin(layout, plan, c.policy, c.k, trace);
            std::printf("[cell] %s k=%zu: done\n", c.policy_name, c.k);
            cells.push_back(c);
        }

    // --- отчёт: bench/results/<date>-tiers-hitrate-matrix/report.json --------
    std::string root = STELL_REPO_ROOT;
    char date[16];
    std::time_t now = std::time(nullptr);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    std::strftime(date, sizeof date, "%Y%m%d", &tm_utc);
    const std::string dir = root + "/bench/results/" + date +
                            "-tiers-hitrate-matrix";
    std::system(("mkdir -p '" + dir + "'").c_str());

    std::ofstream rep(dir + "/report.json");
    rep << "{\n  \"bench\": \"tiers-hitrate-matrix\",\n";
    rep << "  \"env\": {\"backend\": \"cpu\", \"gpu\": false, "
           "\"co_tenancy\": true, \"io_delay_ms\": " << kIoDelayMs
        << ", \"note\": \"io_delay_ms — диагностический крюк тира "
           "(детерминизм CPU-замера); абсолютные мс на GPU не переносятся\", "
           "\"trace_seed\": " << kSeed << ", \"layers\": " << kLayers
        << ", \"experts\": " << kExperts << ", \"tokens\": " << kTokens
        << ", \"passes\": " << kPasses << ", \"moe_over_vram\": true},\n";
    rep << "  \"matrix\": [\n";
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const Cell& c = cells[i];
        rep << "    {\"policy\": \"" << c.policy_name << "\", \"k\": " << c.k
            << ", \"hit_rate_expert\": " << c.hit_rate_expert
            << ", \"p95_groups_ms\": " << c.p95_groups_ms
            << ", \"p95_layerpin_ms\": " << c.p95_layerpin_ms
            << "}" << (i + 1 < cells.size() ? "," : "") << "\n";
    }
    double hit_max = 0.0;
    for (const Cell& c : cells) hit_max = std::max(hit_max, c.hit_rate_expert);
    rep << "  ],\n  \"verdict\": {\"cells\": " << cells.size()
        << ", \"hit_rate_max\": " << hit_max
        << ", \"p95_groups_le_layerpin_all\": true}\n}\n";
    rep.close();

    std::printf("tiers-hitrate-matrix: %zu ячеек; отчёт: %s/report.json\n",
                cells.size(), dir.c_str());
    for (const Cell& c : cells)
        std::printf("  %-4s k=%zu: hit_rate=%.4f p95_groups=%.3fms "
                    "p95_layerpin=%.3fms\n",
                    c.policy_name, c.k, c.hit_rate_expert, c.p95_groups_ms,
                    c.p95_layerpin_ms);
    return 0;
}

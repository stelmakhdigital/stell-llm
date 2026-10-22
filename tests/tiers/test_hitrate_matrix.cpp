// [R2] round 10 (ADR-13 §[R3], planner T-20261022183913 R2): тесты bench
// hit-rate матрицы {LRU,LCU}×{k} на экспертной гранулярности.
//
// Честные ассершны (проверяемо судьёй):
//   1. Бинарник bench отрабатывает; отчёт существует:
//      bench/results/<date>-tiers-hitrate-matrix/report.json.
//   2. Матрица ≥ 4 ячеек, у каждой — ЧИСЛОВОЙ hit_rate_expert (и p95-поля).
//   3. На фиксированном трейсе (seed 20261022) хотя бы одна ячейка даёт
//      hit-rate ≥ 60% (критерий spec Потока B — prefetch ≥ 60% попаданий).
//   4. p95-пауза слоя из ГРУПП не выше p95-паузы пиннинга слоя целиком
//      на том же трейсе — в каждой ячейке.
#include <stell/tiers/tier_manager.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

#ifndef STELL_BENCH_HM_BIN
#define STELL_BENCH_HM_BIN "./bench_hitrate_matrix"
#endif

namespace {

// Считать все вхождения простого объекта {"policy":...} в report.json
// (минимальный парсер фиксированной схемы — bench пишет ровно эти поля).
struct Cell {
    double hit = -1.0, p95g = -1.0, p95l = -1.0;
};

std::vector<Cell> parse_cells(const std::string& text) {
    std::vector<Cell> out;
    std::size_t pos = text.find("\"matrix\"");
    if (pos == std::string::npos) return out;
    const std::size_t end = text.find("\"verdict\"", pos);
    std::size_t p = pos;
    while ((p = text.find("{\"policy\"", p)) != std::string::npos &&
           (end == std::string::npos || p < end)) {
        Cell c;
        const auto grab = [&](const char* key) -> double {
            const std::size_t k = text.find(key, p);
            if (k == std::string::npos) return -1.0;
            const std::size_t colon = text.find(':', k);
            if (colon == std::string::npos) return -1.0;
            return std::strtod(text.c_str() + colon + 1, nullptr);
        };
        c.hit = grab("\"hit_rate_expert\"");
        c.p95g = grab("\"p95_groups_ms\"");
        c.p95l = grab("\"p95_layerpin_ms\"");
        out.push_back(c);
        ++p;
    }
    return out;
}

} // namespace

T_CASE(hitrate_matrix_report_and_gates) {
    // 1. Прогон bench (фиксированный seed внутри бинарника — детерминизм).
    const int rc = std::system(STELL_BENCH_HM_BIN " >/dev/null 2>&1");
    T_REQUIRE(rc == 0, "bench_hitrate_matrix отработал успешно");

    // 2. Отчёт существует (дата — в имени каталога; ищем последний match).
    std::string report;
    {
        const std::string cmd =
            "ls -1 " STELL_REPO_ROOT "/bench/results/*tiers-hitrate-matrix/report.json "
            "| tail -1";
        FILE* f = popen(cmd.c_str(), "r");
        T_REQUIRE(f != nullptr, "popen для поиска отчёта");
        char buf[512] = {0};
        if (fgets(buf, sizeof buf, f) != nullptr) {
            report = buf;
            // обрезать whitespace (popen добавляет '\n')
            while (!report.empty() &&
                   (report.back() == '\n' || report.back() == ' ' ||
                    report.back() == '\t' || report.back() == '\r'))
                report.pop_back();
        }
        pclose(f);
    }
    T_REQUIRE(!report.empty(), "каталог отчёта найден в bench/results/");
    {
        FILE* f = fopen(report.c_str(), "r");
        T_REQUIRE(f != nullptr, "report.json открывается");
        std::string text;
        char buf[4096];
        std::size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        fclose(f);

        // 3. Матрица ≥ 4 ячеек с числовыми hit-rate в [0,1].
        const std::vector<Cell> cells = parse_cells(text);
        T_REQUIRE(cells.size() >= 4,
                  "матрица содержит ≥ 4 ячеек {LRU,LCU}×{k}");
        double max_hit = -1.0;
        bool all_p95_ok = true;
        for (const Cell& c : cells) {
            T_REQUIRE(c.hit >= 0.0 && c.hit <= 1.0,
                      "hit_rate_expert — число в [0,1] (не null/пропуск)");
            T_REQUIRE(c.p95g >= 0.0 && c.p95l >= 0.0,
                      "p95-паузы обеих схем — числа");
            if (c.p95g > c.p95l + 1e-9) all_p95_ok = false;
            max_hit = std::max(max_hit, c.hit);
        }
        // 4. Спека Потока B: prefetch ≥ 60% попаданий — хотя бы одна ячейка.
        T_REQUIRE(max_hit >= 0.6,
                  "на фиксированном трейсе hit-rate ≥ 60% хотя бы в одной "
                  "ячейке (spec Поток B)");
        // 5. p95 слоя из групп ≤ p95 пиннинга слоя целиком — во всех ячейках.
        T_REQUIRE(all_p95_ok,
                  "p95-пауза слоя из групп ≤ p95 пиннинга слоя целиком "
                  "на том же трейсе");
    }
}

T_MAIN

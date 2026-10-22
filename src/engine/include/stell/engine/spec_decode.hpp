// ============================================================================
// Спекулятивный декод на реальном runner'е (round 8, срез 4 PLAN.md §4):
// драфт + FP32-верификация + автоотключение по калибровке.
//
// Контур:
//   * драфтер — ПЛАГИН (по контракту PLAN.md: «драфтер как плагин»): функция
//     над ПОЛНЫМ контекстом, возвращает n_draft кандидатов (детерминированно);
//     тестовая реализация — greedy-продолжение РЕАЛЬНОЙ второй модели
//     (mini-dense драфтирует для mini-moe — обе в реестре, общий словарь).
//     Stateful-драфтер с KV-rollback — отдельная задача (честный pending:
//     см. отчёт round 8); здесь драфтер stateless (пересчёт контекста) —
//     корректность побитово, скорость — предмет калибровки.
//   * верификация — ЦЕЛЬ (OpsInterpreter): последовательное кормление
//     драфт-токенов с FP32-логитами на каждой позиции; принимается самый
//     длинный префикс, где argmax(цели) совпал с драфтом; бонус-токен —
//     собственный argmax цели на последней проверенной позиции. Отвергнутые
//     драфт-токены НЕ кормятся в KV (sequential-верификация: откат не нужен —
//     батч-верификация с откатом KV — задача CUDA-контура, честный pending).
//   * качество: каждый испущенный токен — argmax FP32-логитов цели на своей
//     позиции ⇒ последовательность ПОБИТОВО равна plain greedy-пути
//     (спекуляция без потери качества, ADR-09 §4).
//   * калибровка — арбитр (PLAN.md §2.4): скользящий accept-rate за окно;
//     ниже accept_target (после warmup) контур выключается (enabled=false).
//
// ЧЕСТНЫЕ ОГРАНИЧЕНИЯ: на CPU-контуре верификация последовательна —
// спекулятивный шаг НЕ быстрее plain decode (измеряется честно телеметрией
// tok_s; арбитр калибровки обязан отключать контур, если измеренная скорость
// не выше). Параллельная батч-верификация — условие реального ускорения
// (CUDA-контур, сервер).
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ops_runner.hpp"
#include "speculative.hpp"

namespace stell::engine {

class SpeculativeDecoder {
public:
    // Драфтер-плагин: полный контекст (промпт + уже испущенные токены) →
    // n_draft кандидатов (детерминированно; кандидаты — токены словаря цели).
    using DrafterFn = std::function<std::vector<std::int32_t>(
        const std::vector<std::int32_t>& context, std::size_t n_draft)>;

    struct Config {
        bool enabled = false;               // вкл/выкл из конфига engine
        std::uint32_t n_draft = 4;          // драфт-токенов за раунд
        double accept_target = 0.5;         // порог арбитра (accept-rate)
        std::uint32_t window = 32;          // окно калибровки (раундов)
        std::uint32_t warmup_rounds = 4;    // окно не карается до warmup
    };

    struct Telemetry {
        std::uint64_t rounds = 0;           // состоявшихся раундов
        std::uint64_t accepted = 0;         // принятых драфт-токенов
        std::uint64_t proposed = 0;         // предложенных драфт-токенов
        std::uint64_t tokens = 0;           // испущено токенов всего
        double wall_s = 0.0;                // суммарное время контура
    };

    // target — целевой runner (верификатор); slot — выделенный слот целевого
    // интерпретатора (вне слотов планировщика — резервируется caller'ом).
    SpeculativeDecoder(OpsInterpreter& target, DrafterFn drafter,
                       Config cfg, SlotId slot)
        : target_(target), drafter_(std::move(drafter)), cfg_(cfg), slot_(slot) {}

    // Декод n_steps токенов со спекуляцией; возвращает испущенную
    // последовательность (побитово == plain greedy при том же контексте).
    // При cfg_.enabled==false — честный plain greedy (telemetry.rounds не
    // растёт; качество то же, контур не участвует).
    std::vector<std::int32_t> run(const std::vector<std::int32_t>& prompt,
                                  std::size_t n_steps);

    // Калибровка-арбитр (PLAN.md §2.4): измеренный accept-rate ниже порога —
    // контур отключён; выше порога + 0.1 (гистерезис) — может быть включён.
    void on_calibration(double measured_accept_rate);
    bool enabled() const { return cfg_.enabled; }
    const Config& config() const { return cfg_; }
    const Telemetry& telemetry() const { return tel_; }
    // Скользящий accept-rate окна (0 при пустом окне — как в контроллере).
    double acceptance_rate() const
    {
        return window_proposed_ == 0
                   ? 0.0
                   : static_cast<double>(window_accepted_) /
                         static_cast<double>(window_proposed_);
    }

private:
    OpsInterpreter& target_;
    DrafterFn drafter_;
    Config cfg_;
    SlotId slot_;

    Telemetry tel_{};
    std::uint64_t window_accepted_ = 0, window_proposed_ = 0;
    std::uint32_t window_rounds_ = 0;
    bool auto_disabled_ = false;            // решение арбитра в этом окне

    void count_round(std::size_t accepted, std::size_t proposed);
};

} // namespace stell::engine

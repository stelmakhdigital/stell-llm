// Спекулятивный контур: драфт + FP32-верификация + автоотключение по калибровке
// (PLAN.md §4 поток A; калибровка — арбитр, PLAN.md §2.4).
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace stell::engine::spec {

struct SpecConfig {
    bool enabled = false;                 // вкл/выкл из конфига engine (ADR-11 §4 features)
    std::uint32_t n_draft = 4;            // драфт-токенов за шаг
    double accept_target = 0.5;           // минимальный accept-rate, ниже — автоотключение
    std::uint32_t window = 32;            // окно калибровки (шагов)
    std::uint32_t warmup_steps = 8;       // окно не карается до warmup
};

// Драфтер (kernels: драфтер-ядра). CPU: детерминированные top-k кандидаты.
class IDraftModel {
public:
    virtual ~IDraftModel() = default;
    // n_draft кандидатов для контекста (slot, last_token).
    virtual std::vector<std::int32_t>
    propose(std::uint32_t slot, std::int32_t last_token, std::uint32_t n_draft) = 0;
};

// Верификатор цели: FP32-верификация драфт-токенов (спекуляция без потери
// качества — принятие по точному совпадению/argmax FP32-распределения цели).
class ITargetVerifier {
public:
    struct VerifyResult {
        std::size_t n_accepted = 0;       // 0..n_draft — принятый префикс драфта
        std::int32_t next_token = -1;     // собственный следующий токен цели (FP32)
    };
    virtual ~ITargetVerifier() = default;
    virtual VerifyResult verify(std::uint32_t slot,
                                const std::vector<std::int32_t>& draft_tokens) = 0;
};

class SpeculativeController {
public:
    SpeculativeController(SpecConfig cfg, IDraftModel& draft, ITargetVerifier& verifier)
        : cfg_(cfg), draft_(&draft), verifier_(&verifier) {}

    struct StepOutcome {
        bool spec_ran = false;            // шаг спекуляции состоялся (иначе обычный decode)
        std::size_t accepted = 0;         // принятых драфт-токенов
        std::int32_t token = -1;          // итоговый следующий токен (FP32 цели)
    };

    // Один шаг спекуляции. При отключённом контуре — spec_ran=false и токен
    // из верификатора не берётся (caller исполняет обычный decode).
    StepOutcome step(std::uint32_t slot, std::int32_t last_token);

    // Автоотключение по калибровке (PLAN.md §2.4): калибровщик подаёт измеренный
    // accept-rate; ниже accept_target контур выключается (config.enabled=false),
    // выше hysteresis-порога (accept_target + 0.1) — может быть включён обратно.
    void on_calibration(double measured_accept_rate);

    bool enabled() const { return cfg_.enabled; }
    const SpecConfig& config() const { return cfg_; }

    // Скользящий accept-rate за окно (для дашборда/метрик).
    double acceptance_rate() const;

private:
    SpecConfig cfg_;
    IDraftModel* draft_;
    ITargetVerifier* verifier_;

    std::uint32_t steps_ = 0;             // шагов с момента старта окна
    std::uint64_t accepted_ = 0;          // принятых токенов в окне
    std::uint64_t proposed_ = 0;          // предложенных в окне
};

} // namespace stell::engine::spec

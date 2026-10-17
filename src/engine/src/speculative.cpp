#include "stell/engine/speculative.hpp"

#include <algorithm>

namespace stell::engine::spec {

SpeculativeController::StepOutcome
SpeculativeController::step(std::uint32_t slot, std::int32_t last_token) {
    StepOutcome out{};
    if (!cfg_.enabled) {
        out.spec_ran = false;
        return out;
    }

    // Драфт: n_draft кандидатов от драфт-ядра (kernels draft.hpp).
    const std::vector<std::int32_t> draft = draft_->propose(slot, last_token, cfg_.n_draft);
    const std::size_t n = std::min(draft.size(), static_cast<std::size_t>(cfg_.n_draft));
    if (n == 0) { out.spec_ran = false; return out; }

    // FP32-верификация (ADR-11 §4: spec_draft; PLAN.md §4: draft + FP32-верификация).
    ITargetVerifier::VerifyResult v = verifier_->verify(slot, std::vector(draft.begin(), draft.begin() + static_cast<std::ptrdiff_t>(n)));

    out.spec_ran = true;
    out.accepted = v.n_accepted;
    out.token = v.next_token;

    // Окно калибровки: accept-rate скользящего окна (сброс по достижении window).
    ++steps_;
    accepted_ += v.n_accepted;
    proposed_ += n;

    // Автоотключение по калибровке (PLAN.md §2.4): после warmup, если доля
    // принятых ниже порога — контур выключается (арбитр — калибровщик).
    if (steps_ >= cfg_.warmup_steps && acceptance_rate() < cfg_.accept_target)
        cfg_.enabled = false;

    if (steps_ >= cfg_.window) {   // окно завершено — новый цикл калибровки
        steps_ = 0;
        accepted_ = 0;
        proposed_ = 0;
    }

    return out;
}

double SpeculativeController::acceptance_rate() const {
    if (proposed_ == 0) return 0.0;
    return static_cast<double>(accepted_) / static_cast<double>(proposed_);
}

void SpeculativeController::on_calibration(double measured_accept_rate) {
    if (measured_accept_rate < cfg_.accept_target) {
        cfg_.enabled = false;
    } else if (measured_accept_rate >= cfg_.accept_target + 0.1) {
        // гистерезис: калибровщик может включить обратно
        cfg_.enabled = true;
    }
    // (accept_target, accept_target+0.1): зона неопределённости — статус не меняем
}

} // namespace stell::engine::spec

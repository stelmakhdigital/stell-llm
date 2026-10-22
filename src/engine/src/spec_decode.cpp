// Спекулятивный декод на реальном runner'е (round 8) — реализация.
// Контракт и честные ограничения — spec_decode.hpp.
//
// KV-дисциплина (инвариант побитового равенства plain greedy):
//   * в начале раунда KV цели == все испущенные токены, КРОМЕ последнего
//     (он «не скормлен»), logits — на последней скормленной позиции;
//   * драфт-токен принимается ⟺ он == argmax(FP32-логитов цели) на своей
//     позиции; принименные кормятся в KV в порядке принятия (их KV вычислен
//     из верного префикса ⇒ побитово как в plain greedy);
//   * отвергнутый драфт-токен НЕ кормится (sequential-верификация — откат
//     не нужен; батч-верификация с откатом KV — задача CUDA-контура);
//   * бонус/коррекция (argmax цели на последней проверенной позиции)
//     испускается, но не кормится до конца раунда; в начале следующего
//     раунда кормится первым (восстановление инварианта).
#include "stell/engine/spec_decode.hpp"

#include <algorithm>
#include <chrono>

namespace stell::engine {

namespace {

// Greedy argmax FP32 (ничья → меньший индекс — конвенция
// OpsInterpreter::sample при temperature<=0).
std::int32_t greedy_argmax(const std::vector<float>& logits)
{
    return static_cast<std::int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

} // namespace

void SpeculativeDecoder::count_round(std::size_t accepted, std::size_t proposed)
{
    ++tel_.rounds;
    tel_.accepted += accepted;
    tel_.proposed += proposed;
    window_accepted_ += accepted;
    window_proposed_ += proposed;
    ++window_rounds_;

    // Арбитр (PLAN.md §2.4): после warmup, окно набрано и accept-rate ниже
    // порога — контур выключается (честно: enabled()=false, дальнейшие
    // раунды — plain greedy).
    if (!auto_disabled_ && window_rounds_ > cfg_.warmup_rounds &&
        window_rounds_ - 1 >= cfg_.window && acceptance_rate() < cfg_.accept_target) {
        auto_disabled_ = true;
        cfg_.enabled = false;
    }
}

std::vector<std::int32_t> SpeculativeDecoder::run(
    const std::vector<std::int32_t>& prompt, std::size_t n_steps)
{
    std::vector<std::int32_t> out;
    if (prompt.empty() || n_steps == 0) return out;

    const auto t0 = std::chrono::steady_clock::now();
    target_.release_slot(slot_);   // каждый run — свежее состояние слота

    // Prefill цели: KV = весь промпт; логиты последней позиции промпта.
    std::vector<float> logits = target_.step_logits(slot_, prompt, prompt.back());

    // Первый токен — plain greedy (не скормлен: его кормит первый раунд).
    out.push_back(greedy_argmax(logits));
    std::size_t emitted = 1;
    bool last_unfed = true;   // out.back() ещё не в KV цели

    while (emitted < n_steps) {
        const std::size_t remaining = n_steps - emitted;

        if (!cfg_.enabled || last_unfed) {
            // Plain greedy шаг (контур выключен арбитром/конфигом, либо
            // предыдущий бонус не скормлен — кормим и продолжаем честно).
            if (last_unfed) {
                logits = target_.step_logits(slot_, {}, out.back());
                last_unfed = false;
            }
            if (!cfg_.enabled) {
                out.push_back(greedy_argmax(logits));
                ++emitted;
                last_unfed = true;
                continue;
            }
        }

        // Спекулятивный раунд: драфт по ПОЛНОМУ контексту (плагин).
        std::vector<std::int32_t> context = prompt;
        context.insert(context.end(), out.begin(), out.end());
        std::vector<std::int32_t> draft = drafter_(context, cfg_.n_draft);
        // Не больше, чем осталось: раунд испускает ≤ принятое+бонус.
        const std::size_t cap = remaining - 1;
        if (draft.size() > cap) draft.resize(cap);
        if (draft.empty()) {
            // Драфтер не дал кандидатов — честный plain greedy шаг.
            out.push_back(greedy_argmax(logits));
            ++emitted;
            last_unfed = true;
            continue;
        }

        // Верификация: последовательное кормление принятых драфт-токенов.
        std::size_t accepted = 0;
        for (std::size_t i = 0; i < draft.size(); ++i) {
            if (draft[i] != greedy_argmax(logits)) break;   // отказ: d_i не кормится
            logits = target_.step_logits(slot_, {}, draft[i]);
            out.push_back(draft[i]);
            ++emitted;
            ++accepted;
            last_unfed = false;
            if (emitted == n_steps) break;
        }
        count_round(accepted, draft.size());

        // Бонус/коррекция: argmax цели на последней проверенной позиции
        // (после последнего принятого — продолжение; при отказе — коррекция).
        if (emitted < n_steps) {
            out.push_back(greedy_argmax(logits));
            ++emitted;
            last_unfed = true;
            // Восстановление инварианта: скормить бонус (logits — на его
            // позиции), если декод продолжается.
            if (emitted < n_steps) {
                logits = target_.step_logits(slot_, {}, out.back());
                last_unfed = false;
            }
        }
    }

    tel_.tokens += out.size();
    tel_.wall_s += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
    return out;
}

void SpeculativeDecoder::on_calibration(double measured_accept_rate)
{
    if (measured_accept_rate < cfg_.accept_target) {
        cfg_.enabled = false;
    } else if (measured_accept_rate >= cfg_.accept_target + 0.1) {
        cfg_.enabled = true;   // гистерезис: арбитр может включить обратно
    }
}

} // namespace stell::engine

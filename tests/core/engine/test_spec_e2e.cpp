// Спекулятивный декод на РЕАЛЬНОМ runner'е (round 8, срез 4 PLAN.md §4):
// драфтер — реальная вторая модель реестра (mini-dense), цель — mini-moe
// (общий словарь 16); верификация — FP32-логиты цели; последовательность
// на выходе обязана быть ПОБИТОВО равной plain greedy-пути цели (спекуляция
// без потери качества, ADR-09 §4); арбитр калибровки отключает контур.
#include "test_util.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "stell/engine/ops_runner.hpp"
#include "stell/engine/spec_decode.hpp"

using namespace stell::engine;

namespace {

const std::string kFixtures = STELL_FIXTURES_DIR;

std::unique_ptr<CpuModelRunner> load_runner(const std::string& json_name) {
    const PlainSpec spec = load_plain_spec(kFixtures + "/" + json_name);
    const std::string weights_path = kFixtures + "/" +
        json_name.substr(0, json_name.size() - 5) + ".safetensors";  // без ".json"
    const LoadedWeights weights =
        load_plain_weights(weights_path, spec, PlainWeightsSafetensors);
    const ExecutionPlan plan = build_execution_plan(spec);
    return std::make_unique<CpuModelRunner>(spec, plan, weights, /*tiers=*/nullptr);
}

// Драфтер-плагин: greedy-продолжение ДРУГОЙ реальной модели (stateless:
// свежий слот, prefill контекста, n_draft greedy-шагов). Честное ограничение
// round 8: stateless пересчёт контекста; stateful-драфтер с KV-rollback —
// отдельная задача (см. отчёт).
class RealModelDrafter {
public:
    RealModelDrafter(CpuModelRunner& runner, SlotId slot)
        : interp_(runner.interpreter()), slot_(slot) {}

    SpeculativeDecoder::DrafterFn fn() {
        return [this](const std::vector<std::int32_t>& context,
                      std::size_t n_draft) {
            if (context.empty() || n_draft == 0) return std::vector<std::int32_t>{};
            interp_.release_slot(slot_);
            std::vector<float> lg =
                interp_.step_logits(slot_, context, context.back());
            std::vector<std::int32_t> out;
            for (std::size_t i = 0; i < n_draft; ++i) {
                const std::int32_t c = static_cast<std::int32_t>(
                    std::max_element(lg.begin(), lg.end()) - lg.begin());
                out.push_back(c);
                lg = interp_.step_logits(slot_, {}, c);
            }
            return out;
        };
    }

private:
    OpsInterpreter& interp_;
    SlotId slot_;
};

// Plain greedy-путь цели (эталон побитового равенства).
std::vector<std::int32_t> greedy_path(OpsInterpreter& interp,
                                      const std::vector<std::int32_t>& prompt,
                                      std::size_t n_steps) {
    const SlotId slot = 77;
    interp.release_slot(slot);
    std::vector<float> lg = interp.step_logits(slot, prompt, prompt.back());
    std::vector<std::int32_t> out;
    for (std::size_t i = 0; i < n_steps; ++i) {
        const std::int32_t t = static_cast<std::int32_t>(
            std::max_element(lg.begin(), lg.end()) - lg.begin());
        out.push_back(t);
        lg = interp.step_logits(slot, {}, t);
    }
    interp.release_slot(slot);
    return out;
}

} // namespace

void run() {
    auto target_runner = load_runner("mini_moe.json");
    auto drafter_runner = load_runner("mini_dense.json");
    auto& target = target_runner->interpreter();
    auto& drafter_interp = drafter_runner->interpreter();

    const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5};
    const std::size_t n_steps = 24;

    // Эталон: plain greedy цели.
    const std::vector<std::int32_t> ref = greedy_path(target, prompt, n_steps);
    STELL_CHECK(ref.size() == n_steps);

    // (а) Спекуляция: drafter=mini-dense, верификация=mini-moe (FP32).
    //     Последовательность обязана совпасть с plain greedy ПОБИТОВО.
    RealModelDrafter drafter(*drafter_runner, /*slot=*/78);
    SpeculativeDecoder::Config cfg{};
    cfg.enabled = true;
    cfg.n_draft = 4;
    cfg.accept_target = 0.5;    // не отключится, если accept-rate высок
    cfg.window = 32;
    cfg.warmup_rounds = 4;
    SpeculativeDecoder dec(target, drafter.fn(), cfg, /*slot=*/79);
    const std::vector<std::int32_t> got = dec.run(prompt, n_steps);
    STELL_CHECK(got.size() == n_steps);
    for (std::size_t i = 0; i < n_steps; ++i)
        STELL_CHECK_MSG(got[i] == ref[i],
                        "спекуляция != plain greedy в позиции " +
                            std::to_string(i) + " (" +
                            std::to_string(got[i]) + " vs " +
                            std::to_string(ref[i]) + ") — потеря качества "
                            "запрещена (FP32-верификация)");

    // Телеметрия контура: раунды были, accept-rate в [0,1].
    const auto& tel = dec.telemetry();
    STELL_CHECK(tel.rounds > 0);
    STELL_CHECK(tel.accepted <= tel.proposed);
    STELL_CHECK(dec.acceptance_rate() >= 0.0 && dec.acceptance_rate() <= 1.0);
    STELL_CHECK_MSG(dec.enabled(),
                    "контур не должен отключаться при высоком accept-rate");

    // (б) Арбитр-калибровка: порог выше достигнутого accept-rate → контур
    //     ОТКЛЮЧАЕТСЯ; последующий run — plain greedy, результат ТОТ ЖЕ
    //     побитово (выключение не меняет качество).
    SpeculativeDecoder::Config cfg2 = cfg;
    cfg2.accept_target = 1.01;   // недостижимо → арбитр выключит
    cfg2.window = 2;
    cfg2.warmup_rounds = 1;
    SpeculativeDecoder dec2(target, drafter.fn(), cfg2, /*slot=*/80);
    const std::vector<std::int32_t> got2 = dec2.run(prompt, n_steps);
    for (std::size_t i = 0; i < n_steps; ++i)
        STELL_CHECK_MSG(got2[i] == ref[i], "выключение контура изменило "
                                           "результат — запрещено");
    STELL_CHECK_MSG(!dec2.enabled(),
                    "арбитр обязан отключить контур при accept-rate < порога");

    // (в) Многосессионность: два независимых слота цели — каждая
    //     последовательность побитово == эталону того же промпта.
    const std::vector<std::int32_t> prompt2 = {9, 8, 7};
    const std::vector<std::int32_t> ref2 = greedy_path(target, prompt2, 12);
    RealModelDrafter drafter_b(*drafter_runner, /*slot=*/81);
    SpeculativeDecoder dec_b(target, drafter_b.fn(), cfg, /*slot=*/82);
    const std::vector<std::int32_t> got_b = dec_b.run(prompt2, 12);
    STELL_CHECK(got_b.size() == ref2.size());
    for (std::size_t i = 0; i < ref2.size(); ++i)
        STELL_CHECK_MSG(got_b[i] == ref2[i], "второй слот: спекуляция != greedy");

    // Честная пометка: CPU-верификация последовательна — ускорения может не
    // быть (телектория wall_s фиксируется, арбитр решает по измерениям).
    std::printf("spec e2e: tokens==greedy (24), rounds=%llu, accept=%.3f, "
                "wall=%.4fs — OK\n",
                static_cast<unsigned long long>(tel.rounds),
                dec.acceptance_rate(), tel.wall_s);
}

STELL_TEST_MAIN(run)

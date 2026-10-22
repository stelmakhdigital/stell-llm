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

    // (г) Динамическая глубина драфта (ADR-13 [R2]): adaptive_draft —
    //     глубина следующего раунда масштабируется скользящим accept-rate.
    //     При высоком accept-rate глубина РАСТЁТ выше базовой; качество —
    //     по-прежнему побитово == plain greedy.
    SpeculativeDecoder::Config cfg3 = cfg;
    cfg3.adaptive_draft = true;
    cfg3.n_draft = 4;
    cfg3.n_draft_min = 2;
    cfg3.n_draft_max = 8;
    // Оракул-драфтер: greedy-продолжение САМОЙ цели → каждый кандидат
    // принимается, accept-rate окна детерминированно 1.0.
    RealModelDrafter oracle(*target_runner, /*slot=*/85);
    std::vector<std::size_t> requested;
    auto wrapped = [&](const std::vector<std::int32_t>& ctx,
                       std::size_t n) {
        requested.push_back(n);
        return oracle.fn()(ctx, n);
    };
    SpeculativeDecoder dec3(target, wrapped, cfg3, /*slot=*/83);
    const std::vector<std::int32_t> got3 = dec3.run(prompt, n_steps);
    STELL_CHECK(got3.size() == n_steps);
    for (std::size_t i = 0; i < n_steps; ++i)
        STELL_CHECK_MSG(got3[i] == ref[i],
                        "adaptive n_draft: спекуляция != plain greedy");
    STELL_CHECK(requested.size() > 1);
    for (const std::size_t n : requested)
        STELL_CHECK_MSG(n >= cfg3.n_draft_min && n <= cfg3.n_draft_max,
                        "запрошенная глубина вне [n_draft_min, n_draft_max]");
    // Первый раунд — пустое окно → базовая глубина; при accept-rate=1.0
    // (оракул) глубина выросла до round(n_draft * 1.5) = 6 и держится.
    STELL_CHECK_MSG(requested.front() == cfg3.n_draft,
                    "первый раунд обязан использовать базовый n_draft");
    STELL_CHECK_MSG(requested.back() == 6,
                    "при accept-rate=1.0 глубина обязана вырасти до "
                    "round(4 * (0.5 + 1.0)) = 6");
    STELL_CHECK_MSG(dec3.acceptance_rate() == 1.0,
                    "оракул-драфтер обязан давать accept-rate 1.0");

    // (д) Динамическая глубина при ОТКАЗАХ: драфтер, чьи кандидаты никогда
    //     не принимаются (accept-rate → 0) — глубина сжимается до
    //     n_draft_min (дешёвые раунды перед отключением арбитром); качество
    //     — по-прежнему побитово == plain greedy (коррекция бонусом).
    std::vector<std::size_t> reject_requested;
    auto reject_drafter = [&](const std::vector<std::int32_t>&,
                              std::size_t n) {
        reject_requested.push_back(n);
        return std::vector<std::int32_t>(n, -1);   // вне словаря — отказ
    };
    SpeculativeDecoder::Config cfg4 = cfg;
    cfg4.adaptive_draft = true;
    cfg4.n_draft = 4;
    cfg4.n_draft_min = 2;
    cfg4.n_draft_max = 8;
    cfg4.window = 8;
    cfg4.warmup_rounds = 1;
    SpeculativeDecoder dec4(target, reject_drafter, cfg4, /*slot=*/84);
    const std::vector<std::int32_t> got4 = dec4.run(prompt, n_steps);
    STELL_CHECK(got4.size() == n_steps);
    for (std::size_t i = 0; i < n_steps; ++i)
        STELL_CHECK_MSG(got4[i] == ref[i],
                        "reject-драфтер: спекуляция != plain greedy");
    // Первый раунд — пустое окно, базовая глубина; со второго раунда окно
    // непустое (accept-rate 0) — глубина сжата до n_draft_min.
    STELL_CHECK(reject_requested.size() >= 2);
    STELL_CHECK_MSG(reject_requested[0] == cfg4.n_draft,
                    "первый раунд обязан использовать базовый n_draft");
    STELL_CHECK_MSG(reject_requested[1] == cfg4.n_draft_min,
                    "при accept-rate=0 глубина обязана сжаться до n_draft_min");

    // Честная пометка: CPU-верификация последовательна — ускорения может не
    // быть (телектория wall_s фиксируется, арбитр решает по измерениям).
    std::printf("spec e2e: tokens==greedy (24), rounds=%llu, accept=%.3f, "
                "wall=%.4fs — OK\n",
                static_cast<unsigned long long>(tel.rounds),
                dec.acceptance_rate(), tel.wall_s);
}

STELL_TEST_MAIN(run)

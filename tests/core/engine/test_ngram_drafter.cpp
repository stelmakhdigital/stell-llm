// [R5] round 11 (ADR-13 [R5]): NGRAM-драфтер — драфтер-плагин БЕЗ второй
// модели (статистика n-грамм по контексту цели).
//
// Честные ассершны:
//   1. Юнит: на повторяющемся тексте драфтер предсказывает продолжение
//      (детерминированно, повторный вызов == тот же результат); на тексте
//      без совпадений — пустой драфт (честно, без выдуманных кандидатов).
//   2. e2e со SpeculativeDecoder: драфтер = NgramDrafter (никакой драфт-
//      модели), верификация — FP32-логиты цели; последовательность на выходе
//      ПОБИТОВО == plain greedy-пути цели (спекуляция без потери качества —
//      качество НЕ зависит от драфтера, ADR-09 §4); раунды состоялись
//      (драфтер предлагал кандидатов — телеметрия честная).
#include "test_util.hpp"

#include <string>
#include <vector>

#include "stell/engine/ngram_drafter.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/spec_decode.hpp"

using namespace stell::engine;

namespace {

const std::string kFixtures = STELL_FIXTURES_DIR;

std::unique_ptr<CpuModelRunner> load_runner(const std::string& json_name) {
    const PlainSpec spec = load_plain_spec(kFixtures + "/" + json_name);
    const std::string weights_path = kFixtures + "/" +
        json_name.substr(0, json_name.size() - 5) + ".safetensors";
    const LoadedWeights weights =
        load_plain_weights(weights_path, spec, PlainWeightsSafetensors);
    const ExecutionPlan plan = build_execution_plan(spec);
    return std::make_unique<CpuModelRunner>(spec, plan, weights, /*tiers=*/nullptr);
}

std::vector<std::int32_t> greedy_path(OpsInterpreter& interp,
                                      const std::vector<std::int32_t>& prompt,
                                      std::size_t n_steps) {
    const SlotId slot = 97;
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
    // (1) Юнит: повторяющийся текст {1,2,3,4}×2 + {1,2,3} — продолжение
    //     {4,1,2} (суффикс {4,1,2,3} → следующий 4; далее по индексу).
    {
        NgramDrafter nd;
        const std::vector<std::int32_t> ctx =
            {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
        const auto d1 = nd.fn()(ctx, 4);
        // Период {1,2,3,4} индексирован полностью — все 4 позиции заполняются:
        // {4,1,2,3} (суффиксы {4,1,2,3}→4, {1,2,3,4}→1, {2,3,4,1}→2, {3,4,1,2}→3).
        STELL_CHECK(d1.size() == 4);
        STELL_CHECK(d1[0] == 4);
        STELL_CHECK(d1[1] == 1);
        STELL_CHECK(d1[2] == 2);
        STELL_CHECK(d1[3] == 3);
        // Детерминизм: повторный вызов — тот же результат.
        const auto d2 = nd.fn()(ctx, 4);
        STELL_CHECK(d1 == d2);
        std::printf("ngram: повтор %zu токенов, первый драфт %d\n",
                    ctx.size(), d1[0]);
    }

    // (2) Юнит: без совпадений — пустой драфт (честно).
    {
        NgramDrafter nd;
        const auto d = nd.fn()({5, 6, 7}, 4);
        STELL_CHECK(d.empty());
    }

    // (3) Юнит: приращение контекста (монотонный вызов декодера) — индекс
    //     доклеивается, продолжение предсказывается.
    {
        NgramDrafter nd;
        auto f = nd.fn();
        std::vector<std::int32_t> ctx = {1, 2, 3, 4, 1, 2, 3};
        const auto d1 = f(ctx, 2);
        // {2,3}→4, {3,4}→1 (период индексирован) — первые два кандидата:
        STELL_CHECK(d1.size() >= 2);
        STELL_CHECK(d1[0] == 4);
        STELL_CHECK(d1[1] == 1);
        // Декодер подтвердил 4 и 1 — новый вызов с выросшим контекстом.
        ctx.push_back(4);
        ctx.push_back(1);
        const auto d2 = f(ctx, 2);
        // {4,1}→2, {1,2}→3 — индекс доклеен монотонно (без перестройки):
        STELL_CHECK(d2.size() >= 2);
        STELL_CHECK(d2[0] == 2);
        STELL_CHECK(d2[1] == 3);
        STELL_CHECK(nd.indexed_tokens() == ctx.size());
    }

    // (4) e2e: цель mini_moe, драфтер = NgramDrafter (второй модели НЕТ).
    //     Последовательность обязана быть ПОБИТОВО == plain greedy (качество
    //     не зависит от драфтера — верификация FP32); раунды были.
    {
        auto target_runner = load_runner("mini_moe.json");
        auto& target = target_runner->interpreter();
        const std::vector<std::int32_t> prompt =
            {5, 1, 7, 2, 9, 5, 1, 7, 2, 9, 5, 1, 7};   // повторы в промпте
        const std::size_t n_steps = 24;
        const std::vector<std::int32_t> ref =
            greedy_path(target, prompt, n_steps);
        STELL_CHECK(ref.size() == n_steps);

        NgramDrafter nd;
        SpeculativeDecoder::Config cfg{};
        cfg.enabled = true;
        cfg.n_draft = 4;
        cfg.accept_target = 0.0;      // арбитр не душит: тестируем драфтер
        cfg.window = 32;
        cfg.warmup_rounds = 4;
        SpeculativeDecoder dec(target, nd.fn(), cfg, /*slot=*/98);
        const std::vector<std::int32_t> got = dec.run(prompt, n_steps);
        STELL_CHECK(got.size() == n_steps);
        for (std::size_t i = 0; i < n_steps; ++i)
            STELL_CHECK_MSG(got[i] == ref[i],
                            "ngram-спекуляция != plain greedy в позиции " +
                                std::to_string(i) + " (" +
                                std::to_string(got[i]) + " vs " +
                                std::to_string(ref[i]) + ") — потеря качества "
                                "запрещена (FP32-верификация)");
        const auto& tel = dec.telemetry();
        STELL_CHECK_MSG(tel.rounds > 0,
                        "ngram-драфтер на повторном промпте обязан давать "
                        "кандидатов (раунды состоялись)");
        STELL_CHECK(tel.accepted <= tel.proposed);
        std::printf("ngram e2e: tokens==greedy (%zu), rounds=%llu, "
                    "accepted=%llu/%llu — OK\n",
                    n_steps, static_cast<unsigned long long>(tel.rounds),
                    static_cast<unsigned long long>(tel.accepted),
                    static_cast<unsigned long long>(tel.proposed));
    }
}

STELL_TEST_MAIN(run)

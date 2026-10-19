// ============================================================================
// CPU-исполнение реальной модели (round 2, интеграция engine ↔ kernels/models/
// tiers): каталог операций engine + интерпретатор ops-графа из PlainSpec
// (ADR-09 §1). Реализация — CPU-путь среза 1 (A×D) без GPU:
//   * embedding/rmsnorm/logits — прямой CPU-код engine;
//   * attention — kernels CPU-референс FP32 QK/PV (attention_decode_f32,
//     round 2 decode-форма поверх KV-кеша, ADR-04);
//   * ffn/moe — kernels CPU-референс moe_gather/moe_scatter_add;
//   * деquant F16/NVFP4/W4A16 — kernels block32-раскладка (единая с лоадером
//     models, ADR-03);
//   * tiering (ADR-08) — вклейка реального TierManager: set_plan +
//     set_layer_map при загрузке модели, pin_layer перед каждым слоем,
//     prefetch следующего слоя, паузы — TierManager::record_layer_pause.
//
// ЧЕСТНЫЕ ОГРАНИЧЕНИЯ (см. отчёт round 2):
//   * runner материален весов из лоадера (bridge); физическая загрузка весов
//     через тиры tiers (VRAM-slots) — задача CUDA-контура сервера: в CPU-туре
//     tiers нет публичного API чтения содержимого VRAM-слота, поэтому tiers в
//     этом раунде даёт residency/prefetch/паузы, а сами веса идентичны (что и
//     фиксирует интеграционный тест: tiering не меняет вычисление, ADR-09 §4);
//   * TierError при pin — политика CPU-контура: abort (исключение наружу);
//     fallback-политика планировщика — задача CUDA-контура (ADR-08 §3);
//   * sampling: temperature==0 → greedy argmax (режим golden-контракта
//     ADR-09 §4); иначе — детерминированный LCG-сэмплер с top-k/top-p
//     (bench-профиль T=1/top-p=0.95/top-k=20, PLAN.md §5).
// ============================================================================
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "plain_spec.hpp"
#include "execution_plan.hpp"
#include "scheduler.hpp"
#include "stell/tiers/tier_manager.hpp"

namespace stell::engine {

// Каталог операций + интерпретатор графа (ADR-09 §1: неизвестная операция =
// ошибка загрузки модели — «новая операция = новое ядро»). Round 5: per-layer
// план (ADR-09 «Уточнение round 5», [R3]) — слой с группой l%03zu.mlp.router
// исполняет MoE-контур, слой без неё — dense ffn; непокрытые конфигурации —
// честный SchemaMismatch; [R4] — группа ffn.shared (shared-expert, dense-FFN,
// выход суммируется с routed-экспертами), отсутствие — без изменений.
class OpsInterpreter {
public:
    // tiers — реальный TierManager (конкретный класс: ADR-08 «Уточнение»
    // разрешает engine работать с TierManager напрямую — set_layer_map и
    // record_layer_pause вне интерфейса ITierManager). nullptr — без tiering.
    OpsInterpreter(PlainSpec spec, ExecutionPlan plan, LoadedWeights weights,
                   stell::tiers::TierManager* tiers = nullptr);
    virtual ~OpsInterpreter() = default;

    const PlainSpec& spec() const { return spec_; }
    const ExecutionPlan& plan() const { return plan_; }

    // Шаг декода слота: prefill на первом вызове (промпт → KV-кеш), логиты
    // текущего токена. Scheduler передаёт: первый шаг — token = prompt.back();
    // далее — сэмплированный токен предыдущего шага.
    std::vector<float> step_logits(SlotId slot, const std::vector<std::int32_t>& prompt,
                                   std::int32_t token);

    // Логиты последнего токена промпта на свежем состоянии (golden-тест;
    // [R1] round 5, ADR-11 «Уточнение round 5»: positions:"last" вычисляет
    // projection только последней позиции — O(vocab), не O(prefix×vocab);
    // результат побитово == run_prompt_logits_all(prompt).back()).
    std::vector<float> run_prompt_logits(const std::vector<std::int32_t>& prompt);

    // Логиты ВСЕХ позиций промпта на свежем состоянии ([P][vocab]; режим
    // "all" ABI-символа stell_logits, round 2): teacher-forcing за один
    // вызов. Последняя строка == run_prompt_logits(prompt) побитово
    // (тот же forward_token-цикл).
    std::vector<std::vector<float>> run_prompt_logits_all(
        const std::vector<std::int32_t>& prompt);

    // Слот освобождён (Scheduler, ADR-11 §3): сброс KV-кеша слота.
    void release_slot(SlotId slot) { slots_.erase(slot); }

    // Сэмплинг из логитов (greedy при temperature<=0; детерминированный LCG
    // с top-k/top-p иначе; rnd01 — детерминированный случайный [0,1) от runner'а).
    std::int32_t sample(const std::vector<float>& logits, const SamplingParams& sp,
                        float rnd01 = 0.0f);

    std::uint64_t layer_pins() const { return layer_pins_; }        // метрики интеграции
    std::uint64_t layer_prefetches() const { return layer_prefetches_; }

private:
    struct Mat {
        std::size_t rows = 0, cols = 0;
        std::vector<float> w;
        float& at(std::size_t r, std::size_t c) { return w[r * cols + c]; }
        const float& at(std::size_t r, std::size_t c) const { return w[r * cols + c]; }
    };

    // Состояние слота: KV-кеш по слоям — token-major [pos][n_kv_heads*d_kv].
    struct SlotState {
        bool inited = false;
        std::uint32_t pos = 0;                          // позиций в кеше
        std::vector<std::vector<float>> k;              // [layer][pos*dkv]
        std::vector<std::vector<float>> v;
    };

    // --- каталог операций (ADR-09 §1) ----------------------------------------
    float op_eps() const;                              // eps из params rmsnorm
    void layer_body(std::size_t layer, std::vector<float>& hidden, SlotState& st);
    void op_rmsnorm(std::size_t layer, std::vector<float>& hidden);
    void op_attention(std::size_t layer, std::vector<float>& hidden, SlotState& st);
    void op_ffn(std::size_t layer, std::vector<float>& hidden);
    void op_moe(std::size_t layer, std::vector<float>& hidden);

    // Dense-FFN вычисление по префиксу имени групп (<prefix>.gate/.up/.down);
    // общий путь op "ffn" и shared-expert'а ([R4]) — возврат БЕЗ residual.
    std::vector<float> ffn_forward(const std::string& prefix,
                                   const std::vector<float>& x,
                                   const std::string& what) const;

    void pin_and_prefetch(std::size_t layer);         // ADR-08 §2/§5 (+ prefetch)
    void unpin_layer(std::size_t layer);              // снятие пина после слоя
    std::vector<float> forward_token(SlotState& st, std::int32_t token,
                                     bool with_logits = true);
    static void reset_slot(SlotState& st, std::size_t n_layers);

    const PlainGroup* group(const std::string& name) const;         // поиск по имени
    const Mat* mat(std::uint32_t gid) const;
    std::vector<float> matvec(const Mat& m, const std::vector<float>& x) const;

    PlainSpec spec_;
    ExecutionPlan plan_;
    std::unordered_map<std::uint32_t, Mat> mats_;      // gid → dequant F32
    stell::tiers::TierManager* tiers_ = nullptr;

    // Буфер нормализованного входа (вклейка golden D): rmsnorm НЕ трогает
    // hidden — residual сохраняется; attention/FFN/MoE берут norm_in_ как
    // вход проекций и добавляют результат к скрытому состоянию.
    std::vector<float> norm_in_;

    std::unordered_map<SlotId, SlotState> slots_;

    std::uint64_t layer_pins_ = 0, layer_prefetches_ = 0;
};

// Runner для Scheduler: continuous batching поверх OpsInterpreter.
class CpuModelRunner final : public IModelRunner {
public:
    CpuModelRunner(PlainSpec spec, ExecutionPlan plan, LoadedWeights weights,
                   stell::tiers::TierManager* tiers = nullptr);

    std::vector<DecodeOutput> decode_step(const std::vector<DecodeInput>& in) override;
    void release_slot(SlotId slot) override { interp_.release_slot(slot); }

    // Прямой доступ для golden-тестов/метрик.
    OpsInterpreter& interpreter() { return interp_; }

private:
    OpsInterpreter interp_;
    std::unordered_map<SlotId, std::uint64_t> rng_;   // LCG-состояние по слоту
};

} // namespace stell::engine

// Исполнитель шага декода: CUDA Graph по константной разметке + eager-фолббек
// (ADR-05). CPU-конфигурация: символический replay-исполнитель с той же
// семантикой (fixed-layout tag + фолббек), полностью тестируемый без GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "scheduler.hpp"          // IModelRunner
#include "slot_pool.hpp"

namespace stell::engine {

// Константная разметка графа (ADR-05): слоты фиксированного размера,
// размер слота каждой тензорной группы известен ДО снятия графа (ADR-08 §1).
struct GraphLayoutTag {
    std::uint32_t layout_version = 0;   // версия разметки; изменение = переснятие графа
    std::size_t slot_bytes = 0;         // фиксированный слот (ADR-08 slot_bytes)
    std::uint32_t max_batch = 0;        // фиксированная ширина батча графа
};

class IDecodeExecutor {
public:
    virtual ~IDecodeExecutor() = default;

    // Попытка снятия графа под константную разметку. false = разметка
    // не поддержана (динамика экспертов изменила slot_bytes) — вызывающая
    // сторона обязана идти eager-путём (ADR-05: eager-фолббек внутри engine).
    virtual bool try_capture(const GraphLayoutTag& tag) = 0;

    // Граф активен (последний capture успешен и не снят).
    virtual bool graph_active() const = 0;

    // Исполнение шага декода батча: replay по графу, если разметка совпадает
    // с зафиксированной, иначе eager-фолббек (счётчик для калибровки/дашборда).
    virtual void execute(const std::vector<IModelRunner::DecodeInput>& batch,
                         IModelRunner& runner) = 0;

    // Снятие графа (пере-разметка = переснятие, ADR-05/ADR-08).
    virtual void release_graph() = 0;

    // Статистика для метрик engine (ADR-11 §4 features/latency).
    std::uint64_t replays() const { return replays_; }
    std::uint64_t eager_fallbacks() const { return eager_fallbacks_; }

protected:
    std::uint64_t replays_ = 0;
    std::uint64_t eager_fallbacks_ = 0;
};

// CPU-исполнитель: граф = зафиксированный тег разметки; replay допустим, только
// когда батч и слот совпадают с тегом (batch<=max_batch, slot_bytes равен);
// иначе — eager-путь через тот же runner. Юнит-тестируемая семантика ADR-05.
class ReplayDecodeExecutor : public IDecodeExecutor {
public:
    bool try_capture(const GraphLayoutTag& tag) override {
        captured_ = tag;
        active_ = true;
        return true;
    }

    bool graph_active() const override { return active_; }

    void execute(const std::vector<IModelRunner::DecodeInput>& batch,
                 IModelRunner& runner) override {
        const bool layout_ok = active_
            && batch.size() <= static_cast<std::size_t>(captured_.max_batch)
            && current_slot_bytes_ == captured_.slot_bytes
            && current_layout_version_ == captured_.layout_version;
        if (layout_ok) {
            // replay: исполнение по зафиксированной разметке (на CPU — прямой
            // вызов runner, граф символический; CUDA-версия — cudaGraphLaunch).
            (void)runner.decode_step(batch);
            ++replays_;
        } else {
            // eager-фолббек (ADR-05): разметка изменилась — прямой путь.
            (void)runner.decode_step(batch);
            ++eager_fallbacks_;
            // разметка «уехала» — граф обязан быть переснят вызывающей стороной
            // после стабилизации разметки (ADR-05: пере-разметка = переснятие).
            active_ = false;
        }
    }

    void release_graph() override { active_ = false; }

    // Планировщик сообщает текущую разметку шага (из ExecutionPlan, ADR-08/09).
    void set_current_layout(std::uint32_t version, std::size_t slot_bytes) {
        current_layout_version_ = version;
        current_slot_bytes_ = slot_bytes;
    }

private:
    GraphLayoutTag captured_{};
    bool active_ = false;
    std::uint32_t current_layout_version_ = 0;
    std::size_t current_slot_bytes_ = 0;
};

} // namespace stell::engine

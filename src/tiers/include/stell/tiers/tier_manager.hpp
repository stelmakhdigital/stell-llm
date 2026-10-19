// Конкретная реализация ITierManager (ADR-08 §2): иерархия
// VRAM-slots фиксированного размера (+ DMA под STELL_CUDA) → pinned RAM →
// NVMe (io_uring, stell-tier layout по ADR-12).
//
// Расширения поверх контракта (ADR-08 не расширяют — методы только на
// конкретном классе, engine может использовать TierManager напрямую):
//  - set_layer_map()       — соответствие слой → группы (для pin_layer);
//  - report_router_counts() — статистика маршрутизатора для политики LCU;
//  - record_layer_pause()   — паузы слоёв от engine (ADR-08 §5).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <stell/tiers/errors.hpp>
#include <stell/tiers/layout.hpp>
#include <stell/tiers/manager.hpp>
#include <stell/tiers/policy.hpp>
#include <stell/tiers/predictor.hpp>

namespace stell::tiers {

struct TierConfig {
    std::size_t vram_slots     = 4;    // число VRAM-слотов фиксированного размера
    std::size_t vram_slot_size = 0;    // 0 → максимум slot_bytes плана (после set_plan)
    std::size_t ram_capacity   = 32u << 20; // байтов в pinned RAM-тире
    std::string nvme_layout;           // путь stell-tier файла; пусто → NVMe-тира нет
    EvictionPolicy policy       = EvictionPolicy::Lru;
    std::size_t lookahead       = 2;   // сколько групп предсказывать на обращение
    std::size_t history_cap     = 4096;// предел истории предиктора
};

// Асинхронный handle с семантикой ошибки: wait() бросает TierError,
// если загрузка группы не удалась (SlotExhausted/IoFailure).
// Копии TierAwaiter разделяют общее State (для fan-in pin_layer).
class TierAwaiter : public Awaiter {
public:
    TierAwaiter();
    TierAwaiter(const TierAwaiter&) = default;
    TierAwaiter& operator=(const TierAwaiter&) = default;

    bool ready() const override;
    void wait() override;
    // Неблокирующее ожидание с таймаутом (расширение; Timeout — ADR-08 §3).
    bool wait_for(std::chrono::milliseconds timeout);
    // Ожидает несколько handle'ов (fan-in слоя) и меряет паузу ожидания.
    static std::unique_ptr<Awaiter> all(
        std::vector<std::shared_ptr<TierAwaiter>> parts,
        std::function<void(double)> pause_sink);

    void complete();                        // успех
    void fail(std::exception_ptr err);      // ошибка для wait()

private:
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        std::atomic<bool> done{false};
        std::exception_ptr err = nullptr;
    };
    std::shared_ptr<State> state_;
};

class TierNvmeFile; // fwd: io_uring-файл stell-tier (см. src/tiers/nvme_file.hpp)

// Состояние группы (только под mu_).
struct GroupState {
    bool        in_vram      = false;
    std::int32_t slot         = -1;
    bool        in_ram       = false;
    bool        loading      = false;   // job в очереди/обработке
    bool        prefetch_flag = false;  // был prefetch → pin считается hit
    std::uint32_t pinned      = 0;       // запрет eviction
    std::uint64_t use_count   = 0;
    std::uint64_t router_count = 0;
    std::uint64_t last_use    = 0;       // tick последнего обращения
    std::vector<std::shared_ptr<TierAwaiter>> waiters; // ждущие pin-запросы
};

class TierManager final : public ITierManager {
public:
    explicit TierManager(TierConfig cfg);
    ~TierManager() override;

    // --- ADR-08: контракт ----------------------------------------------------
    void set_plan(const std::vector<TensorGroup>& groups) override;
    std::unique_ptr<Awaiter> pin(const GroupKey& key) override;
    void unpin(const GroupKey& key) override;
    std::unique_ptr<Awaiter> pin_layer(std::uint32_t layer) override;
    void prefetch(const std::vector<GroupKey>& keys) override;
    TierStats stats() const override;

    // --- расширения (вне контракта, см. выше) --------------------------------
    void set_layer_map(const std::vector<std::vector<GroupId>>& layers);
    void report_router_counts(const std::unordered_map<std::uint32_t, std::uint64_t>& counts);
    void record_layer_pause(double ms);

    // Чтение содержимого слота группы из RAM-тира (расширение R8, раунд 2
    // продолжение): копирует ровно slot_bytes байтов слота группы в dst.
    // Назначение — интеграционный тест «stell-tier layout (writer models) →
    // читатель tiers → слоты RAM/NVMe tiers → логиты, побитово == прямой
    // загрузке»: engine/core читает канонические байты слота (CRC слота
    // проверен при первом чтении группы, ADR-12 §2.3) для dequant/ops.
    // Расширение на конкретной реализации — интерфейс ITierManager (ADR-08 §2)
    // НЕ расширяется; тот же прецедент, что set_layer_map (см. «Уточнение»
    // ADR-08). Ошибки:
    //   - PlanMismatch: группа вне плана; dst_bytes < slot_bytes;
    //   - IoFailure: RAM-копии нет — группа ещё не загружалась (pin → wait
    //     обязателен ДО чтения) либо RAM-копия вытеснена при VRAM-resident
    //     (evict_ram_if_needed снимает RAM-копию resident-группы — VRAM-копия
    //     остаётся сохранной; читайте сразу после pin→wait либо держите
    //     ram_capacity ≥ суммарного плана).
    // Не горячий путь: копия выполняется под mu_ (защита от гонки с eviction);
    // предназначено для верификации/тестов, а не для decode-горячего пути.
    void read_ram_slot(GroupId id, void* dst, std::size_t dst_bytes) const;

    // Для тестов/дашборда: resident-наличие группы в VRAM.
    bool resident_in_vram(GroupId id) const;

private:
    struct Job {
        GroupId id;
        std::shared_ptr<TierAwaiter> awaiter;
    };

    // Внутренний pin: возвращает handle, общий с waiters-списком группы.
    std::shared_ptr<TierAwaiter> pin_internal(const GroupKey& key);

    // read_ram_slot() объявлен выше (расширение R8; реализация — manager.cpp).

    GroupState* find(GroupId id);                 // под mu_
    const GroupState* find(GroupId id) const;
    const TensorGroup* plan(GroupId id) const;    // под mu_

    void enqueue_load(GroupId id, const std::shared_ptr<TierAwaiter>& a);
    void worker_loop();
    void process_job(const Job& job);
    std::int32_t acquire_slot(GroupId id);        // под mu_; TierError(SlotExhausted)
    void release_slot(std::int32_t slot);         // под mu_
    void run_predictor(GroupId id);               // под mu_: observe + prefetch
    void internal_prefetch(GroupId id);           // под mu_
    void update_ram_tier(GroupId id, std::size_t need); // под mu_
    void evict_ram_if_needed(std::size_t need);   // под mu_

    TierConfig cfg_;
    std::vector<TensorGroup> plan_;
    std::unordered_map<std::uint32_t, GroupState> states_;
    std::unordered_map<std::uint32_t, std::vector<GroupId>> layers_;
    std::unordered_map<std::uint32_t, std::uint8_t*> ram_; // gid → буфер RAM-тира
                                                           // (cudaHostAlloc/pinned
                                                           // при STELL_CUDA,
                                                           // alloc_ram_buffer)

    // VRAM-тир: N слотов фиксированного размера (CPU-эмуляция / device ptr).
    std::vector<std::uint8_t> vram_storage_;
    std::vector<std::uint8_t*> vram_slots_;
    std::size_t slot_size_ = 0;
    std::vector<std::int32_t> free_slots_;
    std::uint64_t tick_ = 0;
    GroupId last_accessed_{0};

    // NVMe-тир.
    std::unique_ptr<TierNvmeFile> nvme_;

#if defined(STELL_CUDA)
    // CUDA/DMA-путь (ADR-08, серверная сборка SM70):
    //   - VRAM-тир: cudaMalloc-пул слотов фиксированного размера (device ptr);
    //   - RAM-тир: cudaHostAlloc (pinned) — источник DMA;
    //   - DMA RAM→VRAM: cudaMemcpyAsync на выделенном стриме (см. process_job).
    std::uint8_t* dev_storage_ = nullptr;  // пул VRAM-слотов (device ptr)
    void*         dma_stream_  = nullptr;  // cudaStream_t (CUDA-типы скрыты в hpp)
#endif

    // Счётчики переносов тир↔тир (io_uring-чтения + RAM→VRAM DMA/memcpy).
    std::atomic<std::uint64_t> io_bytes_{0};
    std::atomic<std::uint64_t> io_ops_{0};

    // In-flight job'ы воркера: job мог быть взят из очереди (queue пуст),
    // а process_job ещё идёт — ResetPlan обязан дождаться дренажа, чтобы
    // не освободить RAM/VRAM-буферы под живой загрузкой.
    std::atomic<int> active_jobs_{0};

    // Политика/предиктор.
    EvictionPolicy policy_;
    MarkovPredictor predictor_;
    std::size_t lookahead_;

    // Рабочие потоки.
    std::vector<std::thread> workers_;
    std::mutex qmu_;
    std::condition_variable qcv_;
    std::deque<Job> queue_;
    bool shutdown_ = false;

    mutable std::mutex mu_;          // защита плана/состояния (см. выше)
    std::atomic<bool> plan_active_{false};

    // Метрики (атомарные счётчики + снапшот пауз).
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
    mutable std::mutex pauses_mu_;
    std::deque<double> pauses_;
};

// Тесты/утилиты: имя бэкенда I/O NVMe-тира ("io_uring" | "fallback").
std::string nvme_io_backend();

} // namespace stell::tiers

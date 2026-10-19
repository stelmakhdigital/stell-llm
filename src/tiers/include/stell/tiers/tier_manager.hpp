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
    // CRC-политика (ADR-12 §2.3, round 3, planner [Q2]): payload-CRC при
    // pin/read — только при явном флаге (конфиг ИЛИ env STELL_TIER_VERIFY_CRC=1
    // — итоговое значение = config || env), по умолчанию off — hot path без
    // CRC (writer считает CRC при записи; целостность NVMe-данных проверяется
    // вне пути декода: verify_slots() или периодический проход калибровщика).
    // header_crc32 файла layout reader проверяет всегда.
    bool verify_crc             = false;

    // [R2a] round 4 (planner [Q6]): O_DIRECT-бэкенд NVMe-тира — данные слотов
    // читаются на fd, открытом с O_DIRECT (детерминированный cold: page cache
    // не искажает замер под co-tenancy, нет двойного буферирования). Дефолт
    // buffered — fallback-прецедент сохранён. Если FS/ядро не поддерживают
    // O_DIRECT (напр. tmpfs) — честный откат на buffered:
    // TierNvmeFile::odirect_active() == false (тест проверяет активность;
    // bench обязан помечать прогон фактическим режимом). Выравнивание —
    // 4096 (= kTierSlotAlign, консервативно покрывает логический блок
    // распространённых FS; лsb кодом НЕ запрашивается — честно зафиксировано
    // в докстринге nvme_file.hpp): RAM-буферы менеджера выделяются
    // выровненно — O_DIRECT читает прямо в них; произвольные dst — через
    // минимальное выровненное staging-чтение (контракт read_group цел).
    // Итоговый флаг = config || env (STELL_TIER_ODIRECT=1) — та же
    // OR-семантика, что у verify_crc.
    bool odirect                = false;

    // [R2b] round 4 (planner [Q2]): deep-prefetch — глубина k групп в полёте.
    // Воркер тира дренажит очередь загрузок БАТЧАМИ до k групп: k SQE
    // io_uring подаются ОДНИМ submit (k чтений NVMe параллельно — глубина
    // очереди носителя) и завершаются вместе; пауза между слоями
    // перекрывается чтением k слоёв вперёд. k=1 — прежнее поведение
    // (группа за раз). Заделка в tier-профиле: tier_config_from_env
    // (STELL_TIER_PREFETCH_DEPTH) / tier_config_co_tenancy — профиль
    // фиксирует границы тиров, k задаётся поверх (from_env). Честная
    // семантика: батч из k групп материализуется параллельно; при
    // fallback-бэкенде (нет io_uring) батч деградирует до k
    // последовательных pread (фиксировано в докстринге nvme_file.hpp).
    std::size_t deep_prefetch_depth = 1;

    // Диагностический крюк (НЕ прод-параметр): искусственная латентность
    // одной загрузки батча (мс) — только для детерминистических тестов
    // deep-prefetch (k групп в полёте материализуются за ~латентность ОДНОЙ
    // группы, а не k×). Прод/дефолт: 0. env: STELL_TIER_IO_DELAY_MS
    // (tier_config_from_env). Честно: крюк моделирует латентность загрузки;
    // батч-параллелизм с ним детерминирован.
    std::size_t io_delay_ms     = 0;
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
    // [R2] round 5: per-group счётчики prefetch (hit-rate по группам — bench
    // per-group layout; агрегаты hits_/misses_ не дают разрез по группам).
    // Инкремент — в pin_internal под mu_ (та же точка, что агрегаты);
    // снапшот — group_prefetch_stats().
    std::uint64_t pf_hits     = 0;
    std::uint64_t pf_misses   = 0;
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

    // [R2] round 5: снапшот per-group счётчиков prefetch (hits/misses на
    // каждую группу плана; отсортирован по gid). Назначение — hit-rate ПО
    // ГРУППАМ в bench per-group layout (пауза слоя = Σ пауз групп слоя;
    // hit-rate по группам/ролям — агрегат hits_ не даёт этот разрез).
    // Расширение на конкретной реализации — ITierManager (ADR-08 §2) НЕ
    // расширяется; тот же прецедент, что crc_checks/read_ram_slot. Под mu_
    // (снапшот — не горячий путь, для bench/дашборда/тестов).
    struct GroupPrefetchStat {
        GroupId      gid;
        std::uint64_t hits;
        std::uint64_t misses;
    };
    std::vector<GroupPrefetchStat> group_prefetch_stats() const;

    // Чтение содержимого слота группы из RAM-тира (расширение R8, раунд 2
    // продолжение): копирует ровно slot_bytes ГРУППЫ байтов в dst ([R2] round 5:
    // per-group layout — у групп РАЗНЫЕ размеры слотов; uniform — размер группы
    // == slot_size плана, поведение прежнее).
    // Назначение — интеграционный тест «stell-tier layout (writer models) →
    // читатель tiers → слоты RAM/NVMe tiers → логиты, побитово == прямой
    // загрузке»: engine/core читает канонические байты слота (CRC слота
    // проверен при первом чтении группы, ADR-12 §2.3) для dequant/ops.
    // Расширение на конкретной реализации — интерфейс ITierManager (ADR-08 §2)
    // НЕ расширяется; тот же прецедент, что set_layer_map (см. «Уточнение»
    // ADR-08). Ошибки:
    //   - PlanMismatch: группа вне плана; dst_bytes < slot_bytes группы;
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

    // [R5] round 4: чтение содержимого VRAM-СЛОТА группы (после pin→wait)
    // в dst (dst_bytes ≥ slot_bytes ГРУППЫ — [R2] round 5: per-group layout,
    // у групп разные размеры слотов; uniform — прежнее поведение). Назначение
    // — побитовая сверка VRAM-слотов с эталоном прямого лоадера
    // (TierLayoutReader): инвариант «одна группа = один слот» (класс
    // aliasing-багов ba42267 не ловился тестами логитов). Расширение на конкретной реализации — интерфейс
    // ITierManager (ADR-08 §2) НЕ расширяется; тот же прецедент, что
    // read_ram_slot/set_layer_map. Не горячий путь: копия под mu_
    // (защита от гонки с eviction); для верификации/тестов. Ошибки:
    //   - PlanMismatch: группа вне плана; dst_bytes < slot_bytes группы;
    //   - IoFailure: группа не resident в VRAM (pin→wait обязателен до
    //     чтения). При STELL_CUDA — cudaMemcpy D2H из device-слота.
    void read_vram_slot(GroupId id, void* dst, std::size_t dst_bytes) const;

    // --- расширения R1/R7 (раунд 3; интерфейс ITierManager, ADR-08 §2,
    // --- НЕ расширяется — тот же прецедент, что set_layer_map/read_ram_slot)

    // R1 (ADR-12 §2.3, round 3): публичный verify-проход слотов NVMe-тира —
    // внепутевая сверка payload-CRC ВСЕХ групп layout с индексом. Выбор в
    // пользу метода TierManager (а не harness-команды): verify работает с тем
    // же открытым NVMe-файлом и состоянием «проверенных групп», что и горячий
    // путь (единый кодовый путь, без второго канала доступа к layout), а
    // вызвать его может кто угодно — калибровщик, harness, тесты, стартовый
    // проход. Дисциплина «холодной» операции (как set_plan): НЕ вызывать
    // параллельно с pin/prefetch. Возвращает число проверенных слотов;
    // при verify-порче → TierError(IoFailure, "layout: payload CRC mismatch").
    std::size_t verify_slots();

    // Метрика R1: сколько раз payload-CRC фактически пересчитывался (по всем
    // путям: горячий путь при verify=on + verify-проход). При verify=off и без
    // verify-прохода — 0: фиксация «hot path без CRC» (тест R1-а).
    std::uint64_t crc_checks() const;

    // [R2a] round 4: O_DIRECT-тракт данных NVMe-тира РЕАЛЬНО активен (fd
    // открыт с O_DIRECT). false → отката на buffered (FS/ядро не поддерживают
    // O_DIRECT, напр. tmpfs) — bench/отчёты обязаны честно помечать прогон
    // фактическим режимом. Без NVMe-тира — false. Для тестов/дашборда/bench
    // (расширение на конкретной реализации, ITierManager не расширяется —
    // прецедент crc_checks).
    bool nvme_odirect_active() const;

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
    // [R2] round 5: размер слота КОНКРЕТНОЙ группы (per-group layout — у
    // групп разные slot_bytes; uniform — == slot_size_). Под mu_; unknown → 0.
    std::size_t group_slot_bytes_locked(GroupId id) const;

    void enqueue_load(GroupId id, const std::shared_ptr<TierAwaiter>& a);
    void worker_loop();
    void process_batch(const std::vector<Job>& batch); // [R2b]: до k групп в полёте
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

    // [R2] round 5: фактические байты RAM-тира (Σ slot_bytes ГРУПП буферов;
    // per-group layout — группы разных размеров, учёт по slot_size_ завышал
    // бы занятую память в Σmax раз). Под mu_; alloc/evict — единственные
    // точки изменения. tier_bytes[1] = это значение (stats()).
    std::size_t ram_bytes_ = 0;

    // NVMe-тир.
    std::unique_ptr<TierNvmeFile> nvme_;
    bool verify_crc_ = false;        // итоговый флаг R1: config || env (см. TierConfig)

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
    std::size_t batch_depth_;        // [R2b]: k групп в полёте (deep-prefetch)
    bool odirect_;                   // [R2a]: итоговый флаг O_DIRECT (config || env)

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

// --- R7 (раунд 3): конфигурируемые границы тиров -----------------------------

// (1) Параметризация окружением поверх базового конфига (env → поля; не
// заданные переменные оставляют базовое значение):
//   STELL_TIER_VRAM_SLOTS       — число VRAM-слотов;
//   STELL_TIER_VRAM_SLOT_BYTES  — размер VRAM-слота (байтов, кратен 4096);
//   STELL_TIER_RAM_MB           — capacity RAM-тира (МиБ);
//   STELL_TIER_LOOKAHEAD        — глубина router-lookahead prefetch;
//   STELL_TIER_HISTORY_CAP      — предел истории предиктора;
//   STELL_TIER_VERIFY_CRC       — 1 → payload-CRC при pin/read (см. TierConfig);
//   STELL_TIER_POLICY           — "lru" | "lcu";
//   STELL_TIER_NVME_LAYOUT      — путь stell-tier файла;
//   STELL_TIER_ODIRECT          — 1 → O_DIRECT-бэкенд NVMe-тира ([R2a]);
//   STELL_TIER_PREFETCH_DEPTH   — k групп в полёте ([R2b]; 1 — дефолт);
//   STELL_TIER_IO_DELAY_MS      — диагностическая латентность загрузки (тесты).
// Нечисловые/некорректные значения → берётся базовое значение (консервативно).
// Профиль co-tenancy задаётся комбинацией: напр. свободно 2–11 ГБ на V100 →
// STELL_TIER_VRAM_SLOTS небольшим, STELL_TIER_RAM_MB — под основной тир.
TierConfig tier_config_from_env(TierConfig base = {});

// (2) Именованный профиль «co-tenancy на V100» (planner [R7], round 3):
// свободно 2–11 ГБ VRAM (strata держит 21–30 ГБ) → VRAM-тир МАЛ, RAM-тир —
// основной. Параметры:
//   vram_budget_bytes — бюджет tiers под VRAM-тир: доля свободной VRAM,
//       которую калибровщик/engine выделил тиру (с запасом под активации
//       и внутренние буферы CUDA); ДОЛЖЕН быть ≤ фактически свободной VRAM —
//       это ответственность вызывающего (замер free VRAM — вне tiers).
//   plan_bytes        — суммарный размер плана (Σ slot_bytes);
//   slot_bytes        — размер слота плана (кратен 4096, ADR-12 §2.4).
// Результат: vram_slots = budget/slot (мин. 1, больше плана не выделяем —
// при co-tenancy VRAM-тир меньше плана); ram_capacity = план целиком
// (RAM-тир основной: pinned RAM на сервере — сотни ГБ, вмещает план;
// допущение зафиксировано). NVMe-тир: путь stell-tier файла задаёт
// вызывающий (TierConfig::nvme_layout / STELL_TIER_NVME_LAYOUT) — профиль
// фиксирует только ГРАНИЦЫ тиров. slot_bytes вне кратности 4096 →
// TierError(PlanMismatch). Остальные поля — дефолт TierConfig (Lru,
// lookahead 2); уточнение — поверх через tier_config_from_env().
TierConfig tier_config_co_tenancy(std::size_t vram_budget_bytes,
                                  std::size_t plan_bytes,
                                  std::size_t slot_bytes);

} // namespace stell::tiers

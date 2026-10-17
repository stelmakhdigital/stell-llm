# ADR-08: Контракт engine ↔ tiers — тензорные группы с ролями

- **Статус:** Принято (интерфейсный контракт, зафиксирован до старта потоков A и B)
- **Стороны:** `core` (engine) ↔ `tiers` (Tier Manager)
- **Правило изменения:** только через новый/обновлённый ADR и согласование с worker.
- **Связано:** ADR-04 (KV dtype), ADR-05 (графы/слоты), ADR-09, ADR-12

## Контекст

Engine-планировщик (поток A) декодирует MoE/dense слои и должен к началу слоя
гарантированно иметь нужные веса. Tier Manager (поток B) владеет иерархией
VRAM-slots → pinned RAM → NVMe (io_uring). Модули пишутся параллельно, поэтому
стык обязан быть зафиксирован до старта потоков: tiers не знает ни про какую
конкретную модель — он оперирует обезличенными «тензорными группами» с ролями
(expert/attn/kv), engine же отвечает за то, чтобы запросы групп соответствовали
описанию модели (ADR-09) и stell-tier layout (ADR-12).

## Решение

### 1. Базовые типы (namespace `stell::tiers`, заголовок `stell/tiers/types.hpp`)

```cpp
namespace stell::tiers {

enum class Role { Expert, Attn, Kv };

enum class DType { F16, F32, E4M3, NVFP4, W4A16 };  // W4A16 = упакованный W4 + F16 scale

// Непрозрачный идентификатор тензорной группы.
// Присваивается при разметке модели (см. ADR-09), стабилен в рамках сессии модели.
struct GroupId {
    std::uint32_t value;            // монотонный, уникален в рамках одной модели
    bool operator==(const GroupId& o) const { return value == o.value; }
};

// Описание группы: выдаётся engine/tiers при разметке, далее только читается.
struct TensorGroup {
    GroupId   id;
    std::string name;               // человекочитаемое имя, напр. "l012.e003.wqkv"
    Role      role;
    DType     dtype;
    std::vector<std::int64_t> shape; // логическая форма
    std::size_t   slot_bytes;        // РАЗМЕР СЛОТА В БАЙТАХ (выровнен, фиксируется
                                     // ДО старта CUDA Graph — см. ADR-05; гарантия tiers)
};

struct GroupKey {       // минимальный запрос tiers со стороны engine
    GroupId   id;
    Role      role;             // дублируется для читабельности метрик/логов
};
} // namespace stell::tiers
```

Ключевое правило: **`slot_bytes` каждой группы фиксирован при разметке модели**
(до снятия CUDA Graph, до старта потоков декода). Tiers не изменяет разметку на лету;
изменение разметки = пересоздание плана (переснятие графа, см. ADR-05).

### 2. Интерфейс Tier-менеджера (заголовок `stell/tiers/manager.hpp`)

```cpp
namespace stell::tiers {

// Асинхронный handle ожидания готовности групп (для планировщика).
// Реализация: future/fan-in от io_uring и DMA-очередей; CPU-версия — thread-pool.
class Awaiter {
public:
    virtual ~Awaiter() = default;
    virtual bool ready() const = 0;                    // неблокирующая проверка
    virtual void wait() = 0;                            // блокирующее ожидание
};

class ITierManager {
public:
    virtual ~ITierManager() = default;

    // Разметка: engine передаёт полный план групп одной модели (см. ADR-09/ADR-12).
    // Должна вызываться ДО старта decode-потоков; повторный вызов = ResetPlan.
    virtual void set_plan(const std::vector<TensorGroup>& groups) = 0;

    // --- pin / unpin -----------------------------------------------------
    // pin: гарантироватьresident-ность группы в VRAM-тире (блокирует eviction).
    // Возвращает Awaiter; блокировать поток планировщика можно только через wait().
    virtual std::unique_ptr<Awaiter> pin(const GroupKey& key) = 0;
    virtual void unpin(const GroupKey& key) = 0;       // снять запрет eviction
    // Массовые операции для слоя (частый случай планировщика):
    virtual std::unique_ptr<Awaiter> pin_layer(std::uint32_t layer) = 0;

    // --- prefetch ----------------------------------------------------------
    // Асинхронный, идемпотентный, НЕ блокирует; политика/предсказатель внутри tiers.
    // Вызывается engine заранее (router-lookahead — см. PLAN.md §4, поток B).
    virtual void prefetch(const std::vector<GroupKey>& keys) = 0;

    // --- stats --------------------------------------------------------------
    // Потокобезопасный снапшот метрик (для дашборда/bench — см. ADR-11, PLAN.md §5).
    virtual TierStats stats() const = 0;
};

struct PrefetchStats {
    std::uint64_t hits;                 // prefetch попал: группа уже в VRAM к использованию
    std::uint64_t misses;
    double        hit_rate;             // hits / (hits + misses)
};

struct PauseStats {                     // паузы между слоями (ожидание весов)
    double p50_ms;
    double p95_ms;                      // целевой порог ≤ 5 мс на FP16-профиле (PLAN.md §4)
    double p99_ms;
};

struct TierStats {
    PrefetchStats prefetch;
    PauseStats    layer_pause;
    std::array<std::uint64_t, 3> tier_bytes;   // [VRAM, pinned RAM, NVMe] занято
    std::uint64_t io_bytes;                     // суммарно перенесено тир↔тир
    std::uint64_t io_ops;                       // количество I/O-операций
};

} // namespace stell::tiers
```

### 3. Ошибки — выбираем исключения

Единственный механизм ошибок tiers — **исключения**, не коды возврата:

```cpp
namespace stell::tiers {
class TierError : public std::runtime_error {  // базовая; код — для логов/дашборда
public:
    enum class Code { PlanMismatch, SlotExhausted, IoFailure, NotPinned, Timeout };
    TierError(Code c, const std::string& what);
    Code code() const noexcept;
};
}
```

Ошибки, которые возможны в steady-state декода (`SlotExhausted` и т.п.), — это
протокол-нарушения или деградация носителя, т.е. исключительные ситуации;
успешный путь сигнатур не загрязняется. Коды `Code` — для дашборда/логов.
Планировщик обязан оборачивать `TierError` в свою политику (fallback/abort),
схема обработки — внутреннее дело engine, на стык не влияет.

### 4. Потокобезопасность

- Все методы `ITierManager` **потокобезопасны**: `pin/prefetch/stats` вызываются
  из потоков планировщика и фоновых потоков tiers одновременно — ожидается.
- `set_plan` — единственная «холодная» операция: обязана вызываться до старта
  decode-потоков и до снятия CUDA Graph; одновременный вызов с рабочими методами
  недопустим по контракту (проверяется assert'ами и CPU-тестами).
- `stats()` — неблокирующий снапшот (copy-on-write или атомарные счётчики);
  не является консистентным срезом «в момент t», это асимптотическая статистика.
- Реализация обязана корректно работать при `prefetch` → `pin` той же группы
  concurrently (идемпотентность prefetch фиксируется CPU-тестами tiers).

### 5. Кто за что отвечает (сводка обязанностей на стыке)

- **Engine:** разметка групп (из ADR-09/ADR-12), вызовы `set_plan/pin/unpin/prefetch`,
  измерение пауз между слоями (передача таймингов — через stats-снапшот tiers,
  т.к. планировщик знает границы слоёв).
- **Tiers:** владение тирами (VRAM-slots/pinned RAM/NVMe), политика eviction
  (LRU/LCU + router-lookahead), io_uring, метрики hit-rate и пропускной способности.

## Альтернативы

- **Коды возврата / `std::expected`** — чистые сигнатуры, но загрязняют каждый
  вызов проверками в горячем пути планировщика; выбраны исключения (см. выше).
- **Tiers знает про слои/модели** (как в project-maya) — нарушает принцип
  PLAN.md §2.3 («tiering не знает про модель»); заменено ролью `Role` и
  обезличенными группами.
- **Синхронный `pin` с блокировкой до готовности** — просто, но убивает
  router-lookahead prefetch; выбран асинхронный `Awaiter`.

## Последствия

- Потоки A и B могут писать код независимо: A — против `ITierManager`, B — реализует его.
- CPU-тест tiers (RAM/NVMe-файлы без CUDA) и CUDA-тест (DMA) обязаны проходить
  один и тот же интерфейс — это фиксируется в тестах tiers.
- Изменение сигнатур после старта потоков — только через ADR и согласование с worker.

## Уточнение (round 1, задача T-20261017200354)

Контракт не меняется; ниже — фиксация решений, принятых при реализации
(`src/tiers`, `src/engine`), по открытым вопросам раунда 1.

- **Соответствие слой → группы (`pin_layer`).** Задаётся методом
  `TierManager::set_layer_map(const std::vector<std::vector<GroupId>>&)` —
  это расширение на конкретной реализации (`tier_manager.hpp`), **не** новый
  метод `ITierManager` (интерфейс §2 не расширяется; engine вправе работать
  с `TierManager` напрямую). Вызывающий — только engine, один раз при загрузке
  модели, сразу после `set_plan`; пара `set_plan` → `set_layer_map` — «холодная»
  дисциплина §4 (до decode-потоков и до снятия CUDA Graph). `pin_layer(layer)`
  без карты слоя или без группы плана = `TierError::PlanMismatch`
  (реализация `manager.cpp`). Обоснование: слои — понятие engine; tiers лишь
  хранит обезличенную таблицу слой→группы, чтобы не нарушать
  «tiers не знает про модель» (§5 и PLAN.md §2.3).
- **`layout_version` и переснятие графа.** Версия разметки — величина engine;
  tiers её не хранит и не знает. `ExecutionPlan` вычисляет `layout_version`
  детерминированно из состава плана (FNV-1a по `GroupId` + `slot_bytes`,
  `execution_plan.cpp`); `DecodeExecutor` сверяет `current_layout_version_`
  с захваченным при снятии графа — любое изменение состава/размеров групп →
  переснятие графа (ADR-05). При пересоздании плана engine повторяет пару
  `set_plan` → `set_layer_map` **до** переснятия графа; одновременная работа
  методов недопустима (§4).

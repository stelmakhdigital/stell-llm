# ADR-09: Контракт engine ↔ models — декларативные описания, реестр, golden-тесты

- **Статус:** Принято (интерфейсный контракт, зафиксирован до старта потоков A и D)
- **Стороны:** `core` (engine) ↔ `models` (реестр архитектур, лоадеры весов)
- **Правило изменения:** только через ADR и согласование с worker.
- **Связано:** ADR-03 (форматы), ADR-08 (тензорные группы), ADR-10 (шардирование), ADR-12 (stell-tier)

## Контекст

Принцип PLAN.md §2.2: «новая модель = файл описания (+ ядро, если новая операция)».
Engine-планировщик не должен знать конкретные архитектуры; он исполняет операции,
которые ему выписывает описание модели, и передаёт tiers разметку тензорных групп.
Модули пишутся параллельно (потоки A и D), значит формат описания и интерфейс
загрузчика весов должны быть зафиксированы до старта.

## Решение

### 1. Формат описания модели — JSON (выбор и обоснование)

Выбран **JSON**, не YAML:

- конфиги архитектур (HF-референсы, `config.json`) уже JSON — golden-сравнение
  и миграция с референсов проще;
- в ядре C++20 JSON-парсер (nlohmann/json — header-only) проще и детерминированнее,
  чем YAML: меньше зависимостей, нет YAML-подвохов (типы `yes/no`, неоднозначные
  числа), CI на CPU-пути не тянет libyaml;
- описания моделей машинно-генерируются конвертерами (из `config.json`/GGUF-метаданных),
  человеку их редактировать редко.

Схема (версионная, поле `schema` — мажор.минор; неизвестные поля — ошибка парсинга):

```json
{
  "schema": "1.0",
  "name": "glm-5.3-flash",
  "class": "moe",                      // "dense" | "moe"
  "dtype_weights": "NVFP4",            // целевой профиль: F16 | NVFP4 | W4A16 (ADR-03)
  "vocab_size": 131072,               // числа — обычные int
  "d_model": 4096,
  "n_layers": 32,
  "n_heads": 64,
  "d_head": 128,
  "kv": {"dtype": "F16", "n_kv_heads": 8, "d_kv": 128, "quant": null}, // quant: "E4M3" (ADR-04)
  "moe": {                             // только для class=moe
    "n_experts": 128,
    "n_experts_active": 8,
    "expert_d_ff": 1024,
    "router": "softmax-topk"
  },
  "ops": [                             // операции engine, порядок = граф исполнения слоя
    {"op": "rmsnorm",   "io": ["hidden"],  "params": {"eps": 1e-5}},
    {"op": "attention", "io": ["hidden", "kv"], "params": {"kernel": "sm70_fp32_qk_pv"}},
    {"op": "moe_gather_scatter", "io": ["hidden", "experts"], "params": {"kernel": "sm70_moe_gs"}}
  ],
  "tensor_groups": [                   // разметка → tiers (ADR-08) и stell-tier (ADR-12)
    {"name": "l{layer}.attn.wq",  "role": "attn",   "shape": [4096, 4096],
     "dtype": "NVFP4", "source_pattern": "layers.{layer}.self_attn.q_proj.weight"},
    {"name": "l{layer}.kv",       "role": "kv",     "shape": [2048, 4096],
     "dtype": "F16",   "source_pattern": "layers.{layer}.kv_cache", "kv_block": true},
    {"name": "l{layer}.e{expert}.ffn", "role": "expert", "shape": [1024, 4096],
     "dtype": "NVFP4", "source_pattern": "layers.{layer}.mlp.experts.{expert}.weight",
     "cardinality": {"layer": 32, "expert": 128}}
  ]
}
```

Правила разметки:

- `tensor_groups` — единственный источник ролей `expert/attn/kv` (ADR-08);
  engine не выводит роли сам.
- `cardinality` — шаблон размножается по слоям/экспертам; `slot_bytes` каждой
  группы фиксируется при загрузке описания (до CUDA Graph, ADR-05/ADR-08).
- `source_pattern` — имя тензора во входном формате (safetensors/GGUF),
  привязка к лоадеру (§3).
- `ops` — операции engine; если операция не найдена в реестре ядра engine —
  ошибка загрузки модели («новая операция = новое ядро», PLAN.md §2.2).

### 2. Реестр архитектур (заголовок `stell/models/registry.hpp`)

```cpp
namespace stell::models {

struct ModelDescription {   // результат парсинга JSON (строго типизирован)
    std::string name;
    bool is_moe;
    // ... поля 1:1 из схемы §1; tensor_groups уже размножены по cardinality
    std::vector<stell::tiers::TensorGroup> groups;  // ADR-08 — готово к set_plan()
};

class IRegistry {
public:
    virtual ~IRegistry() = default;
    // Загрузка/валидация описания: schema-версия, операции, роли, cardinality.
    virtual ModelDescription load(const std::string& json_path) = 0;
    // Список встроенных архитектур реестра (цель PLAN.md: 1 dense + 2 MoE).
    virtual std::vector<std::string> available() const = 0;
};

} // namespace stell::models
```

Реестр растит архитектуру за архитектурой (PLAN.md §6); engine не имеет
if'ов по архитектурам — только `ops` и `groups`.

### 3. Интерфейс загрузчика весов (заголовок `stell/models/weights.hpp`)

```cpp
namespace stell::models {

// Лоадер переводит «входной формат» → заполненные тензорные группы (ADR-08).
// Форматы: Safetensors | GGUF | NVFP4W4A16 | StellTier (ADR-03).
enum class WeightsFormat { Safetensors, GGUF, NVFP4W4A16, StellTier };

class IWeightsLoader {
public:
    virtual ~IWeightsLoader() = default;
    virtual WeightsFormat format() const = 0;

    // Открывает источник весов и возвращает план групп, согласованный с описанием.
    // Ошибки — исключениями (единая политика с tiers, ADR-08 §3):
    //   LoaderError::Code { SchemaMismatch, TensorNotFound, ChecksumMismatch,
    //                       UnsupportedQuant, IoFailure }
    virtual std::unique_ptr<IWeightsSource> open(
        const std::string& uri, const ModelDescription& desc) = 0;
};

// Источник весов: групповое чтение для tiers (может быть файлом, pinned-памятью
// или уже размещёнными в VRAM буферами — CPU-путь без GPU обязан работать).
class IWeightsSource {
public:
    virtual ~IWeightsSource() = default;
    // Чтение группы по её GroupId (роль/dtype/shape берутся из плана, не из файла).
    virtual void read(const stell::tiers::GroupKey& key, void* dst, std::size_t dst_bytes) = 0;
    virtual void close() = 0;
};

} // namespace stell::models
```

Контракт отображения «формат → группы»: лоадер обязан по `source_pattern`
найти тензор, проверить dtype/shape против плана и заполнить буфер группы
(в целевом dtype профиля, dequant — ядрами engine/kernels, не лоадером).
Mismatch = исключение `SchemaMismatch`/`UnsupportedQuant` (тишина запрещена).

### 4. Golden-тесты логитов — часть контракта

Для каждой архитектуры реестра обязателен golden-тест (`tests/golden/`):

- фиксированный вход: промпт-токены, seed=0, greedy (T=1) — как в bench-контракте
  PLAN.md §5;
- эталон: логиты последнего слоя против HF-референса этой модели
  (фикстура `logits_ref.json`, допуски: rel ≤ 1e-3 для FP16-профиля,
  отдельные пороги для NVFP4/W4A16 — фиксируются в заголовке фикстуры);
- CPU-путь — обязателен и проходит локально без GPU; CUDA-вариант — на сервере.
- Тест прогоняется на трёх путях: обычный лоадер, stell-tier layout (ADR-12),
  tiering с принудительными промахами в NVMe-тир — логиты совпадают со входом
  того же профиля. Это фиксирует, что tiering не меняет вычисление.

Изменение формата описания или лоадера, ломающее golden-тест, запрещено.

## Альтернативы

- **YAML** — человечески приятнее, но: лишняя зависимость в C++-ядре, недетерминизм
  типов, нет выигрыша: описания генерируются конвертерами, а не пишутся руками.
- **Императивный код-описание модели (C++-класс на архитектуру)** — так сделано
  в тяжёлых стеках (vLLM-реестр): быстрое, но новая модель = компиляция, что
  противоречит принципу «файл описания» (PLAN.md §2.2).
- **Engine сам парсит конфиги моделей** — размазывает знание об архитектурах
  по engine; выбран отдельный модуль `models` с реестром.

## Последствия

- Потоки A и D пишут независимо: A — против `ModelDescription`/`ops`/`groups`,
  D — реализует реестр/лоадеры по фиксированной JSON-схеме.
- Разметка `tensor_groups` — единственный вход `ITierManager::set_plan` (ADR-08),
  и она же конвертируется в stell-tier layout (ADR-12) — один источник истины.
- Новая модель: JSON-описание + (если нужна новая операция) ядро; engine/kernels
  не трогаются для архитектур из существующих операций.

## Уточнение (round 1, задача T-20261017200354)

Контракт не меняется; ниже — фиксация конвенций, принятых при реализации
(`src/models`), по открытым вопросам раунда 1.

- **Имена тензоров GGUF (§3).** `source_pattern` всегда записывается в конвенции
  имён **входного формата**; лоадер сравнивает имя тензора с `source_pattern`
  буквально — никакого неявного маппинга HF-style ↔ GGUF (`blk.*`) нет
  (`loader_gguf.cpp`). Конвенция для генерации GGUF-описаний: паттерны —
  нативные имена llama.cpp-GGUF (`token_embd.weight`, `output.weight`,
  `output_norm.weight`, `blk.{layer}.attn_q.weight`, `blk.{layer}.ffn_up.weight`,
  `blk.{layer}.ffn_norm.weight`, …); описание для GGUF-источника генерируется
  конвертером как отдельный файл описания (реестр может держать несколько
  описаний одной архитектуры под разные семейства имён; `llama-class.json` —
  HF-стиль для safetensors). Обоснование: конвенции имён GGUF и HF несводимы
  простой заменой префикса (разная структура имён attn/ffn), маппинг в лоадере —
  скрытое знание о форматах; честнее генерировать описание под источник.
- **Greedy vs «T=1» (§4).** Поле `greedy: true` фикстуры golden — отдельный явный
  режим argmax: temperature не применяется, сэмплинг детерминирован.
  «T=1» bench-контракта PLAN.md §5 — другой режим: `temperature = 1.0`,
  `top_p = 0.95`, `top_k = 20`, `seed = 0` (дефолты `Scheduler::Sampling`,
  `scheduler.hpp`). Комментарий «T=1» в README golden-фикстуры — ссылка на
  bench-профиль, **не** синоним greedy; смешение режимов устранено:
  golden — greedy, bench — T=1.

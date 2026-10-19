# ADR-11: Контракт engine ↔ serve — постановка запросов, stream-события, метрики

- **Статус:** Принято (интерфейсный контракт, зафиксирован до старта потоков A и E)
- **Стороны:** `core` (engine) ↔ `serve` (Python control-plane: API, дашборд, калибровщик, bench)
- **Правило изменения:** только через ADR и согласование с worker.
- **Связано:** ADR-01 (Python — тонкий control-plane), ADR-06 (порт 8080), ADR-08 (stats), PLAN.md §5

## Контекст

Serve — Python (FastAPI, ADR-01): оба API (`/v1/chat/completions`, `/v1/messages`),
стриминг SSE, tool-calls, CORS, health/метрики, дашборд. Engine — C++20 ядро.
Нужен один способ: как serve ставит запросы в планировщик continuous batching,
как события декода идут обратно в SSE-потоки и как engine отдаёт метрики.
Всё это фиксируется до старта потоков, чтобы serve не придумывал API ядра.

## Решение

### 1. Встраивание engine: C ABI динамической библиотеки

**Выбор:** engine собирается в `libstell_core.so` и exposes **стабильный C ABI**
(не C++-интерфейс — стабильность бинарная, не исходная). Serve связывается через
`ctypes`/`cffi` (один процесс, см. ADR-06, port 8080: Python-процесс грузит .so;
отдельного процесса engine и IPC-сокетов между serve и engine нет).

```c
// stell/engine/capi.h — единственная точка стыка (никаких C++-типов в ABI)
typedef struct stell_engine stell_engine_t;      // непрозрачный handle

// Лifecycle: cfg — JSON-строка (конфиг: модель, tier-границы, parallel, профили).
int  stell_engine_create(const char* cfg_json, stell_engine_t** out);
void stell_engine_destroy(stell_engine_t* e);

// Постановка запроса в планировщик (continuous batching, см. §2).
// Возвращает непрозрачный id сессии (uint64), >= 0. 0 — ошибка (msg_out — текст).
uint64_t stell_submit(stell_engine_t* e, const char* request_json, char** msg_out);

// Неблокирующее выкачивание событий сессии (§3). Возвращает число событий,
// 0 — событий нет, < 0 — сессия закрыта/ошибка. out_json — массив событий.
int  stell_poll(stell_engine_t* e, uint64_t session_id, char** out_json);

// Отмена: планировщик выкидывает сессию из батча на ближайшем шаге.
int  stell_cancel(stell_engine_t* e, uint64_t session_id);

// Метрики/health — JSON (§4). Снапшот, потокобезопасно.
int  stell_metrics(stell_engine_t* e, char** out_json);
```

События и запросы — **JSON-строки через границу ABI** (никаких бинарных структур:
мажорные версии ABI не меняются из-за полей; поля только добавляются).
Свойства: `libstell_core.so` обязан иметь CPU-путь (CPU-эмуляции ADR-08/10),
чтобы весь контур serve собирался и тестировался локально без GPU (ADR-07).

### 2. Постановка запросов: формат request_json

```json
{
  "session": {"max_tokens": 1024, "temperature": 0.7, "top_p": 0.95,
              "top_k": 20, "seed": 0, "stop": ["\n"]},
  "tools": [{"name": "get_weather", "schema": {...}}],        // tool-calls
  "messages": [{"role": "user", "content": "..."}],           // chat-шаблон применяет serve
  "stream": true
}
```

- serve применяет chat-шаблон модели и передаёт готовые токены/строку; engine
  не знает ни Anthropic, ни OpenAI — только запрос §2 и события §3.
- Планировщик continuous batching обязан обслуживать **≥ 8 одновременных
  сессий** (PLAN.md §4, поток A); постановка — `stell_submit`, сессия сразу
  получает slot (ADR-08) и входит в ближайший шаг батча.
- `seed`/`top_p`/`top_k`/`temperature` — контракты bench-протокола PLAN.md §5
  (T=1, top-p=0.95, top-k=20, seed=0) берутся отсюда же.

### 3. Формат stream-событий (stell_poll → SSE)

Каждое событие — JSON-объект с `seq` (монотонно в рамках сессии), `type`:

```json
{"seq": 12, "type": "token",       "text": "…", "token_id": 4312}
{"seq": 13, "type": "tool_call",   "id": "call_a1", "name": "get_weather",
 "arguments": "{\"city\":\"…\"}"}
{"seq": 14, "type": "tool_result", "id": "call_a1", "content": "..."}   // serve → engine
{"seq": 15, "type": "finish",      "reason": "stop", "usage": {"prompt_tokens": 12,
 "completion_tokens": 34}}
{"seq": 16, "type": "error",       "code": "slot_overflow", "message": "…"}
```

Правила:

- `token` — инкрементально, в порядке декодирования; `tool_call` — когда
  модель эмитит вызов инструмента (спец-токены → структурный вывод);
  `tool_result` — serve доставляет результат обратно через повторный
  `stell_submit` того же session_id (продолжение сессии, слот сохранён);
  `finish` — терминальное; `reason`: `stop | length | tool_call | cancelled`.
- `finish` с `usage` обязателен всегда — bench-контракт (PLAN.md §5) и метрики
  API на нём держатся.
- serve превращает события в SSE (`data: {...}\n\n`) для обоих API; маппинг
  `finish.reason` → `finish_reason` OpenAI / `stop_reason` Anthropic —
  на стороне serve (engine не знает API-диалекты).

### 4. Метрики: /health и /metrics (единый снапшот)

`stell_metrics` возвращает JSON-снапшот; serve отдаёт его через `/health`
(кратко) и `/metrics` (полностью) на том же порту 8080 (ADR-06):

```json
{
  "engine": {"state": "serving", "sessions_active": 12, "slots_busy": 12,
             "slots_total": 16, "batch_size_max": 16},
  "latency": {"ttft_ms": {"p50": 90, "p95": 180, "p99": 240},
              "tpot_ms":  {"p50": 14, "p95": 22, "p99": 35},
              "tok_s": {"decode": 53.0, "prefill": 266.0, "spec_decode": 0.0}},
  "tiers": {"prefetch": {"hit_rate": 0.63}, "layer_pause_ms": {"p50": 2.1, "p95": 4.4,
            "p99": 6.0}, "tier_bytes": [8589934592, 0, 1099511627776]},
  "parallel": {"tp": 4, "ep": 2, "nccl": true},
  "features": {"spec_draft": false, "kv_quant": "F16"},
  "build": {"version": "…", "cuda": "12.x", "sm": "70"}
}
```

- `latency.ttft/tpot/tok_s` — измеряет engine (таймер планировщика), форматы
  гистограмм-перцентилей — как в tiers-статистике (ADR-08 §2 `PauseStats`).
- `tiers` — passthrough `ITierManager::stats()` (ADR-08 §2); дашборд рендерит
  это 1:1 (VRAM-карта, hit-rate, паузы слоёв).
- Целевые пороги (p95-пауза ≤ 5 мс, hit-rate ≥ 60% — PLAN.md §4) проверяются
  bench-харнессом, не serve; serve только публикует значения.

## Альтернативы

- **Отдельный процесс engine + gRPC/IPC** — лишний hop и ещё один процесс
  против «одного процесса на порту 8080» (ADR-06); выбран in-process .so.
- **Бинарные структуры через ABI (protobuf-сериализация и т.п.)** — хрупкий
  мажорный ABI; выбраны JSON-строки (скорость границы — не проблема: 1 вызов
  на батч событий, а не на токен).
- **Serve сам вносит токены в планировщик через разделяемую память** — быстрее,
  но дублирует планировщик; выбрана явная постановка `stell_submit` + poll.

## Последствия

- Потоки A и E пишут независимо: A — реализует `stell_engine_create/submit/poll/...`
  и планировщик ≥8 сессий; E — FastAPI-обёртку, SSE, оба API, дашборд.
- CPU-путь всего стыка обязателен: CI (ADR-07) поднимает serve + CPU-движок
  без GPU и гоняет контрактные тесты событий/метрик.
- Все численные форматы метрик едины с ADR-08 (перцентили p50/p95/p99) —
  дашборд и bench потребляют один снапшот.

## Уточнение (round 1, задача T-20261017200354)

Контракт не меняется; ниже — фиксация решений, принятых при реализации
(`src/engine/capi.*`, `src/serve/mock`, `python/stell_serve`), по открытым
вопросам раунда 1.

- **Владение строками C ABI (§1).** Все `char**`-выходы (`msg_out`, `out_json`)
  — `malloc`-строки со стороны engine (`strdup`), освобождает **вызывающий**
  (`free()`; пустое сообщение — тоже выделенная строка `""`, free обязателен).
  Единое правило границы, документировано в `capi.h`; serve освобождает
  сразу после копирования (`python/stell_serve/engine.py`). Обоснование:
  ownership на вызывающем устраняет двусмысленность «кто освобождает» на
  границе языков; строки короткие (сообщения, снапшоты), стоимость копии
  несущественна.
- **`stell_step` (§1).** Round-1 расширение ABI — новый символ («поля/символы
  только добавляются», мажор ABI не меняется): однопоточный CPU-контур, serve
  ведёт шаги декода вызовами `stell_step`, пока у engine нет фоновой нити.
  Старт-стоп фоновой нити в ABI раунда 1 **не вводится**: прод-контур ведёт
  шаги сам, потребности в управлении потоком через ABI пока нет; при
  доказанной потребности — новый символ (ABI-совместимо).
- **«T=1» в sampling (§2).** Подтверждено: `temperature = 1.0` (дефолт
  `Scheduler::Sampling`, `scheduler.hpp`), НЕ greedy-argmax; greedy —
  отдельный явный режим (уточнение ADR-09 §4). Сэмплинг принадлежит runner'у
  (`IModelRunner::decode_step`); scheduler передаёт `Sampling` из request_json.
- **Chat-шаблон (§2).** Владелец — serve (подтверждено): реализован
  детерминированный линейный шаблон (`chat_template.render_prompt`) как
  заглушка до готовности реестра моделей; при появлении в описании модели
  (ADR-09) поля шаблона serve подменяет реализацию на шаблон модели —
  контракт с engine (`request_json`: `prompt_tokens`/`eos_token`) не меняется.
  Engine не знает ни шаблонов, ни API-диалектов.
- **Маппинг `cancelled` (§3).** `finish.reason = cancelled` → OpenAI
  `finish_reason: "stop"`, Anthropic `stop_reason: "end_turn"` (в обоих
  диалектах нет терминальной отмены; реализовано в `dialects.py`). Факт
  отмены не кодируется в тексте ответа, а публикуется в `/metrics` как
  `cancelled_total` — агентные клиенты (Claude Code-класс) отличают
  клиентскую отмену по метрике, а не по семантике finish_reason.
  SSE-поток завершается терминальным кадром с маппированным значением;
  `usage` в `finish` обязателен (§3) — метрики serve на нём держатся.

## Уточнение (round 2, задача T-20261017200354)

Контракт не ломается; ниже — фиксация решений по открытым вопросам раунда 2
(вердикт planner: approve, Q1/Q2/Q7).

- **`stell_logits` (§1) — ABI-совместимое расширение.** Новый символ по тому
  же прецеденту, что и `stell_step` («поля/символы только добавляются»,
  мажор ABI не меняется):

  ```c
  int stell_logits(stell_engine_t* e, const char* request_json, char** out_json);
  ```

  - `request_json`: `{"prompt_tokens":[...], "positions":"last"|"all"}`
    (по умолчанию `"last"`); токены — готовые id (serve применяет chat-шаблон
    и токенизацию, engine не знает строк).
  - `out_json`: `{"logits":[[float,...],...], "n_positions":P, "n_vocab":V}` —
    логиты финального слоя, до softmax; `"all"` покрывает ≥2 позиций
    (teacher-forcing за один вызов — нужен quality-гейтам PPL). При ошибке
    `out_json` содержит `{"error":...}`; отдельная `msg`-модель не вводится.
  - Владение строк — конвенция round 1 (см. выше): `out_json` — `malloc`-строка
    со стороны engine, освобождает вызывающий (`free()`).
  - Возврат: **0 — успех, -1 — ошибка**.

  Обоснование: ADR-11 фиксирует границу «только C ABI + JSON-строки»
  (`capi.h`); сырой указатель на буфер ломал бы владение. PPL/HumanEval/GSM8K
  прогоны в `fetch_quality_gates.py` разблокированы именно этим символом.

- **Семантика `stell_cancel` (§1) — выравнивание на конвенцию create/submit.**
  В ABI уже две функции с конвенцией `0 = успех, -1 = ошибка`
  (`stell_engine_create`, `stell_submit`); `1/0`-bool у cancel — случайное
  отступление от собственной конвенции модуля, а не осознанный выбор.
  Канон: **`stell_cancel` возвращает 0 при успехе и -1, если сессия не найдена
  или уже закрыта**. Реальный движок (`src/engine/src/capi.cpp`, был `1/0` bool)
  приводится к этому; mock (`src/serve/mock/stell_mock_engine.c`, `0/-1`)
  уже соответствует. Внутренний `Scheduler::cancel → bool` не меняется —
  маппинг выполняется в capi-слое.

- **Прод-токенизатор (§2) — byte-level BPE в src/models.** Ядро не знает
  формат `tokenizer.json` (минимальные зависимости, PLAN.md §2.3): полная
  реализация `ITokenizer` живёт в `src/models` и загружает `tokenizer.json`
  из каталога модели (файл указывается в описании модели — реестр ADR-09
  владеет лоадерами); в engine токенизатор инжектируется существующей
  контрактной точкой `Scheduler::set_tokenizer`. Byte-fallback engine в
  реестре моделей остаётся контрактной точкой (fallback при отсутствии
  `tokenizer.json`). Контракт качества: encode→decode идемпотентны на
  фикс.наборе строк; расхождение с HF-токенизатором на ASCII-текстах = 0
  (эталонные id коммитятся в фикстурах — без сети). Детокенизация token-событий
  — на стороне serve (уже реализована). Eval-контур (PPL/quality-gates)
  работает на готовых token id и от токенизатора не зависит.

## Уточнение (round 3, задача T-20261017200354)

Контракт не ломается; фиксация решения planner [Q5] по формату снапшота
`logits` в `/metrics` ([R5]).

- **Поле `logits` снапшота `/metrics` (§4) — строгий формат, только
  статистика вызовов `stell_logits`, без payload:**

  ```json
  {"requests_total": 0, "errors_total": 0, "last_n_vocab": 0, "last_positions": 0}
  ```

  Payload логитов (массивы) в `/metrics` **не передаётся никогда** (размер;
  логиты доступны только через `stell_logits`). Счётчики достаточны
  дашборду и не зависят от vocab-размера модели.
- **Дашборд** рендерит поле ключ→значение таблицей (текущий generic-рендер
  `app.js` совместим; при несовпадении имён — микроправка).
- **Тест serve** обязан проверять: (а) наличие всех четырёх полей
  в `/metrics` реального движка; (б) отсутствие payload — поле не содержит
  массивов.
- Обоснование: ADR-11 §4 — контракт снапшота; нефиксированное поле —
  скрытый контракт между serve и ui (нарушение принципа явных
  интерфейсов).

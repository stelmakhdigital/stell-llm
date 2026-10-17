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

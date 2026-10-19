# stell-llm control-plane

Тонкий Python-слой (ADR-01): FastAPI-приложение `stell_serve`, один процесс
на порту 8080 (ADR-06):

- `POST /v1/chat/completions` — OpenAI-совместимый (SSE-стриминг, tool-calls);
- `POST /v1/messages` — Anthropic-совместимый;
- `GET /health`, `GET /metrics` — снапшот движка + serve-метрики (ADR-11 §4);
- `/ui` — дашборд-монитор (статика из `src/ui`).

## Конфигурация (round 2)

`python/stell_serve/config.py` — файл конфигурации + валидация:

- источники: аргументы CLI → `STELL_SERVE_CONFIG` (или `--config`) →
  `python/stell_serve/config.json` → значения по умолчанию;
- ключи: `host`, `port` (8080, ADR-06), `model`, `engine_lib` (путь .so),
  `cors_origins`, `ui_dir`, `log_metrics`, `calibrate` (пороги арбитра
  src/calibrate: спек-драфт / KV-квант / NVMe-тир);
- неизвестные ключи и невалидные пороги — ValueError (честная валидация);
  пороги `calibrate` делегируются `CalibrationConfig.validate()`.

## Движок (round 2: вклейка реального движка)

Движок подключается строго через C ABI (ADR-11): `libstell_core.so` через
ctypes. Разрешение пути (`engine.resolve_engine_lib`):

1. `--engine-lib` / `STELL_ENGINE_LIB`;
2. **реальный движок** `build/cpu/src/engine/libstell_core.so` — SHARED-таргет
   `stell_core` потока core (round 2): классифицируется по наличию
   `stell_step` (ABI-совместимое расширение round 1 — serve сам ведёт шаги
   декода однопоточного CPU-контура); token-события несут только `token_id`
   (engine не владеет детокенизатором) — serve подставляет текст
   decimal-детокенизатором; запрос кодируется в формат engine
   (`{"sampling":…, "prompt_tokens":…, "eos_token":-1}`);
   расширения ABI round 2: `stell_logits` (логиты финального слоя по готовым
   токенам, "positions":"last"|"all" — гейты качества/дашборд) —
   доступность классифицируется по наличию символа (`supports_logits`);
   `stell_cancel` — конвенция ADR-11 0/-1 (legacy 1/0 pre-merge сборок
   распознаётся по отсутствию `stell_logits` — оба изменения приходят
   одним merge core);
3. локальная ABI-копия `build/cpu/src/serve/libstell_core_serve.so`
   (таргет `stell_core_serve_check` из src/serve — те же исходники
   src/engine для контрактных тестов, пока core не вклеил свой таргет);
4. **CPU-fallback mock** `build/cpu/src/serve/libstell_mock_core.so`
   (`src/serve/mock`, тот же ABI; события несут готовый текст).

Сборка:

```bash
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
cmake -S . -B build/cpu -G Ninja -DSTELL_CUDA=OFF
cmake --build build/cpu   # stell_mock_core + stell_core_serve_check
```

Запуск (Python-префикс `~/.local/stell-conda/envs/serve`):

```bash
PYTHONPATH=python ~/.local/stell-conda/envs/serve/bin/python -m stell_serve --port 8080
```

## Метрики и дашборд (round 2)

- `/metrics` = снапшот движка (passthrough) + serve-секция (счётчики,
  TTFT/TPOT/tok/s p50/p95/p99, лог-кольцо);
- **`cancelled_total`** — факт отмены в `/metrics` (уточнение ADR-11 round 2:
  `finish.reason=cancelled` → OpenAI `stop` / Anthropic `end_turn`, факт
  отмены не кодируется в finish_reason, а публикуется метрикой);
  serve-счётчик — единственный источник для реального движка (у mock есть
  additive-поле `engine.cancelled_total`);
- passthrough tier-статистики (hit-rate, паузы p50/p95/p99, tier_bytes) —
  от mock; у реального движка round 1 — min-снапшот (только `engine`-блок),
  дашборд рендерит через optional chaining;
- скриншоты до/после — инструментов нет, честно pending (см. отчёт round 2).

## Calibrate ↔ bench (round 2)

- `python -m stell_calibrate` — прогон арбитра на CPU-замерах mock-движка
  (A/B фич: спек-драфт / KV-квант / NVMe-тир), отчёт в
  `bench/results/local/` (gitignored); входы PPL/accept-rate — синтетические
  (ABI не отдаёт логитов — расхождение в отчёте);
- `python -m stell_bench` — bench-прогон по §5 (mock: pending=true);
- `python -m stell_bench.quality_gates_runner` — GSM8K-32 / HumanEval-32 /
  PPL: датасеты РЕАЛЬНЫЕ (выборка 32 задач seed=42, скрипт
  `stell_bench.fetch_quality_gates`), значения — честный pending.

Зависимости: `pip install -r requirements.txt` (в префикс serve, не в
envs/stell и не в системные места).

## Замечания по решению SSE

sse-starlette установлен как зависимость, но wire-формат SSE форматируется
вручную (`dialects.sse`): оба диалекта требуют точного формата кадра
(OpenAI: `data: {...}` + `data: [DONE]`; Anthropic: именованные `event:`),
которым ручной генератор даёт полный контроль. sse-starlette оставлен для
будущих keepalive/heartbeat политик.

## Статус

Реализовано и покрыто тестами (`tests/serve/`): контракты обоих API,
стриминг, tool-calls, CORS, health/metrics (включая `cancelled_total`),
биндинг ABI **реального движка** (src/engine, включая `stell_step` —
ABI-копия `stell_core_serve_check`) и mock. Реальный SHARED-таргет
`libstell_core.so` от потока core ещё не вклеен в master — реальный путь
покрыт ABI-копией тех же исходников; при вклейке путь подхватывается
автоматически (`build/cpu/src/engine/libstell_core.so`).

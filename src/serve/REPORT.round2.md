# Отчёт раунда 2 — serve (Поток E), T-20261017200354

Статус: раунд 2 выполнен; ветка `agent/serve/2` (от master `2a00229`),
коммиты локальные (НЕ запушены). pytest 75 (40 round 1 + 35 новых) зелёный;
ctest 29/29 зелёный.

## 1. Вклейка реального движка (C ABI, ADR-11)

- `python/stell_serve/engine.py` — биндинг расширен до реального движка:
  `resolve_engine_lib()` — порядок: явный путь → `STELL_ENGINE_LIB` →
  `build/cpu/src/engine/libstell_core.so` (SHARED-таргет `stell_core`
  потока core, когда он вклеен в master) → локальная ABI-копия
  `build/cpu/src/serve/libstell_core_serve.so` → CPU-fallback mock.
  Классификация `kind` по наличию `stell_step` (ABI-совместимое расширение
  round 1): у реального движка есть — serve сам ведёт шаги декода
  (`backend.step()`), у mock его нет — шаги ведёт движок.
- Реальный движок: token-события несут только `token_id` (engine не владеет
  детокенизатором, scheduler.hpp) — serve подставляет текст
  `decimal_detokenizer` (token_id → десятичная цифра, детерминированно);
  запрос кодируется в формат engine (`{"sampling":…, "prompt_tokens":…,
  "eos_token":-1}`); токенизатор — byte-level заглушка до реестра моделей
  (ADR-09). `app.py`: drive-шаги в drain_events и обоих SSE-генераторах,
  кап `MAX_POLL_ROUNDS`.
- ABI-совместимость с `capi.h` из master проверена тестом
  (`test_abi_symbols_match_master_capi_h`): все объявленные символы
  (включая `stell_step`) доступны из .so; ABI-копия собрана из тех же
  исходников `src/engine` таргетом `stell_core_serve_check` в
  `src/serve/CMakeLists.txt` (каталог src/engine НЕ редактировался).
- Тесты: `tests/serve/test_real_engine.py` (15): ABI-символы, kind,
  submit/poll/step, cancel→cancelled, slot-overflow, метрики, API обоих
  диалектов на реальном движке, `/metrics` реального снапшота, дашборд,
  resolve-пути. Честный SKIP: если ABI-копия не собирается (нет тулчейна) —
  `pytest.skip`, не молчаливый фейк.

## 2. Конфигурация (round 2)

- `python/stell_serve/config.py` + `config.json` (коммитнут): host/port
  (8080, ADR-06), model, engine_lib, cors_origins, ui_dir, log_metrics,
  `calibrate` (пороги арбитра src/calibrate — делегируются
  `CalibrationConfig.validate()`); неизвестные ключи/порти/пороги —
  ValueError; источники: CLI → `STELL_SERVE_CONFIG`/`--config` →
  config.json → дефолты. `__main__.py` переписан под конфиг.
- Тесты: `tests/serve/test_config.py` (11).

## 3. cancelled_total + passthrough метрик

- `/metrics` = passthrough снапшот движка + serve-секция;
  `serve.cancelled_total` — счётчик serve (уточнение ADR-11 round 2:
  `cancelled` → `stop`/`end_turn`, факт отмены — метрикой, не в
  finish_reason); тест на cancel-endpoint: счётчики инкрементируются.
- Расширение метрик движка (ABI позволяет — additive-поле, контракт не
  менялся): mock публикует `engine.cancelled_total` (реальный движок
  round 1 — min-снапшот, поля не отдаёт; служебный счётчик serve —
  единственный источник факта отмены). Тир-статистика passthrough
  (hit-rate, паузы p50/p95/p99, tier_bytes) покрыта тестами round 1 +
  реальный снапшот (без tiers — честно, дашборд рендерит optional chaining).

## 4. Calibrate ↔ bench интеграция

- `src/calibrate/stell_calibrate/runner.py` (+`__main__.py`): прогон
  арбитра на CPU-замерах через mock-движок (C ABI): A/B фич
  (спек-драфт / KV-квант / NVMe-тир); отчёт — `bench/results/local/`
  (gitignored): `bench/results/local/20261017-calibrate-local/report.json`.
- Результат (mock, синтетика): spec_draft=on (speedup 1.39 ≥ 1.05, accept
  0.7), kv_quant=on E4M3 (speedup 1.17 ≥ 1.03, penalty 0.5% ≤ 1%),
  nvme_tier=on (ratio 0.4 ≥ 0.25), граница тиров 3 GiB. Честность в notes:
  таймер — синтетическая модель декода (sleep по заявленной tpot mock,
  НЕ GPU-честно); accept-rate/PPL/ram-nvme/max_group_bytes — синтетические
  входы (нет модели/логитов). Прогон повторяем: тесты
  `tests/serve/test_calibrate_runner.py` (4) воспроизводят решение.

## 5. Quality-gates (PLAN.md §5)

- Датасеты РЕАЛЬНЫЕ (round 1 были синтетическими заглушками): скачаны
  скриптом `src/bench/stell_bench/fetch_quality_gates.py` (только
  urllib/gzip stdlib, seed=42, 32 задачи из каждого открытого датасета
  OpenAI): GSM8K (test.jsonl, 1319) → `data/gsm8k_32.jsonl` (10 КиБ),
  HumanEval (HumanEval.jsonl.gz, 164) → `data/humaneval_32.jsonl` (45 КиБ);
  файлы коммитнуты (объём малый), schema.json/README обновлены (status:
  real).
- Runner `src/bench/stell_bench/quality_gates_runner.py`: честный
  pending=True — нет модели/весов; ABI ADR-11 round 1 не отдаёт логитов
  (PPL) и полных генераций (проверка HumanEval/GSM8K) — расхождение
  контракта зафиксировано в notes; отчёт в
  `bench/results/local/20261017-quality-gates-local/report.json`.
- PPL: текст round 1 не менялся (схема §5: сравнение версий), прогон
  pending.

## 6. Дашборд

- Соответствие /metrics реальному движку проверено тестом
  (`test_dashboard_serves_on_real_engine`): у реального движка нет
  tiers/latency — app.js рендерит через optional chaining, /metrics честно
  отдаёт только engine+serve; статика /ui отдаётся.
- Скриншоты до/после: инструментов нет в окружении (headless-браузер/снимки
  недоступны) — честно **pending**; верификация снапшота — через
  контрактные тесты /metrics выше.

## Расхождения ABI/контракта (обнаружены, зафиксированы, не меняли контракт)

1. **Коды возврата cancel**: mock — `0` ок / `<0` неизвестная сессия;
   реальный движок — `1` ок / `0` неизвестная (scheduler.cancel → bool).
   Биндинг обрабатывает оба по `kind` (engine.py); контракт (ADR-11 §1)
   не менялся — для выравнивания нужен клариф ADR или фиксы с обеих сторон.
2. **Метрики реального движка — min-снапшот**: только `engine`-блок; нет
   latency/tiers/features/cancelled_total (см. ADR-11 §4 — полный формат).
   `/metrics` честно отдаёт что есть; passthrough не выдумывает данные.
3. **ABI без логитов/полных генераций**: PPL и проверка HumanEval/GSM8K
   не вычислимы — гейты pending до нового ABI-символа (совместимо) и
   весов.
4. **Детокенизатор/токенизатор serve — заглушки** (decimal/byte) до
   реестра моделей (ADR-09); контракт request_json не меняется.

## Не сделано и почему

- **SHARED-таргет `stell_core` от потока core не вклеен в master**
  (на момент раунда master — `2a00229`, src/engine STATIC): реальный путь
  покрыт ABI-копией тех же исходников (`stell_core_serve_check`) + честный
  SKIP; при вклейке путь подхватывается (`build/cpu/src/engine/`).
- **Прогон bench/gates на реальной модели**: нет весов/модели — честный
  pending (см. §4/§5).
- **Скриншоты дашборда**: инструментов нет — pending (§6).

## Открытые вопросы

- Выравнивание кодов возврата `stell_cancel` (см. расхождение 1) — нужно
  решение через ADR (mock-сторона или реальный движок).
- Расширение ABI логитами (для PPL/гейтов) — предложение round 3:
  новый символ (ABI-совместимо), например `stell_logits(e, session_id,
  char** out_json)`.
- Формат token-событий реального движка: только `token_id` — нужен
  договор с core о детокенизаторе (serve-свой vs поле `text` в событии).
- Владелец SHARED-таргета `stell_core` в master — поток core; биндинг
  готов к вклейке (`STELL_ENGINE_LIB`/`build/cpu/src/engine/`).

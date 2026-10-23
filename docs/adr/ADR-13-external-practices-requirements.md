# ADR-13: Внешние практики — требования к спекуляции, tiering'у и планировщику

- **Статус:** Принято (round 9, задача T-20261022183913; источник — исследование
  `external-practices-notes.md` в задаче: 1Cat-LLM/1cat-vllm, SGLang, DeepInfer,
  project-maya, Strata OSDI'26)
- **Стороны:** `engine` (спекуляция, планировщик), `tiers` (экспертный tier-flow),
  `serve`/`bench` (метрики, quality-gates)
- **Правило изменения:** только через ADR и согласование с worker.
- **Связано:** ADR-08 (continuous batching), ADR-09 (драфтер-плагин),
  ADR-11 (метрики/C ABI), ADR-12 (тензорные группы)

## Контекст

Round 8 закрыл спекулятивный контур (драфт + FP32-верификация + арбитр).
Исследование внешних прод-систем (см. источники) подтвердило архитектуру и
выявило класс практик «configured ≠ executed», динамической глубины драфта,
экспертно-гранулярного offloading и mixed-prefill бюджета, которые у нас
отсутствуют или вырождены. Настоящий ADR фиксирует их как требования
следующих раундов (нумерация [R1]…[R6] — из `external-practices-notes.md`).
Сопоставление с внешними системами — только здесь и в PLAN.md (по правилам
spec: в коде/UI/конфигах названий чужих стеков нет).

## Решение

### [R1] features: configured ≠ executed (сделано в этом раунде)

Снапшот `stell_metrics` различает **настроенный** путь и **фактическое
исполнение** (прецедент capability-report 1Cat):

- `features.spec_draft` — bool, остался для обратной совместимости
  (bench `harness.py`, python control-plane, UI);
- `features.spec_draft_configured` — контур собран и подключён (`ctx->spec`);
- `features.spec_draft_executed` — был хотя бы один состоявшийся спек-раунд
  (`telemetry().rounds > 0`);
- `features.spec_draft_reason` — `not_configured | arbiter_disabled |
  disabled_by_config | enabled` (арбитр-отключение различается от
  конфиг-отключения: `SpeculativeDecoder::auto_disabled()`).
- `spec.n_draft_effective` — действующая глубина следующего раунда
  (при adaptive_draft отличается от базовой, см. [R2]).

### [R2] Динамическая глубина драфта n_draft (сделано в этом раунде)

`SpeculativeDecoder::Config` получил `adaptive_draft` (default false —
поведение существующих контуров не меняется), `n_draft_min` (1),
`n_draft_max` (8). Политика (`effective_n_draft()`): при пустом окне —
базовый `n_draft`; далее
`n_eff = round(n_draft * (0.5 + accept_rate))`, зажат в
`[n_draft_min, n_draft_max]` — при высоком accept-rate глубина растёт,
при низком сжимается (дешёвые раунды перед возможным отключением арбитром).
Покрыто e2e-тестами (оракул-драфтер: рост до 6 при ar=1.0; reject-драфтер:
сжатие до `n_draft_min` при ar=0; качество — побитово == plain greedy).
Дальнейшее (батч-верификация на CUDA, NGRAM/MTP-драфтеры) — см. [R5]/pending.

### [R3] Экспертно-гранулярный tier-flow + hit-rate матрица (tiers, будущие раунды)

Требование (прецедент project-maya: sparse router offloading): выгружать
ВЫБРАННЫЕ роутером эксперты, а не слои целиком. Текущий layer-пиннинг делает
матрицу `{LRU,LCU}×{k}` вырожденной — это основной pending round 8.
Требования: (а) tier-операции на уровне отдельных экспертных групп внутри
слоя (ADR-08: группы с ролью `expert` уже существуют); (б) trace-driven
замер hit-rate prefetch для матрицы `{LRU,LCU}×{k}` с фиксацией в
`bench/results/` (контракт PLAN.md §5). Критерий: MoE > VRAM, p95-пауза
≤ 5 мс, prefetch ≥ 60% попаданий (spec, Поток B) — на экспертной гранулярности.

### [R4] Mixed-prefill budget для continuous batching (engine, будущие раунды)

Префиллы не должны блокировать resident-декодеров (прецедент 1Cat #860:
6.88 → 30.34 tok/s resident при входящих 32K-промптах). Требование: бюджет
префилл-токенов на шаг планировщика, сохраняющий decode-ёмкость; soft target,
не latency-гарантия; честная телеметрия (доля шагов, прижатых бюджетом).
Связано с p95-паузами tier-контурa (spec Поток B) и ADR-08 §5.

### [R5] Драфтер-плагины: NGRAM / lookup (engine; NGRAM реализован в round 11)

Семейство драфтеров с общим интерфейсом — `SpeculativeDecoder::DrafterFn`
(плагин). **Round 11 (T-20261023040642 R1): NGRAM-драфтер реализован**
(`ngram_drafter.hpp/cpp`): статистика n-грамм (хвостовое окно ≤ max_order,
FNV-1a ключ, инкрементальный индекс по подтверждённому контексту) — БЕЗ
второй модели; lookup-продолжение = suffix-поиск в индексе (включён в
NGRAM). Выбор: `spec.drafter:"model"(default)|"ngram"` (оба сразу — ошибка
конфигурации); снапшот честно помечает `spec.drafter`. Кандидаты
верифицируются целью (FP32) — качество побитово == greedy; accept-rate
решает арбитр-калибровка (PLAN.md §2.4). Тесты: `test_engine_ngram_drafter`
(юнит: период/пустой драфт/приращение; e2e: побитово == greedy, раунды
состоялись) + C ABI-кейс в `test_engine_spec_capi`. **Pending:** MTP- и
block-diffusion-драфтеры, батч-верификация на CUDA.

### [R6] PPL-delta gate спекуляции (bench/quality-gates, будущие раунды)

Помимо побитового равенства greedy-путей (уже покрыто тестами): гейт
PPL-дельты цели между plain и спекулятивным путём на семплинг-конфигах
(прецедент 1Cat: delta +0.00092%) — фиксация в bench-отчётах как отдельное
поле (контракт PLAN.md §5). Гейт применяется к будущим драфтерам с
семплингом (NGRAM/MTP), где побитовое равенство не гарантировано.

## Альтернативы

- Не фиксировать кандидаты до появления CUDA-контура — отклонено: требования
  [R3]/[R4] меняют контракты ADR-08/ADR-12 (гранулярность tier-операций,
  телеметрия планировщика), их надо знать заранее, чтобы не ломать API.
- Сломать совместимость `features.spec_draft` (объект вместо bool) —
  отклонено: ломает bench harness.py, python control-plane и UI; выбран
  аддитивный набор полей (прецедент round 8: аддитивные ABI-расширения).

## Последствия

- [R1]/[R2] реализованы в round 9 (engine); тесты — `test_engine_spec_e2e`;
  базелайны сохранены (ctest 56/56, pytest 194+2skip).
- Round 10 (T-20261022183913 R2): [R3] экспертно-гранулярный tier-flow —
  реализован (TierManager::pin_groups + MoE-контур ops_runner: пин не-
  экспертных групп + роутера + shared, выбранные эксперты — после
  маршрутизации, невыбранные вытесняемы; hit-rate матрица {LRU,LCU}×{k}
  измерена — `bench/results/20261022-tiers-hitrate-matrix/`, ≥60% хотя бы
  в одной ячейке, p95 групп ≤ p95 слоя целиком); [R4] mixed-prefill budget —
  реализован (Scheduler::prefill_token_budget + телеметрия
  prefill_budget_deferrals); [R6] PPL-delta gate — реализован
  (bench-поле `spec_ppl_delta`, planner R2 [R4]). [R5] (NGRAM/lookup-дрáфтеры)
  — остаётся pending. Qwen3_5Moe (планировщик R2 [R5], models) — синтетическая
  поддержка, см. ADR-09. CPU-прогоны: ctest 58/58,
  pytest 202+2skip; GPU-контуры (NCCL/CUDA-окно) — честный pending
  (co-tenancy).
- Честность: в раундах 9–10 не заявлено ускорение CPU-спекуляции
  (sequential-верификация — ограничение задокументировано в
  `spec_decode.hpp`); [R2] меняет только глубину драфта, не скорость.

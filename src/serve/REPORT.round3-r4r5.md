# Отчёт раунда 3 — serve (Поток E), T-20261017200354 — [R4]/[R5]
# (подагент serve-4; ветка agent/serve/4 от master 545ef99; R1/R2-интеграцию
# раунда см. REPORT.round3.md от serve-3)

## 1. [R4] Песочница HumanEval

- `src/bench/stell_bench/code_sandbox.py` — process-sandbox-раннер:
  subprocess (`sys.executable -I`, stdin/stdout DEVNULL/DEVNULL),
  `preexec_fn` ставит **до exec**: RLIMIT_CPU=5с, RLIMIT_AS=512 МиБ,
  RLIMIT_NOFILE=64; wall-таймаут 10с (kill + reap — раннер не висит);
  cwd — изолированный temp-dir (удаляется после кейса); env пустой
  (ничего из родителя; интерпретатор сам ставит LC_CTYPE для stdio).
  **Честная пометка в докстринге: process-sandbox, НЕ security-граница**
  (права пользователя, сеть не отключается — unprivileged seccomp/
  namespaces недоступны; граница достаточна bench-этапу, планировщик [Q4]).
  Классификация исхода: `ok` (exit 0) / `timeout` (wall) / `error`
  (исключение, sys.exit(n), сигнал — вкл. SIGXCPU от RLIMIT_CPU).
- Gates-runner `--live`: HumanEval-32 больше НЕ pending. Конвейер кейса:
  жадное декодирование (argmax, ≤32 токенов, проекция байта → vocab) →
  детокенизация → `extract_code` (первый markdown-fence ```…```, иначе
  весь ответ; пустой ответ → None) → candidate = prompt + completion →
  исполнение в песочнице против test-кейса задачи + `check(entry_point)`
  → accuracy = доля exit 0. Протокол зафиксирован константами
  HUMANEVAL_PROMPT_TOKENS=64, HUMANEVAL_MAX_STEPS=32 и в report.json
  (`protocol.humaneval_sandbox`).
- Результат закоммичен: `bench/results/20261019-quality-gates-live/`
  (обновление существующей директории того же дня; provenance в report
  и run.log). Числа: humaneval_32[mini_dense]=0.0, humaneval_32[mini_moe]=0.0
  (исходы песочницы error=32 — mini-фикстуры декодируют цифры/разделители,
  не код: честная механическая точка регрессии пайплайна, НЕ оценка LLM);
  PPL/GSM8K совпали с базовым прогоном round 2 до последней цифры
  (18.3928/30.0858; 0.0625/0.0) — детерминизм прогона сохранён.

## 2. [R5] Строгий формат logits-снапшота

- `python/stell_serve/metrics.py`: `LOGITS_SNAPSHOT_FIELDS` =
  {requests_total, errors_total, last_n_vocab, last_positions} +
  `is_valid_logits_snapshot` (ровно 4 поля, значения — числа; массивы/
  объекты/bool/str/None — нарушение: payload в /metrics не проходит).
- `tests/serve/test_logits_metrics.py`: чистая валидация формата на
  синтетике (6 тестов — контракт проверен уже сейчас) + контракт на
  РЕАЛЬНОМ движке: `/metrics` содержит logits ровно с 4 полями без
  массивов. Реальный движок (ABI-копия build/cpu/src/engine/libstell_core.so,
  stell_metrics = min-снапшот) поля пока НЕ имеет → **честный SKIP по
  наличию поля, включится после merge core-4 автоматически** (2 SKIP).
- Дашборд (`src/ui/app.js`): renderLogits рендерит СТРОГО 4 KV
  (фиксированная схема LOGITS_FIELDS с подписями; лишние ключи игнорируются,
  отсутствующие — «—», отсутствующий блок — честное «нет данных»).
  Generic-рендер Object.entries(logits) удалён — формат теперь контракт
  ADR, а не скрытый договор serve↔ui. Тест в test_dashboard.py.

## 3. ADR-11 §4

Каталог docs/ вне зоны раунда (запрет в постановке) → готовый текст
вклейки «Уточнение (round 3): строгий формат logits-снапшота» —
`src/serve/ADR-11-round3-amendment.md` (для потока docs).

## 4. Тесты до/после

- До: pytest 112 passed + 2 skipped (merge core), ctest 44/44.
- После: pytest **141 passed + 4 skipped**, ctest 44/44 (C++ не менялся).
  Новые: test_code_sandbox 20, test_logits_metrics 8 (2 — SKIP [R5]),
  test_dashboard +1, test_quality_gates_live: humaneval pending→live
  (+2 конвейер/протокол). Старые 112 не сломаны.

## 5. НЕ сделано / причины

- Поле `logits` в stell_metrics src/engine — зона core (merge core-4);
  serve-сторона (валидатор, тесты с честным SKIP, строгий дашборд) готова.
- ADR-11 §4 напрямую — docs/ запрещён к правкам в этой постановке
  (материал передан, см. п.3).
- Сеть в песочнице не отключается (см. честную пометку) — контейнерная
  изоляция вне рамок [Q4].

## 6. Открытые вопросы

- Потоку core-4: поле `logits` в stell_metrics — строго 4 числовых поля
  (формат в ADR-11-round3-amendment.md); после merge SKIP-тесты [R5]
  включатся сами.
- Потоку docs: вклеить amendment в ADR-11 §4.

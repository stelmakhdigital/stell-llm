# serve — раунд 5, [R8-координация] (Поток E): судьба фонового прогона, обёртка перезапуска, wall до/после [R1]

Ветка: `agent/serve/6` (от master `7a6b770`). Зона: `src/bench/stell_bench`,
`tests/serve`. Чужие каталоги (src/engine, src/kernels, src/tiers, src/models,
src/parallel, src/ui, src/calibrate, docs), корневые CMakeLists.txt/.gitignore —
не тронуты. Тулчейн: python env `~/.local/stell-conda/envs/serve`.

## 1. [R8-координация]: статус фонового прогона — ЧЕСТНЫЙ ФАКТ: УБИТ ПО TIMEOUT

Фоновый R8-прогон на tiny-qwen2moe был запущен worker'ом (раунд 4→5) на
сервере `ssh arkalaust@192.168.1.114:2214` (серверный `/tmp`, не локальный —
локально логов нет):

- **запуск**: 2026-10-19 **16:39** по серверному времени (TZ **+05**;
  в задаче «16:39 UTC+?» — это 11:39 UTC). Точная команда (из `ps`):
  ```
  setsid bash -c 'export PYTHONPATH=python:src/bench; timeout 4800 \
      python3 -m stell_bench.quality_gates_runner --real --live \
      --real-model-dir ~/models/tiny-qwen2moe --co-tenancy \
      --results-dir bench/results/20261019-quality-gates-real-tiny-qwen2moe \
      > /tmp/r8_real.log 2>&1; echo $? > /tmp/r8_rc.flag'
  ```
  (репо сервера: `/home/arkalaust/Code/stell-llm`; лог `/tmp/r8_real.log`,
  rc-флаг `/tmp/r8_rc.flag`, wall-бюджет timeout(1) **4800 с**).
- **код прогона** (проверено по требованию задачи): серверный чекаут —
  master `7a6b770`, содержащий merge `10e8434` → фиксап **incremental-last в
  прогоне ЕСТЬ** (PPL не зависает на 300+ МиБ JSON); движковая оптимизация
  [R1] в master **НЕ входит** — `src/engine/src/ops_runner.cpp:508`:
  `run_prompt_logits = run_prompt_logits_all(prompt)` материализует все
  строки (O(n²) на повторных `positions:"last"`). Прогон шёл на
  **неоптимизированном движке**, как и планировалось для «before»-цифры.
- **исход**: процесс жил `elapsed 79:57` при CPU 99.9% и был **убит
  timeout'ом 4800 с** — `rc=124` в `/tmp/r8_rc.flag`, лог пуст (runner
  печатает только в конце), `report.json` НЕ создан, каталог результатов
  не появился. Наблюдение закрыто в 18:00 +05 (= 13:00 UTC).
- **следствие**: «если завершится — worker закоммитит точку регрессии»
  НЕ реализовалось: фоновый прогон **не сложился в точку регрессии**.
  Wall-факт «до» [R1]: **wall ≈ 4800 с, убит по бюджету timeout (rc=124),
  НЕ завершён** — честная «before»-цифра-факт (не измерение завершённого
  прогона).

**Решение по «before»-цифре** (допущение зафиксировано): неоптимизированный
прогон с расширенным бюджетом НЕ перезапускать. O(n²) мог бы тянуть много
часов; [Q1] planner'а ровно про это — оптимизация [R1] устраняет сам этот
путь, и сжигание сервера под co-tenancy ради цифры противоречит духу
planner-решений (ср. [Q2] «дальнейшие попытки … запрещены»). «Before» = факт
убийства по бюджету (4800 с, rc=124); «after» = перезапуск обёрткой после
слияния [R1] — обе цифры в артефакте через wall-поля provenance.

## 2. Что сделано

### 2.1 Обёртка перезапуска: `src/bench/stell_bench/run_real_gates.sh` (NEW)

Единый протокол запуска/перезапуска R8-прогона (worker'ом, из корня репо):

- **env-управление** (все параметры необязательны): `STELL_R8_MODEL_DIR`
  (по умолчанию `~/models/tiny-qwen2moe`; пустая строка → честный pending),
  `STELL_R8_RESULTS_DIR` (по умолчанию `bench/results/<дата>-quality-gates-real-<слаг модели>`),
  `STELL_R8_TIMEOUT` (4800), `STELL_R8_CO_TENANCY=1` → `--co-tenancy`,
  `STELL_R8_ENGINE_LIB`, `STELL_R8_DRY_RUN=1` (печать команды без запуска);
- **корректные env**: `PYTHONPATH=<корень>/python:<корень>/src/bench`
  (корень вычисляется от местоположения скрипта — работает и в worktree,
  и на серверном чекауте); экспортирует `STELL_R8_REAL_MODEL_DIR` по прецеденту R7;
- **wall-замер date-диффом**: `date +%s.%N` до/после, разница awk'ом;
- **ко-tenancy-пометка**: `STELL_R8_CO_TENANCY=1` → `--co-tenancy` в runner
  (provenance.co_tenancy);
- **запись provenance ВСЕГДА**: `wall.json` в results-dir пишется даже при
  отказе/таймауте (`exit_code`, `killed_by_timeout` для rc=124, model_dir,
  engine_lib, `git_commit`, hostname, `date_utc`, co_tenancy, `wall_seconds`,
  `wall_phase`, `wall_before_seconds`) — урок убитого прогона: report.json не
  успел создаться, и wall-факт обязан переживать отказ;
- **обе цифры в артефакте**: после завершения обёртка вплетает
  `wall_seconds`/`wall_phase`/`wall_before_seconds` в provenance
  **report.json** (коммитимой точки регрессии);
- веса в git не попадают (каталог модели — вне репо; в results-dir — только отчёты).

**Точная команда перезапуска worker'ом после [R1]** (обе цифры):

```bash
STELL_R8_MODEL_DIR="$HOME/models/tiny-qwen2moe" \
STELL_R8_CO_TENANCY=1 STELL_R8_TIMEOUT=14400 \
STELL_R8_WALL_PHASE=after STELL_R8_WALL_BEFORE=4800 \
bash src/bench/stell_bench/run_real_gates.sh
# → bench/results/<дата>-quality-gates-real-tiny-qwen2moe/{report.json,wall.json,run.log}
#   report.json.provenance: ppl_mode=incremental-last, co_tenancy=true,
#   wall_phase=after, wall_seconds=<замер>, wall_before_seconds=4800
#   («до»: убит timeout'ом 4800, НЕ завершён — см. раздел 1);
#   артефакт закоммитить как точку регрессии (веса не в git).
```

`STELL_R8_TIMEOUT=14400` — запас: после [R1] прогон должен уложиться в минуты,
но honest-бюджет не должен снова убить завершённый прогон; wall-замер — не
бюджет, цифра «после» будет фактической.

### 2.2 Готовность кода принять/обогатить provenance — `quality_gates_runner.py`

- **wall-поля provenance** ([R8]/planner [R1]): новая чистая функция
  `wall_provenance(wall_seconds, wall_phase, wall_before_seconds)` — только
  переданные поля («не измерено — не записано»), rounding, громкая ошибка на
  чужой фазе; CLI-флаги `--wall-seconds`, `--wall-phase {before,after}`,
  `--wall-before-seconds` — применяются и к live-real, и к pending
  provenance; протокол-константы дополнены `wall_measurement` (метод замера).
  Семантика фаз: `before` — неоптимизированный движок (O(n²) на
  `positions:"last"`), `after` — после [R1] (проекция только последней позиции).
- **dtype-происхождение из [R5] models**: `real_provenance()` теперь несёт
  `dtype_weights` (профиль описания, определяет weights_format) и `dtype_orig`
  (исходный dtype весов до конверсии; по ADR-03 round 5 — BF16-источник →
  `"bf16"` с clamp-сатурацией ±65504). Пока описание не несёт поле — честный
  `None` (не догадки): код готов принять значение, как только поток models
  ([R5]) начнёт его писать.
- `ppl_mode` (incremental-last/all) и `co_tenancy` — уже в provenance (round 4):
  артефакт-приёмка дополнительных полей не требует.

## 3. Тесты (`tests/serve/test_r8_wrapper.py`, NEW — 9)

- обёртка: валидный shell + исполняемость; dry-run (PYTHONPATH = ровно
  `<корень>/python:<корень>/src/bench`, команда `timeout … --real --live`,
  артефакты не создаются); полный pending-контур (report.json
  `real-model-pending` + wall.json: exit_code/wall_seconds/git_commit/
  co_tenancy/date_utc); вплетение wall-полей в provenance report.json
  (wall_phase=after, wall_before_seconds=4797.5 — один источник с wall.json);
  отказ (битый источник, rc=1) — wall.json ВСЁГДА записан, report.json нет;
  results-dir по умолчанию — слаг из имени каталога модели;
- runner: `wall_provenance()` (поля/validation/пустота), CLI-флаги →
  provenance pending-отчёта; `dtype_weights`/`dtype_orig`/`weights_format`
  в real_provenance (в т.ч. описание с `dtype_orig: bf16` — как от [R5] models).

## 4. Протокол (spec)

- pytest: до — **175 passed + 2 honest SKIP**, после — **184 passed + 2 SKIP**
  (+9 новых, старые не сломаны);
- ctest: **47/47** (код C++ не менялся; на ветке 2 теста потока models
  сначала «Not Run» — бинарники не собраны в этом worktree, собраны и
  пройдены: `models_weights_dir_mixed`, `models_hf_description`);
- веса/датасеты в git не попадают (обёртка пишет только отчёты).

## 5. Ключевые решения

1. **Wall-факт переживает отказ** (`wall.json` всегда + патч provenance
   report.json): фоновый прогон умер без report.json — повторять этот паттерн
   недопустимо; артефакт обязан фиксировать и отказ (killed_by_timeout).
2. **«Before» = факт убийства по бюджету (4800 с, rc=124), не завершённый
   замер**: неоптимизированный перезапуск с расширенным бюджетом не делаем
   (допущение выше; против [Q2]-духа сжигания сервера) — семантика цифры
   зафиксирована в отчёте и в wall.json (`killed_by_timeout`).
3. **Wall-поля — в provenance, не в отдельном бокфайле**: обе цифры живут в
   самой точке регрессии (требование planner [R1]); wall.json — raw-дубль
   с полным контекстом (commit, hostname, timeout).
4. **dtype_orig — «принять, не выдумывать»**: поле читается из описания ADR-09;
   отсутствие — честный None до слияния [R5] models.
5. **Обёртка — в `src/bench/stell_bench`** (рядом с runner'ом, зона serve —
   bench-инфраструктура качества; python-часть потока E не менялась).

## 6. НЕ сделано / почему

- **Перезапуск прогона после [R1] и коммит точки регрессии tiny-qwen2moe** —
  worker: [R1] (движковая оптимизация core-6) ещё не слит в master (серверный
  чекаут `7a6b770`), перезапускать до слияния — снова O(n²). Команда — в §2.1.
- **Артефакт убитого прогона в bench/results не коммитится**: report.json не
  создался, замеренного значения нет; факт (rc=124, 4800 с, команда, код)
  зафиксирован здесь и попадёт в артефакт через `STELL_R8_WALL_BEFORE`.
- **NCCL [R7]** — не в зоне; только чистое окно (planner [Q2]).

## 7. Открытые вопросы

1. **«Before»-цифра незавершённого прогона** — достаточен ли факт «убит по
   бюджету 4800 с (rc=124)» для сравнения до/после, или planner требует
   завершённый before-замер (тогда — отдельное окно на неоптимизированном
   движке с расширенным бюджетом, часы под co-tenancy)?
2. **`dtype_orig`**: подтвердить имя поля/значения у потока models ([R5]) —
   код читает `description.json → dtype_orig` (ожидается `"bf16"`).
3. **Слаг results-dir**: дефолт обёртки — `<дата>-quality-gates-real-<имя
   каталога модели>`; для повторного прогона той же модели в другой день
   каталог будет новым (дата в имени) — сравнение через wall_before_seconds
   в provenance; устраивает ли такой протокол идентификации точек?

## 8. Коммиты

- `src/bench/stell_bench/run_real_gates.sh` — NEW: обёртка (env, date-дифф,
  co-tenancy, wall.json всегда, патч provenance report.json, dry-run);
- `src/bench/stell_bench/quality_gates_runner.py` — wall_provenance() +
  CLI wall-флаги + `wall_measurement` в протокол-константах +
  `dtype_weights`/`dtype_orig` в real_provenance;
- `tests/serve/test_r8_wrapper.py` — NEW: 9 тестов;
- хеши: см. git log ветки `agent/serve/6` (заполняется при коммите).

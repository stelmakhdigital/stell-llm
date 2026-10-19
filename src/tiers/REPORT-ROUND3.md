# tiers — отчёт раунда 3 (T-20261017200354, Поток B, роль tiers)

Ветка `agent/tiers/4` (база `dedb67e`; в постановке указан master `545ef99` —
у моей ветки родитель `dedb67e`, расхождение — на merge worker'а, конфликтов
по моим файлам не ожидается: все изменения в src/tiers, tests/tiers и двух
тестах tests/core/engine). Коммиты локально, НЕ пушено:

- `ba42267` — fix(tiers): корректный учёт VRAM-слотов при eviction + LRU-ранг
  префетченных групп (+ синхронные правки тестов-потребителей);
- `8f8d989` — feat(tiers): [R1] CRC-политика round 3 (payload-CRC вне
  горячего пути, ADR-12 §2.3);
- `55e42e2` — feat(tiers): [R7] конфигурируемые границы тиров (профиль
  co-tenancy) + [R2] cold-замер p95 через posix_fadvise.

## 1. Критический bugfix (найден при подготовке R7-фикстуры) — `ba42267`

Два бага в `src/tiers/src/manager.cpp`, оба на master (`dedb67e`), оба
проявляются при churn (ровно сценарий среза 3: MoE > VRAM):

1. **Aliasing VRAM-слотов.** `acquire_slot` (eviction-путь) возвращал
   `free_slots_.back()` после `release_slot(victim)` БЕЗ `pop_back` → слот
   жертвы оставался в списке свободных → следующий `acquire_slot` выдавал
   ТОТ ЖЕ слот другой группе → две группы в одном слоте (порча VRAM-копий,
   resident-группы > vram_slots — фактически 11 resident при 6 слотах).
   Фикс: слот жертвы забирается (push + pop), инвариант
   `tier_bytes[0] == resident × slot_bytes` восстановлен.
2. **Prefetch сам себя вытеснял.** Префетченная группа имела `last_use = 0`
   → была первой жертвой LRU при следующей eviction: prefetch(l+1) второй
   загрузкой вытеснял первую свежезагруженную → гарантированные промахи
   prefetch. Фикс: `internal_prefetch` обновляет `last_use` (префетченная
   группа = «недавно использованная»).

Синхронные правки тестов-потребителей — три теста проходили на master
ТОЛЬКО из-за aliasing-бага (resident завышался, вытеснение фактически не
происходило):
- `tests/tiers/test_engine_mimic.cpp`: vram_slots 3 → 6 (≥ 2 слоёв групп —
  детерминизм prefetch-хитов; интент churn сохранён: 9 групп > 6 слотов);
- `tests/core/engine/test_tier_integration.cpp` и `test_weight_path.cpp`:
  vram_slots 16 → 32 (всех групп плана mini_dense = 21; интент —
  интеграция/residency, не churn). **Внимание worker'а**: это минимальные
  правки в tests/core/engine (каталог НЕ входит в запрещённый список, но
  фиксирую явно — причина: старые ассерты «все группы resident» были
  истинными только на багованном учёте слотов).
- НОВЫЙ регресс-тест `manager_vram_slot_accounting_under_churn`
  (1 слот, 4 раунда churn: resident ≤ vram_slots; tier_bytes[0] ==
  resident × slot).

## 2. [R1] CRC-политика — `8f8d989`

По planner [Q2] («CRC из hot path, проверка — на границах»):

- reader ВСЕГДА проверяет `header_crc32` (без изменений);
- payload-CRC при pin/read — ТОЛЬКО при явном флаге: `TierConfig::verify_crc`
  (дефолт **false**) ИЛИ env `STELL_TIER_VERIFY_CRC=1` (итог = config || env).
  Дефолт off — hot path без CRC; writer считает CRC при записи (точка
  гарантии целостности, layout.cpp `add_group` — без изменений);
- публичный verify-проход: **`TierManager::verify_slots()`** — внепутевая
  сверка payload-CRC всех групп layout с индексом; успех помечает группу
  проверенной (hot path с verify=on не перепроверяет — CRC-once);
- **выбор: метод на TierManager, НЕ harness-команда** (обоснование): verify
  работает с тем же открытым NVMe-файлом и состоянием «проверенных групп»,
  что и горячий путь — единый кодовый путь, без второго канала доступа к
  layout; вызвать может кто угодно (калибровщик, harness, тесты, стартовый
  проход). Интерфейс `ITierManager` (ADR-08 §2) НЕ расширяется — прецедент
  `read_ram_slot`/`set_layer_map`. Дисциплина «холодной» операции (как
  set_plan): не вызывать параллельно с pin/prefetch;
- метрика `TierManager::crc_checks()` — сколько раз payload-CRC фактически
  пересчитывался (0 при verify=off и без verify-прохода);
- сообщения порчи унифицированы: ровно `TierError(IoFailure,
  "layout: payload CRC mismatch")` (об paths: io_uring и fallback;
  `ChecksumMismatch` отсутствует в enum ADR-08 §3 — интерфейс не расширяется);
- сырые `TierLayoutReader`/`TierNvmeFile` по умолчанию verify=**true**
  (вне горячего пути: тесты/интеграция с models — все прежние CRC-тесты
  зелёные); менеджер пробрасывает свой флаг вниз (TierManager →
  TierNvmeFile → TierLayoutReader).
- Файлы: `src/tiers/include/stell/tiers/layout.hpp`, `src/tiers/src/layout.cpp`,
  `src/tiers/src/nvme_file.hpp`, `src/tiers/src/nvme_file.cpp`,
  `src/tiers/include/stell/tiers/tier_manager.hpp`, `src/tiers/src/manager.cpp`.

Тесты (`tests/tiers/test_manager.cpp`, +4 кейса):
- (а) `manager_verify_crc_off_hot_path` — pin всех групп при verify=off:
  `crc_checks()==0` (фиксация по метрике), `io_ops>0` (контроль негатива);
- (б) `manager_verify_crc_on_catches_corruption` — испорченный байт payload →
  `wait()` бросает `TierError(IoFailure, "layout: payload CRC mismatch")`
  (точная строка проверена);
- (в) `manager_verify_slots_pass_marks_groups` — verify-проход покрывает все
  слоты, метрика = n_groups, последующий hot path (verify=on) не
  перепроверяет; без NVMe-тира → PlanMismatch;
- (в) `manager_verify_slots_catches_corruption` — verify-проход ловит порчу
  при verify=off (внепутевая верификация работает и при off);
- (в) все существующие layout/CRC-тесты зелёные (ctest 44/44).
- `tests/tiers/bench_pause.cpp` печатает «payload-CRC пересчётов» для
  отчёта [R2].

## 3. [R7-часть] Конфигурируемые границы тиров — `55e42e2`

Профиль «свободно 2–11 ГБ на V100 при co-tenancy» (RAM-тир основной,
VRAM-тир мал). Файлы: `src/tiers/include/stell/tiers/tier_manager.hpp`,
`src/tiers/src/manager.cpp`.

- База: `TierConfig` уже параметризует `vram_slots`/`vram_slot_size`/
  `ram_capacity`. Поверх — два механизма:
  1) `tier_config_from_env(base)` — env поверх базового конфига:
     `STELL_TIER_VRAM_SLOTS`, `STELL_TIER_VRAM_SLOT_BYTES`,
     `STELL_TIER_RAM_MB`, `STELL_TIER_LOOKAHEAD`, `STELL_TIER_HISTORY_CAP`,
     `STELL_TIER_VERIFY_CRC`, `STELL_TIER_POLICY` (lru|lcu),
     `STELL_TIER_NVME_LAYOUT`; нечисловые значения → базовое (консервативно);
  2) `tier_config_co_tenancy(vram_budget_bytes, plan_bytes, slot_bytes)` —
     именованный профиль: VRAM-тир МАЛ (`vram_slots = budget/slot`, мин. 1,
     больше плана не выделяем), RAM-тир ОСНОВНОЙ (`ram_capacity` = план
     целиком), NVMe-тир — весь план. `slot_bytes` вне кратности 4096 →
     `TierError(PlanMismatch)`.
- Граница с models-4: потоковая конвертация — зона models; моя часть —
  конфиг тиров + замеры p95-пауз/hit-rate prefetch на профиле (тест с
  фикстурой).
- Тест `manager_co_tenancy_profile_full_cycle` (фикстура в масштабе 1:2048:
  64 КиБ-слоты, 16 групп = 8 слоёв, план 1 МиБ, VRAM-бюджет 384 КиБ → 6
  слотов): полный цикл pin/read — слот через NVMe→RAM→VRAM побитово ==
  payload writer'а layout; замер p95-пауз cold/warm (печатаются честно:
  cold p95 ≈ 0.29 мс, warm ≈ 1.1 мс локально); hit-rate prefetch на профиле
  = 1.0 в warm-проходе (16/16), 0 промахов. Порядок engine-цикла
  зафиксирован детерминистичным: pin(l) → prefetch(l+1) → wait(pin) →
  wait_resident(l+1) при закреплённом l (victim'ы загрузки l+1 —
  непиновенные старые группы; в реальном engine на это окно уходит compute
  слоя). Также `manager_env_config_override` (env-параметризация + fallback
  нечисловых значений).

## 4. [R2] bench_p95_nvme.sh — `55e42e2`

`src/tiers/bench_p95_nvme.sh` (исполняемый):
- cold-замер через `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` по файлу
  тира (unprivileged, planner [Q3]); путь sudo//proc/sys/vm/drop_caches
  УДАЛЁН; `DROP_CACHE_WARM=1` остался только как честная пометка
  «тёплый (page cache)» (саму эвикию при этом флаге скрипт не выполняет);
- в шапке скрипта: точная команда серверного прогона worker'ом (ПОСЛЕ [R1]
  и merge, в клоне `~/Code/stell-llm`) и ожидаемые поля отчёта:
  1) p50/p95/p99 (prefetch+cold, мс);
  2) verdict против цели p95 ≤ 5 мс (PLAN.md §4) — по prefetch-режиму;
  3) профиль остатка БЕЗ CRC-once (raw-паузы по проходам; `VERIFY_CRC=1` —
     дополнительный сравнительный прогон с CRC-once в hot path);
  4) co-tenancy-пометка (nvidia-smi; strata НЕ трогать; NVMe общий).
- Локальный smoke: ext4-каталог (`~`), 8 групп × 256 КиБ × 4 слоя — verdict
  PASS (p95 ≈ 0.35 мс, сырые паузы в логе); guard tmpfs отработал (rc=2);
  fallback-пометка DROP_CACHE_WARM и VERIFY_CRC-проход проверены
  (verify=on дал p95 1.31 мс против 0.34 off уже на 256 КиБ-слотах —
  цена CRC видна, масштабируется на 16 МиБ).
- **Серверный прогон [R2] — worker после merge и после [R1]** (порядок
  planner'а соблюдён: [R1] уже в этой ветке). Точная команда — в шапке
  скрипта; кратко:
  ```
  export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
  export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
  cmake -S . -B build/cpu -G Ninja && cmake --build build/cpu
  ./src/tiers/bench_p95_nvme.sh build/cpu ~/tiers_bench
  ```
  (опционально `build/cuda` — DMA-путь; логи сохранить в `bench/results/`
  с пометкой co-tenancy).

## 5. Тесты

- До (базлайн ветки, `dedb67e`): **ctest 44/44** зелёные.
- После: **ctest 44/44** зелёные (ctest-записей столько же — добавлены кейсы
  внутрь существующих бинарников; tiers/test_manager: 16 кейсов, из них
  новых: 4×R1 + 1 регрессия slot-accounting + 2×R7); оба прогона подряд —
  стабильно, без флаков.
- Каждый коммит ветки собирается и проходит ctest изолированно (проверено
  выборочной сборкой промежуточных состояний).
- Серверные прогоны ([R2] p95-NVMe, CUDA-конфигурация) — **pending: worker
  после merge** (push на сервер запрещён; команды в шапке bench-скрипта).

## 6. Ключевые решения

1. Verify-проход — метод `TierManager::verify_slots()` (не harness-команда):
   единый кодовый путь с горячим (тот же NVMe-файл и verified_-состояние);
   интерфейс не расширяется (прецедент R8 `read_ram_slot`).
2. Итоговый флаг CRC = `config || env` (OR-семантика): конфиг для движка/
   калибровщика, env для эксплуатационного включения без пересборки.
3. Сырой layout-reader по умолчанию verify=true — старые CRC-тесты и
   models-интеграция не ломаются; политика «off» применяется именно в
   горячем пути менеджера (по постановке [R1]).
4. Профиль co-tenancy: VRAM-бюджет — явный параметр (замер свободной VRAM и
   запас под активации — ответственность вызывающего; за пределами tiers);
   RAM-тир = весь план (допущение «host RAM — сотни ГБ» зафиксировано в
   докстринге).
5. Детерминизм R7-фикстуры: 6 VRAM-слотов (3 слоя) + порядок
   pin(l) → prefetch(l+1) → wait_resident(l+1) при закреплённом l —
   victim-выбор не зависит от гонок unpin/worker'а.

## 7. НЕ сделано / почему

- Серверный прогон [R2] и CUDA-конфигурация: push на сервер запрещён —
  worker после merge (команды готовы, скрипт смоукен локально).
- Фиксация CRC-политики в ADR-12 §2.3: docs/ — чужий каталог (по
  постановке фиксирует docs-4; в моей копии ADR-12 round-3-правки нет —
  ссылаюсь на текст постановки).
- `docs/design/server-env.md`: в моём worktree (от `dedb67e`) файла нет —
  читать было нечего; серверные пометки [R6]/co-tenancy учтены в скрипте
  и отчёте. Допущение: файл появится от docs-потока, содержательных
  расхождений с моей частью нет.
- Потоковая конвертация весов: зона models-4 (граница зафиксирована в [R7]).
- Аппаратный CRC32-IEEE (PCLMUL): с CRC-off дефолте горячий путь чист —
  вопрос ускорения verify-прохода остаётся опциональным (открытый вопрос
  round-2 §8.1, требует ADR-кларификации — формат IEEE фиксирован).
- O_DIRECT в NVMe-тире: как и в round-2 — отдельное изменение (ADR), на
  политику CRC не влияет.

## 8. Открытые вопросы

1. Серверный [R2]: достигается ли p95 ≤ 5 мс на реальном NVMe после [R1]
   (профиль остатка: NVMe-чтение + DMA/memcpy) — pending worker.
2. Срез 3 (MoE > VRAM): с bugfix'ами `ba42267` churn-сценарий корректен;
   рекомендую интеграционному прогону среза 3 добавить побитовую сверку
   слотов VRAM-тира (aliasing-баг не обнаруживался существующими тестами —
   только фикстурой R7 и новым регресс-тестом).
3. Правки `tests/core/engine/*` (vram_slots 16→32) и
   `tests/tiers/test_engine_mimic.cpp` (3→6): на обзор worker'а — это
   единственные правки за пределами src/tiers/tests/tiers, мотивировка
   в §1.
4. `n_tensors` в заголовке (конвертер models пишет n_groups) — известное
   зафиксированное расхождение (ADR-12 «Уточнение», round 1) — без изменений.

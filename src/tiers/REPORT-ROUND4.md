# tiers — отчёт раунда 4 (T-20261017200354, Поток B, роль tiers)

Ветка `agent/tiers/5` (чекаутнута от master `c51da0e` — round-3 работа,
включая bugfix `ba42267`, уже в master). Коммиты локально, НЕ пушено.
Запрещённые каталоги не тронуты: изменения только `src/tiers`, `tests/tiers`.

Коммиты (поверх master `c51da0e`):
- `b2673a7` — feat(tiers): [R2a] O_DIRECT-бэкенд TierNvmeFile + [R2b]
  deep-prefetch (k групп в полёте) + [R5] побитовая сверка VRAM-слотов +
  [R2-матрица] в bench_p95_nvme.sh (код + тесты + bench);
- docs-коммит этого отчёта (см. git log — следующий за `b2673a7`).

## 1. [R2a] O_DIRECT-бэкенд TierNvmeFile (planner [Q6])

**Конфигурация** (`TierConfig`, `src/tiers/include/stell/tiers/tier_manager.hpp`):
- `TierConfig::odirect` (дефолт `false` — buffered, fallback-прецедент
  сохранён); итоговый флаг = `config || env` — та же OR-семантика, что у
  `verify_crc` ([R1] round 3); env: `STELL_TIER_ODIRECT=1`
  (`tier_config_from_env`, `env_odirect()` в manager.cpp).

**Механика** (`src/tiers/src/nvme_file.cpp/.hpp`):
- O_DIRECT — ОТДЕЛЬНЫЙ fd данных (`open(O_RDONLY|O_DIRECT)`): заголовок/
  индекс читает `TierLayoutReader` через обычный buffered fd (маленькие
  невыровненные чтения); O_DIRECT-тракт — только данные слотов (io_uring
  SQE(READ) на этом fd, батч и одиночный путь).
- Если O_DIRECT открыть не удалось (FS/ядро не поддерживают, напр. tmpfs) —
  **честный откат на buffered**: `TierNvmeFile::odirect_active() == false`
  (проверяется тестом; bench печатает режим в каждой строке прогона —
  «odirect_requested / odirect_active / k_in_flight»).
- `TierManager::nvme_odirect_active()` — расширение на конкретной реализации
  для bench/тестов (ITierManager, ADR-08 §2, НЕ расширяется — прецедент
  `crc_checks`).

**Выравнивание — честно зафиксировано**:
- Требование O_DIRECT: кратность буфера, офсета и длины логическому блоку
  файловой системы. Layout v1 (ADR-12) гарантирует `data_offset` кратно 4096
  и `slot_bytes` кратно 4096 — офсет/длина ВСЕГДА валидны.
- Логический блок FS кодом НЕ запрашивается (ioctl на regular-файле не
  универсален) — консервативная фиксация **4096 = kTierSlotAlign** (покрывает
  lsb 512/2048/4096 распространённых FS: ext4/xfs). Если реальный lsb больше —
  O_DIRECT вернёт EINVAL → честная ошибка чтения, не порча данных.
- **Буферы**: RAM-буферы менеджера теперь выделяются выровненно
  (`posix_memalign(4096)` на CPU; `cudaHostAlloc` page-aligned при
  STELL_CUDA — свойство зафиксировано) → O_DIRECT читает ПРЯМО в RAM-тир,
  без staging-копии (полный выигрыш O_DIRECT). Произвольные (невыровненные)
  dst сырых вызовов `read_group` — **минимальное выровненное staging-чтение**
  (`posix_memalign`-буфер + memcpy): контракт `read_group` (произвольный dst)
  не меняется.
- Честное свойство O_DIRECT: отсутствие readahead — компенсируется
  deep-prefetch ([R2b]); детерминированный cold (page cache не искажает замер
  под co-tenancy — главная мотивация [Q6]).

**Тест** (`tests/tiers/test_nvme_tier.cpp::nvme_odirect_bitwise_equals_buffered`):
один layout читается buffered (эталон) и O_DIRECT — побитово равно, ОБА пути:
(а) staging-чтение в гарантированно невыровненный dst (сдвиг +1, шум 0xEE до
чтения — проверено отсутствие записи за границей); (б) прямое чтение в
выровненный `posix_memalign(4096)` буфер (прод-путь). Плюс
`tests/tiers/test_manager.cpp::manager_odirect_full_cycle`: полный цикл тиров
при `odirect=true` (O_DIRECT прямо в RAM-буферы менеджера) — RAM-слот побитово
== payload writer'а; OR-семантика env (`STELL_TIER_ODIRECT=1` → on; дефолт —
buffered). Локально O_DIRECT активен (ext4): `odirect_active=true` в логах.

## 2. [R2b] Deep-prefetch: k групп в полёте (planner [Q2])

- `TierConfig::deep_prefetch_depth` (k=1/2/4; дефолт 1 — прежнее поведение);
  env `STELL_TIER_PREFETCH_DEPTH` через `tier_config_from_env` (tier-профиль).
- **Семантика — честно**: воркер тира дренажит очередь загрузок БАТЧАМИ до k
  групп (`worker_loop` → `process_batch`): k SQE io_uring подаются ОДНИМ
  `io_uring_enter` (новый `IoUring::submit_and_wait_all` — собирает ВСЕ CQE,
  чужие не дропает) → k чтений NVMe параллельно (глубина очереди носителя k).
  k групп материализуются в фоне одновременно; пауза между слоями
  перекрывается чтением k слоёв вперёд.
- Ошибки изолированы по группам батча: `TierNvmeFile::read_groups` возвращает
  исход на запрос (`BatchOutcome{bytes, err}`) — промах/порча одной группы не
  роняет соседние по батчу (отказ кольца/файла — весь батч, честно).
- Фазы A (RAM-тир) и C (RAM→VRAM DMA/memcpy) — прежние, по группам батча.
- **Fallback-бэкенд** (нет io_uring): батч честно деградирует до k
  последовательных pread (фиксировано в докстринге nvme_file.hpp).
- **Диагностический крюк** `TierConfig::io_delay_ms` (env
  `STELL_TIER_IO_DELAY_MS`; прод: 0): модель латентности ОДНОЙ загрузки,
  батч из k в полёте перекрывает её — детерминизм тестов deep-prefetch.
  Не прод-параметр, задокументирован в докстринге.

**Тест** (`manager_deep_prefetch_k_in_flight`): k=4 — 4 группы (слои 1–2)
материализуются в фоне параллельно: замер 61.5 мс ≈ латентность ОДНОЙ группы
(delay=60 мс) против k=1 — 242 мс = 4× последовательно (5 прогонов подряд —
стабильно); декод префетченных групп — без пауз загрузки (пауза < delay,
prefetch-hit 4/4 по метрике). Порог параллельности 3.5×delay допускает
разбиение очереди гонкой worker'а на ≤3 батча — любая конфигурация строго
быстрее полностью последовательной (честно отмечено в тесте).

## 3. [R5] Побитовая сверка VRAM-слотов (planner [Q5])

- Новое расширение **`TierManager::read_vram_slot(id, dst, dst_bytes)`** —
  чтение содержимого VRAM-СЛОТА resident-группы (CPU: host-слот
  `vram_storage_`; STELL_CUDA: cudaMemcpy D2H из device-слота). Интерфейс
  ITierManager (ADR-08 §2) НЕ расширяется — прецедент `read_ram_slot`.
- **Тест** `manager_vram_slots_bitwise_direct_loader` — интеграционный,
  resident > slots (6 групп > 2 VRAM-слота, churn гарантирован):
  - эталон = **прямой лоадер** (`TierLayoutReader`, чтение layout ВНЕ тиров);
  - сценарий жертвы = **префетченная группа**: g0/g1 префетчены и
    материализованы в фоне → префетч g2/g3 → жертвы эвиции именно
    префетченные; побитовая сверка ВСЕХ resident-слотов после каждой
    материализации;
  - полный churn-цикл (3 раунда pin/unpin всех групп): сверка ПОСЛЕ КАЖДОГО
    pin/DMA (инвариант «одна группа = один слот»; регрессия aliasing-бага
    `ba42267` — resident > vram_slots и/или несовпадение байтов ловятся
    немедленно); итог: 40 сверок слотов;
  - контрольные инварианты: resident ≤ vram_slots,
    `tier_bytes[VRAM] == resident × slot_bytes`;
  - error-кейсы расширения: не-resident → IoFailure (на детерминированном
    менеджере с 1 слотом и закреплённой группой), unknown → PlanMismatch,
    dst < slot → PlanMismatch.
- **Честная пометка про CPU-конфигурацию**: VRAM-слот в CPU-сборке
  эмулируется host-памятью, но кодовый путь записи слота ТОТ ЖЕ (memcpy
  RAM→VRAM в `process_batch` — та же ветка, что и перед DMA). DMA-путь
  `cudaMemcpyAsync` при STELL_CUDA покрыт серверной сборкой — срез 3 обязан
  прогнать этот же тест под build/cuda (co-tenancy bench) + bench_tiers_cuda.
  Без этого тест НЕ считается покрытием DMA-пути.

## 4. [R2-матрица] Серверный прогон (worker после merge)

`src/tiers/bench_p95_nvme.sh`, режим `MATRIX=1` (перенесён ДО warmup —
ячейки матрицы самодостаточны: создают layout под свой slot → fadvise-cold →
замер; устаревший layout от предыдущего прогона не мешает):

```
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
cmake -S . -B build/cpu -G Ninja && cmake --build build/cpu
MATRIX=1 ./src/tiers/bench_p95_nvme.sh build/cpu ~/tiers_bench
```

Матрица: **slot {16 МиБ, 4 МиБ} × I/O {buffered, O_DIRECT} × k {1, 4}** —
8 ячеек, логи `~/tiers_bench/matrix_slot*_io*_k*.log` → сохранить в
`bench/results/`. В каждой ячейке:
- co-tenancy-пометка (`nvidia-smi` в логе; strata НЕ трогать; NVMe общий);
- cold через `posix_fadvise(DONTNEED)` (userspace, [Q3]);
- фактический режим I/O зафиксирован строкой `bench io mode:`
  (`odirect_requested/odirect_active/k_in_flight`) — `odirect_active=false`
  означает ЧЕСТНЫЙ откат на buffered (FS не поддерживает O_DIRECT);
- профиль остатка: сырые паузы (`BENCH_DEBUG` → `raw pauses`) в логе ячейки;
- **пометка обязательна**: замер под co-tenancy — «НЕ ВЕРДИКТ ПО ЦЕЛИ §4
  (p95 ≤ 5 мс) — вердикт только по чистому окну»; цель не отменяется;
  целевой профиль [Q2] — «слоты 4 МиБ + prefetch k≥4 + O_DIRECT» (последняя
  ячейка матрицы).

Локальный smoke матрицы (ext4, 8 групп × 256 КиБ — масштаб не серверный):
все 8 ячеек прошли; O_DIRECT активен (`odirect_active=true` в 4 ячейках);
на слотах 16 МиБ p95 ≈ 24–28 мс, на 4 МиБ ≈ 7 мс (локальный диск, НЕ серверный
NVMe — числа только как смоук плумбинга, не замер цели).

## 5. Тесты

- **До** (master `c51da0e`, базлайн): ctest **45/45** зелёные.
- **После**: ctest **45/45** зелёные (новые кейсы добавлены внутрь существующих
  бинарников — число ctest-записей не изменилось):
  - `test_manager`: 19 кейсов (+3: `manager_odirect_full_cycle`,
    `manager_deep_prefetch_k_in_flight`, `manager_vram_slots_bitwise_direct_loader`);
  - `test_nvme_tier`: 4 кейса (+1: `nvme_odirect_bitwise_equals_buffered`).
- Тайминг-тест [R2b] прогнан 5× подряд — стабильно (61.4–61.6 мс k=4 против
  241.7–242.6 мс k=1).
- Сборка чистая (`-Wall -Wextra`, без warning'ов; reorder-семантика
  конструктора выправлена).
- CUDA-конфигурация локально не собирается (нет GPU в работе) — серверный
  прогон worker'а покроет DMA-ветки (`read_vram_slot` D2H, `process_batch`
  cudaMemcpyAsync) — см. пометку среза 3 в §3.

## 6. Ключевые решения

1. **O_DIRECT — отдельный fd данных**: заголовок/индекс остаются на buffered
   fd (TierLayoutReader не тронут — все layout/CRC-тесты зелёные без правок);
   O_DIRECT-тракт — только слоты (горячий путь).
2. **Выравнивание**: консервативные 4096 (= kTierSlotAlign), lsb FS кодом не
   запрашивается (честно); RAM-буферы менеджера выровнены → staging-копия в
   прод-пути не возникает; произвольные dst — минимальное staging (контракт
   read_group цел).
3. **Честный откат**: если O_DIRECT недоступен — buffered +
   `odirect_active()==false`; тест требует активности, bench помечает каждую
   строку прогона (никакой «молчаливой» подмены режима).
4. **Deep-prefetch = батч-глубина очереди** (k SQE одним io_uring submit):
   параллелизм именно на носителе, а не на потоках; изоляция ошибок по
   группам батча; k=1 — точный прецедент прежнего поведения.
5. **Диагностический крюк io_delay_ms** — детерминизм теста [R2b] без
   флаков; прод-дефолт 0, честно задокументирован как НЕ прод-параметр.
6. **[R5] через read_vram_slot** (расширение concrete-класса): прямая сверка
   именно VRAM-памяти (не RAM-копии) — единственный прямой тест инварианта
   «одна группа = один слот» (класс aliasing `ba42267`).

## 7. НЕ сделано / почему

- **Серверный прогон матрицы [R2] и CUDA-конфигурация**: push на сервер
  запрещён — worker после merge (команды в §4; логи → `bench/results/`).
- **DMA-путь [R5] под STELL_CUDA**: локально GPU нет; тест обязан быть
  повторён под build/cuda на сервере (срез 3) — пометка в §3 и в тесте.
- **docs/** (tiers.md, server-env.md, ADR-12): чужой каталог по постановке —
  фиксацию решений раунда 4 там делает docs-поток; формулировки для дубля —
  в этом отчёте (§1–§4).
- **Fallback + O_DIRECT** (pread на O_DIRECT-fd через staging): код-путь
  реализован, локально не исполним (io_uring доступен) — покрыт ревью +
  структурой кода; честная пометка.
- **Параметризация slot size в `tier_config_co_tenancy`**: профиль
  фиксирует только границы тиров (round 3); slot/k/I-O задаются вызывающим
  через `TierConfig`/env — расширение сигнатуры именованного профиля не
  потребовалось.

## 8. Открытые вопросы

1. Серверная матрица [R2]: достигает ли профиль «4 МиБ + k=4 + O_DIRECT»
   p95 ≤ 5 мс на реальном NVMe в чистом окне — pending worker (verdict §4 —
   только по чистому окну, co-tenancy-замер — точки регрессии).
2. Срез 3: повторить `manager_vram_slots_bitwise_direct_loader` под
   build/cuda (покрытие DMA) — пункт приёмки среза 3 ([Q5]).
3. `IoUring::submit_and_wait_all` ждёт все теги батча блокирующим enter —
   при отказе носителя посреди батча CQE ошибок приходят для каждого SQE
   (res<0) — исходы честные; дедлок невозможен (каждый поданный SQE даёт
   CQE). Если ядро даст частичную подачу (submit вернул меньше) — pending_
   учтён в submit(); сценарий покрыть серверным прогоном.
4. Право O_DIRECT на серверном NVMe (ext4/xfs — поддерживается; если
   том на LVM/DMCrypt — всё равно поддерживается; если нет — матрица
   честно пометит ячейки `odirect_active=false`).

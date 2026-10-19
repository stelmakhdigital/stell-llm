# tiers — отчёт раунда 5 (T-20261017200354, Поток B, роль tiers)

Ветка `agent/tiers/6` (создана от master `7a6b770`). Коммиты локально,
НЕ пушено. Запрещённые каталоги не тронуты: изменения только `src/tiers`,
`tests/tiers`, артефакт в `bench/results/20261020-tier-pergroup-r5/`.

## Коммиты

- `5ab40dd` — feat(tiers): [R2] round 5 — per-group bench (слоты переменной
  ёмкости) — bench + менеджер + тесты + скрипт;
- `…` — docs/report: отчёт раунда + артефакт `bench/results/20261020-tier-
  pergroup-r5/` (README с командами worker'а + локальный смоук) — см. git log.

## 1. [R2] Per-group bench (planner [Q3])

**Режим pergroup в `bench_tiers_pause`** (8-й аргумент; 6-аргументная форма —
прежний uniform-режим без изменений, обратная совместимость):

- **Реальный stell-tier layout v1**: план читается `TierLayoutReader`'ом
  (группы/роли/dtype/ёмкости/шейпы — из индекса; header-CRC всегда).
  Слоты групп **переменной ёмкости** — как в реальном конвертированном
  layout («по 1 тензору на группу», ёмкости от КиБ до десятков МиБ).
- **Карта слоёв выводится из имён групп**: префикс `l<N>.` → слой N+1;
  группы без префикса слоя (embed/norm/head) → слой 0 (глобальные, читаются
  один раз в начале декода). Имена в индексе усечены до 16 байт (ADR-12
  §2.2) — префикс парсится и в усечённом виде. **Допущение зафиксировано**:
  в проде карту слоёв знает engine (ADR-08 §5); для bench вывод из имён —
  детерминированный и честный, для tiny-qwen2moe даёт слои ровно по
  плану (проверено на фикс­туре).
- **Пауза слоя = Σ пауз групп слоя** ([Q3], честная семантика цели §4):
  каждая группа слоя пинится отдельно (`pin`), пауза группы — время до
  готовности её handle; пауза слоя = сумма. Wall-пауза fan-in печатается
  справочно (`bench-side wall`). Загрузки групп идут в очереди воркера
  батчами до k в полёте (deep-prefetch [R2b] round 4) — в cold-режиме
  честная последовательная Σ-пауза, в prefetch-режиме k групп следующего
  слоя в полёте.
- **Hit-rate prefetch по группам**: новое расширение
  `TierManager::group_prefetch_stats()` (per-group счётчики pf_hits/pf_misses
  в `GroupState`; инкремент в той же точке, что агрегатные hits_/misses_;
  расширение на concrete-классе — ITierManager/ADR-08 НЕ расширяется,
  прецедент `crc_checks`). Bench печатает общий hit-rate и разрез по ролям
  плана (expert/attn/kv, ADR-09), плюс per-group таблицу под BENCH_DEBUG.
- **Режим I/O и глубина k — как в uniform** (`STELL_TIER_ODIRECT`,
  `STELL_TIER_PREFETCH_DEPTH` через `tier_config_from_env`); строка
  `bench io mode:` (odirect_requested/odirect_active/k_in_flight) — та же.
- **Протокол bench per-group** (детерминизм hit-rate, помечено в коде):
  префетч слоя и ожидание его материализации ДО паузы — в прод-контуре
  загрузку перекрывает compute слоя; пауза меряется в steady-state (hit-путь).
  Cold-режим — без префетча: честная холодная Σ-пауза.

**Поддержка per-group ёмкостей в менеджере** (необходимая честность, не
опция): у групп разные `slot_bytes`, раньше всё считалось по
`slot_size_` (max-слот плана):

- буферы RAM-тира выделяются на `slot_bytes` ГРУППЫ; счётчик занятой памяти
  RAM-тира `ram_bytes_` (Σ gb) — раньше учёт по max-слоту завышал занятую
  память в Σ(slot_size_)/Σ(gb) раз и не давал плану влезть в `ram_capacity`;
- memcpy фазы C (RAM→VRAM) и DMA `cudaMemcpyAsync` — `gb` байтов группы
  (хвост max-слота за границей данных группы не читается и не пишется —
  O_DIRECT читает ровно gb в выровненный буфер);
- учёт `io_bytes` — по gb (2 переноса на холодную загрузку: NVMe→RAM +
  RAM→VRAM); `tier_bytes[VRAM]`/`tier_bytes[RAM]` = Σ slot_bytes групп
  resident/буферов;
- `read_ram_slot`/`read_vram_slot` копируют ровно `slot_bytes` группы
  (граница dst — gb; uniform: gb == slot_size_, прежнее поведение);
  `vram_slot_size = 0` → авто (max slot плана) — штатный путь set_plan.

## 2. [R2-прогон] Серверный прогон на реальном layout — worker после merge

`src/tiers/bench_p95_nvme.sh`: режим **`PER_GROUP=1`** + **авто-детект**
(если файл `$REAL_LAYOUT` существует, а PER_GROUP не задан — per-group
выбирается автоматически; `PER_GROUP=0` — принудительно uniform).

Ячейки: **(1) целевой профиль {O_DIRECT + k=4}** — cold по протоколу
server-env round 5 ([Q8]); **(2) сравнительная buffered-ячейка** (k=4) —
тёплая категория (fadvise best-effort), помечена.

Команды worker'а (полный блок — в шапке скрипта и
`bench/results/20261020-tier-pergroup-r5/README.md`):

```bash
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
cmake -S . -B build/cpu -G Ninja && cmake --build build/cpu
PER_GROUP=1 ./src/tiers/bench_p95_nvme.sh build/cpu ~/tiers_bench
# логи ~/tiers_bench/pergroup_io{odirect,buffered}_k4.log →
#   bench/results/20261020-tier-pergroup-r5/
```

- **co-tenancy-пометка обязательна** (nvidia-smi в логе ячейки; strata НЕ
  трогать; NVMe общий) и **«НЕ ВЕРДИКТ ПО ЦЕЛИ §4 (p95 ≤ 5 мс)»** — вердикт
  только по чистому окну; цель не отменяется.
- **Сравнение замеров только внутри категории I/O** (server-env round 5,
  [Q8]): odirect = cold, buffered+fadvise = warm; cross-категорийное
  сравнение в вердикты не идёт — скрипт печатает это правило в каждой ячейке.
- **Честная пометка профиля «4 МиБ»**: целевой «4 МиБ-слоты» относится к
  uniform-матрице round 4; в per-group режиме VRAM-слот = max slot_bytes
  группы layout (tiny-qwen2moe: embed F16 [151646, 32] ≈ 9.3–9.7 МиБ —
  слот 4 МиБ меньше крупнейшей группы, `set_plan` честно отверг бы такой
  профиль: PlanMismatch). Фактические max slot/слоты/VRAM-бюджет печатаются
  строкой `pergroup layout:` в логе каждой ячейки; профиль {O_DIRECT + k=4}
  применяется как есть. Если носитель layout — tmpfs/ramfs, ячейки помечаются
  «носитель RAM-disk — НЕ NVMe-замер» (скрипт автоматически).
- Профиль остатка: `raw layer sums` + per-group таблица в логах (BENCH_DEBUG=1).

## 3. Тесты

- **До** (master `7a6b770`): ctest **47/47** зелёные.
- **После**: ctest **48/48** (новая ctest-запись `tiers/test_bench_per_group`;
  число кейсов внутри существующих бинарников: test_manager +1 = 25):
  - `test_bench_per_group` (новый файл, спавнит бинарник bench по абсолютному
    пути из генератора CMake):
    - фикс­тура-layout (8 групп, переменные слоты 4/8/12 КиБ, 3 слоя,
      роли attn/expert/kv) — **побитово == паттернам** через прямой лоадер
      (header+payload CRC);
    - bench pergroup (prefetch, passes=3): p50/p95/p99 считаются; **per-group
      hit-rate побитово == ожидаемому паттерну** — каждая группа hits=3/
      misses=0, overall/expert/attn/kv = 1.0000 (детерминизм: префетч
      материализуется до паузы); `vram_slots=3` = max групп на слой
      («VRAM-тир вмещает слой»);
    - cold: все группы misses=3/hits=0, overall=0.0000 — честные промахи;
    - O_DIRECT+k4 через env: `odirect_requested=1`, `k_in_flight=4`
      (odirect_active зависит от FS — на tmpfs честный откат, ассершн на
      requested);
    - **обратная совместимость uniform**: старая форма аргументов строит
      синтетический layout и печатает прежний вывод; pergroup-строк в
      uniform-выводе нет; явный 8-й аргумент «uniform» работает.
  - `test_manager::manager_per_group_variable_slot_sizes`: переменные
    ёмкости (3× разницы) — полный цикл: холодный pin каждой группы →
    **побитовая сверка RAM- и VRAM-копий с payload writer'а** (ровно gb
    байтов); учёт `io_bytes == 2×Σ gb`, `io_ops == 2n`, RAM-тир занят ровно
    Σ gb (≤ capacity), NVMe = Σ слотов layout; per-group pf-счётчики
    (холодный pin → misses=1; prefetch+pin → hits=1); граница dst по gb
    (dst < gb → PlanMismatch).
- Смоуки плумбинга (НЕ замеры): per-group bench на 30 группах/8 слоях с
  O_DIRECT+k4 (5 проходов) — без дедлоков/тайм-аутов; скрипт PER_GROUP=1,
  одиночный uniform-прогон и MATRIX=1 (все 8 ячеек round 4) — работают.

## 4. Ключевые решения

1. **Per-group семантика цели §4 — как у planner [Q3]**: слой = множество
   групп; пауза слоя = Σ пауз групп слоя (последовательные ожидания готовности
   каждой группы), hit-rate — по группам/ролям. Uniform-режим сохранён как
   синтетический профиль (обратная совместимость — протестирована).
2. **Карта слоёв из имён** (допущение зафиксировано): префикс `l<N>.` →
   слой N+1, глобальные → слой 0; в проде карту знает engine — bench-вывод
   честен и детерминирован, для tiny-qwen2moe совпадает со структурой плана.
3. **Менеджер обязан поддерживать переменные ёмкости честно**: per-group
   буферы/копии/учёт по gb группы (иначе bench печатал бы завышенные
   io_bytes/RAM и мусорные хвосты в сверках). ITierManager/ADR-08 не
   расширяется (прецедент `crc_checks`/`read_ram_slot`).
4. **«VRAM-тир вмещает слой»** — дефолт per-group bench (слоты = max групп
   на слой): steady-state hit-путь, детерминизм паттерна hit-rate;
   переопределяется env (STELL_TIER_VRAM_SLOTS). RAM-тир дефолтно = план
   целиком (как профиль co-tenancy round 3).
5. **Профиль «4 МиБ-слоты» — честная пометка неприменимости напрямую** к
   per-group layout (embed крупнейшей группы > 4 МиБ; set_plan отверг бы
   PlanMismatch): VRAM-слот per-group = max slot группы, фактические числа
   фиксируются в логе ячейки; {O_DIRECT + k=4} применяется как есть.
6. **Протокол [Q8] в скрипте**: cold = O_DIRECT штатно; buffered = тёплая
   категория (fadvise best-effort), помечена; сравнение только внутри
   категории I/O — напечатано в каждой ячейке; co-tenancy-пометка и «НЕ
   вердикт §4» обязательны.

## 5. НЕ сделано / почему

- **Серверный прогон на реальном layout tiny-qwen2moe (65 групп, 18.8 МиБ)**:
  сервер read-only для агента, push запрещён — **worker после merge**
  (команды — в §2 и в README артефакта). Локальный смоук плумбинга — в
  `bench/results/20261020-tier-pergroup-r5/smoke-local-pergroup.log` с
  честной пометкой «НЕ серверный замер».
- **CUDA-сборка per-group пути**: локально GPU/conda-CUDA нет; DMA-ветки
  (`cudaMemcpyAsync` по gb, D2H `read_vram_slot` по gb) покрыты структурно
  и серверным прогоном build/cuda (прецедент round 4) — пометка для среза 3.
- **docs/ (tiers.md, server-env.md)**: чужой каталог — дубль решений делает
  docs-поток; формулировки — в этом отчёте (§1, §2, §4) и в README артефакта.
- **VRAM-слоты переменного размера в менеджере**: слоты фиксированного
  размера (max slot) — архитектура round 1; per-group честность достигнута
  учётом/копиями по gb, без переустройства слотов (ADR-08 не менялся).

## 6. Открытые вопросы

1. Серверный per-group прогон: p95 Σ-пауз слоёв tiny-qwen2moe в профиле
   {O_DIRECT + k=4} — pending worker (НЕ вердикт §4; co-tenancy-замер —
   точки регрессии; сравнение buffered-ячейки — только как «тёплая»
   категория, без cross-вердикта).
2. Карта слоёв реального layout: имена групп обрезаны до 16 байт — для
   tiny-qwen2moe парсинг подтверждён на фикс­туре; если реальные имена
   дадут иные префиксы (проверить строкой `pergroup layout:` в логе —
   число слоёв/групп на слой) — worker фиксирует факт в отчёте.
3. Срез 3: повторить per-group тесты под build/cuda (DMA-ветки по gb).
4. Deep-prefetch k > группы слоя: батч-глубина очереди k=4 при слоях из
   десятков групп — очередь дренажится батчами по k; на серверном прогоне
   честно фиксируется фактическое параллельное перекрытие (сырые Σ-паузы).

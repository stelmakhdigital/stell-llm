# tiers — отчёт раунда 2 (T-20261017200354, Поток B, роль tiers)

Ветка `agent/tiers/2` (от master `2a00229`). Коммиты: см. git log; НЕ пушено.

## 1. Критический код-фикс: CRC-граница заголовка (ADR-12 «Уточнение»)

- `src/tiers/src/layout.cpp`: writer (`finalize`) и reader — граница
  `header_crc32` заменена `0x28 → 0x2C` (канон — CRC32 байтов [0x00..0x2C),
  поля 0x00..0x28 включительно, как у конвертера models
  `stelltier_format.cpp::stelltier_header_crc32`). Комментарии приведены к
  кларификации (512-байтовый заголовок с нулевым паддингом 0xA0..0x1FF,
  имя группы 16 байт, CRC группы — по слоту целиком — уже были верны).
- `src/tiers/include/stell/tiers/layout.hpp`: уточнён комментарий поля
  `crc32` записи индекса (CRC по слоту целиком, payload+padding).
- Тесты: `tests/tiers/test_layout.cpp`
  - комментарий порчи заголовка обновлён под канон [0x00..0x2C);
  - НОВЫЙ кейс `layout_header_crc_boundary_0x2c`: (a) roundtrip tiers→tiers
    на канонической границе; (b) файл с CRC по СТАРОМУ канону [0x00..0x28)
    reader'ом ОТКЛОНЯЕТСЯ (байты n_tensors 0x28..0x2B входят в диапазон);
    (c) порча первого байта n_tensors (0x28) обнаруживается.
- До фикса кросс-модульный тест честно падал: `[FAIL]
  cross_models_converter_to_tiers_reader: layout: header CRC mismatch`
  (файл конвертера models не принимался старым reader'ом — ровно задокумент
  рованное отклонение); после фикса — зелёный.

## 2. Кросс-модульный тест tiers↔models (ADR-12, интеграционный срез)

Файлы: `tests/tiers/test_cross_models.cpp` (сторона tiers, реальные типы
`stell/tiers/types.hpp`) + `tests/tiers/cross_models_shim.cpp` (сторона
models) + `tests/tiers/cross_models_abi.h` (плоский C-ABI).

- **Почему шим**: заголовки models тянут локальное зеркало типов ADR-08
  (`tiers_compat.hpp`), несовместимое в одной TU с реальным `types.hpp`
  (`enum class Role`/`DType` декларированы по-разному — ODR-нарушение,
  подтверждено компиляцией: `multiple definition of 'enum class Role'`).
  Модели-сторона живёт в отдельной TU, наружу — C-фасад. При вклейке
  (ADR-09: заменить зеркало на реальный `types.hpp`) шим сливается в одну TU.
- Сценарий: реестр models (fixtures `mini_dense.json` F16 + `mini_moe.json`
  NVFP4) → синтетический safetensors (writer тестов models, `synth_writers.hpp`)
  → конвертер `StellTierConverter` → stell-tier layout → читатель tiers:
  - `TierLayoutReader` принимает файл конвертера (CRC-канон [0x00..0x2C));
    dtype_profile 1/2 сверён; индекс сверён с планом (role/dtype/slot_bytes/
    имя ≤16, выравнивания);
  - канон сверён в обе стороны: `crc32(hdr,0x2C)==stored` и
    `crc32(hdr,0x28)!=stored`; negative-тест: файл с CRC по старому канону
    → `TierError::IoFailure`;
  - полный NVMe-контур: `TierManager::set_plan` (сверка slot_bytes/role/dtype
    с layout, ADR-12 §2.2) → pin всех групп через io_uring → resident;
    io_bytes == 2×план×слот (NVMe→RAM + RAM→VRAM);
  - roundtrip данных: слот из tiers == слот, прочитанный models reader'ом
    (`stelltier_read_header_and_index` — CRC-валидация со стороны models);
    padding нулевой;
  - мини-симуляция engine→tiers: `set_layer_map` (слой l по имени `l000…`)
    → `prefetch(l+1)` → `pin_layer(l)` → wait → unpin; prefetch-хиты ≥ число
    слоёв−1; `pin_layer` вне карты → `TierError::PlanMismatch`.

## 3. CUDA/DMA-путь под STELL_CUDA (device-слоты + cudaMemcpyAsync)

Реализовано в `src/tiers/src/manager.cpp` + `tier_manager.hpp`:

- VRAM-тир: `cudaMalloc`-пул слотов фиксированного размера (device ptr в
  `vram_slots_`), освобождение в деструкторе и на ResetPlan; host-эмуляция
  (`vram_storage_`) в CUDA-конфигурации не используется.
- RAM-тир: буферы — `cudaHostAlloc` (pinned) как источник DMA
  (`alloc_ram_buffer`/`free_ram_buffer`; CPU-конфигурация — heap).
- DMA RAM(pinned)→VRAM(device): `cudaMemcpyAsync` на выделенном стриме
  (`dma_stream_`, создаётся в конструкторе) + `cudaStreamSynchronize` до
  завершения Awaiter'а группы. CPU-путь — прежний memcpy (без регрессий).
- **Bug-fix по ходу (ODR/ABI)**: дефиниция `STELL_CUDA=1` была `PRIVATE` на
  `stell_tiers` → TU, включающий `tier_manager.hpp` (тесты), видели класс
  БЕЗ device-полей, а `manager.cpp` — С НИМИ → раскладка класса разъехалась,
  ctor писал `pauses_` за пределы объекта → `stack smashing detected` на
  CUDA-сборке всех manager-тестов. Исправлено: `target_compile_definitions
  (stell_tiers PUBLIC STELL_CUDA=1)` — интерфейсная дефиниция (все TU,
  включающие hpp, видят одну раскладку).
- **Bug-fix по ходу (безопасный ResetPlan)**: воркер мог взять job из очереди
  (queue пуста, `process_job` ещё идёт) → `set_plan` освобождал RAM/VRAM
  под живой загрузкой. Добавлен `active_jobs_` (дренаж-счётчик) + ожидание
  дренажа в `set_plan` (до 30 с → `PlanMismatch`); benign-окно «pop без
  инкремента» безопасно (set_plan при job от prefetch не трогает буферы).
- Серверная сборка (ssh arkalaust@192.168.1.114, копия worktree в
  `~/tiers_r2` — СВОЯ scratch-директория, клон `~/Code/stell-llm` не тронут;
  push на сервер запрещён, поэтому ветка доставлена tar'ом):
  - CPU-конфигурация копии: ctest **31/31** зелёные;
  - CUDA-конфигурация (`-DSTELL_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`,
    nvcc 12.8, gcc 13.3): все 9 tiers-тестов зелёные (DMA-копии проходят
    через V100 при co-tenancy: strata держит 21–30 GB VRAM, свободно ~10.7 GB;
    тир-использование — единицы MiB);
  - сборка полного дерева CUDA падает в **чужом** `src/parallel`
    (`nccl_communicator.cpp: cuda_runtime.h not found` — нет include-dirs;
    не мой каталог — зафиксировано для интеграции); сборка велась таргетами.
  - Локальная CUDA-сборка невозможна (нет toolkit на локальной машине) —
    честно.
- GPU-smoke DMA: `tests/tiers/bench_cuda.cpp` (exec, не add_test):
  - V100 (device 0, co-tenancy ~10.7 GB свободно): H→D/D→H ≈ 3.3/3.5 GiB/s
    (64 KiB), 5.7/5.9 (1 MiB), 6.1/6.0 (16 MiB), 6.1/6.1 (64 MiB);
  - RTX 2060 (device 4, свободно ~5.6 GB): 4.5/4.9, 10.0/10.4, 8.9/9.0,
    8.6/8.5 GiB/s; утечек VRAM не осталось (cudaMemGetInfo до/после).

## 4. Измерение пауз слоёв на сервере (реальный NVMe, co-tenancy)

Бенч: `tests/tiers/bench_pause.cpp` (exec, не add_test; режимы `prefetch` и
`cold` — без lookahead-prefetch). Файл: `~/tiers_bench/pause_layout.stelltier`
(root NVMe `/dev/nvme0n1p2`, 512 MiB = 32 группы × 16 MiB; 16 слоёв × 2
группы; 2 VRAM-слота → churn; RAM-тир 4 слота; io_uring).

Ключевые результаты (raw pauses в логе, двойной замер bench-side + stats):

- **Проход 1 (холодный, NVMe + CRC-once)**: ~196–255 мс на слой.
  Доминанта — **одноразовая CRC-верификация группы** (ADR-12 §2.3): software
  IEEE CRC32 ~0.3 GiB/s → **~47–59 мс на 16 MiB-слот** (замерено отдельным
  микро-тестом), ~50 мс×2 групп ≈ половина паузы; NVMe-чтение само — быстрое
  (~5 мс на 16 MiB из page cache; O_DIRECT dd — 1.2–1.3 GiB/s; io_uring
  микротест — 5.4–6.2 GiB/s). ВАЖНО: CRC считается один раз на группу за
  жизнь файла в кеше tiers — это НЕ повторяется в passes 2+ (иначе это был
  бы перманентный tax).
- **Проход 2**: ~9–10 мс на слой (RAM-tier хиты: RAM→VRAM memcpy 32 MiB
  ≈ 3.5 GiB/s).
- **Проходы 3–4**: ~0 мс (VRAM resident).
- Сводка p50/p95/p99 по всем 64 паузам (смешанная статистика):
  CPU: p50 0.001 / p95 199–206 / p99 209–223 мс;
  CUDA (DMA): p50 0.001 / p95 223–237 / p99 227–242 мс; aggregate
  throughput 0.48–0.58 GiB/s (co-tenancy: NVMe общий с нагрузкой сервера —
  зафиксировано честно).
- Цель PLAN.md §4 (p95 ≤ 5 мс) в этой churn-конфигурации (2 VRAM-слота,
  16 MiB-слоты, lookahead-промахи) недостижима; холодная пауза слоя
  доминируется CRC-once → **открытый вопрос для следующего раунда**:
  аппаратный CRC32-IEEE (PCLMUL folding; полином формата — IEEE, не CRC32C,
  поэтому просто `_mm_crc32_u32` не подходит — нужна реализация/ADR) либо
  вынос верификации с decode-пути (верифицировать на prefetch/idle).

## 5. Совместимость с engine (вклейка core-2)

- Контракт сверён по ADR-08 + «Уточнение»: интерфейс `ITierManager`
  (`manager.hpp`) соответствует `src/engine/include/stell/engine/adapters/
  stub_tiers.hpp` (типы/сигнатуры дословно; stub — build aid, после вклейки
  заменяется на реальный `stell/tiers/types.hpp`+`manager.hpp`).
  `ExecutionPlan::pin_layer(ITierManager&, layer)` (src/engine/src/
  execution_plan.cpp) совместим с `TierManager::pin_layer`.
- `set_layer_map` — расширение на реализации (`TierManager`), НЕ новый метод
  интерфейса — совпадает с кларификацией ADR-08; расхождений API нет.
- `layout_version` — величина engine; tiers её не хранит (совпадает).
- Тест-двойник: `tests/tiers/test_engine_mimic.cpp` — мини-симуляция
  decode-цикла по контракту: set_plan → pin_layer до set_layer_map →
  PlanMismatch; set_layer_map с группой вне плана → PlanMismatch;
  prefetch(l+1) → pin_layer(l) → unpin; set_plan при pinned-группе →
  PlanMismatch; после unpin — допустим (ResetPlan), карта слоёв сброшена.
- Вклейка (замена stub, вызовы из DecodeExecutor) — задача интеграции
  worker'а/следующего раунда; контракт не менялся.

## 6. Тесты (итог)

- Локально (CPU-конфигурация): **31/31** зелёные (7 тестов раунда 1 остались
  зелёными; +`test_cross_models`, +`test_engine_mimic` → 9 tiers-тестов;
  +2 bench-executable, не тесты).
- Сервер CPU: **31/31**; сервер CUDA: **9/9 tiers** (полное дерево CUDA не
  собирается из-за чужого src/parallel — см. §3).
- До фикса CRC: кросс-модульный тест честно `[FAIL] … header CRC mismatch`;
  после — зелёный. Тесты layout правлены честно под новый канон.

## 7. Что НЕ сделано / почему

- Пуш на сервер и CUDA-сборка master на сервере с моей веткой: push запрещён;
  вместо этого — scratch-копия `~/tiers_r2` (СВОЯ директория, клон не тронут).
- Локальная CUDA-сборка: нет toolkit.
- Аппаратный CRC / вынос CRC с decode-пути: только зафиксировано как открытый
  вопрос (ADR-12 §2.3; изменение формата CRC32-IEEE→CRC32C требует ADR).
- docs/ — чужой каталог: дизайн-документ `docs/design/tiers.md` НЕ обновлял
  (пункты «открытое исправление кода» уже есть в master; фиксация сделана
  кодом и этим отчётом).
- Fan-in по событиям CUDA (Awaiter на cudaMemcpyAsync-событиях вместо
  per-copy stream sync) — развитие следующего раунда; сейчас честный
  `cudaStreamSynchronize` после постановки копии.

## 8. Открытые вопросы

1. CRC-once: первый pin каждой группы платит ~50–60 мс/16 MiB (software CRC
   0.29–0.35 GiB/s). Предложение: SIMD/PCLMUL CRC32-IEEE или off-path
   верификация — требует ADR-кларификации (формат CRC32-IEEE фиксирует
   байтовый полином; ускорение не меняет формат).
2. `n_tensors` в заголовке: конвертер models пишет `n_groups`, tiers writer
   аккумулирует фактические `tensor_count` — расхождение метаданных
   (некритично, зафиксировано в ADR-12 «Уточнение»; tiers поле не читает).
3. Чужой каталог `src/parallel` не собирается под CUDA (include cuda_runtime.h)
   — блокирует полную CUDA-верификацию дерева на сервере.
4. Имя записи индекса: `layout.hpp` и `stelltier_format.hpp` комменты уже
   сведены к 16 байтам; тесты моделей используют `substr(0,16)` — согласовано.

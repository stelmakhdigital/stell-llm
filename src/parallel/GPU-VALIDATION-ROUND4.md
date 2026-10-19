# parallel — GPU-валидация [R1] round 4: третья попытка (socket/NET-транспорт) + диагностика 4b

## Серверный прогон [R1] (worker, пара 0+4) — исход: честный PENDING

Все 3 попытки (дефолт → без P2P → без P2P+SHM) → одна и та же ошибка
на ОБОИХ ранках: `expert_alltoall: ncclAllGather(counts): unhandled cuda
error` → CommError → честный pending (checks=0 failures=0). Логи сервера:
`/tmp/r4_nccl.log`, `/tmp/r4_bench04.log`.

**Слепое пятно диагностики (закрыто в 4b):** `ncclGetErrorString` даёт только
класс ошибки — реальная причина скрыта в NCCL. В `check_nccl`
(src/parallel/src/nccl_communicator.cpp) добавлен вывод `ncclGetLastError(comm)`
в сообщение CommError (`[ncclGetLastError: …]`) — underlying CUDA-ошибка
NCCL вскрывается без NCCL_DEBUG=INFO; comm прокидывается на все
коллективные вызовы (allreduce/allgather/expert_alltoall/kv-p2p; на пути
бутстрапа комма ещё нет — там безопасное «no communicator»).

Статус: код и протокол готовы; исполнение — worker на сервере ПОСЛЕ merge
(push в ветку запрещён, серверный прогон из worktree недоступен). Источники
истины: planner round 3 [Q1] + [R1] round 4, docs/design/server-env.md
(«NCCL-транспорт: ограничение среды» — placeholder исхода попытки 3
заполняется worker'ом по факту прогона; каталог docs — вне зоны этого потока).

## [R1] Ретрай-протокол — РОВНО три попытки

Цепочка транспортов (planner round 3 [Q1]):

1. **Попытка 1 — дефолт** (P2P как есть). Round 3: fail-fast `ncclAllGather`
   → shm-attach failure.
2. **Попытка 2 — `NCCL_P2P_DISABLE=1`:** NCCL сам выбирает SHM-транспорт для
   intra-node P2P (ring через host/shm). Round 3: снова shm-attach errno
   115/11. Честый pending.
3. **Попытка 3 — `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`:** запрещает и SHM —
   NCCL вынуждается на NET/sockets-транспорт между картами (штатная
   альтернатива SHM в NCCL 2.26.2). НОВОЕ в round 4.

**Транспортная sanity (round 4b) — диагностическая развилка «транспорт vs
код».** Серверный прогон round 4 упёрся: падает только expert-контракт,
простые коллективы на паре НЕ проверялись вовсе (bench-фаза заблокирована
pending-исходом expert через bench_allowed_after) — транспорт-vs-код не
различить. Добавлен самостоятельный TEST `parallel_gpu_nccl_transport_sanity`
(отдельная фаза оркестратора ДО expert-фазы, вне bench_allowed_after-логики):
минимальные allreduce+allgather F32 с побитовым сравнением против
CPU-паттернов, БЕЗ latency-замеров (латентности остаются в bench-фазе с
co-tenancy-пометкой); тот же ретрай-протокол (3 попытки) и watchdog.
Итоговый вердикт — чистая функция `gpu_validation_verdict(transport, expert)`
(p2p_retry.hpp):
- транспорт ОК + expert не подтверждён (pending/пустой/fail) →
  **честный FAIL**: баг нашей обвязки (counts-allgather), починка в потоке C;
- простые коллективы падают тем же 'unhandled cuda error' → транспорт/
  среда → **честный pending** (checks=0 failures=0);
- подтверждённое битовое несовпадение (fail:) на любом из тестов →
  **честный FAIL** (маскировать запрещено);
- корректность подтверждена / «<2 карт» → Ok → bench по
  bench_allowed_after (как было).

Ровно три попытки; исход попытки 3 окончателен (pending или fail).

**Инварианты сохранены:**
- недоступность/зависание/OOM/сбой бутстрапа → **честный pending**
  (`checks=0 failures=0`), тест зелёный, проверок нет;
- прогон дошёл до проверок и побитовое равенство НЕ подтвердилось →
  **честный FAIL** (маскировать pending'ом запрещено);
- `bench_allowed_after` — по ИСХОДУ корректности (pass или «<2 карт»),
  номер попытки на допуск bench не влияет; после GiveUp попытки 3 исход
  остаётся pending:/fail: → bench запрещён (результаты латентностей на
  сломанном контуре недостоверны). Согласованность покрыта юнит-тестом
  `p2p_retry_max_attempts_is_three`.

**Реализация:**
- `src/parallel/include/stell/parallel/p2p_retry.hpp` — вердикты
  `Pass / RetryWithP2pDisabled / RetryWithShmDisabled / GiveUpPending`;
  `kGpuRetryMaxAttempts = 3`; `p2p_disable_env(attempt)` (2+: «1») и НОВОЕ
  `shm_disable_env(attempt)` (только 3: «1»). Заголовок БЕЗ CUDA.
- `tests/parallel/test_nccl_gpu.cpp` — оркестратор: фаза 0 «transport»
  (транспортная sanity) → фаза 1 «correctness» (expert) — обе с циклом
  попыток 1→2→3; `run_child_phase(..., attempt, ...)` прокидывает обе
  NCCL_*-переменных в окружение дочернего процесса; итоговый вердикт по
  `gpu_validation_verdict`; bench-фаза — с попыткой 1 (без
  NCCL_*-переопределений, как было). Watchdog и честная классификация не
  тронуты.
- `tests/parallel/test_p2p_retry.cpp` — юнит-тесты расширены: цепочка
  1→2→3→GiveUp (для pending:, для fail: и для пустого исхода «зависание»),
  env-пары всех трёх попыток, «четвёртой попытки нет», согласованность
  bench_allowed_after, развилка `gpu_validation_verdict` (транспорт OK +
  expert сломан → FailExpert; оба сломаны → Pending; fail: → FailConfirmed;
  pass/fewDevices → Ok).
- `src/parallel/src/nccl_communicator.cpp` — `check_nccl` выводит
  `ncclGetLastError(comm)` (underlying CUDA-ошибка NCCL) в сообщение
  CommError.

## Команды worker'а (сервер, после merge)

```bash
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
cmake -S . -B build/gpu -G Ninja -DSTELL_CUDA=ON && cmake --build build/gpu

# Пара [Q1] GPU0+GPU4 — ВСЕ фазы внутри теста: транспортная sanity →
# expert-контракт (обе с 3 попытками) → bench при подтверждении:
ctest --test-dir build/gpu -R parallel_gpu --output-on-failure
# или бинарником:
./build/gpu/tests/parallel/test_parallel_gpu

# Ручной вариант явного контроля одной попытки (если нужен отдельный шаг):
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 STELL_GPU_DEVICES=0,4 \
  ctest --test-dir build/gpu -R parallel_gpu --output-on-failure
```

**Чтение исхода (диагностическая развилка round 4b):**
- `GPU-VALIDATION: транспортная sanity OK …` + `GPU-VALIDATION FAIL: транспорт
  ОК … expert-контракт НЕ подтверждён … баг нашей expert_alltoall (поток C)`
  → транспорт работает, баг в нашей counts-allgather обвязке — починка в
  потоке C (underlying-ошибка теперь в тексте CommError);
- `транспортная sanity не подтверждена` тем же `unhandled cuda error` →
  транспорт/среда → честный pending, протокол в docs/design/server-env.md;
- при подъёме expert-контракта: `GPU-VALIDATION OK` — побитово == CPU-эталон;
  артефакт логов прогона положить в `bench/results/20261019-nccl-gpu-r4/`
  (co-tenancy-пометка `COTENANCE:` обязательна) и заполнить placeholder в
  docs/design/server-env.md.
- Watchdog фазы — 240 с (переопределение `STELL_GPU_TEST_TIMEOUT_S`);
  полный прогон — до 6 фаз-попыток (2 фазы × 3 попытки) — учитывайте тайм-аут
  окна.

- **При подъёме попытки 3:** тест печатает `GPU-VALIDATION: корректность
  подтверждена БЕЗ P2P и БЕЗ SHM (...)` и `NCCL alltoall: rank…` —
  побитовое равенство == CPU-эталон подтверждено. Артефакт логов прогонa
  положить в `bench/results/20261019-nccl-gpu-r4/` (co-tenancy-пометка
  `COTENANCE:` обязательна) и заполнить placeholder в
  docs/design/server-env.md.
- **При неудаче всех трёх попыток:** ограничение среды фиксируется
  окончательно (раздел «ограничение среды» server-env.md): межкарточная
  корректность — блокер среза 2, ждёт чистого окна (пара V100+V100 без
  strata; `NCCL_DEBUG=INFO` + strace attach — там). Никакого «фолбэка» на
  неподтверждённую корректность.
- Watchdog попытки — 240 с (переопределение `STELL_GPU_TEST_TIMEOUT_S`).

## Статус: ЧЕСТНЫЙ PENDING (серверный прогон round 4: pending; развилка 4b — ждёт повторного прогона)

Серверный прогон round 4 (worker, пара 0+4): все 3 попытки → честный
PENDING, `expert_alltoall: ncclAllGather(counts): unhandled cuda error` на
обоих ранках; простые коллективы на паре не проверялись (блокировка
bench_allowed_after) — развилка «транспорт vs код» не различима. Round 4b
закрывает слепые пятна (ncclGetLastError в CommError + транспортная sanity
вне bench_allowed_after); повторный серверный прогон — по командам выше;
развилка решает: баг нашей обвязки (FAIL, поток C) vs транспорт/среда
(pending, ограничение среды).

Локальная валидация round 4b:
- полный ctest **45/45** зелёный (эмуляторные golden побитово не тронуты);
  юнит-тесты p2p_retry/вердикта: 10 кейсов OK, checks=87945 failures=0;
- синтакс-проверка `test_nccl_gpu.cpp` И `nccl_communicator.cpp` на стабах
  cuda/nccl 2.26.2 — OK;
- смок оркестратора на стабах с РЕАЛЬНОЙ CPU-библиотекой parallel
  (NCCL→эмулятор; стаб-обвязка с падающим expert_alltoall): 4 пути —
  (А) всё OK → bench → exit 0; (Б) транспорт OK + expert CommError →
  честный FAIL exit 1; (В) оба сломаны средой → 3+3 попытки → честный
  PENDING exit 0; (Г) «<2 карт» → bench TP1 → exit 0.

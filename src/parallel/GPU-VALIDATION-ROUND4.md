# parallel — GPU-валидация [R1] round 4: третья попытка (socket/NET-транспорт)

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
- `tests/parallel/test_nccl_gpu.cpp` — оркестратор: цикл попыток 1→2→3,
  `run_child_phase(..., attempt, ...)` прокидывает обе NCCL_*-переменных в
  окружение дочернего процесса (унаследованные значения заменяются);
  bench-фаза исполняется с попыткой 1 (без NCCL_*-переопределений, как и
  было). Watchdog и честная классификация не тронуты.
- `tests/parallel/test_p2p_retry.cpp` — юнит-тесты расширены: цепочка
  1→2→3→GiveUp (для pending:, для fail: и для пустого исхода «зависание»),
  env-пары всех трёх попыток, «четвёртой попытки нет», согласованность
  bench_allowed_after.

## Команды worker'а (сервер, после merge)

```bash
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
cmake -S . -B build/gpu -G Ninja -DSTELL_CUDA=ON && cmake --build build/gpu

# Пара [Q1] GPU0+GPU4 — ВСЕ три попытки внутри теста автоматические:
ctest --test-dir build/gpu -R parallel_gpu --output-on-failure
# или бинарником:
./build/gpu/tests/parallel/test_parallel_gpu

# Ручной вариант явного контроля одной попытки (если нужен отдельный шаг):
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 STELL_GPU_DEVICES=0,4 \
  ctest --test-dir build/gpu -R parallel_gpu --output-on-failure
```

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

## Статус: ЧЕСТНЫЙ PENDING

Исполнение требует сервера (CUDA 12.8 + libnccl + GPU0/GPU4) — после merge.
Локальная валидация round 4:
- полный ctest **45/45** зелёный (эмуляторные golden побитово не тронуты);
  юнит-тесты p2p_retry: 7 кейсов OK, checks=87934 failures=0;
- синтакс-проверка `test_nccl_gpu.cpp` на стабах cuda/nccl 2.26.2 — OK;
- смок оркестратора на стабах (стаб-коммуникатор недоступен):
  - pending-путь: попытка 1 → попытка 2 → попытка 3 → честный PENDING
    (checks=0 failures=0), exit 0;
  - путь «<2 карт» (`STELL_GPU_DEVICES=0`): три попытки → «корректность
    неприменима» → bench-фаза → честный pending замера, exit 0;
  - bench-defer (`STELL_GPU_BENCH=0`): bench пропущен, exit 0.

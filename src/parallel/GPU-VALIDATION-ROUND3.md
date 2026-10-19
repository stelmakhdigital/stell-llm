# parallel — GPU-валидация [R3] + scaling-бенч [R8] (раунд 3, T-20261017200354)

Статус: код и протокол готовы; исполнение — worker на сервере ПОСЛЕ merge
(push в ветку запрещён, серверный прогон из worktree недоступен). Источники
истины: planner round 2 [Q1]/[Q6] + [R3]/[R8] round 3, docs/design/server-env.md.

## [R3] NCCL GPU-валидация корректности — протокол и код

**Цель:** контракты `expert_alltoall` (token_elems) и `kv_stream` на реальном
NCCL-backend, побитово == CPU-эталону; при недоступности карт — честный
PENDING (`checks=0 failures=0`), не молчание.

**Реализация (tests/parallel/test_nccl_gpu.cpp, оркестратор):**

- Бинарник `test_parallel_gpu` сам является оркестратором: фазы корректности
  и bench исполняются ДОЧЕРНИМИ процессами (`fork`+`execve` себя) под
  watchdog'ом (зависший NCCL-коллектив не перехватывается изнутри процесса;
  тайм-аут — `STELL_GPU_TEST_TIMEOUT_S`, дефолт 240 с, после него SIGKILL).
- **Ретрай-протокол [Q1]:** попытка 1 — дефолтное окружение; при любой
  неудаче (зависание P2P-коллективов при `ncclCommInitRank COMPLETE` —
  типичный признак вмешательства co-tenancy в direct-pointer P2P; OOM; сбой
  бутстрапа) — **попытка 2 с `NCCL_P2P_DISABLE=1`**: корректность обязана
  подтверждаться и без P2P (ring через host/shm). Третьей попытки нет.
- Дефолтная пара — **GPU0 (V100) + GPU4 (RTX 2060)** (`STELL_GPU_TEST_DEVICES
  "0,4"` по протоколу [Q1]); переопределение — env `STELL_GPU_DEVICES`.
- **Честный pending vs честный FAIL:** исключение (CommError: OOM/карты
  недоступны/бутстрап) → PENDING, тест зелёный, проверок нет. Если же прогон
  дошёл до проверок и побитовое равенство НЕ подтвердилось — тест КРАСНЫЙ
  (маскировать pending'ом запрещено: «корректность обязана подтверждаться»).
- Логика решений ретрая вынесена в чистый заголовок
  `src/parallel/include/stell/parallel/p2p_retry.hpp` (без CUDA) и покрыта
  юнит-тестом на CPU: `tests/parallel/test_p2p_retry.cpp` (5 кейсов:
  pass→без ретрая; неудача попытки 1 → ровно один ретрай с
  `NCCL_P2P_DISABLE=1`; третьей попытки нет; confirmed-несовпадение → FAIL,
  не pending; bench-фаза разрешена после pass и «<2 карт»).
- **Latency-бенч (б)** — отдельная фаза, запускается ТОЛЬКО при наличии окна:
  после подтверждения корректности (или в TP1-режиме, когда карт < 2);
  пропуск — env `STELL_GPU_BENCH=0`. Co-tenancy-пометка печатается
  (`COTENANCE:`); замеры с RTX 2060/без NVLink с целями §5 не сравниваются
  ([Q6]/server-env).

**Точная команда worker'а (в git-клоне, после merge):**

```bash
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
cmake -S . -B build/gpu -G Ninja -DSTELL_CUDA=ON && cmake --build build/gpu
# пара [Q1] GPU0+GPU4 (ретрай с NCCL_P2P_DISABLE=1 внутри теста автоматический):
ctest --test-dir build/gpu -R parallel_gpu --output-on-failure
# вручную, при желании явного контроля попытки без P2P:
NCCL_P2P_DISABLE=1 STELL_GPU_DEVICES=0,4 ctest --test-dir build/gpu -R parallel_gpu
# окно V100↔V100 для латентностей (после освобождения сервера):
STELL_GPU_DEVICES=0,1 ctest --test-dir build/gpu -R parallel_gpu --output-on-failure
```

**Статус: ЧЕСТНЫЙ PENDING** — исполнение невозможно до merge (push запрещён);
после merge worker исполняет по командам выше; при недоступности карт тест
выдаст честный pending с причиной (механика проверена локально на стабах:
путь «карт нет → попытка 1 → ретрай без P2P → pending, exit 0»).

## [R8] Scaling-бенчмарк TP1/TP2/TP4 dense (контракт PLAN.md §5)

**Скрипт:** `src/parallel/bench_scaling.sh` (по образцу
src/tiers/bench_p95_nvme.sh; только git-клон — проверяется `git rev-parse`;
strata не затрагивается; зависимости user-level: awk/join/timeout/nvidia-smi).

- Матрица: TP1 (`STELL_SCALING_TP1`, дефолт `0`), TP2 (`0,1`), TP4 (`0,1,2,3`).
- Метрики: латентность allreduce F32 (медиана 100 итераций, худший ранок) по
  размерам 1…262144 floats; scaling-фактор TP1→TPn по каждому размеру;
  «allreduce на токен» (вектор `D_MODEL`=4096 floats; до 2×/слой).
- Отчёт: `bench/results/scaling_tp_<штамп>.md` с co-tenancy-пометкой
  (nvidia-smi) и честным вердиктом; **итоговый вердикт §5 (TP4 ≥3.2×) —
  на end-to-end decode (tok/s)**, скрипт даёт allreduce-компоненту.
- Замеры на RTX 2060/смешанных парах с целями §5 не сравниваются.

**Команда worker'а (окно ≥4 свободных V100 без помех):**

```bash
cmake -S . -B build/gpu -G Ninja -DSTELL_CUDA=ON && cmake --build build/gpu
./src/parallel/bench_scaling.sh build/gpu bench/results
```

**Статус: ЧЕСТНЫЙ PENDING** — причина: исполнение требует ≥4 свободных
V100-SXM2 без co-tenancy (strata держит 3 из 4 V100; TP4 на смешанных/занятых
картах недостоверен, а RTX 2060 без NVLink для scaling не годится [Q6]).
Скрипт отработан локально на эмуляции вывода (полный прогон, частичный
pending при падении шага, pending при отсутствии бинарника — во всех случаях
честная причина в отчёте, exit 0, результаты не выдумываются).

## Что сделано / не сделано (честно)

Сделано:
- ретрай-протокол [Q1] в самом тесте (дочерние процессы + watchdog) + чистая
  логика в `p2p_retry.hpp` + CPU-юнит-тест (5 кейсов, checks=24 failures=0);
- дефолт пары [Q1] → `"0,4"` (GPU0 V100 + GPU4 RTX 2060); env
  `STELL_GPU_DEVICES` сохранён; честный PENDING сохранён и уточнён
  (подтверждённое нарушение корректности → честный FAIL);
- latency-бенч — отдельной фазой «только при наличии окна» с co-tenancy
  пометкой и возможностью пропуска (`STELL_GPU_BENCH=0`);
- скрипт scaling-бенча [R8] закоммичен, смоукнут на эмуляции;
- локальная проверка: полный ctest 44/44 зелёный (эмуляторные golden
  побитово не тронуты), синтакс-проверка GPU-файла на стабах cuda/nccl
  2.26.2, смоук оркестратора (pending-путь, путь «<2 карт», bench-defer).

НЕ сделано (исполнение — worker после merge):
- серверный прогон корректности на GPU0+GPU4 — pending (push запрещён);
- серверный прогон scaling-бенча TP1/TP2/TP4 — pending (нужно окно ≥4
  свободных V100; co-tenancy);
- latency-замер V100↔V100 по NVLink — pending (только в чистом окне;
  co-tenancy-пометка обязательна).

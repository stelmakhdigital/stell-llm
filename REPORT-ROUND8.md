# REPORT — round 8, поток C (parallel), итерация 9

Ветка: `agent/parallel/9` (от master `49e19ae`). Коммитов в чужие ветки нет, `.pi/` не тронут.
Прецедент: `REPORT-ROUND7.md` (round 7, поток C).

## [NCCL-окно] — честный PENDING, изменений нет

**Статус: pending, без изменений по требованию planner.** Попытки NCCL/TP-прогонов под
co-tenancy не выполнялись — они **запрещены** (чужой процесс `strata` на сервере;
GPU-нагрузку создавать нельзя).

### Почему невозможно

- Co-tenancy зафиксирована round 4–6 (`docs/design/server-env.md`, раздел «NCCL-транспорт:
  ограничение среды»): все три попытки протокола (P2P → `NCCL_P2P_DISABLE=1` →
  `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`) не подняли межкарточный транспорт —
  shm-attach errno 11/115, затем `Failed to CUDA host alloc 9641984 bytes` на обоих
  ранках. Single-card world=1 (RTX 2060) — GPU-VALIDATION OK, `checks=1`: наш код и NCCL
  корректны до межранкового bootstrap.
- Ограничение среды фиксируется **окончательно** (решение planner round 5 [Q2]);
  дальнейшие попытки под co-tenancy **запрещены**; обходы (не-NCCL транспорт,
  самописные P2P) и «фолбэк» на неподтверждённую корректность запрещены протоколом.
- Никаких новых артефактов `bench/results/` по NCCL-окну в этой итерации не создавалось
  (запрещено) — это отсутствие по правилам, а не упущение.

### Что готово (проверено read-only в этой итерации)

- **Протокол чистого окна** в `docs/design/server-env.md` (раздел «NCCL-транспорт:
  ограничение среды»): 3 попытки + транспортная sanity-фаза (round 4b) + expert-фаза,
  `NCCL_DEBUG=INFO` + strace attach, развилка `gpu_validation_verdict`
  (Ok / Pending / FailExpert / FailConfirmed).
- **Сверка протокола с кодом `src/parallel` (read-only):** команды протокола соответствуют
  коду — `src/parallel/include/stell/parallel/p2p_retry.hpp`:
  `kGpuRetryMaxAttempts = 3`, `p2p_retry_decide` (попытка 1 → retry P2P_DISABLE,
  попытка 2 → retry +SHM_DISABLE, попытка 3+ → GiveUpPending), `p2p_disable_env`,
  `shm_disable_env`, `bench_allowed_after`, `gpu_validation_verdict`; env-флаги попыток
  1/2/3 в точности как в протоколе. `nccl_communicator.cpp` содержит `ncclGetLastError`
  в тексте `CommError` (round 4b) — соответствует доке.
- **Сверка ссылок и фактов доки с артефактами:**
  - `bench/results/20261019-nccl-gpu-r3/` существует; `pair_04_attempts.log` (исходы
    попыток 1–3), `pair_04_nccl_debug.log`, `tp1_rtx2060.log` — ссылка валидна;
  - `tp1_rtx2060.log`: `GPU-VALIDATION OK`, `checks=1 failures=0`, allreduce 0.8 µs,
    NCCL 2.26.2, пометка co-tenancy/strata в логе — утверждения доки
    «checks=1 (allreduce ≈ 0.8 µs)» и «NCCL 2.26.2» подтверждены;
  - `expert_alltoall` / `kv_stream` — присутствуют в `comm.hpp` (ADR-10 §2/§4);
  - PLAN.md §5 («Бенчмарк-контракт») и ADR-03/ADR-09/ADR-11 существуют.
- **Итог сверки: фактических ошибок и битых ссылок в NCCL-протоколе `server-env.md`
  не найдено — правки не нужны** (тот же вывод, что и в round 7; дока уже
  перегенерирована под round 7 и консистентна с кодом).

### Что нужно

- **Чистое окно — пункт эскалации пользователю.** Пара V100+V100 без strata; исполнение
  протокола `docs/design/server-env.md` (3 попытки + транспортная sanity +
  `NCCL_DEBUG=INFO` + strace attach). Межкарточная корректность (`expert_alltoall`,
  `kv_stream`) — блокер GPU-этапа среза 4.
- Окно **не блокирует** срез 4: остальной объём среза (CPU-контракты, tiers, quality-gates
  с честными co-tenancy-пометками) исполним без NCCL-окна; pending честный, вердикт не
  подменяется.

## Итог

- [NCCL-окно] — pending, изменений нет; прогоны запрещены до чистого окна; потребность
  окна — эскалация пользователю (не блокирует срез 4).
- `docs/design/server-env.md` — сверен с `src/parallel` и артефактами
  `bench/results/20261019-nccl-gpu-r3/`: ошибок/битых ссылок нет, правки не потребовались.
- `src/parallel` — read-only, код не менялся.

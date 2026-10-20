# REPORT — round 7, поток C (parallel), итерация 8

Ветка: `agent/parallel/8` (от master `3ea93d1`). Коммитов в чужие ветки нет, `.pi/` не тронут.

## [R6] NCCL-окно — честный PENDING, изменений нет

**Статус: pending, без изменений по требованию planner round 6.** Попытки NCCL/TP-прогонов
под co-tenancy не выполнялись — они **запрещены** (чужой процесс `strata` держит ~21–30 GB
VRAM на каждой V100 сервера 192.168.1.114; GPU-нагрузку создавать нельзя).

### Почему невозможно

- Ограничение среды зафиксировано окончательно ещё в round 4–5 (`docs/design/server-env.md`,
  раздел «NCCL-транспорт: ограничение среды»): все три попытки протокола (P2P →
  `NCCL_P2P_DISABLE=1` → `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`) не подняли межкарточный
  транспорт (shm-attach errno 11/115; затем `Failed to CUDA host alloc 9641984 bytes` на
  обоих ранках). Single-card world=1 (RTX 2060) — GPU-VALIDATION OK, `checks=1`: наш код и
  NCCL корректны до межранкового bootstrap.
- Planner round 6 запретил любые прогоны под co-tenancy; «фолбэк» на неподтверждённую
  корректность и обходы (не-NCCL транспорт, самописные P2P) запрещены протоколом.

### Что готово (в репо, проверено в этой итерации)

- **Протокол чистого окна** в `docs/design/server-env.md` (раздел «NCCL-транспорт:
  ограничение среды»): полная схема 3 попыток + транспортная sanity-фаза (round 4b) +
  expert-контракт-фаза, `NCCL_DEBUG=INFO` + strace attach, развилка
  `gpu_validation_verdict` (Ok / Pending / FailExpert / FailConfirmed). Протокол актуален
  при чистом окне — команды соответствуют коду (`src/parallel/include/stell/parallel/p2p_retry.hpp`:
  `kGpuRetryMaxAttempts = 3`, `p2p_retry_decide`, `p2p_disable_env`, `shm_disable_env`,
  `bench_allowed_after`, `gpu_validation_verdict`). Дополнение round 7 в server-env
  подтверждает: «NCCL-окно ([R6]). Статус pending с условием не меняется».
- Проверка ссылок: `bench/results/20261019-nccl-gpu-r3/` существует, содержит
  `pair_04_attempts.log`, `pair_04_nccl_debug.log`, `tp1_rtx2060.log` — ссылка из
  server-env валидна. **Фактических ошибок и битых ссылок не найдено — правки не нужны.**

### Чего нет (честно, без выдумываний)

- Чистого окна (пара V100+V100 без strata) — **не существует на текущий момент**; поэтому
  самих прогонов, логов и вердикта межкарточной корректности (`expert_alltoall`,
  `kv_stream`) нет ни в этой итерации, ни ранее.
- Никаких новых артефактов `bench/results/` по [R6] в этой итерации не создавалось
  (запрещено) — это отсутствие по правилам, а не упущение.

### Что нужно

- **Чистое окно — пункт эскалации пользователю.** Исполнение протокола
  `docs/design/server-env.md` (3 попытки + транспортная sanity + `NCCL_DEBUG=INFO` +
  strace attach) на паре V100+V100 без strata. Межкарточная корректность — блокер среза 2
  GPU-этапа.
- Окно **не блокирует** [R1]–[R4] раунда 7 (перегенерация описаний qk-norm/router_renorm,
  I/O-протокол, формальный критерий §4 — всё исполняемо на CPU/co-tenancy с честными
  пометками).

## Проверка консистентности src/parallel ↔ ADR-10 (read-only, код не менялся)

Проверено соответствие `src/parallel` контракту `docs/adr/ADR-10-engine-parallel.md` —
расхождений не найдено:

- `comm.hpp` — интерфейс `ICommunicator` совпадает с ADR-10 §2 + уточнениями round 2:
  `expert_alltoall(..., std::size_t token_elems, CommDType t)`;
  `ICommunicator* subgroup(const std::vector<rank_t>& ranks)`; `CommError::Code
  { NcclFailure, RankMismatch, BufferTooSmall, Timeout }`; `CommDType { F16, F32, E4M3 }`.
- `plan.hpp` — формула ранка соответствует ADR-10 §3: `rank = pp_rank * tp_ep +
  tp_ep_rank` (`compose_rank`, `tp_ep = tp*ep`, `tp_ep_rank = ep_rank*tp + tp_rank`);
  порядок PP → TP/EP; `expert_owner` = `expert_id % ep`; `check_kv_heads_divisible`
  (n_kv_heads % tp == 0 — обязательная проверка); планировщик на ранке 0.
- `emulated_communicator.cpp` — CPU-эмулятор (ADR-10 §4): барьерные коллективы,
  `BufferTooSmall` при нехватке recv-ёмкости (молчаливого сокращения нет), `Timeout`
  поддерживается, владение подгруппой у родителя, переиспользование под-комма по ranks.
- `nccl_communicator.cpp` — NCCL-backend реализует token_elems, subgroup (bootstrap-реестр
  под-коммов), `ncclGetLastError` в CommError (round 4b).
- `p2p_retry.hpp` — ровно 3 попытки, развилка вердикта и honest pending/fail-семантика
  соответствуют протоколу server-env.

## Итог

- [R6] — pending, изменений нет; прогоны запрещены до чистого окна; потребность окна —
  эскалация пользователю (не блокирует [R1]–[R4]).
- `docs/design/server-env.md` — проверен, ошибок/битых ссылок нет, правки не требовались.
- `src/parallel` — консистентен с ADR-10, код не менялся (read-only проверка).

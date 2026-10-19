# parallel ↔ engine — сверка контракта ICommunicator (задача 4, раунд 2)

Контекст: вклейка core-2 (engine) с модулем parallel. ADR-10 §2 — контракт;
`src/engine/include/stell/engine/adapters/stub_parallel.hpp` — выписанная ДО
реализации parallel копия интерфейса («stub»). Раунд 2 сверил реальный
`src/parallel/include/stell/parallel/comm.hpp` с ожиданиями scheduler
(`src/engine/src/scheduler.cpp::broadcast_step_meta`).

## 1. Сигнатуры операций — СОВПАДАЮТ

`allreduce/allgather/expert_alltoall/chain_send/chain_recv/kv_stream_push/kv_stream_pop`
+ `world_size()/rank()` — сигнатуры в stub и в реальном заголовке идентичны
(имена, параметры, порядок). Мета-протокол: scheduler вызывает
`allgather(meta, recv, 8, parallel::CommDType::F32)` — 8 слотов × 4 байта =
32 байта, ровно как в уточнении ADR-10 (раунд 1): `CommDType::Meta` отклонён,
meta — непрозрачный для parallel буфер.

## 2. РАСХОЖДЕНИЕ (не молча): ёмкость recv-буфера allgather

- Контракт `comm.hpp::allgather`: recv получает **world × count** элементов
  (конкатенация send-буферов всех ранков).
- scheduler.cpp (раунд 1, до появления comm): `float meta[8], recv[8]` —
  ёмкость **count**, а не world×count.
- Эффект при реальном бэкенде: **переполнение recv на стеке scheduler**
  (NCCL ncclAllGather и CPU-эмулятор пишут world×count). Поймано ASan в
  tests/parallel/test_meta_broadcast.cpp (stack-buffer-overflow, WRITE size 32,
  memcpy в EmulatedCommunicator::allgather).
- Требуемая правка во вклейке core-2 (engine): `float recv[8 * world]` —
  либо (семантически чище) использовать отдельный broadcast-примитив, но
  расширять ICommunicator ради меты не планируется (ADR-10: асинхронные
  хендлы/новые операции — только через ADR).

## 3. РАСХОЖДЕНИЕ: все ранки обязаны участвовать в allgather

scheduler.cpp вызывает `broadcast_step_meta` в `step()` — вызов исполняется
на том ранке, где крутится планировщик; последователи (ранги ≠ 0) обязаны
вызывать `allgather` с теми же (count, dtype) каждый шаг, иначе ранг 0
блокируется до Timeout (эмулятор) или виснет (NCCL — коллективная операция
на всех ранках комм-группы). В раунде 1 followers-логики нет; во вклейке:
ранг 0 рассылает мету, остальные ранки в своём цикле исполнения вызывают
`allgather` и исполняют полученную разметку.

## 4. РАСХОЖДЕНИЕ типов: ODR-конфликт двух ParallelConfig/CommError

stub_parallel.hpp и comm.hpp объявляют ОДНОИМЕННЫЕ типы в одном
namespace stell::parallel (ICommunicator, ParallelConfig, CommError,
CommDType) с разной семантикой полей:

| Поле | stub (engine) | реальный (parallel) |
|---|---|---|
| tp/pp/ep | `std::uint32_t` | `rank_t` (`std::int32_t`) |
| mode, rank, group_id, timeout_ms, kv_ring_capacity, cuda_stream | отсутствуют | есть |
| CommError::ctor | `(Code, const std::string&)` | `(Code, const std::string& = "")` |

Если оба заголовка попадут в одну единицу трансляции — ODR-нарушение/линк-
конфликт. Во вклейке core-2: заменить `#include "adapters/stub_parallel.hpp"`
в scheduler.hpp на `#include <stell/parallel/comm.hpp>` и удалить stub
(единственный источник истины — comm.hpp); ParallelConfig строить в engine
из конфига (mode=Nccl, device_ids, rank=порядковый ранок процесса, timeout).

## 5. Прочее сверка

- `broadcast_step_meta` по умолчанию выключен (Config::broadcast_step_meta=false,
  CPU-раунд 1) — во вклейке включается при передаче ICommunicator.
- Эмулятор: группа после ошибки Timeout не гарантирует состояние — engine
  обязан пересоздавать комм-группу (новый group_id) при CommError(Timeout).
- kv_stream_push/pop в NCCL-бэкенде пока бросают NcclFailure («завершить при
  серверной валидации») — engine-стык KV-streaming (ADR-10 §2) на GPU-ранге
  нельзя подключать до завершения; в отчёте раунда 2 помечено pending.

## 6. Тесты раунда 2, фиксирующие семантику

- tests/parallel/test_meta_broadcast.cpp — паттерн scheduler на РЕАЛЬНОМ
  ICommunicator (broadcast-as-allgather, многошаговый follower-режим);
  recv-буфер контрактной ёмкости world×count.
- tests/parallel/test_stress_multi_rank.cpp — нагрузка/тайм-ауты/deadlock;
  обнаружен мёртвый защитный путь BufferTooSmall в expert_alltoall
  (см. отчёт) и исправлены два дефекта эмулятора (rethrow последнего
  прибывшего; симметричная ёмкость KV-кольца).

## 7. Дополнения раунда 2 (золотой прогон mini_dense/mini_moe)

- **ODR-конфликт stub_models/stub_tiers**: engine (`adapters/stub_models.hpp`,
  `adapters/stub_tiers.hpp`) выписал те же одноимённые типы
  (`stell::models::ModelDescription/IRegistry`, `stell::tiers::TensorGroup/
  GroupKey/GroupId/Awaiter`) — теперь реальные модули (`src/models`,
  `src/tiers`) их определяют. Во вклейке: заменить все три stub-включения на
  реальные заголовки (models/registry.hpp+description.hpp, tiers/types.hpp,
  parallel/comm.hpp) и удалить stub-каталог adapters (иначе ODR/линк-конфликт).
  Расхождения stub_models ↔ реальное описание: нет `ops`, `schema`,
  `group_source_patterns`, `kv`-структура (плоские поля), `moe` (плоские поля) —
  `build_execution_plan` потребует адаптации.
- **Лоадер NVFP4W4A16 отклоняет F32/F16 группы** смешанного плана
  (UnsupportedQuant в do_fill) — mini_moe (NVFP4-профиль, F32-нормы и
  F32-роутер) читается ТОЛЬКО двумя источниками (plain safetensors +
  quant-лоадер). Либо лоадеру нужен fallback для неквантованных групп, либо
  смешанный план фиксируется как «два источника» в ADR-03/09.
- **Расхождение раскладки NVFP4**: лоадер/конвертер models упаковывают блок
  32 (nibbles + scale E4M3 на 32 элемента), kernels `dequant_nvfp4` ждёт блок
  16 (`scales ceil(n/16)`) — де-квантование kernels над payload лоадера даст
  мусор. Интеграция engine↔kernels обязана зафиксировать единую раскладку
  (блок 32 — как в quant.hpp/weights.hpp) или две ветки.
- **expert_alltoall не параметризует ширину токена**: контракт —
  «max_tokens элементов dtype» (эмулятор раунда 1 гонит скаляры); реальный
  MoE-токен — d_model-вектор. Тест раунда гонит вектор поэлементно
  (slot_id кодирует позицию — допущение A3b) — работает, но нужен ADR:
  либо параметр `token_width`, либо «токен = max_tokens×d_model элементов».
- **allgather-as-broadcast**: recv-буфер обязан быть world×count (scheduler.cpp
  выделяет count — переполнение, ASan-поймано; см. §2).
- **expert_alltoall recv-ёмкость**: при контрактной ёмкости max_tokens×world
  переполнение (BufferTooSmall) недостижимо (fan-out ≤ tp даёт inflow ≤
  ёмкость ровно в worst-case) — либо ревизия контракта, либо ёмкость по плану.

## 8. Раунд 2 (R4/R5/R6, T-20261017200354): token_elems, subgroup, NCCL

Реализовано в `src/parallel` (ветка agent/parallel/3); уточнения внесены в
`comm.hpp` (контракт ADR-10) и в `docs/design/parallel.md` (дубль docs-3).
Текст ниже — материал для «Уточнения round 2» в ADR-10 (docs/ закрыт для
потока C — переносит worker/planner при merge).

- **[R4] token_elems**: `expert_alltoall(..., max_tokens, token_elems, t)`;
  token_elems — элементов dtype на ОДИН токен (hidden → d_model);
  token_elems = 1 — совместимость со скалярной семантикой раунда 1 (старые
  вызовы/тесты остаются зелёными без порогов-послаблений). send_tokens —
  max_tokens*token_elems элементов; recv — max_tokens*world ТОКЕНОВ
  (= max_tokens*world*token_elems элементов) и max_tokens*world мет; входящие
  — в порядке (источник ↑, индекс ↑); хвост — нули + TokenMeta{-1,-1};
  переполнение → CommError(BufferTooSmall).
- **[R4] golden**: mini_moe EP2 в `test_golden_real_models.cpp` переписан на
  ЦЕЛЫЕ токены (token_elems = dm, мета — одна на токен, slot_id кодирует пару
  (источник,(t,j))) — побитовое равенство TP1 сохранено (88000+ CHECK).
  Юнит-тест маршрутизации целых токенов: `tok_width_expert_alltoall_dmodel_ep2`.
- **[R5] subgroup**: `ICommunicator::subgroup(ranks)` — коллективы и p2p
  подгруппы затрагивают только её ранки; ranks — уникальные глобальные ранки,
  содержащие собственный (иначе CommError(RankMismatch)); перенумерация — по
  возрастанию глобального ранка; ВЛАДЕНИЕ У РОДИТЕЛЯ (хранится в родителе,
  повторный вызов с тем же ranks возвращает тот же объект); вложенные
  подгруппы допустимы. Эмулятор — фильтрация ранков с ОТДЕЛЬНЫМ состоянием
  рандеву на подгруппу (коллективы подгруппы не пересекаются с родителем).
- **[R5] BufferTooSmall**: инвариант inflow(d) ≤ Σ_s max_tokens = контрактной
  ёмкости (max_tokens×world) означает, что в ПОЛНОЙ группе ошибка недостижима
  (подтверждает вывод §7). Достижимая ветвь — ПОДГРУППА: маршрут токена за
  пределы состава подгруппы → CommError(BufferTooSmall) у ВСЕХ членов
  (молчаливое сокращение пересылки запрещено); тест
  `sub_route_outside_buffer_too_small`. NCCL-путь: матрица счётчиков через
  ncclAllGather (строка = [counts по позициям, out_sub]) — решения об ошибках
  глобально согласованы между членами.
- **[R6] NCCL**: `expert_alltoall` — обмен матрицей счётчиков (ncclAllGather)
  + байтовый p2p ncclSend/ncclRecv по парам ранков (токены и меты отдельными
  проходами, fused в ncclGroupStart/ncclGroupEnd); локальная доставка — D2D;
  сборка recv по возрастанию источника. `kv_stream` — ncclSend/ncclRecv по
  кольцу (next/prev) на ОТДЕЛЬНОЙ kv-комме (не мешает коллективам приложения);
  блок = заголовок uint64 + payload; фоновый поток-приёмник обеспечивает
  неблокирующий pop; push даёт backpressure до приёма получателем.
  `subgroup` — свежий комм-клик по бутстрап-реестру подгрупп (ключ — отсорти
  рованные члены; NCCL 2.26.2 на сервере).
- **pending GPU-валидации [R6]** (worker после merge): корректность + latency
  expert_alltoall/kv_stream на V100 (тест `parallel_gpu` — добавлен
  `parallel_gpu_nccl_expert_alltoall_kv_stream`, при OOM/контенанс — честный
  PENDING); бутстрап ncclUniqueId между процессами; замена host-маршрутизации
  на device-ядра MoE gather/scatter (kernels); Timeout-семантика kv-кольца на
  NCCL-пути (сейчас — backpressure).
- **Синтакс-контроль NCCL-файла** локально (файл собирается только при
  STELL_CUDA=ON): прогнан через g++ -fsyntax-only со минимальными стубами
  cuda_runtime.h/nccl.h (сигнатуры сверены с серверным nccl.h 2.26.2) — чисто,
  -Wall -Wextra.
- Тесты: 53 → 59 кейсов test_parallel (добавлено 6: tok_width ×1, sub ×5);
  ctest локально 40/40; golden побитовые TP2/TP4/EP2 == TP1 — без послаблений.

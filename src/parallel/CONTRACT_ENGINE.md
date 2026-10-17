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

// stell-llm — parallel: NCCL-backend (CUDA/NVLink). Заголовок.
//
// Собирается ТОЛЬКО при STELL_CUDA=ON и наличии libnccl (сервер: cuda 12.8,
// /usr/include/nccl.h, libnccl.so, 4×V100-SXM2 NVLink). Локально (CPU) —
// не компилируется и не тестируется; реальный TP2/TP4 прогон — на сервере.
//
// Раунд 2: expert_alltoall и kv_stream реализованы (ncclSend/ncclRecv);
// subgroup(ranks) — свежий комм-клик по бутстрап-реестру подгрупп, ВЛАДЕНИЕ
// У РОДИТЕЛЯ (хранится в родителе, разрушается его деструктором, повторный
// вызов с тем же ranks возвращает тот же объект). GPU-валидация [R6] —
// pending: worker прогонит после merge (co-tenancy: strata ~21–30 GB).
#pragma once

#include <memory>
#include "stell/parallel/comm.hpp"

namespace stell::parallel {

// Требования к cfg для Nccl-режима:
//   - cfg.mode == CommMode::Nccl;
//   - cfg.rank/world — топология плана (rank = pp_rank*(tp*ep)+ep_rank*tp+tp_rank);
//   - cfg.device_ids[rank] — CUDA-устройство этого ранка;
//   - cfg.cuda_stream — CUDA-stream, на котором обязаны исполняться
//     коллективы и chain/kv-p2p приложения (ADR-10 §2: синхронная семантика
//     на stream); kv-кольцо крутится на внутренних CUDA-streams (фоновый
//     приёмник — pending-серверной валидации);
//   - буферы операций — device-указатели на device_ids[rank].
// Бутстрап ncclUniqueId: rank 0 создаёт id (основной + отдельный kv-комм);
// для ранков одного процесса id передаётся через общее состояние; межпроцессный
// бутстрап (IPC) — отдельная задача интеграции с engine (pending).
std::unique_ptr<ICommunicator> make_nccl_communicator(const ParallelConfig& cfg);

} // namespace stell::parallel

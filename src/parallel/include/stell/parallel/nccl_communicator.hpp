// stell-llm — parallel: NCCL-backend (CUDA/NVLink). Заголовок.
//
// Собирается ТОЛЬКО при STELL_CUDA=ON и наличии libnccl (сервер: cuda 12.8,
// /usr/include/nccl.h, libnccl.so, 4×V100-SXM2 NVLink). Локально (CPU) —
// не компилируется и не тестируется; реальный TP2/TP4 прогон — на сервере.
#pragma once

#include <memory>
#include "stell/parallel/comm.hpp"

namespace stell::parallel {

// Требования к cfg для Nccl-режима:
//   - cfg.mode == CommMode::Nccl;
//   - cfg.rank/world — топология плана (rank = pp_rank*(tp*ep)+ep_rank*tp+tp_rank);
//   - cfg.device_ids[rank] — CUDA-устройство этого ранка;
//   - cfg.cuda_stream — CUDA-stream, на котором обязаны исполняться операции
//     (ADR-10 §2: синхронная семантика на stream);
//   - буферы операций — device-указатели на device_ids[rank].
// Бутстрап ncclUniqueId: rank 0 создаёт id; для ранков одного процесса id
// передаётся через общее состояние; межпроцессный бутстрап (IPC) — отдельная
// задача интеграции с engine (помечено pending-серверной валидации).
std::unique_ptr<ICommunicator> make_nccl_communicator(const ParallelConfig& cfg);

} // namespace stell::parallel

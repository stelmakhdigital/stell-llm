// stell-llm — parallel: фабрика комм-бэкендов (ADR-10 §2/§4).
#include "stell/parallel/comm.hpp"
#include "stell/parallel/emulated.hpp"

#if defined(STELL_HAVE_NCCL)
#include "stell/parallel/nccl_communicator.hpp"
#endif

#include <memory>

namespace stell::parallel {

std::unique_ptr<ICommunicator> make_communicator(const ParallelConfig& cfg) {
#if defined(STELL_HAVE_NCCL)
    if (cfg.mode == CommMode::Nccl)
        return make_nccl_communicator(cfg);
#else
    if (cfg.mode == CommMode::Nccl)
        // NCCL-backend собирается только при STELL_CUDA + libnccl (сервер,
        // cuda 12.8 + libnccl.so; локально недоступен — см. ADR-07 CPU-путь).
        throw CommError(CommError::Code::NcclFailure,
                        "NCCL-backend не собран: требуется STELL_CUDA=ON и libnccl");
#endif
    return std::make_unique<EmulatedCommunicator>(cfg);
}

} // namespace stell::parallel

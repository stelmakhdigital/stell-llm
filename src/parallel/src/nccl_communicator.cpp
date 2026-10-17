// stell-llm — parallel: NCCL-backend (ADR-02, ADR-10). Реализация.
//
// ВАЖНО: этот файл компилируется только при STELL_CUDA=ON + libnccl
// (CMakeLists задаёт STELL_HAVE_NCCL). Локальная CPU-конфигурация его не
// собирает и не тестирует: реальный прогон TP2/TP4 — на сервере (V100×4,
// NVLink, cuda 12.8), с учётом co-tenancy (чужой процесс ~21–30 GB VRAM).
//
// pending-серверной валидации:
//   - бутстрап ncclUniqueId между процессами (сейчас — только один процесс
//     с потоками-ранками, id через общее состояние);
//   - E4M3-редукция через F32-стадию (host-конвертация; при интеграции
//     с kernels заменить на device-ядра MoE gather/scatter);
//   - производительность ncclSend/ncclRecv для expert_alltoall (план ≥3.2×
//     на TP4 dense фиксируется в bench/results по контракту PLAN.md §5).
#include "stell/parallel/nccl_communicator.hpp"

#if defined(STELL_HAVE_NCCL)

#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/plan.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace stell::parallel {
namespace {

ncclDataType_t nccl_dtype(CommDType t) {
    switch (t) {
        case CommDType::F16: return ncclHalf;
        case CommDType::F32: return ncclFloat;
        case CommDType::E4M3: break;   // стадия F32 (см. коммент. в шапке)
    }
    return ncclFloat;
}

void check_nccl(ncclResult_t r, const char* what) {
    if (r != ncclSuccess)
        throw CommError(CommError::Code::NcclFailure,
                        std::string(what) + ": " + ncclGetErrorString(r));
}
void check_cuda(cudaError_t r, const char* what) {
    if (r != cudaSuccess)
        throw CommError(CommError::Code::NcclFailure,
                        std::string(what) + ": " + cudaGetErrorString(r));
}

// Бутстрап id между ранками одного процесса (см. коммент. в шапке).
struct Bootstrap {
    std::mutex m;
    std::condition_variable cv;
    ncclUniqueId id{};
    bool ready = false;
};
std::map<int, Bootstrap*>& bootstrap_registry() {
    static std::map<int, Bootstrap*> map;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    return map;   // регистрация под mutex ниже
}
std::mutex& bootstrap_map_mutex() {
    static std::mutex m;
    return m;
}

class NcclCommunicator final : public ICommunicator {
public:
    explicit NcclCommunicator(const ParallelConfig& cfg) : cfg_(cfg) {
        check_topology(cfg);
        if (cfg.mode != CommMode::Nccl || cfg.device_ids.empty() ||
            rank_t(cfg.device_ids.size()) != cfg.world())
            throw CommError(CommError::Code::RankMismatch, "NCCL: нужен device_ids по ранку");
        const rank_t my = cfg.rank;
        check_cuda(cudaSetDevice(cfg.device_ids[std::size_t(my)]), "cudaSetDevice");
        stream_ = static_cast<cudaStream_t>(cfg.cuda_stream);
        if (stream_ == nullptr) {
            check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
            own_stream_ = true;
        }

        // Бутстрап: rank 0 создаёт id; остальные ждут.
        {
            std::lock_guard<std::mutex> lk(bootstrap_map_mutex());
            auto& slot = bootstrap_registry()[cfg.group_id];
            if (!slot) slot = new Bootstrap();
            bs_ = slot;
        }
        if (my == 0) {
            check_nccl(ncclGetUniqueId(&bs_->id), "ncclGetUniqueId");
            std::lock_guard<std::mutex> lk(bs_->m);
            bs_->ready = true;
            bs_->cv.notify_all();
        } else {
            std::unique_lock<std::mutex> lk(bs_->m);
            if (!bs_->ready)
                if (bs_->cv.wait_for(lk, std::chrono::milliseconds(cfg.timeout_ms)) ==
                    std::cv_status::timeout)
                    throw CommError(CommError::Code::Timeout, "nccl bootstrap timeout");
        }
        check_nccl(ncclCommInitRank(&comm_, cfg.world(), bs_->id, my), "ncclCommInitRank");
    }
    ~NcclCommunicator() override {
        if (comm_) ncclCommDestroy(comm_);
        if (own_stream_) cudaStreamDestroy(stream_);
    }

    rank_t world_size() const override { return cfg_.world(); }
    rank_t rank() const override { return cfg_.rank; }

    void allreduce(void* buf, std::size_t count, CommDType t) override {
        if (t != CommDType::E4M3) {
            check_nccl(ncclAllReduce(buf, buf, count, nccl_dtype(t), ncclSum, comm_, stream_),
                       "ncclAllReduce");
            sync_check();
            return;
        }
        // E4M3: стадия F32 (host-конвертация; заменить на device-ядро при
        // интеграции с kernels — pending-серверной валидации).
        std::vector<std::uint8_t> raw(count);
        std::vector<float> host(count);
        check_cuda(cudaMemcpy(raw.data(), buf, count, cudaMemcpyDeviceToHost), "D2H");
        for (std::size_t i = 0; i < count; ++i) host[i] = e4m3_to_float(raw[i]);
        float* stage = nullptr;
        check_cuda(cudaMalloc(&stage, count * sizeof(float)), "cudaMalloc");
        check_cuda(cudaMemcpy(stage, host.data(), count * sizeof(float), cudaMemcpyHostToDevice),
                   "H2D");
        check_nccl(ncclAllReduce(stage, stage, count, ncclFloat, ncclSum, comm_, stream_),
                   "ncclAllReduce(F32 stage)");
        check_cuda(cudaStreamSynchronize(stream_), "sync");
        check_cuda(cudaMemcpy(host.data(), stage, count * sizeof(float), cudaMemcpyDeviceToHost),
                   "D2H");
        for (std::size_t i = 0; i < count; ++i) raw[i] = float_to_e4m3(host[i]);
        check_cuda(cudaMemcpy(buf, raw.data(), count, cudaMemcpyHostToDevice), "H2D");
        cudaFree(stage);
    }

    void allgather(void* send, void* recv, std::size_t count, CommDType t) override {
        if (t == CommDType::E4M3)
            throw CommError(CommError::Code::NcclFailure,
                            "ncclAllGather для E4M3 не поддерживается (используйте F16/F32)");
        check_nccl(ncclAllGather(send, recv, count, nccl_dtype(t), comm_, stream_), "ncclAllGather");
        sync_check();
    }

    // EP: alltoall поверх ncclSend/ncclRecv парами ранков (ADR-10 §1):
    // для каждой пары (src, dst) — ncclSend токенов с owner==dst и зеркальный
    // ncclRecv; метки (TokenMeta) — отдельным проходом. Групповой обмен через
    // ncclGroupStart/ncclGroupEnd.
    void expert_alltoall(void* send_tokens, void* send_meta, void* recv_tokens,
                         void* recv_meta, std::size_t max_tokens, CommDType t) override {
        // TODO(сервер): точный обмен по рангам с учётом slot_id (ADR-08);
        // семантика — как в CPU-эмуляторе (порядок: источник ↑, индекс ↑).
        throw CommError(CommError::Code::NcclFailure,
                        "expert_alltoall(NCCL): завершить при серверной валидации");
    }

    void chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) override {
        group_sendrecv(peer, buf, count, t, /*send=*/true);
    }
    void chain_recv(void* buf, std::size_t count, CommDType t, rank_t peer) override {
        group_sendrecv(peer, buf, count, t, /*send=*/false);
    }

    void kv_stream_push(const void* kv_block, std::size_t bytes) override {
        // Кольцевая очередь поверх chain send/recv: push шлёт блок следующему
        // ранку и принимает от предыдущего (ротация кольца).
        throw CommError(CommError::Code::NcclFailure,
                        "kv_stream(NCCL): завершить при серверной валидации");
    }
    bool kv_stream_pop(void* dst, std::size_t bytes) override {
        throw CommError(CommError::Code::NcclFailure,
                        "kv_stream(NCCL): завершить при серверной валидации");
    }

private:
    void sync_check() {
        check_cuda(cudaStreamSynchronize(stream_), "nccl stream sync");
    }
    void group_sendrecv(rank_t peer, void* buf, std::size_t count, CommDType t, bool send) {
        check_nccl(ncclGroupStart(), "ncclGroupStart");
        if (send)
            check_nccl(ncclSend(buf, int(count), nccl_dtype(t), peer, comm_, stream_), "ncclSend");
        else
            check_nccl(ncclRecv(buf, int(count), nccl_dtype(t), peer, comm_, stream_), "ncclRecv");
        check_nccl(ncclGroupEnd(), "ncclGroupEnd");
        sync_check();
    }

    ParallelConfig cfg_;
    ncclComm_t comm_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool own_stream_ = false;
    Bootstrap* bs_ = nullptr;
};

} // namespace

std::unique_ptr<ICommunicator> make_nccl_communicator(const ParallelConfig& cfg) {
    return std::make_unique<NcclCommunicator>(cfg);
}

} // namespace stell::parallel

#endif // STELL_HAVE_NCCL

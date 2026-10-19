// stell-llm — parallel: NCCL-backend (ADR-02, ADR-10). Реализация.
//
// ВАЖНО: этот файл компилируется только при STELL_CUDA=ON + libnccl
// (CMakeLists задаёт STELL_HAVE_NCCL). Локальная CPU-конфигурация его не
// собирает и не тестирует: реальный прогон TP2/TP4 — на сервере (V100×4,
// NVLink, cuda 12.8), с учётом co-tenancy (чужой процесс ~21–30 GB VRAM).
//
// Раунд 2 (уточнения ADR-10): expert_alltoall реализован поверх ncclSend/
// ncclRecv (обмен счётчиками через ncclAllGather, затем p2p по парам ранков;
// p2p — байтовый, ncclUint8); kv_stream — через ncclSend/ncclRecv по кольцу
// (ранок next/prev) с фоновым потоком приёмника (неблокирующий pop);
// subgroup(ranks) — свежий комм-клик по реестру бутстрапа подгрупп (владение
// у родителя). Серверная GPU-валидация [R6] — pending: worker прогонит после
// merge (co-tenancy честная, свободная VRAM ~2–11 GB).
//
// pending-серверной валидации (помечено по ходу файла):
//   - бутстрап ncclUniqueId МЕЖДУ процессами (сейчас — ранки-потоки одного
//     процесса, id через общее состояние);
//   - маршрутизация expert_alltoall через host-метаданные (D2H) и D2D-упаковка
//     токенов — при интеграции с kernels заменить на device-ядра MoE
//     gather/scatter;
//   - производительность ncclSend/ncclRecv (план ≥3.2× на TP4 dense —
//     bench/results по контракту PLAN.md §5);
//   - kv_stream: тайм-ауты кольца (Timeout) на NCCL-пути недостижимы —
//     push даёт backpressure до приёма получателем; pop неблокирующий.
#include "stell/parallel/nccl_communicator.hpp"

#if defined(STELL_HAVE_NCCL)

#include "stell/parallel/dtype_codec.hpp"
#include "stell/parallel/plan.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
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

void check_cuda(cudaError_t r, const char* what) {
    if (r != cudaSuccess)
        throw CommError(CommError::Code::NcclFailure,
                        std::string(what) + ": " + cudaGetErrorString(r));
}

// Причина ошибки NCCL — в двух уровнях (диагностика round 4b): слепое пятно
// серверного прогона было в том, что ncclGetErrorString даёт только класс
// («unhandled cuda error» / «run with NCCL_DEBUG=INFO») — реальная причина
// скрыта в NCCL. ncclGetLastError(comm) вскрывает underlying ошибку
// (CUDA-статус, файл/строка NCCL). comm==nullptr безопасно («no communicator»):
// на пути бутстрапа (ncclGetUniqueId/ncclCommInitRank) комма ещё нет.
void check_nccl(ncclResult_t r, const char* what, const ncclComm_t comm = nullptr) {
    if (r == ncclSuccess) return;
    std::string msg = std::string(what) + ": " + ncclGetErrorString(r);
    if (comm) {
        const char* last = ncclGetLastError(comm);
        if (last && *last && std::string(last) != "no error")
            msg += std::string(" [ncclGetLastError: ") + last +"]";
    }
    throw CommError(CommError::Code::NcclFailure, msg);
}

// Бутстрап id между ранками одного процесса (см. коммент. в шапке).
// Два id: основной комм и kv-комма (kv-потоки не смешиваются с коллективами
// приложения на одной комме — NCCL-коммы используются из нескольких потоков).
struct Bootstrap {
    std::mutex m;
    std::condition_variable cv;
    ncclUniqueId id{};
    ncclUniqueId id_kv{};
    bool ready = false;
};
Bootstrap* bootstrap_slot(int group_id) {
    static std::mutex m;
    static std::map<int, Bootstrap*> map;
    std::lock_guard<std::mutex> lk(m);
    auto& slot = map[group_id];
    if (!slot) slot = new Bootstrap();
    return slot;
}

// Бутстрап подгруппы: ключ — отсортированные глобальные ранки-члены (один
// процесс — ранки-потоки; межпроцессный бутстрап — pending-интеграции).
struct SubBootstrap {
    std::mutex m;
    std::condition_variable cv;
    ncclUniqueId id{};
    ncclUniqueId id_kv{};
    bool ready = false;
};
SubBootstrap* sub_bootstrap_slot(const std::vector<rank_t>& members) {
    static std::mutex m;
    static std::map<std::vector<rank_t>, SubBootstrap*> map;
    std::lock_guard<std::mutex> lk(m);
    auto& slot = map[members];
    if (!slot) slot = new SubBootstrap();
    return slot;
}

} // namespace

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

        // Бутстрап: rank 0 создаёт id (основной + kv); остальные ждут.
        bs_ = bootstrap_slot(cfg.group_id);
        if (my == 0) {
            check_nccl(ncclGetUniqueId(&bs_->id), "ncclGetUniqueId");
            check_nccl(ncclGetUniqueId(&bs_->id_kv), "ncclGetUniqueId(kv)");
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

        // Полная группа: члены 0..world-1, позиция == глобальному ранку.
        for (rank_t r = 0; r < cfg.world(); ++r) members_.push_back(r);
        my_pos_ = my;
        fill_positions();

        // kv-комма — отдельная (kv-потоки и коллективы приложения не делят
        // одну комму: p2p-операции kv не блокируют коллективы).
        if (cfg.world() > 1) {
            check_nccl(ncclCommInitRank(&kv_comm_, cfg.world(), bs_->id_kv, my),
                       "ncclCommInitRank(kv)");
            check_cuda(cudaStreamCreate(&kv_send_stream_), "cudaStreamCreate(kv send)");
            check_cuda(cudaStreamCreate(&kv_recv_stream_), "cudaStreamCreate(kv recv)");
            check_cuda(cudaMalloc(&kv_hdr_send_dev_, sizeof(std::uint64_t)),
                       "cudaMalloc(kv hdr send)");
            check_cuda(cudaMalloc(&kv_hdr_recv_dev_, sizeof(std::uint64_t)),
                       "cudaMalloc(kv hdr recv)");
            kv_recv_thread_ = std::thread([this] { kv_recv_loop(); });
        }
    }

    ~NcclCommunicator() override {
        // Под-коммы принадлежат родителю (ADR-10 round 2) — разрушаются первыми.
        owned_subs_.clear();
        if (kv_recv_thread_.joinable()) {
            kv_stop_ = true;
            // Разблокировать приёмник: ncclCommAbort прерывает операции коммы
            // И освобождает её ресурсы (nccl.h: «frees resources … and aborts»),
            // поэтому Destroy для kv-коммы после Abort НЕ вызывается.
            ncclCommAbort(kv_comm_);
            kv_comm_ = nullptr;
            kv_recv_thread_.join();
        }
        if (kv_hdr_send_dev_) cudaFree(kv_hdr_send_dev_);
        if (kv_hdr_recv_dev_) cudaFree(kv_hdr_recv_dev_);
        if (kv_send_stream_) cudaStreamDestroy(kv_send_stream_);
        if (kv_recv_stream_) cudaStreamDestroy(kv_recv_stream_);
        if (comm_) ncclCommDestroy(comm_);
        if (own_stream_) cudaStreamDestroy(stream_);
    }

    rank_t world_size() const override { return rank_t(members_.size()); }
    rank_t rank() const override { return my_pos_; }

    void allreduce(void* buf, std::size_t count, CommDType t) override {
        if (t != CommDType::E4M3) {
            check_nccl(ncclAllReduce(buf, buf, count, nccl_dtype(t), ncclSum, comm_, stream_),
                       "ncclAllReduce", comm_);
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
                   "ncclAllReduce(F32 stage)", comm_);
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
        check_nccl(ncclAllGather(send, recv, count, nccl_dtype(t), comm_, stream_),
                   "ncclAllGather", comm_);
        sync_check();
    }

    // EP: alltoall поверх ncclSend/ncclRecv парами ранков (ADR-10 §1).
    // Протокол (одинаковый на всех членах, без переговоров):
    //   1. Метаданные (TokenMeta) — D2H на host (pending: device-ядро MoE
    //      gather/scatter в kernels); маршрутизация по родительской топологии
    //      (fan-out на tp-ранки группы-владельца эксперта).
    //   2. Матрица счётчиков ncclAllGather: строка ранка = [count → позиция
    //      p<n, out_sub] (диагональ — локальная доставка). Матрица одинакова
    //      на всех членах → ошибки (маршрут вне подгруппы; переполнение
    //      recv-ёмкости) видны всем одновременно → CommError(BufferTooSmall)
    //      у ВСЕХ членов, без молчаливого сокращения пересылки.
    //   3. Упаковка исходящих токенов (D2D по токенам из send_tokens) и меток
    //      (8 байт = 2×int32 на токен); p2p-обмен — ncclUint8 (байтовый),
    //      fused в ncclGroupStart/ncclGroupEnd; peer — позиция в подгруппе.
    //   4. Сборка recv_tokens/recv_meta: блоки по возрастанию источника
    //      (источник ↑, индекс ↑ — детерминированный контракт ADR-10 §2),
    //      локальная доставка — D2D из send_tokens; хвост ёмкости — нули +
    //      TokenMeta{expert_id = -1}.
    void expert_alltoall(void* send_tokens, void* send_meta, void* recv_tokens,
                         void* recv_meta, std::size_t max_tokens,
                         std::size_t token_elems, CommDType t) override {
        if (token_elems == 0)
            throw CommError(CommError::Code::RankMismatch,
                            "expert_alltoall: token_elems должен быть > 0");
        const rank_t n = world_size();
        const rank_t my = my_pos_;
        const std::size_t es = dtype_size(t);
        const std::size_t tb = token_elems * es;              // байтов на токен
        const std::size_t cap_tokens = std::size_t(max_tokens) * std::size_t(n);

        // RAII device-буферов вызова (нулевой размер — без аллокации).
        struct DevBufs {
            ~DevBufs() { for (void* p : ptrs) cudaFree(p); }
            void* alloc(std::size_t bytes) {
                if (bytes == 0) return nullptr;
                void* p = nullptr;
                check_cuda(cudaMalloc(&p, bytes), "expert_alltoall: cudaMalloc");
                ptrs.push_back(p);
                return p;
            }
            std::vector<void*> ptrs;
        } dev;

        // 1. Метаданные — на host.
        std::vector<TokenMeta> hmeta(max_tokens);
        if (max_tokens)
            check_cuda(cudaMemcpy(hmeta.data(), send_meta,
                                  max_tokens * sizeof(TokenMeta), cudaMemcpyDeviceToHost),
                       "expert_alltoall: D2H send_meta");

        // 2. Маршрут: route[p] — индексы моих токенов на позицию p (p != my),
        //    local — токены, владелец которых я сам (диагональ), out_sub —
        //    отправления за пределы подгруппы.
        std::vector<std::vector<std::size_t>> route{std::size_t(n)};
        std::vector<std::size_t> local;
        std::size_t out_sub = 0;
        const RankDecomp sd = decompose_rank(members_[std::size_t(my)], cfg_);
        std::vector<rank_t> dstv(std::size_t(cfg_.tp));
        for (std::size_t i = 0; i < max_tokens; ++i) {
            const std::int32_t eid = hmeta[i].expert_id;
            if (eid < 0) continue;                            // пустой слот
            expert_alltoall_route(rank_t(eid), sd.pp_rank, cfg_, dstv.data());
            for (rank_t k = 0; k < cfg_.tp; ++k) {
                const rank_t d = dstv[std::size_t(k)];
                if (d < 0 || d >= cfg_.world())
                    throw CommError(CommError::Code::RankMismatch, "route вне world");
                const rank_t dpos = position_of(d);
                if (dpos < 0) ++out_sub;                      // приёмник вне подгруппы
                else if (dpos == my) local.push_back(i);
                else route[std::size_t(dpos)].push_back(i);
            }
        }

        // 3. Матрица счётчиков: строка n+1 int32 → ncclAllGather.
        std::vector<int> row(std::size_t(n) + 1, 0);
        for (rank_t p = 0; p < n; ++p)
            row[std::size_t(p)] = int(p == my ? local.size() : route[std::size_t(p)].size());
        row[std::size_t(n)] = int(out_sub);
        int* d_row = static_cast<int*>(dev.alloc((std::size_t(n) + 1) * sizeof(int)));
        int* d_mat = static_cast<int*>(dev.alloc(
            std::size_t(n) * (std::size_t(n) + 1) * sizeof(int)));
        check_cuda(cudaMemcpy(d_row, row.data(), (std::size_t(n) + 1) * sizeof(int),
                              cudaMemcpyHostToDevice), "expert_alltoall: H2D row");
        check_nccl(ncclAllGather(d_row, d_mat, std::size_t(n) + 1, ncclInt, comm_, stream_),
                   "expert_alltoall: ncclAllGather(counts)", comm_);
        sync_check();
        std::vector<int> mat(std::size_t(n) * (std::size_t(n) + 1));
        check_cuda(cudaMemcpy(mat.data(), d_mat, mat.size() * sizeof(int),
                              cudaMemcpyDeviceToHost), "expert_alltoall: D2H counts");
        // Ошибки — глобально согласованные (решение по общей матрице):
        long long out_total = 0;
        for (rank_t s = 0; s < n; ++s)
            out_total += mat[std::size_t(s) * (std::size_t(n) + 1) + std::size_t(n)];
        if (out_total > 0)
            throw CommError(CommError::Code::BufferTooSmall,
                            "expert_alltoall: маршрут вне подгруппы (recv-ёмкость подгруппы "
                            "не покрывает поток); молчаливое сокращение запрещено");
        for (rank_t p = 0; p < n; ++p) {
            long long inflow = 0;
            for (rank_t s = 0; s < n; ++s)
                inflow += mat[std::size_t(s) * (std::size_t(n) + 1) + std::size_t(p)];
            if (inflow > static_cast<long long>(cap_tokens))
                throw CommError(CommError::Code::BufferTooSmall,
                                "expert_alltoall: recv-ёмкость меньше входящего потока");
        }

        // 4. Упаковка исходящих: токены (tb байт каждый, порядок индекса ↑) и
        //    метки (8 байт каждая) — в раздельные device-буферы, смещения по
        //    позициям назначения.
        std::size_t nsend = 0;
        for (rank_t p = 0; p < n; ++p)
            if (p != my) nsend += route[std::size_t(p)].size();
        std::uint8_t* pack_tok = static_cast<std::uint8_t*>(dev.alloc(nsend * tb));
        std::uint8_t* pack_meta = static_cast<std::uint8_t*>(dev.alloc(nsend * sizeof(TokenMeta)));
        std::vector<std::size_t> send_off(std::size_t(n), 0);
        {
            std::size_t acc = 0;
            for (rank_t p = 0; p < n; ++p) {
                if (p == my) continue;
                send_off[std::size_t(p)] = acc;
                acc += route[std::size_t(p)].size();
            }
        }
        // Метки упаковываются на host (порядок совпадает с токенами).
        std::vector<TokenMeta> hpack(nsend);
        for (rank_t p = 0; p < n; ++p) {
            if (p == my) continue;
            std::size_t off = send_off[std::size_t(p)];
            for (std::size_t k = 0; k < route[std::size_t(p)].size(); ++k) {
                const std::size_t i = route[std::size_t(p)][k];
                check_cuda(cudaMemcpyAsync(pack_tok + (off + k) * tb,
                                           static_cast<std::uint8_t*>(send_tokens) + i * tb,
                                           tb, cudaMemcpyDeviceToDevice, stream_),
                           "expert_alltoall: pack token");
                hpack[off + k] = hmeta[i];
            }
        }
        if (nsend)
            check_cuda(cudaMemcpy(pack_meta, hpack.data(), nsend * sizeof(TokenMeta),
                                  cudaMemcpyHostToDevice), "expert_alltoall: H2D pack meta");

        // 5. Приёмные стейджинги: единые буферы со смещениями по источникам
        //    (блоки по возрастанию источника, включая «дыру» под локальную
        //    доставку — она не пишется через NCCL).
        std::size_t inflow_me = 0;         // весь входящий поток (включая local)
        std::vector<std::size_t> recv_off(std::size_t(n), 0);
        for (rank_t s = 0; s < n; ++s) {
            const std::size_t cnt = (s == my)
                ? local.size()
                : std::size_t(mat[std::size_t(s) * (std::size_t(n) + 1) + my]);
            recv_off[std::size_t(s)] = inflow_me;
            inflow_me += cnt;
        }
        std::uint8_t* stage_tok = static_cast<std::uint8_t*>(dev.alloc(inflow_me * tb));
        std::uint8_t* stage_meta =
            static_cast<std::uint8_t*>(dev.alloc(inflow_me * sizeof(TokenMeta)));

        // 6. P2P-обмен (байтовый, fused): peer — позиция в подгруппе.
        check_nccl(ncclGroupStart(), "expert_alltoall: ncclGroupStart", comm_);
        for (rank_t p = 0; p < n; ++p) {
            if (p == my) continue;
            const std::size_t cnt = route[std::size_t(p)].size();
            if (cnt == 0) continue;
            check_nccl(ncclSend(pack_tok + send_off[std::size_t(p)] * tb, cnt * tb,
                                ncclUint8, p, comm_, stream_), "expert_alltoall: ncclSend tokens", comm_);
            check_nccl(ncclSend(pack_meta + send_off[std::size_t(p)] * sizeof(TokenMeta),
                                cnt * sizeof(TokenMeta), ncclUint8, p, comm_, stream_),
                       "expert_alltoall: ncclSend meta", comm_);
        }
        for (rank_t s = 0; s < n; ++s) {
            if (s == my) continue;
            const std::size_t cnt = std::size_t(mat[std::size_t(s) * (std::size_t(n) + 1) + my]);
            if (cnt == 0) continue;
            check_nccl(ncclRecv(stage_tok + recv_off[std::size_t(s)] * tb, cnt * tb,
                                ncclUint8, s, comm_, stream_), "expert_alltoall: ncclRecv tokens", comm_);
            check_nccl(ncclRecv(stage_meta + recv_off[std::size_t(s)] * sizeof(TokenMeta),
                                cnt * sizeof(TokenMeta), ncclUint8, s, comm_, stream_),
                       "expert_alltoall: ncclRecv meta", comm_);
        }
        check_nccl(ncclGroupEnd(), "expert_alltoall: ncclGroupEnd", comm_);
        sync_check();

        // 7. Сборка recv: блоки по возрастанию источника; локальная доставка —
        //    D2D из send_tokens; меты — host-сборка → H2D; хвост — нули/{-1,-1}.
        auto* rtok = static_cast<std::uint8_t*>(recv_tokens);
        for (rank_t s = 0; s < n; ++s) {
            const std::size_t cnt = std::size_t(mat[std::size_t(s) * (std::size_t(n) + 1) + my]);
            const std::size_t off = recv_off[std::size_t(s)];
            if (s == my) {
                for (std::size_t k = 0; k < local.size(); ++k)
                    check_cuda(cudaMemcpyAsync(rtok + (off + k) * tb,
                                               static_cast<std::uint8_t*>(send_tokens) +
                                                   local[k] * tb,
                                               tb, cudaMemcpyDeviceToDevice, stream_),
                               "expert_alltoall: local token");
            } else if (cnt) {
                check_cuda(cudaMemcpyAsync(rtok + off * tb, stage_tok + off * tb, cnt * tb,
                                           cudaMemcpyDeviceToDevice, stream_),
                           "expert_alltoall: unpack tokens");
            }
        }
        std::vector<TokenMeta> hrmeta(cap_tokens, TokenMeta{});   // заполнитель expert_id=-1
        {
            std::vector<TokenMeta> hin(inflow_me);
            if (inflow_me)
                check_cuda(cudaMemcpy(hin.data(), stage_meta, inflow_me * sizeof(TokenMeta),
                                      cudaMemcpyDeviceToHost), "expert_alltoall: D2H recv meta");
            for (rank_t s = 0; s < n; ++s) {
                const std::size_t cnt = (s == my) ? local.size()
                                                  : std::size_t(mat[std::size_t(s) * (std::size_t(n) + 1) + my]);
                const std::size_t off = recv_off[std::size_t(s)];
                for (std::size_t k = 0; k < cnt; ++k)
                    hrmeta[off + k] = (s == my) ? hmeta[local[k]] : hin[off + k];
            }
            if (cap_tokens)
                check_cuda(cudaMemcpy(recv_meta, hrmeta.data(), cap_tokens * sizeof(TokenMeta),
                                      cudaMemcpyHostToDevice), "expert_alltoall: H2D recv meta");
        }
        if (cap_tokens > inflow_me)
            check_cuda(cudaMemsetAsync(rtok + inflow_me * tb, 0,
                                       (cap_tokens - inflow_me) * tb, stream_),
                       "expert_alltoall: memset filler");
        sync_check();
    }

    void chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) override {
        group_sendrecv(peer, buf, count, t, /*send=*/true);
    }
    void chain_recv(void* buf, std::size_t count, CommDType t, rank_t peer) override {
        group_sendrecv(peer, buf, count, t, /*send=*/false);
    }

    // KV-streaming (ADR-10 §2): кольцевой канал между (rank+1)%n и (rank-1+n)%n
    // поверх ncclSend/ncclRecv на ОТДЕЛЬНОЙ kv-комме (см. шапку). Блок передаётся
    // двумя байтовыми ncclSend: заголовок (uint64 размер) + payload; приёмник —
    // фоновый поток, постящий ncclRecv заранее (неблокирующий pop из локальной
    // очереди принятых блоков). push даёт backpressure до приёма получателем;
    // ёмкость kv_ring_capacity на NCCL-пути проявляется как backpressure
    // (Timeout при переполнении недостижим — pending-серверной валидации).
    void kv_stream_push(const void* kv_block, std::size_t bytes) override {
        if (bytes == 0)
            throw CommError(CommError::Code::NcclFailure,
                            "kv push: блоки нулевого размера не поддержаны на NCCL-пути");
        const rank_t n = world_size();
        if (n == 1) {                                   // кольцо в себя
            void* buf = nullptr;
            check_cuda(cudaMalloc(&buf, bytes), "kv push: cudaMalloc");
            check_cuda(cudaMemcpy(buf, kv_block, bytes, cudaMemcpyDeviceToDevice),
                       "kv push: D2D");
            std::lock_guard<std::mutex> lk(kv_in_mu_);
            kv_in_q_.push_back({buf, bytes});
            return;
        }
        std::lock_guard<std::mutex> lk(kv_push_mu_);    // сериализация push (общий hdr)
        const std::uint64_t sz = bytes;
        check_cuda(cudaMemcpy(kv_hdr_send_dev_, &sz, sizeof(sz), cudaMemcpyHostToDevice),
                   "kv push: H2D size");
        check_nccl(ncclGroupStart(), "kv push: ncclGroupStart", kv_comm_);
        check_nccl(ncclSend(kv_hdr_send_dev_, 1, ncclUint64, next_rank(), kv_comm_,
                            kv_send_stream_), "kv push: ncclSend hdr", kv_comm_);
        check_nccl(ncclSend(kv_block, bytes, ncclUint8, next_rank(), kv_comm_,
                            kv_send_stream_), "kv push: ncclSend block", kv_comm_);
        check_nccl(ncclGroupEnd(), "kv push: ncclGroupEnd", kv_comm_);
        check_cuda(cudaStreamSynchronize(kv_send_stream_), "kv push: sync");
    }

    bool kv_stream_pop(void* dst, std::size_t bytes) override {
        if (bytes == 0)
            throw CommError(CommError::Code::NcclFailure,
                            "kv pop: блоки нулевого размера не поддержаны на NCCL-пути");
        std::lock_guard<std::mutex> lk(kv_in_mu_);
        if (kv_in_q_.empty()) return false;
        const KvBlock& front = kv_in_q_.front();
        if (front.size != bytes)
            throw CommError(CommError::Code::BufferTooSmall,
                            "kv_stream_pop: размер блока не совпал");
        check_cuda(cudaMemcpy(dst, front.dev, bytes, cudaMemcpyDeviceToDevice),
                   "kv pop: D2D");
        cudaFree(front.dev);
        kv_in_q_.pop_front();
        return true;
    }

    // Подгруппа (ADR-10 round 2): новый комм-клик по бутстрап-реестру подгрупп
    // (ключ — отсортированные члены; ранки-потоки одного процесса —
    // межпроцессный бутстрап pending). ВЛАДЕНИЕ У РОДИТЕЛЯ: под-комм хранится
    // в родителе, разрушается его деструктором; повторный вызов с тем же ranks
    // возвращает тот же объект. Перенумерация: ранок i подгруппы — i-й член.
    ICommunicator* subgroup(const std::vector<rank_t>& ranks) override {
        const rank_t my_global = members_[std::size_t(my_pos_)];
        std::vector<rank_t> sorted = ranks;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw CommError(CommError::Code::RankMismatch, "subgroup: повторяющиеся ранки");
        for (rank_t r : sorted)
            if (r < 0 || !std::binary_search(members_.begin(), members_.end(), r))
                throw CommError(CommError::Code::RankMismatch,
                                "subgroup: ранок не входит в текущий комм");
        if (!std::binary_search(sorted.begin(), sorted.end(), my_global))
            throw CommError(CommError::Code::RankMismatch,
                            "subgroup: собственный ранок не входит в подгруппу");
        const rank_t n = rank_t(sorted.size());
        const rank_t my_pos = rank_t(std::find(sorted.begin(), sorted.end(), my_global) -
                                     sorted.begin());

        SubBootstrap* slot = sub_bootstrap_slot(sorted);
        if (my_pos == 0) {
            check_nccl(ncclGetUniqueId(&slot->id), "subgroup: ncclGetUniqueId");
            check_nccl(ncclGetUniqueId(&slot->id_kv), "subgroup: ncclGetUniqueId(kv)");
            std::lock_guard<std::mutex> lk(slot->m);
            slot->ready = true;
            slot->cv.notify_all();
        } else {
            std::unique_lock<std::mutex> lk(slot->m);
            if (!slot->ready)
                if (slot->cv.wait_for(lk, std::chrono::milliseconds(cfg_.timeout_ms)) ==
                    std::cv_status::timeout)
                    throw CommError(CommError::Code::Timeout, "subgroup: bootstrap timeout");
        }
        // Устройство ранка подгруппы = устройство его глобального ранка;
        // контекст устройства потока восстанавливается после инициализации.
        int prev_dev = -1;
        check_cuda(cudaGetDevice(&prev_dev), "subgroup: cudaGetDevice");
        check_cuda(cudaSetDevice(cfg_.device_ids[std::size_t(my_global)]),
                   "subgroup: cudaSetDevice");
        ncclComm_t c = nullptr, kc = nullptr;
        cudaStream_t st = nullptr;
        check_nccl(ncclCommInitRank(&c, n, slot->id, my_pos), "subgroup: ncclCommInitRank");
        check_nccl(ncclCommInitRank(&kc, n, slot->id_kv, my_pos),
                   "subgroup: ncclCommInitRank(kv)");
        check_cuda(cudaStreamCreate(&st), "subgroup: cudaStreamCreate");
        check_cuda(cudaSetDevice(prev_dev), "subgroup: cudaSetDevice(restore)");

        // Владение у родителя: хранится в owned_subs_, разрушается деструктором.
        auto& owned = owned_subs_[sorted];
        if (!owned)
            owned.reset(new NcclCommunicator(cfg_, sorted, my_pos, c, kc, st));
        return owned.get();
    }

private:
    struct KvBlock {
        void* dev;          // device-буфер блока
        std::size_t size;
    };

    // Конструктор подгруппы: коммы уже созданы (бутстрап-реестр подгрупп),
    // стримы свои; маршрутизация — по родительской топологии cfg_.
    NcclCommunicator(const ParallelConfig& cfg, const std::vector<rank_t>& members,
                     rank_t my_pos, ncclComm_t comm, ncclComm_t kv_comm,
                     cudaStream_t stream)
        : cfg_(cfg), comm_(comm), kv_comm_(kv_comm), stream_(stream), own_stream_(true),
          members_(members), my_pos_(my_pos) {
        fill_positions();
        if (rank_t(members_.size()) > 1) {
            check_cuda(cudaStreamCreate(&kv_send_stream_), "cudaStreamCreate(kv send)");
            check_cuda(cudaStreamCreate(&kv_recv_stream_), "cudaStreamCreate(kv recv)");
            check_cuda(cudaMalloc(&kv_hdr_send_dev_, sizeof(std::uint64_t)),
                       "cudaMalloc(kv hdr send)");
            check_cuda(cudaMalloc(&kv_hdr_recv_dev_, sizeof(std::uint64_t)),
                       "cudaMalloc(kv hdr recv)");
            kv_recv_thread_ = std::thread([this] { kv_recv_loop(); });
        }
    }

    void fill_positions() {
        rank_t maxr = 0;
        for (rank_t r : members_) maxr = std::max(maxr, rank_t(r + 1));
        pos_of_.assign(std::size_t(maxr), -1);
        for (std::size_t i = 0; i < members_.size(); ++i)
            pos_of_[members_[i]] = rank_t(i);
    }
    rank_t position_of(rank_t global) const {
        if (std::size_t(global) >= pos_of_.size()) return -1;
        return pos_of_[std::size_t(global)];
    }
    rank_t next_rank() const {
        return rank_t((my_pos_ + 1) % std::size_t(members_.size()));
    }
    rank_t prev_rank() const {
        const rank_t n = rank_t(members_.size());
        return rank_t((my_pos_ + rank_t(n) - 1) % n);
    }

    // Фоновый приёмник kv-кольца: постит ncclRecv заголовка, затем payload
    // (размер из заголовка) и кладёт блок в локальную очередь для pop.
    void kv_recv_loop() {
        for (;;) {
            if (kv_stop_) break;
            {
                std::lock_guard<std::mutex> lk(kv_mu_);
                if (ncclRecv(kv_hdr_recv_dev_, 1, ncclUint64, prev_rank(), kv_comm_,
                             kv_recv_stream_) != ncclSuccess)
                    break;
            }
            const cudaError_t err = cudaStreamSynchronize(kv_recv_stream_);
            if (err != cudaSuccess || kv_stop_) break;   // abort/остановка
            std::uint64_t size = 0;
            // Поток не бросает исключений: ошибка — тихий выход (pending:
            // диагностика kv-потерь при серверной валидации).
            if (cudaMemcpy(&size, kv_hdr_recv_dev_, sizeof(size), cudaMemcpyDeviceToHost)
                != cudaSuccess)
                break;
            if (size == 0) continue;
            void* buf = nullptr;
            if (cudaMalloc(&buf, size) != cudaSuccess) break;
            {
                std::lock_guard<std::mutex> lk(kv_mu_);
                if (ncclRecv(buf, size, ncclUint8, prev_rank(), kv_comm_,
                             kv_recv_stream_) != ncclSuccess) {
                    cudaFree(buf);
                    break;
                }
            }
            if (cudaStreamSynchronize(kv_recv_stream_) != cudaSuccess || kv_stop_) {
                cudaFree(buf);
                break;
            }
            {
                std::lock_guard<std::mutex> lk(kv_in_mu_);
                kv_in_q_.push_back({buf, size});
            }
        }
    }

    void sync_check() {
        check_cuda(cudaStreamSynchronize(stream_), "nccl stream sync");
    }
    void group_sendrecv(rank_t peer, void* buf, std::size_t count, CommDType t, bool send) {
        check_nccl(ncclGroupStart(), "ncclGroupStart", comm_);
        if (send)
            check_nccl(ncclSend(buf, int(count), nccl_dtype(t), peer, comm_, stream_), "ncclSend", comm_);
        else
            check_nccl(ncclRecv(buf, int(count), nccl_dtype(t), peer, comm_, stream_), "ncclRecv", comm_);
        check_nccl(ncclGroupEnd(), "ncclGroupEnd", comm_);
        sync_check();
    }

    ParallelConfig cfg_;         // родительская топология — для маршрутизации
    ncclComm_t comm_ = nullptr;
    ncclComm_t kv_comm_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaStream_t kv_send_stream_ = nullptr, kv_recv_stream_ = nullptr;
    bool own_stream_ = false;
    Bootstrap* bs_ = nullptr;

    std::vector<rank_t> members_;          // глобальные ранки-члены (по возрастанию)
    rank_t my_pos_ = 0;                    // позиция данного ранка в members_
    std::vector<rank_t> pos_of_;           // глобальный ранок -> позиция (или -1)
    std::map<std::vector<rank_t>, std::unique_ptr<NcclCommunicator>> owned_subs_;

    // kv-кольцо: фоновый приёмник + очередь принятых блоков.
    std::thread kv_recv_thread_;
    std::atomic<bool> kv_stop_{false};
    std::mutex kv_mu_;                     // постинг nccl-операций kv
    std::mutex kv_push_mu_;                // сериализация push (общий hdr-буфер)
    std::mutex kv_in_mu_;                  // очередь принятых блоков
    std::deque<KvBlock> kv_in_q_;
    void* kv_hdr_send_dev_ = nullptr;
    void* kv_hdr_recv_dev_ = nullptr;
};

std::unique_ptr<ICommunicator> make_nccl_communicator(const ParallelConfig& cfg) {
    return std::make_unique<NcclCommunicator>(cfg);
}

} // namespace stell::parallel

#endif // STELL_HAVE_NCCL

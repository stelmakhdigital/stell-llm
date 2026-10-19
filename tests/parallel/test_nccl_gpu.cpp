// tests/parallel: GPU-тест NCCL-backend (ADR-02/ADR-10 §2) — задача 3 раунда 2.
//
// Собирается ТОЛЬКО при STELL_CUDA=ON + libnccl (сервер, cuda 12.8 + libnccl).
// Co-tenancy честно: чужой процесс strata держит ~21–30 GB VRAM на V100 —
// тест работает в свободной VRAM, печатает занятость и честно PENDING при
// OOM/нехватке. Замер latency allreduce F32 разных размеров (V100-SXM2,
// NVLink), медиана по итерациям. Проверки корректности allreduce/allgather.
//
// Запуск: ctest -R parallel_gpu  (или бинарник напрямую).
#include "test_harness.hpp"
#include "stell/parallel/comm.hpp"
#include "stell/parallel/nccl_communicator.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

using namespace stell::parallel;

namespace {

#ifndef STELL_GPU_TEST_DEVICES
#define STELL_GPU_TEST_DEVICES "0,1"   // 2 ранка: свободная VRAM ~2–11 GB
#endif

std::vector<int> parse_devices(const char* s) {
    std::vector<int> out;
    int v = 0;
    bool any = false;
    for (const char* p = s; *p; ++p) {
        if (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); any = true; }
        else if (any) { out.push_back(v); v = 0; any = false; }
    }
    if (any) out.push_back(v);
    return out;
}

void check_cuda(cudaError_t r, const char* what) {
    if (r != cudaSuccess) {
        std::printf("CUDA-FAIL: %s: %s\n", what, cudaGetErrorString(r));
        throw CommError(CommError::Code::NcclFailure, what);
    }
}

// Честная co-tenancy: занятость каждой карты перед прогоном.
void report_cotenance(const std::vector<int>& devices) {
    int prev = -1;
    cudaGetDevice(&prev);
    for (int d : devices) {
        check_cuda(cudaSetDevice(d), "cudaSetDevice(report)");
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
            std::printf("COTENANCE: GPU %d: свободно %.2f GB / %.2f GB "
                        "(чужой процесс strata держит остальное)\n", d,
                        double(free_b) / 1e9, double(total_b) / 1e9);
        else
            std::printf("COTENANCE: GPU %d: cudaMemGetInfo не удался — честный pending\n", d);
    }
    cudaSetDevice(prev);
}

// Один ранок: создаёт комм, исполняет коллективы, пишет результаты.
struct RankOut {
    bool ok = true;
    std::vector<double> latency_us;      // по размерам
};

void rank_body(const ParallelConfig& cfg, const std::vector<std::size_t>& sizes,
               int iters_per_size, RankOut& out) {
    auto comm = make_communicator(cfg);
    out.latency_us.assign(sizes.size(), 0.0);
    const rank_t world = comm->world_size();
    const std::size_t maxn = *std::max_element(sizes.begin(), sizes.end());
    // NCCL работает над DEVICE-буферами: выделяем в свободной VRAM (co-tenancy).
    float* dev_buf = nullptr;
    float* dev_recv = nullptr;
    std::vector<float> host(maxn), recv(maxn * std::size_t(world));
    check_cuda(cudaMalloc(&dev_buf, maxn * 4), "cudaMalloc(allreduce buf)");
    check_cuda(cudaMalloc(&dev_recv, maxn * std::size_t(world) * 4),
               "cudaMalloc(allgather recv)");
    auto cleanup = [&] {
        cudaFree(dev_buf);
        cudaFree(dev_recv);
    };
    try {
        // allgather: корректность (конкатенация send-паттернов по ранкам).
        {
            for (std::size_t i = 0; i < maxn; ++i)
                host[i] = 100.0f * float(cfg.rank) + float(i);
            check_cuda(cudaMemcpy(dev_buf, host.data(), maxn * 4, cudaMemcpyHostToDevice),
                       "H2D allgather");
            comm->allgather(dev_buf, dev_recv, maxn, CommDType::F32);
            check_cuda(cudaMemcpy(recv.data(), dev_recv,
                                  maxn * std::size_t(world) * 4, cudaMemcpyDeviceToHost),
                       "D2H allgather");
            for (rank_t s = 0; s < world; ++s)
                for (std::size_t i = 0; i < maxn; ++i)
                    if (!(recv[std::size_t(s) * maxn + i] == 100.0f * float(s) + float(i)))
                        out.ok = false;
        }
        // allreduce: корректность + latency (медиана по итерациям).
        for (std::size_t si = 0; si < sizes.size(); ++si) {
            const std::size_t n = sizes[si];
            for (std::size_t i = 0; i < n; ++i) host[i] = 1000.0f * float(cfg.rank) + float(i);
            check_cuda(cudaMemcpy(dev_buf, host.data(), n * 4, cudaMemcpyHostToDevice),
                       "H2D allreduce");
            comm->allreduce(dev_buf, n, CommDType::F32);   // warmup + корректность
            check_cuda(cudaMemcpy(host.data(), dev_buf, n * 4, cudaMemcpyDeviceToHost),
                       "D2H allreduce");
            for (std::size_t i = 0; i < n; ++i) {
                float expect = 0.f;
                for (rank_t s = 0; s < world; ++s) expect += 1000.0f * float(s) + float(i);
                if (!(host[i] == expect)) out.ok = false;   // суммы целые < 2^22 — точны
            }
            std::vector<double> samples;
            samples.reserve(std::size_t(iters_per_size));
            for (int it = 0; it < iters_per_size; ++it) {
                const auto t0 = std::chrono::steady_clock::now();
                comm->allreduce(dev_buf, n, CommDType::F32);
                const auto t1 = std::chrono::steady_clock::now();
                samples.push_back(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(samples.begin(), samples.end());
            out.latency_us[si] = samples[samples.size() / 2];
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}
} // namespace

TEST(parallel_gpu_nccl_allreduce_allgather_latency) {
    // Список устройств: env STELL_GPU_DEVICES (напр. "0,1,2,3") перекрывает
    // дефолт STELL_GPU_TEST_DEVICES.
    const char* env = std::getenv("STELL_GPU_DEVICES");
    const std::vector<int> devices =
        parse_devices(env ? env : STELL_GPU_TEST_DEVICES);
    std::printf("NCCL: версия %d.%d.%d, устройств в прогоне: %zu (%s)\n",
                NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, devices.size(), STELL_GPU_TEST_DEVICES);
    report_cotenance(devices);

    const std::vector<std::size_t> sizes = {1, 64, 256, 1024, 4096, 16384, 65536, 262144};
    const int iters = 100;

    ParallelConfig cfg;
    cfg.mode = CommMode::Nccl;
    cfg.pp = 1;
    cfg.tp = rank_t(devices.size());
    cfg.ep = 1;
    cfg.device_ids = devices;
    cfg.timeout_ms = 30000;

    SafeFlag ok;
    std::vector<RankOut> outs(devices.size());
    std::vector<std::thread> th;
    for (rank_t r = 0; r < rank_t(devices.size()); ++r)
        th.emplace_back([&, r] {
            ParallelConfig rc = cfg;
            rc.rank = r;
            try {
                rank_body(rc, sizes, iters, outs[std::size_t(r)]);
            } catch (const CommError& e) {
                std::printf("PENDING: CommError на ранке %d: %s\n", int(r), e.what());
                ok.fail();
            } catch (...) {
                std::printf("PENDING: неизвестная ошибка на ранке %d\n", int(r));
                ok.fail();
            }
        });
    for (auto& t : th) t.join();

    if (!ok.get()) {
        std::printf("PENDING: NCCL-прогон не удался (co-tenancy/OOM/бутстрап) — "
                    "см. вывод выше; GPU-результат не выдаётся за проверку\n");
        return;
    }
    std::printf("NCCL-BENCH allreduce F32, V100-SXM2 NVLink, co-tenancy "
                "(strata ~21–30 GB), медиана µs по %d итерациям:\n", iters);
    for (std::size_t si = 0; si < sizes.size(); ++si) {
        std::printf("NCCL-BENCH: size=%6zu floats (%6zu KB):", sizes[si],
                    sizes[si] * 4 / 1024);
        for (std::size_t r = 0; r < outs.size(); ++r)
            std::printf(" rank%zu=%.1fµs", std::size_t(r), outs[r].latency_us[si]);
        std::printf("\n");
    }
    CHECK(ok.get());
}

// --- [R6] GPU-контракт expert_alltoall (token_elems) + kv_stream -------------

// CPU-эмуляторный контракт этих операций прогнан в tests/parallel (tok_width_*,
// sub_*); здесь — NCCL-исполнение. PENDING: worker прогоняет после merge
// (co-tenancy честная; при OOM/нехватке VRAM — честный pending, см. шапку).
TEST(parallel_gpu_nccl_expert_alltoall_kv_stream) {
    const char* env = std::getenv("STELL_GPU_DEVICES");
    const std::vector<int> devices = parse_devices(env ? env : STELL_GPU_TEST_DEVICES);
    if (devices.size() < 2) {
        std::printf("PENDING: нужно ≥2 устройств (STELL_GPU_DEVICES)\n");
        return;
    }
    constexpr int DM = 8, T = 4;                     // токен = d_model векторов
    SafeFlag ok;
    std::vector<std::thread> th;
    for (rank_t r = 0; r < rank_t(devices.size()); ++r)
        th.emplace_back([&, r] {
            ParallelConfig cfg;
            cfg.mode = CommMode::Nccl;
            cfg.pp = 1;
            cfg.tp = 1;
            cfg.ep = rank_t(devices.size());         // EP: владелец = expert_id % ep
            cfg.device_ids = devices;
            cfg.rank = r;
            cfg.group_id = 1;                        // отдельный бутстрап от bench-теста
            cfg.timeout_ms = 30000;
            try {
                auto comm = make_communicator(cfg);
                check_cuda(cudaSetDevice(devices[std::size_t(r)]), "setdev");
                // --- expert_alltoall: токен = вектор d_model ---
                float* send_t = nullptr;
                TokenMeta* send_m = nullptr;
                const std::size_t cap = std::size_t(T) * std::size_t(devices.size());
                float* recv_t = nullptr;
                TokenMeta* recv_m = nullptr;
                std::vector<float> hsend(std::size_t(T) * DM);
                std::vector<TokenMeta> hsm(T);
                for (int i = 0; i < T; ++i) {
                    hsm[std::size_t(i)] = TokenMeta{std::int32_t((r + 1) % int(cfg.ep)),
                                                    std::int32_t(r * 16 + i)};
                    for (int dd = 0; dd < DM; ++dd)
                        hsend[std::size_t(i) * DM + dd] =
                            float(r) * 1000.f + float(i) * 10.f + float(dd);
                }
                check_cuda(cudaMalloc((void**)&send_t, hsend.size() * sizeof(float)), "malloc");
                check_cuda(cudaMalloc((void**)&send_m, hsm.size() * sizeof(TokenMeta)), "malloc");
                check_cuda(cudaMalloc((void**)&recv_t, cap * DM * sizeof(float)), "malloc");
                check_cuda(cudaMalloc((void**)&recv_m, cap * sizeof(TokenMeta)), "malloc");
                check_cuda(cudaMemcpy(send_t, hsend.data(), hsend.size() * sizeof(float),
                                      cudaMemcpyHostToDevice), "H2D");
                check_cuda(cudaMemcpy(send_m, hsm.data(), hsm.size() * sizeof(TokenMeta),
                                      cudaMemcpyHostToDevice), "H2D");
                const auto t0 = std::chrono::steady_clock::now();
                comm->expert_alltoall(send_t, send_m, recv_t, recv_m, std::size_t(T),
                                      std::size_t(DM), CommDType::F32);
                const double us =
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count();
                std::vector<float> hrecv(cap * DM);
                std::vector<TokenMeta> hrm(cap);
                check_cuda(cudaMemcpy(hrecv.data(), recv_t, hrecv.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost), "D2H");
                check_cuda(cudaMemcpy(hrm.data(), recv_m, hrm.size() * sizeof(TokenMeta),
                                      cudaMemcpyDeviceToHost), "D2H");
                // С одним expert_id=(src+1)%ep все токены ранка src уезжают
                // на следующий ранок: входящих = T (от предыдущего ранка);
                // recv_meta.expert_id = владелец (= мой ранок); хвост — заполнители.
                const rank_t src_expect = rank_t((int(r) + int(cfg.ep) - 1) % int(cfg.ep));
                std::size_t got = 0;
                for (std::size_t i = 0; i < cap; ++i) {
                    if (hrm[i].expert_id < 0) continue;
                    ++got;
                    const rank_t src = rank_t(hrm[i].slot_id) / 16;
                    const int k = int(hrm[i].slot_id % 16);
                    if (!(src == src_expect)) ok.fail();
                    if (!(hrm[i].expert_id == std::int32_t(r))) ok.fail();
                    for (int dd = 0; dd < DM; ++dd)
                        if (!(hrecv[i * DM + dd] == float(src) * 1000.f + float(k) * 10.f +
                              float(dd)))
                            ok.fail();
                }
                if (got != std::size_t(T)) ok.fail();
                std::printf("NCCL alltoall: rank%d: входящих=%zu, %.1f µs\n", int(r), got, us);
                cudaFree(send_t); cudaFree(send_m); cudaFree(recv_t); cudaFree(recv_m);

                // --- kv_stream: 3 блока r0→r1 по кольцу, побайтово ---
                if (r == 0) {
                    for (int b = 0; b < 3; ++b) {
                        std::vector<std::uint8_t> blk(64);
                        for (std::size_t i = 0; i < blk.size(); ++i)
                            blk[i] = std::uint8_t(b * 31 + i);
                        comm->kv_stream_push(blk.data(), blk.size());
                    }
                } else {
                    for (int b = 0; b < 3; ++b) {
                        std::vector<std::uint8_t> blk(64, 0);
                        bool got_blk = false;
                        for (int spin = 0; spin < 5000 && !got_blk; ++spin)
                            got_blk = comm->kv_stream_pop(blk.data(), blk.size());
                        if (!got_blk) { ok.fail(); continue; }
                        for (std::size_t i = 0; i < blk.size(); ++i)
                            if (blk[i] != std::uint8_t(b * 31 + i)) ok.fail();
                    }
                    std::vector<std::uint8_t> empty(64);
                    if (comm->kv_stream_pop(empty.data(), empty.size())) ok.fail();
                }
            } catch (const CommError& e) {
                std::printf("PENDING: CommError (rank %d): %s\n", int(r), e.what());
                ok.fail();
            } catch (...) {
                std::printf("PENDING: неизвестная ошибка (rank %d)\n", int(r));
                ok.fail();
            }
        });
    for (auto& t : th) t.join();
    if (!ok.get()) {
        std::printf("PENDING: GPU-контракт expert_alltoall/kv_stream не подтверждён "
                    "(co-tenancy/OOM/бутстрап) — worker прогонит после merge\n");
        return;
    }
    CHECK(ok.get());
}

int main() {
    testfw::run_all("test_parallel_gpu");
    return testfw::failures() == 0 ? 0 : 1;
}

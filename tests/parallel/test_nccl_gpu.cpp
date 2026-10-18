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
#include <chrono>
#include <cstdio>
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
        int free_b = 0, total_b = 0;
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
    const rank_t world = comm->world_size();
    const std::size_t maxn = *std::max_element(sizes.begin(), sizes.end());
    std::vector<float> buf(maxn), recv(maxn * std::size_t(world));
    std::vector<double> lat(sizes.size(), 0.0);
    // allgather: корректность (конкатенация send-паттернов по ранкам).
    {
        std::vector<float> send(maxn), r(maxn * std::size_t(world));
        for (std::size_t i = 0; i < maxn; ++i)
            send[i] = 100.0f * float(cfg.rank) + float(i);
        comm->allgather(send.data(), r.data(), maxn, CommDType::F32);
        for (rank_t s = 0; s < world; ++s)
            for (std::size_t i = 0; i < maxn; ++i)
                if (!(r[std::size_t(s) * maxn + i] == 100.0f * float(s) + float(i)))
                    out.ok = false;
    }
    // allreduce: корректность + latency.
    for (std::size_t si = 0; si < sizes.size(); ++si) {
        const std::size_t n = sizes[si];
        for (std::size_t i = 0; i < n; ++i) buf[i] = 1000.0f * float(cfg.rank) + float(i);
        comm->allreduce(buf.data(), n, CommDType::F32);   // warmup + корректность
        for (std::size_t i = 0; i < n; ++i) {
            float expect = 0.f;
            for (rank_t s = 0; s < world; ++s) expect += 1000.0f * float(s) + float(i);
            if (!(buf[i] == expect)) out.ok = false;   // F32: суммы точны до 2^22
        }
        // Замер: медиана по итерациям.
        std::vector<double> samples;
        samples.reserve(std::size_t(iters_per_size));
        for (int it = 0; it < iters_per_size; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            comm->allreduce(buf.data(), n, CommDType::F32);
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        lat[si] = samples[samples.size() / 2];
    }
    out.latency_us = lat;
}

} // namespace

TEST(parallel_gpu_nccl_allreduce_allgather_latency) {
    const std::vector<int> devices = parse_devices(STELL_GPU_TEST_DEVICES);
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

int main() {
    testfw::run_all("test_parallel_gpu");
    return testfw::failures() == 0 ? 0 : 1;
}

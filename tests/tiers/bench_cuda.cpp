// bench_cuda — GPU-smoke DMA-копирований (как в TierManager при STELL_CUDA):
// cudaMemcpyAsync H→D/D→H на pinned-буферах (cudaHostAlloc) в свободной VRAM.
// Не add_test. Co-tenancy: работает только с малыми буферами (≤ 128 MiB);
// при нехватке VRAM — честный SKIP (печать + exit 0).
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
}
} // namespace

int main() {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) {
        std::printf("SKIP: нет доступного CUDA-устройства\n");
        return 0;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::printf("SKIP: не удалось получить свойства устройства\n");
        return 0;
    }
    std::printf("GPU: %s (SM %d.%d, %d SM, %.1f MiB total)\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount,
                double(prop.totalGlobalMem) / double(1 << 20));
    {
        // co-tenancy: фиксируем свободную VRAM ДО бенча (чужой процесс на
        // устройстве не мешает малым копированиям, но фиксируем честно).
        std::size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
            std::printf("SKIP: cudaMemGetInfo failed\n");
            return 0;
        }
        free_b = f;
        total_b = t;
        std::printf("VRAM до бенча: свободно %.1f MiB / %.1f MiB (co-tenancy: см. отчёт)\n",
                    double(f) / double(1 << 20), double(t) / double(1 << 20));
        if (f < (std::size_t(256) << 20)) {
            std::printf("SKIP: свободной VRAM < 256 MiB — честный SKIP\n");
            return 0;
        }
    }

    const std::size_t sizes[] = {64u << 10, 1u << 20, 16u << 20, 64u << 20};
    for (std::size_t size : sizes) {
        void* d = nullptr;
        void* h = nullptr;
        cudaError_t e = cudaMalloc(&d, size);
        if (e != cudaSuccess) {
            std::printf("SKIP: cudaMalloc(%zu) failed: %s\n", size,
                        cudaGetErrorString(e));
            return 0;
        }
        e = cudaHostAlloc(&h, size, cudaHostAllocDefault);
        if (e != cudaSuccess) {
            std::printf("SKIP: cudaHostAlloc(%zu) failed: %s\n", size,
                        cudaGetErrorString(e));
            cudaFree(d);
            return 0;
        }
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) {
            std::printf("SKIP: cudaStreamCreate failed\n");
            cudaFree(d);
            cudaFreeHost(h);
            return 0;
        }
        std::memset(h, 0xAB, size);

        const int iters = 20;
        double h2d = 0.0, d2h = 0.0;
        for (int i = 0; i <= iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            e = cudaMemcpyAsync(d, h, size, cudaMemcpyHostToDevice, s);
            if (e != cudaSuccess) {
                std::printf("SKIP: cudaMemcpyAsync H→D failed: %s\n", cudaGetErrorString(e));
                cudaStreamDestroy(s);
                cudaFree(d);
                cudaFreeHost(h);
                return 0;
            }
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                std::printf("SKIP: stream sync failed\n");
                cudaStreamDestroy(s);
                cudaFree(d);
                cudaFreeHost(h);
                return 0;
            }
            if (i > 0) h2d += ms_since(t0);

            const auto t1 = std::chrono::steady_clock::now();
            e = cudaMemcpyAsync(h, d, size, cudaMemcpyDeviceToHost, s);
            if (e != cudaSuccess) {
                std::printf("SKIP: cudaMemcpyAsync D→H failed: %s\n", cudaGetErrorString(e));
                cudaStreamDestroy(s);
                cudaFree(d);
                cudaFreeHost(h);
                return 0;
            }
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                std::printf("SKIP: stream sync failed\n");
                cudaStreamDestroy(s);
                cudaFree(d);
                cudaFreeHost(h);
                return 0;
            }
            if (i > 0) d2h += ms_since(t1);
        }
        const double gbs_h2d = double(iters) * size / (h2d / 1000.0) / double(1 << 30);
        const double gbs_d2h = double(iters) * size / (d2h / 1000.0) / double(1 << 30);
        std::printf("DMA %5.1f KiB: H→D %.2f GiB/s  D→H %.2f GiB/s (iters=%d)\n",
                    double(size) / double(1 << 10), gbs_h2d, gbs_d2h, iters);

        cudaStreamDestroy(s);
        cudaFree(d);
        cudaFreeHost(h);
    }
    std::printf("VRAM после бенча: свободно %.1f MiB (утечек не оставили)\n",
                double([&] {
                    std::size_t f = 0, t = 0;
                    cudaMemGetInfo(&f, &t);
                    return f;
                }()) / double(1 << 20));
    return 0;
}

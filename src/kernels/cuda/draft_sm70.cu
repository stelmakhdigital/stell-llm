// Драфтер-ядра (SM70): argmax по строкам + softmax-вероятности top-k
// (отбор k — на host в round 1; перенос в ядро — по профилированию).
// НЕ компилировался локально — сервер.
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "dispatch.hpp"

namespace stell::kernels::cuda {

namespace {

// block per row: max через shared reduction → argmax
__global__ void argmax_kernel(const float* __restrict__ logits, int n,
                              std::int32_t* __restrict__ out_ids) {
    extern __shared__ float sh[];      // [blockDim.x]
    const int row = blockIdx.x;
    const int j = threadIdx.x;
    const float* l = logits + (std::size_t)row * n;

    float v = (j < n) ? l[j] : -FLT_MAX;
    sh[j] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (j < off) {
            if (sh[j + off] > sh[j])   // тай: строгое сравнение сохраняет первый
                sh[j] = sh[j + off];
        }
        __syncthreads();
    }
    // после reduction argmax восстанавливаем линейным поиском по найденному
    // max (round 1; оптимизация — индексы в shared)
    if (j == 0) {
        const float mx = sh[0];
        int best = 0;
        for (int i = 0; i < n; ++i)
            if (l[i] == mx) { best = i; break; }
        out_ids[row] = best;
    }
}

// block per row: softmax-вероятности (потоки по vocab)
__global__ void probs_kernel(const float* __restrict__ logits, int n,
                             float* __restrict__ probs) {
    extern __shared__ float sh[];
    const int row = blockIdx.x;
    const int j = threadIdx.x;
    const float* l = logits + (std::size_t)row * n;

    float mx = (j < n) ? l[j] : -FLT_MAX;
    sh[j] = mx;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (j < off) sh[j] = fmaxf(sh[j], sh[j + off]);
        __syncthreads();
    }
    const float mxr = sh[0];
    __syncthreads();

    float e = (j < n) ? __expf(l[j] - mxr) : 0.0f;
    sh[j] = e;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (j < off) sh[j] += sh[j + off];
        __syncthreads();
    }
    const float sum = sh[0];
    if (j < n) probs[(std::size_t)row * n + j] = e / sum;
}

} // namespace

void argmax_rows(const float* logits, std::size_t rows, std::uint32_t n,
                 std::int32_t* out_ids) {
    int threads = 32;
    while (threads < (int)n && threads < 1024) threads <<= 1;
    argmax_kernel<<<(int)rows, threads, threads * sizeof(float)>>>(
        logits, (int)n, out_ids);
}

void topk_probs(const float* logits, std::size_t rows, std::uint32_t n,
                std::uint32_t k, float* out_probs) {
    (void)k;   // вероятности для всего vocab; отбор k — на host
    int threads = 32;
    while (threads < (int)n && threads < 1024) threads <<= 1;
    probs_kernel<<<(int)rows, threads, threads * sizeof(float)>>>(
        logits, (int)n, out_probs);
}

} // namespace stell::kernels::cuda

// SM70-attention FP32 QK/PV (PLAN.md §4 поток A). Один блок на (head, query-row):
// threadIdx.x — позиция ключа/значения; d_head <= 1024 (V100 warp-луп).
// ВАЖНО: файл не компилировался локально (нет nvcc) — верификация на сервере,
// сравнение с CPU-референсом через replay-фикстуры (tests/core/kernels).
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "dispatch.hpp"

namespace stell::kernels::cuda {

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

namespace {

__global__ void qk_scores_kernel(const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 int n_heads, int T, int d_head,
                                 float* __restrict__ scores) {
    const int h = blockIdx.x;
    const int i = blockIdx.y;
    const int j = threadIdx.x;
    if (j >= T) return;

    const float* qh = q + (std::size_t)h * T * d_head + (std::size_t)i * d_head;
    const float* kh = k + (std::size_t)h * T * d_head;
    float acc = 0.0f;
    for (int d = 0; d < d_head; ++d)
        acc += qh[d] * kh[(std::size_t)j * d_head + d];
    scores[((std::size_t)h * T + i) * T + j] =
        acc / sqrtf((float)d_head);
}

// softmax по строке (block per row): max и сумма через shared reduction
__global__ void softmax_kernel(float* __restrict__ scores, int T) {
    extern __shared__ float sh[];   // [blockDim.x]
    const int row = blockIdx.x;
    const int j = threadIdx.x;

    // max
    float s = (j < T) ? scores[(std::size_t)row * T + j] : -FLT_MAX;
    sh[j] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (j < off) sh[j] = fmaxf(sh[j], sh[j + off]);
        __syncthreads();
    }
    const float mx = sh[0];
    __syncthreads();

    // exp + сумма
    float e = (j < T) ? __expf(scores[(std::size_t)row * T + j] - mx) : 0.0f;
    sh[j] = e;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (j < off) sh[j] += sh[j + off];
        __syncthreads();
    }
    const float sum = sh[0];

    if (j < T)
        scores[(std::size_t)row * T + j] = e / sum;
}

__global__ void pv_kernel(const float* __restrict__ probs,
                          const float* __restrict__ v,
                          int n_heads, int T, int d_head,
                          float* __restrict__ out) {
    const int h = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= d_head) return;

    const float* ph = probs + (std::size_t)h * T * T + (std::size_t)i * T;
    const float* vh = v + (std::size_t)h * T * d_head;
    float* oh = out + (std::size_t)h * T * d_head + (std::size_t)i * d_head;
    float acc = 0.0f;
    for (int j = 0; j < T; ++j)
        acc += ph[j] * vh[(std::size_t)j * d_head + d];
    oh[d] = acc;
}

} // namespace

void qk_scores(const float* q, const float* k, std::uint32_t n_heads,
               std::uint32_t T, std::uint32_t d_head, float* scores) {
    dim3 grid(n_heads, T);
    qk_scores_kernel<<<grid, (int)T>>>((const float*)q, (const float*)k,
                                       (int)n_heads, (int)T, (int)d_head, scores);
}

void softmax_rows(float* scores, std::size_t rows, std::uint32_t cols) {
    // blockDim — степень двойки >= cols (shared reduction), минимум warp (32)
    int bs = 32;
    while (bs < (int)cols) bs <<= 1;
    softmax_kernel<<<(int)rows, bs, bs * sizeof(float)>>>(scores, (int)cols);
}

void pv_output(const float* probs, const float* v, std::uint32_t n_heads,
               std::uint32_t T, std::uint32_t d_head, float* out) {
    dim3 grid(n_heads, T);
    pv_kernel<<<grid, (int)d_head>>>(probs, v, (int)n_heads, (int)T,
                                     (int)d_head, out);
}

} // namespace stell::kernels::cuda

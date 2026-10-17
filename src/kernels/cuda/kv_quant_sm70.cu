// KV-квант E4M3 (эмуляция FP8, ADR-04; per-block scale) — SM70.
// НЕ компилировался локально — сервер (см. attention_sm70.cu).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "dispatch.hpp"

namespace stell::kernels::cuda {

namespace {

__device__ float e4m3_decode(std::uint8_t code) {
    const bool neg = code & 0x80;
    const int e = (code >> 3) & 0xF;
    const int m = code & 0x7;
    float v;
    if (e == 15 && m == 7)
        v = nanf("");
    else if (e == 0)
        v = (float)m * 0x1p-9f;
    else
        v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return neg ? -v : v;
}

// RNE (round-to-nearest-even) кодирование float → E4M3; переполнение — кламп
// на максимум нормы (0x7E) — софт-квант политика ADR-04.
__device__ std::uint8_t e4m3_encode_dev(float v) {
    if (isnan(v)) return 0x7F;
    const std::uint8_t sign = (v < 0.0f) ? 0x80 : 0x00;
    const float a = fabsf(v);
    if (a == 0.0f) return 0x00;

    int ee;                                   // a = m * 2^ee, m ∈ [0.5, 1)
    (void)frexp(a, &ee);
    const int exp = ee + 6;                   // поле экспоненты (bias 7), см. dequant.cpp

    // RNE целого значения x >= 0
    auto rne = [](float x) -> int {
        int i = (int)x;
        const float frac = x - (float)i;
        if (frac > 0.5f || (frac == 0.5f && (i & 1))) ++i;
        return i;
    };

    if (exp < 1) {
        const int man = rne(ldexpf(a, 9));    // subnormal: man = RNE(a * 2^9)
        if (man >= 8) return (std::uint8_t)(sign | 0x08);
        return (std::uint8_t)(sign | man);
    }

    int man = rne((ldexpf(a, 1 - ee) - 1.0f) * 8.0f);
    int expc = exp;
    if (man == 8) { man = 0; ++expc; }
    if (expc > 15) return (std::uint8_t)(sign | 0x7E);   // кламп: 448
    return (std::uint8_t)(sign | (expc << 3) | man);
}

// блок 32 значений: amax через shared reduction → scale → коды
__global__ void quantize_kernel(const float* __restrict__ kv, int n,
                                int block_size, std::uint8_t* __restrict__ codes,
                                float* __restrict__ scales) {
    extern __shared__ float sh[];   // [blockDim.x]
    const int g = blockIdx.x;       // индекс блока KV
    const int j = threadIdx.x;
    const int begin = g * block_size;
    const int end = min(n, begin + block_size);
    const int span = end - begin;
    const int jj = j;

    float local_amax = 0.0f;
    // поток j обрабатывает элементы jj, jj+blockDim, ... (span может быть < bs)
    for (int i = jj; i < span; i += blockDim.x)
        local_amax = fmaxf(local_amax, fabsf(kv[begin + i]));
    sh[jj] = local_amax;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (jj < off) sh[jj] = fmaxf(sh[jj], sh[jj + off]);
        __syncthreads();
    }
    const float amax = sh[0];
    if (jj == 0) scales[g] = amax / 448.0f;
    __syncthreads();

    const float scale = scales[g];
    const float inv = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    for (int i = jj; i < span; i += blockDim.x)
        codes[begin + i] = (scale == 0.0f) ? (std::uint8_t)0x00
                                           : e4m3_encode_dev(kv[begin + i] * inv);
}

__global__ void dequantize_kernel(const std::uint8_t* __restrict__ codes,
                                  const float* __restrict__ scales, int n,
                                  int block_size, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = e4m3_decode(codes[i]) * scales[i / block_size];
}

} // namespace

void quantize_kv_e4m3(const float* kv, std::size_t n, std::uint32_t block_size,
                      std::uint8_t* out_codes, float* out_scales) {
    const int bs = (int)(block_size ? block_size : 1u);
    const int n_groups = (int)((n + bs - 1) / bs);
    int threads = 32;
    while (threads < bs && threads < 1024) threads <<= 1;
    quantize_kernel<<<n_groups, threads, threads * sizeof(float)>>>(
        kv, (int)n, bs, out_codes, out_scales);
}

void dequantize_kv_e4m3(const std::uint8_t* codes, const float* scales,
                        std::size_t n, std::uint32_t block_size, float* out) {
    const int bs = (int)(block_size ? block_size : 1u);
    const int blocks = (int)((n + 255) / 256);
    dequantize_kernel<<<blocks, 256>>>(codes, scales, (int)n, bs, out);
}

} // namespace stell::kernels::cuda

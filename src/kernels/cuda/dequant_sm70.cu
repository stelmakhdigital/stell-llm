// Dequant NVFP4 / W4A16 (SM70) — device-таблицы E2M1/E4M3 дублируют
// CPU-референс (dequant.hpp). НЕ компилировался локально — сервер.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "dispatch.hpp"

namespace stell::kernels::cuda {

namespace {

__device__ float e2m1_value(std::uint8_t nibble) {
    static const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool neg = nibble >= 8;
    const std::uint8_t mag = neg ? (std::uint8_t)(nibble - 8) : nibble;
    const float v = table[mag & 0x7];
    return neg ? -v : v;
}

__device__ float e4m3_value(std::uint8_t code) {
    const bool neg = code & 0x80;
    const int e = (code >> 3) & 0xF;
    const int m = code & 0x7;
    float v;
    if (e == 15 && m == 7)
        v = nanf("");
    else if (e == 0)
        v = (float)m * 0x1p-9f;                       // subnormal
    else
        v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return neg ? -v : v;
}

__device__ float half_value(std::uint16_t h) {
    const bool neg = h & 0x8000;
    const int e16 = (h >> 10) & 0x1F;
    const int m = h & 0x3FF;
    float v;
    if (e16 == 0x1F)
        v = m ? nanf("") : INFINITY;
    else if (e16 == 0)
        v = m ? (float)m * 0x1p-24f : 0.0f;           // denormal half
    else
        v = ldexpf((float)m + 1024.0f, e16 - 15 - 10);
    return neg ? -v : v;
}

__global__ void dequant_nvfp4_kernel(const std::uint8_t* __restrict__ packed,
                                     const std::uint8_t* __restrict__ scales,
                                     float global_scale, int n,
                                     float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint8_t byte = packed[i / 2];
    const std::uint8_t nib = (i % 2 == 0) ? (std::uint8_t)(byte & 0x0F)
                                          : (std::uint8_t)(byte >> 4);
    out[i] = e2m1_value(nib) * e4m3_value(scales[i / 16]) * global_scale;
}

__global__ void dequant_w4a16_kernel(const std::uint8_t* __restrict__ packed,
                                     const std::uint16_t* __restrict__ scales_f16,
                                     int n, int group_size,
                                     float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::uint8_t byte = packed[i / 2];
    const int nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
    const int s4 = (nib & 0x8) ? (nib - 16) : nib;    // two's complement
    out[i] = half_value(scales_f16[i / group_size]) * (float)s4;
}

} // namespace

void dequant_nvfp4(const std::uint8_t* packed, const std::uint8_t* scales,
                   float global_scale, std::size_t n, float* out) {
    const int blocks = (int)((n + 255) / 256);
    dequant_nvfp4_kernel<<<blocks, 256>>>(packed, scales, global_scale, (int)n, out);
}

void dequant_w4a16(const std::uint8_t* packed, const std::uint16_t* scales_f16,
                   std::size_t n, std::uint32_t group_size, float* out) {
    const int blocks = (int)((n + 255) / 256);
    dequant_w4a16_kernel<<<blocks, 256>>>(packed, scales_f16, (int)n,
                                          (int)group_size, out);
}

} // namespace stell::kernels::cuda

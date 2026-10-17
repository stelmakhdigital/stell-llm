// MoE gather/scatter (SM70). Токен-мажор раскладка как у CPU-референса.
// НЕ компилировался локально — верификация на сервере (см. attention_sm70.cu).
#include <cuda_runtime.h>

#include <cstdint>

#include "dispatch.hpp"

namespace stell::kernels::cuda {

namespace {

__global__ void moe_gather_kernel(const float* __restrict__ x,
                                  const std::int32_t* __restrict__ expert_ids,
                                  int n_tokens, int k_top, int d_model,
                                  float* __restrict__ gathered) {
    // один поток на (строка t*k_top+j, элемент d); expert_ids не влияет на
    // содержимое копии (идентично CPU-референсу), но читается для трассировки
    const std::size_t row = (std::size_t)blockIdx.x;
    const int d = threadIdx.x;
    if (d >= d_model) return;
    (void)expert_ids;   // id эксперта фиксирует, в какой слот уходит строка (EP)
    const std::size_t t = row / k_top;
    gathered[row * d_model + d] = x[t * d_model + d];
}

__global__ void moe_scatter_kernel(const float* __restrict__ expert_out,
                                   const std::int32_t* __restrict__ expert_ids,
                                   const float* __restrict__ weights,
                                   int n_tokens, int k_top, int d_model,
                                   float* __restrict__ y) {
    // один поток на (токен t, элемент d): аккумулируем по всем k_top
    const std::size_t t = (std::size_t)blockIdx.x;
    const int d = threadIdx.x;
    if (d >= d_model) return;
    float acc = 0.0f;
    for (int j = 0; j < k_top; ++j) {
        const std::size_t row = t * k_top + j;
        acc += weights[row] * expert_out[row * d_model + d];
    }
    y[t * d_model + d] += acc;
    (void)expert_ids;
}

} // namespace

void moe_gather(const float* x, const std::int32_t* expert_ids,
                std::uint32_t n_tokens, std::uint32_t k_top, std::uint32_t d_model,
                float* gathered) {
    const int rows = (int)(n_tokens * k_top);
    moe_gather_kernel<<<rows, (int)d_model>>>(x, expert_ids, (int)n_tokens,
                                              (int)k_top, (int)d_model, gathered);
}

void moe_scatter_add(const float* expert_out, const std::int32_t* expert_ids,
                     const float* weights, std::uint32_t n_tokens,
                     std::uint32_t k_top, std::uint32_t d_model, float* y) {
    moe_scatter_kernel<<<(int)n_tokens, (int)d_model>>>(expert_out, expert_ids,
                                                        weights, (int)n_tokens,
                                                        (int)k_top, (int)d_model, y);
}

} // namespace stell::kernels::cuda

// MoE gather/scatter — CPU-референс (CUDA-версия — sm70_moe_gs, ADR-09 §1 ops).
#pragma once

#include <cstdint>

namespace stell::kernels {

// Роутер уже вычислен: expert_ids[t][k], weights[t][k] (softmax-topk, ADR-09 §1).

// Gather: копия активаций токена в буферы слотов экспертов.
//   x            — [n_tokens][d_model]
//   expert_ids   — [n_tokens][k_top]
//   gathered     — [n_tokens * k_top][d_model]; строка t*k_top+j
void moe_gather(const float* x, const std::int32_t* expert_ids,
                std::uint32_t n_tokens, std::uint32_t k_top, std::uint32_t d_model,
                float* gathered);

// Scatter-add: y[t] += w[t][j] * expert_out[t*k_top + j]  (аккумуляция).
//   y            — [n_tokens][d_model] (вход: нулями или partial sums)
void moe_scatter_add(const float* expert_out, const std::int32_t* expert_ids,
                     const float* weights,
                     std::uint32_t n_tokens, std::uint32_t k_top, std::uint32_t d_model,
                     float* y);

} // namespace stell::kernels

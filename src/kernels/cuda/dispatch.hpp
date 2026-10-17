// CUDA-исходники пишутся и проверяются ТОЛЬКО на целевом сервере
// (ssh -p 2214 192.168.1.114, SM70, cuda 12.8): локальная машина без nvcc.
// Статус в отчёте потока core: «CUDA-сборка не выполнена локально».
#pragma once

#include <cstddef>
#include <cstdint>

namespace stell::kernels::cuda {

// Есть ли CUDA-контекст (runtime check; на CPU-пути всегда false).
bool cuda_available();

// --- attention FP32 QK/PV (SM70) --------------------------------------------
// Раскладка как у CPU-референса: q/k/v — [heads][T][d], scores — [heads][T][T].
void qk_scores(const float* q, const float* k, std::uint32_t n_heads,
               std::uint32_t T, std::uint32_t d_head, float* scores);
void softmax_rows(float* scores, std::size_t rows, std::uint32_t cols);
void pv_output(const float* probs, const float* v, std::uint32_t n_heads,
               std::uint32_t T, std::uint32_t d_head, float* out);

// --- MoE gather/scatter ------------------------------------------------------
void moe_gather(const float* x, const std::int32_t* expert_ids,
                std::uint32_t n_tokens, std::uint32_t k_top, std::uint32_t d_model,
                float* gathered);
void moe_scatter_add(const float* expert_out, const std::int32_t* expert_ids,
                     const float* weights, std::uint32_t n_tokens,
                     std::uint32_t k_top, std::uint32_t d_model, float* y);

// --- dequant NVFP4 / W4A16 ---------------------------------------------------
void dequant_nvfp4(const std::uint8_t* packed, const std::uint8_t* scales,
                   float global_scale, std::size_t n, float* out);
void dequant_w4a16(const std::uint8_t* packed, const std::uint16_t* scales_f16,
                   std::size_t n, std::uint32_t group_size, float* out);

// --- KV-квант E4M3 (эмуляция FP8, ADR-04) ------------------------------------
void quantize_kv_e4m3(const float* kv, std::size_t n, std::uint32_t block_size,
                      std::uint8_t* out_codes, float* out_scales);
void dequantize_kv_e4m3(const std::uint8_t* codes, const float* scales,
                        std::size_t n, std::uint32_t block_size, float* out);

// --- драфтер -----------------------------------------------------------------
// argmax по строкам [rows][n]; top-k вероятности (softmax) — ядром, отбор k —
// на host (round 1; при профилировании перенести в ядро).
void argmax_rows(const float* logits, std::size_t rows, std::uint32_t n,
                 std::int32_t* out_ids);
void topk_probs(const float* logits, std::size_t rows, std::uint32_t n,
                std::uint32_t k, float* out_probs);

} // namespace stell::kernels::cuda

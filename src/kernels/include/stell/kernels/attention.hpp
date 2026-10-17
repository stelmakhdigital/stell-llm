// SM70-attention FP32 QK/PV — CPU-референс-реализация (replay-тесты сравнивают
// с CUDA-версией; сама по себе — операторная фиксура).
#pragma once

#include <cstddef>
#include <cstdint>

namespace stell::kernels {

// Раскладка (layout): q, k, v — [heads][T][d_head], scores — [heads][T][T].
// Всё FP32 (SM70-attention FP32 QK/PV, PLAN.md §4 поток A).

// S[h][i][j] = sum_d Q[h][i][d] * K[h][j][d] / sqrt(d_head)
void qk_scores_f32(const float* q, const float* k,
                   std::uint32_t n_heads, std::uint32_t T, std::uint32_t d_head,
                   float* scores);

// softmax по последней размерности (строкам длиной cols): rows * cols подряд.
void softmax_rows_f32(float* scores, std::size_t rows, std::uint32_t cols);

// O[h][i][d] = sum_j P[h][i][j] * V[h][j][d]
void pv_output_f32(const float* probs, const float* v,
                   std::uint32_t n_heads, std::uint32_t T, std::uint32_t d_head,
                   float* out);

// Композиция одного слоя attention (QK -> softmax -> PV).
void attention_forward_f32(const float* q, const float* k, const float* v,
                           std::uint32_t n_heads, std::uint32_t T, std::uint32_t d_head,
                           float* out);

// Decode-вариант (round 2, интеграция engine): ОДНА query-строка (текущий
// токен decode-шага) против кеша K/V длины T. Это операторная форма, в которой
// decode-шаг обращается к полной истории (ADR-04 KV-кеш) без пере-вычисления
// Q прошлых позиций:
//   q        — [n_heads][d_head]           (текущий токен, после проекции Wq);
//   k, v     — [n_kv_heads][T][d_head]     (кеш; GQA — n_heads % n_kv_heads == 0);
//   out      — [n_heads][d_head]           (строка текущего токена).
// Соответствие голов (GQA): голова h использует kv-голову h / (n_heads/n_kv_heads).
// Раскладка и масштаб идентичны attention_forward_f32 (FP32 QK/PV, scale 1/sqrt(d)).
void attention_decode_f32(const float* q, const float* k, const float* v,
                          std::uint32_t n_heads, std::uint32_t n_kv_heads,
                          std::uint32_t T, std::uint32_t d_head,
                          float* out);

} // namespace stell::kernels

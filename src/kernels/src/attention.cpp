#include "stell/kernels/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace stell::kernels {

void qk_scores_f32(const float* q, const float* k,
                   std::uint32_t n_heads, std::uint32_t T, std::uint32_t d_head,
                   float* scores) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    const std::size_t row = T;
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        const float* qh = q + static_cast<std::size_t>(h) * T * d_head;
        const float* kh = k + static_cast<std::size_t>(h) * T * d_head;
        float* sh = scores + static_cast<std::size_t>(h) * row * row;
        for (std::uint32_t i = 0; i < T; ++i) {
            for (std::uint32_t j = 0; j < T; ++j) {
                float acc = 0.0f;
                for (std::uint32_t d = 0; d < d_head; ++d)
                    acc += qh[i * d_head + d] * kh[j * d_head + d];
                sh[i * T + j] = acc * scale;
            }
        }
    }
}

void softmax_rows_f32(float* scores, std::size_t rows, std::uint32_t cols) {
    for (std::size_t r = 0; r < rows; ++r) {
        float* row_ptr = scores + r * cols;
        float mx = row_ptr[0];
        for (std::uint32_t j = 1; j < cols; ++j)
            mx = std::max(mx, row_ptr[j]);
        float sum = 0.0f;
        for (std::uint32_t j = 0; j < cols; ++j) {
            row_ptr[j] = std::exp(row_ptr[j] - mx);
            sum += row_ptr[j];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (std::uint32_t j = 0; j < cols; ++j) row_ptr[j] *= inv;
    }
}

void pv_output_f32(const float* probs, const float* v,
                   std::uint32_t n_heads, std::uint32_t T, std::uint32_t d_head,
                   float* out) {
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        const float* ph = probs + static_cast<std::size_t>(h) * T * T;
        const float* vh = v + static_cast<std::size_t>(h) * T * d_head;
        float* oh = out + static_cast<std::size_t>(h) * T * d_head;
        for (std::uint32_t i = 0; i < T; ++i) {
            for (std::uint32_t d = 0; d < d_head; ++d)
                oh[i * d_head + d] = 0.0f;
            for (std::uint32_t j = 0; j < T; ++j) {
                const float p = ph[i * T + j];
                for (std::uint32_t d = 0; d < d_head; ++d)
                    oh[i * d_head + d] += p * vh[j * d_head + d];
            }
        }
    }
}

void attention_forward_f32(const float* q, const float* k, const float* v,
                           std::uint32_t n_heads, std::uint32_t T, std::uint32_t d_head,
                           float* out) {
    std::vector<float> scores(static_cast<std::size_t>(n_heads) * T * T, 0.0f);
    qk_scores_f32(q, k, n_heads, T, d_head, scores.data());
    softmax_rows_f32(scores.data(), static_cast<std::size_t>(n_heads) * T, T);
    pv_output_f32(scores.data(), v, n_heads, T, d_head, out);
}

void attention_decode_f32(const float* q, const float* k, const float* v,
                          std::uint32_t n_heads, std::uint32_t n_kv_heads,
                          std::uint32_t T, std::uint32_t d_head,
                          float* out) {
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return;   // нарушение контракта
    const std::uint32_t per_kv = n_heads / n_kv_heads;          // голов attn на kv-голову
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        const std::uint32_t kvh = h / per_kv;
        const float* qh = q + static_cast<std::size_t>(h) * d_head;
        const float* kh = k + static_cast<std::size_t>(kvh) * T * d_head;
        const float* vh = v + static_cast<std::size_t>(kvh) * T * d_head;
        float* oh = out + static_cast<std::size_t>(h) * d_head;
        // scores по кешу: S[j] = q_h · K[j] * scale (FP32 QK)
        std::vector<float> p(T, 0.0f);
        float mx = -std::numeric_limits<float>::infinity();
        for (std::uint32_t j = 0; j < T; ++j) {
            float acc = 0.0f;
            for (std::uint32_t d = 0; d < d_head; ++d)
                acc += qh[d] * kh[static_cast<std::size_t>(j) * d_head + d];
            p[j] = acc * scale;
            mx = std::max(mx, p[j]);
        }
        // softmax (max-subtraction, идентично softmax_rows_f32)
        float sum = 0.0f;
        for (std::uint32_t j = 0; j < T; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        // PV: out = Σ_j p_j * V[j]
        for (std::uint32_t d = 0; d < d_head; ++d) oh[d] = 0.0f;
        for (std::uint32_t j = 0; j < T; ++j) {
            const float w = p[j] * inv;
            for (std::uint32_t d = 0; d < d_head; ++d)
                oh[d] += w * vh[static_cast<std::size_t>(j) * d_head + d];
        }
    }
}

} // namespace stell::kernels

#include "stell/kernels/draft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace stell::kernels {

std::int32_t argmax_f32(const float* logits, std::size_t n) {
    std::size_t best = 0;
    float bv = logits[0];
    for (std::size_t i = 1; i < n; ++i)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return static_cast<std::int32_t>(best);
}

void draft_topk_f32(const float* logits, std::size_t n, std::uint32_t k,
                    std::int32_t* out_ids, float* out_probs) {
    const std::uint32_t kk = std::min(k, static_cast<std::uint32_t>(n));
    // softmax-нормализация по всему vocab (численно стабильная)
    float mx = logits[0];
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
    std::vector<float> exps(n);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        exps[i] = std::exp(logits[i] - mx);
        sum += exps[i];
    }

    // top-k по вероятности: простая сортировка индексов (CPU-референс;
    // CUDA-версия — драфтер-ядро с параллельным отбором).
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(n),
              [&](std::size_t a, std::size_t b) {
                  if (exps[a] != exps[b]) return exps[a] > exps[b];
                  return a < b;                    // детерминизм при равенстве
              });

    for (std::uint32_t j = 0; j < kk; ++j) {
        out_ids[j] = static_cast<std::int32_t>(idx[j]);
        out_probs[j] = exps[idx[j]] / sum;
    }
}

} // namespace stell::kernels

#include "stell/kernels/moe_gather_scatter.hpp"

#include <cstddef>

namespace stell::kernels {

void moe_gather(const float* x, const std::int32_t* expert_ids,
                std::uint32_t n_tokens, std::uint32_t k_top, std::uint32_t d_model,
                float* gathered) {
    (void)expert_ids;  // раскладка токен-мажор: gather не зависит от id эксперта
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        for (std::uint32_t j = 0; j < k_top; ++j) {
            const std::size_t row = static_cast<std::size_t>(t) * k_top + j;
            const float* src = x + static_cast<std::size_t>(t) * d_model;
            float* dst = gathered + row * d_model;
            for (std::uint32_t d = 0; d < d_model; ++d)
                dst[d] = src[d];
        }
    }
}

void moe_scatter_add(const float* expert_out, const std::int32_t* expert_ids,
                     const float* weights,
                     std::uint32_t n_tokens, std::uint32_t k_top, std::uint32_t d_model,
                     float* y) {
    (void)expert_ids;  // scatter-add не зависит от id (веса уже соответствуют gather)
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        float* yt = y + static_cast<std::size_t>(t) * d_model;
        for (std::uint32_t j = 0; j < k_top; ++j) {
            const float w = weights[static_cast<std::size_t>(t) * k_top + j];
            const float* out = expert_out + (static_cast<std::size_t>(t) * k_top + j) * d_model;
            for (std::uint32_t d = 0; d < d_model; ++d)
                yt[d] += w * out[d];
        }
    }
}

} // namespace stell::kernels

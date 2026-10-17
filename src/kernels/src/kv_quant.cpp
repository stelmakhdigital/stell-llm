#include "stell/kernels/kv_quant.hpp"

#include "stell/kernels/dequant.hpp"

#include <algorithm>
#include <cmath>

namespace stell::kernels {

void quantize_kv_e4m3(const float* kv, std::size_t n, std::uint32_t block_size,
                      std::uint8_t* out_codes, float* out_scales) {
    const std::size_t bs = block_size ? block_size : 1u;
    const std::size_t n_groups = (n + bs - 1) / bs;
    for (std::size_t g = 0; g < n_groups; ++g) {
        const std::size_t begin = g * bs;
        const std::size_t end = std::min(n, begin + bs);

        float amax = 0.0f;
        for (std::size_t i = begin; i < end; ++i) amax = std::max(amax, std::fabs(kv[i]));

        // scale = amax / 448 (максимум нормы E4M3 = 448); amax==0 → scale=0, коды 0
        const float scale = amax / 448.0f;
        out_scales[g] = scale;
        if (scale == 0.0f) {
            for (std::size_t i = begin; i < end; ++i) out_codes[i] = 0x00;
            continue;
        }
        const float inv = 1.0f / scale;
        for (std::size_t i = begin; i < end; ++i)
            out_codes[i] = e4m3_encode(kv[i] * inv);
    }
}

void dequantize_kv_e4m3(const std::uint8_t* codes, const float* scales,
                        std::size_t n, std::uint32_t block_size, float* out) {
    const std::size_t bs = block_size ? block_size : 1u;
    const std::size_t n_groups = (n + bs - 1) / bs;
    for (std::size_t g = 0; g < n_groups; ++g) {
        const std::size_t begin = g * bs;
        const std::size_t end = std::min(n, begin + bs);
        const float scale = scales[g];
        for (std::size_t i = begin; i < end; ++i)
            out[i] = e4m3_value(codes[i]) * scale;
    }
}

} // namespace stell::kernels

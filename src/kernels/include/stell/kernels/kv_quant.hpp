// KV-квант E4M3 (ADR-04: софт-квант, эмуляция на CPU; V100 без нативного FP8).
#pragma once

#include <cstddef>
#include <cstdint>

namespace stell::kernels {

// Per-block scale квантование KV (ADR-04: scale per-block):
//   квант:  для блока из block_size значений — scale = max|v| / 448;
//           коды = e4m3_encode(v / scale)
//   деквант: out = e4m3_value(code) * scale
// scales — [ceil(n/block_size)] FP32.
void quantize_kv_e4m3(const float* kv, std::size_t n, std::uint32_t block_size,
                      std::uint8_t* out_codes, float* out_scales);
void dequantize_kv_e4m3(const std::uint8_t* codes, const float* scales,
                        std::size_t n, std::uint32_t block_size, float* out);

} // namespace stell::kernels

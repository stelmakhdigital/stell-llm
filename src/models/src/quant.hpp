// ============================================================================
// Референсное (CPU) квантование профилей ADR-03 для лоадера NVFP4W4A16 и
// конвертера stell-tier. Инференсный dequant выполняют ядра engine/kernels
// (ADR-09 §3); здесь — единственная упаковка/распаковка модуля models:
//   - лоадер NVFP4W4A16 использует quantize_* на read();
//   - конвертер stell-tier — при квантованном профиле;
//   - dequant_* нужны тестам корректности (roundtrip) как референс.
//
// Блок = 32 элемента (см. weights.hpp).
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace stell::models {

// --- float16 (IEEE 754 half) -----------------------------------------------
std::uint16_t f32_to_f16_bits(float v);
float f16_bits_to_f32(std::uint16_t h);

// --- float8 e4m3 (scale NVFP4) ---------------------------------------------
std::uint8_t f32_to_e4m3(float v);
float e4m3_to_f32(std::uint8_t b);

// --- квантование тензора (payload без выравнивания) ------------------------
// Требуется n элементов во входе; буфер dst размером payload_bytes_for(dtype, {n}).
void quantize_nvfp4(const float* src, std::size_t n, std::uint8_t* dst);
void quantize_w4a16(const float* src, std::size_t n, std::uint8_t* dst);
void dequant_nvfp4(const std::uint8_t* src, std::size_t n, float* dst);
void dequant_w4a16(const std::uint8_t* src, std::size_t n, float* dst);

}  // namespace stell::models

// Dequant-ядра NVFP4 / W4A16 — CPU-референс (ADR-03, ADR-09 §1 dtype_weights).
#pragma once

#include <cstddef>
#include <cstdint>

namespace stell::kernels {

// --- FP4 (E2M1): nibble → значение ------------------------------------------
// Таблица E2M1 (знак | exp(2) | man(1), bias 1):
//   0..7:  0, 0.5, 1, 1.5, 2, 3, 4, 6
//   8..15: -0, -0.5, -1, -1.5, -2, -3, -4, -6
float e2m1_value(std::uint8_t nibble);

// --- E4M3 (OCP): код → значение ---------------------------------------------
// sign(1) | exp(4) | man(3), bias 7; subnormal при exp==0; NaN = 0x7F (и 0xFF).
// Макс. нормальное: 448 (код 0x7E).
float e4m3_value(std::uint8_t code);
// Кодирование float → E4M3 (round-to-nearest-even; переполнение — кламп
// на 448: софт-квант политика KV-кеша, ADR-04).
std::uint8_t e4m3_encode(float v);

// --- NVFP4 ------------------------------------------------------------------
// Раскладка: группы по 16 элементов; на группу — 16 байт (2 nibble/элемент) +
// 1 байт scale E4M3; глобальный scale — FP32 на тензор.
//   value = e2m1(nibble) * e4m3(scale_byte) * global_scale
//   packed    — [ceil(n/2)] байт (младший nibble — чётный элемент)
//   scales    — [ceil(n/16)] байт E4M3
void dequant_nvfp4(const std::uint8_t* packed, const std::uint8_t* scales,
                   float global_scale, std::size_t n, float* out);

// --- W4A16 ------------------------------------------------------------------
// Упакованный W4 (int4, two's complement, младший nibble — чётный элемент) +
// scale F16 на группу group_size элементов.
//   value = scale_f16 * int4(nibble)
void dequant_w4a16(const std::uint8_t* packed, const std::uint16_t* scales_f16,
                   std::size_t n, std::uint32_t group_size, float* out);

// IEEE half → float (для scale F16; без libdevice/F16-хардваря CPU-путь).
float half_to_float(std::uint16_t h);

} // namespace stell::kernels

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
#include <cstring>
#include <string>

namespace stell::models {

// --- квантованные GGML-блоки (subset, round 2; R3/R9 — срезы стековных
// тензоров в file_weights_source.hpp и dequant в loader_gguf.cpp) ------------
constexpr const char* kGgmlQ40Name = "Q4_0";
constexpr const char* kGgmlQ80Name = "Q8_0";

inline bool is_quantized_ggml(const std::string& dtype_str) {
    return dtype_str == kGgmlQ40Name || dtype_str == kGgmlQ80Name;
}

// Размер блока из 32 элементов: Q4_0 = 2 Б F16 delta (НАЧАЛО блока) + 16 Б
// нибблов; Q8_0 = 32 Б int8 + 2 Б F16 delta (КОНЕЦ блока).
inline std::size_t ggml_block_bytes(const std::string& dtype_str) {
    return dtype_str == kGgmlQ40Name ? 18 : 34;
}

// --- float16 (IEEE 754 half) -----------------------------------------------
std::uint16_t f32_to_f16_bits(float v);
float f16_bits_to_f32(std::uint16_t h);

// --- bfloat16 (round 4, R3-подготовка: BF16-тензора HF-чекпоинтов Qwen/
// Mixtral/OLMoE) ------------------------------------------------------------
// BF16 → F32: биты — старшая половина F32 (мантисса усекается, экспонента
// та же — конверсия точная по значению). Обратная конверсия — f32_to_f16_bits
// после привязки к F16 (см. plain_fill_slot).
inline float bf16_bits_to_f32(std::uint16_t h) {
    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

// --- R5 ([Q6], ADR-03 round 5): clamp-сатурация конверсий F32/BF16 → F16 ----
// Конверсия в F16 сатурсирируется к ±65504 (максимум F16): inf в весах не
// порождается (inf заражает вычисления NaN-пространством и недиагностируем
// в golden-логитах); clamp — детерминированный, видимый (счётчик) и
// наблюдаемый. Обычные значения (|v| ≤ 65504) — побитово == прежней
// конверсии (round-to-nearest-even не меняется).
inline constexpr float kF16Max = 65504.0f;

// Счётчик сатураций (plain_fill_slot: F32/BF16→F16 — общий путь лоадеров и
// конвертера). CPU-пути однопоточные — атомарность не требуется; читается/
// сбрасывается тестами и CLI конвертера (лог: «сатураций … N»).
inline std::uint64_t g_f16_saturation_count = 0;
inline void f16_saturation_reset() { g_f16_saturation_count = 0; }
inline std::uint64_t f16_saturation_count() { return g_f16_saturation_count; }

// Clamp-сатурация к ±65504; насыщенные значения учитываются в глобальном
// счётчике. NaN/|v| ≤ kF16Max — без изменений (NaN обрабатывает
// f32_to_f16_bits: canonical NaN half — прежнее поведение).
inline float saturate_f16(float v) {
    if (v > kF16Max) {
        ++g_f16_saturation_count;
        return kF16Max;
    }
    if (v < -kF16Max) {
        ++g_f16_saturation_count;
        return -kF16Max;
    }
    return v;
}

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

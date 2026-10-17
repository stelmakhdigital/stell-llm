#include "quant.hpp"

#include <cmath>
#include <algorithm>
#include <cstring>

namespace stell::models {

// --- IEEE half: round-to-nearest-even, компактная реализация ----------------
std::uint16_t f32_to_f16_bits(float v) {
    if (std::isnan(v)) return 0x7E00;  // canonical NaN half
    std::uint32_t b = 0;
    std::memcpy(&b, &v, 4);
    std::uint32_t sign = (b >> 16) & 0x8000u;
    std::int32_t e = static_cast<std::int32_t>((b >> 23) & 0xFF) - 127;
    std::uint32_t mant = b & 0x7FFFFFu;
    if (e >= 16) {  // переполнение → Inf
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (e < -10) {
        // subnormal half / нуль: округляем mant к 10 битам со сдвигом
        std::uint32_t shift = static_cast<std::uint32_t>(-(e + 1) + 13);
        if (shift >= 24) return static_cast<std::uint16_t>(sign);  // → 0
        std::uint32_t m = (mant | 0x800000u);                      // скрытая единица
        std::uint32_t half_m = m >> shift;
        std::uint32_t rem = m & ((1u << shift) - 1);
        std::uint32_t half = half_m;
        if (rem > (1u << shift)) half += 1;                 // > половины
        else if (rem == (1u << shift) && (half & 1)) half += 1;  // half-even
        if (half >= 0x400u)  // перенос в минимальный normal half
            return static_cast<std::uint16_t>(sign | (1u << 10));
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half_mant = mant >> 13;
    std::uint32_t rem13 = mant & 0x1FFFu;
    std::uint32_t h = static_cast<std::uint32_t>(e + 15) << 10;
    if (rem13 > 0x1000u) half_mant += 1;
    else if (rem13 == 0x1000u && (half_mant & 1)) half_mant += 1;
    if (half_mant >= 0x400u) {  // перенос в next exponent
        ++e;
        h = static_cast<std::uint32_t>(e + 15) << 10;
        half_mant = 0;
        if (e >= 16) return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    return static_cast<std::uint16_t>(sign | h | half_mant);
}

float f16_bits_to_f32(std::uint16_t h) {
    std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    std::uint32_t b;
    if (exp == 0x1F) {
        b = sign | 0x7F800000u | (mant << 13);  // Inf/NaN
        if (mant) b |= 0x400000u;               // make it a NaN
    } else if (exp == 0) {
        // subnormal half: value = mant * 2^-24
        if (mant == 0) b = sign;
        else {
            int e = -1;
            std::uint32_t m = mant;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            m &= 0x3FFu;
            b = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else {
        b = sign | (static_cast<std::uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &b, 4);
    return out;
}

// --- float8 e4m3 (1 sign, 4 exp, 3 mant; bias 7; max 448; NaN = 0x7F) -------
std::uint8_t f32_to_e4m3(float v) {
    if (std::isnan(v)) return 0x7F;
    bool sign = std::signbit(v) && v != 0.0f;
    float a = std::fabs(v);
    if (a >= 448.0f) return sign ? 0xFEu : 0x7Eu;  // saturate (без NaN-протеза)
    int e = 0;
    float m = std::frexp(a, &e);  // a = m * 2^e, m ∈ [0.5, 1)
    if (a == 0.0f) return 0;
    // value = m * 2^(e-1); exp_field = e - 1 + 7 = e + 6
    if (e + 6 <= 0) {
        // denorm: value = mant_field * 2^-9 (mant_field 0..7) или перенос
        std::int32_t mant = static_cast<std::int32_t>(std::lround(a * 512.0));
        if (mant <= 0) return 0;
        if (mant >= 8) return sign ? 0x80u : 0x08u;  // min normal 2^-6
        return static_cast<std::uint8_t>((sign ? 0x80u : 0) | static_cast<std::uint32_t>(mant));
    }
    std::int32_t exp_field = e + 6;
    if (exp_field > 15) return sign ? 0xFEu : 0x7Eu;
    std::int32_t mant = static_cast<std::int32_t>(std::lround(m * 8.0f));  // ∈ [4..8]
    if (mant >= 8) {
        ++exp_field;
        mant = 0;
        if (exp_field > 15) return sign ? 0xFEu : 0x7Eu;
    }
    return static_cast<std::uint8_t>((sign ? 0x80u : 0) |
                                     (static_cast<std::uint32_t>(exp_field) << 3) |
                                     static_cast<std::uint32_t>(mant));
}

float e4m3_to_f32(std::uint8_t b) {
    int sign = (b & 0x80) ? -1 : 1;
    int exp_field = (b >> 3) & 0xF;
    int mant = b & 7;
    if (exp_field == 15 && mant == 7) return std::numeric_limits<float>::quiet_NaN();
    if (exp_field == 0) return sign * static_cast<float>(mant) / 8.0f * std::ldexp(1.0f, -6);
    float v = std::ldexp(1.0f, exp_field - 7) * (1 + static_cast<float>(mant) / 8.0f);
    return sign * v;
}

// --- E2M1 (NVFP4 nibble): idx 0..7 → 0, 0.5, 1, 1.5, 2, 3, 4, 6 -------------
static float e2m1_value(std::uint8_t idx) {
    static const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return table[idx];
}

static std::uint8_t e2m1_quantize(float scaled) {  // scaled = |x|/scale ∈ [0..6]
    static const float boundaries[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    std::uint8_t idx = 0;
    while (idx < 7 && scaled >= boundaries[idx]) ++idx;
    return idx;
}

void quantize_nvfp4(const float* src, std::size_t n, std::uint8_t* dst) {
    const std::size_t n_blocks = (n + 31) / 32;
    std::uint8_t* nibbles = dst;                        // ceil(n/2) байт
    std::uint8_t* scales = dst + (n + 1) / 2;           // n_blocks байт
    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const std::size_t begin = blk * 32;
        const std::size_t cnt = std::min<std::size_t>(32, n - begin);
        float maxabs = 0.0f;
        for (std::size_t i = 0; i < cnt; ++i) maxabs = std::max(maxabs, std::fabs(src[begin + i]));
        const std::uint8_t scale_byte = f32_to_e4m3(maxabs / 6.0f);
        const float scale = e4m3_to_f32(scale_byte);
        scales[blk] = scale_byte;
        for (std::size_t i = 0; i < cnt; ++i) {
            float a = std::fabs(src[begin + i]) / (scale == 0.0f ? 1.0f : scale);
            std::uint8_t nib = e2m1_quantize(std::min(a, 6.0f));
            if (std::signbit(src[begin + i])) nib |= 0x8u;
            std::size_t byte_i = (begin + i) / 2;
            if ((begin + i) % 2 == 0) nibbles[byte_i] = nib;  // младший ниббл — чётный элемент
            else nibbles[byte_i] = static_cast<std::uint8_t>((nibbles[byte_i] & 0x0Fu) | (nib << 4));
        }
        // Хвост блока (кратность n не блоку) = 0 nibble (значение 0.0) — детерминизм.
    }
}

void dequant_nvfp4(const std::uint8_t* src, std::size_t n, float* dst) {
    const std::size_t n_blocks = (n + 31) / 32;
    const std::uint8_t* nibbles = src;
    const std::uint8_t* scales = src + (n + 1) / 2;
    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const std::size_t begin = blk * 32;
        const std::size_t cnt = std::min<std::size_t>(32, n - begin);
        const float scale = e4m3_to_f32(scales[blk]);
        for (std::size_t i = 0; i < cnt; ++i) {
            std::uint8_t byte = nibbles[(begin + i) / 2];
            std::uint8_t nib = ((begin + i) % 2 == 0) ? (byte & 0x0Fu) : (byte >> 4);
            float mag = e2m1_value(nib & 0x7u);
            dst[begin + i] = ((nib & 0x8u) ? -1.0f : 1.0f) * mag * scale;
        }
    }
}

void quantize_w4a16(const float* src, std::size_t n, std::uint8_t* dst) {
    const std::size_t n_blocks = (n + 31) / 32;
    std::uint8_t* nibbles = dst;                        // ceil(n/2) байт
    std::uint8_t* scales = dst + (n + 1) / 2;           // 2*n_blocks байт (F16)
    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const std::size_t begin = blk * 32;
        const std::size_t cnt = std::min<std::size_t>(32, n - begin);
        float maxabs = 0.0f;
        for (std::size_t i = 0; i < cnt; ++i) maxabs = std::max(maxabs, std::fabs(src[begin + i]));
        const std::uint16_t scale_bits = f32_to_f16_bits(maxabs / 7.0f);
        const float scale = f16_bits_to_f32(scale_bits);
        scales[2 * blk] = static_cast<std::uint8_t>(scale_bits & 0xFF);
        scales[2 * blk + 1] = static_cast<std::uint8_t>((scale_bits >> 8) & 0xFF);
        for (std::size_t i = 0; i < cnt; ++i) {
            float a = src[begin + i] / (scale == 0.0f ? 1.0f : scale);
            int q = static_cast<int>(std::lround(a));
            q = std::clamp(q, -8, 7);
            std::uint8_t nib = static_cast<std::uint8_t>(q + 8);
            std::size_t byte_i = (begin + i) / 2;
            if ((begin + i) % 2 == 0) nibbles[byte_i] = nib;
            else nibbles[byte_i] = static_cast<std::uint8_t>((nibbles[byte_i] & 0x0Fu) | (nib << 4));
        }
    }
}

void dequant_w4a16(const std::uint8_t* src, std::size_t n, float* dst) {
    const std::size_t n_blocks = (n + 31) / 32;
    const std::uint8_t* nibbles = src;
    const std::uint8_t* scales = src + (n + 1) / 2;
    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const std::size_t begin = blk * 32;
        const std::size_t cnt = std::min<std::size_t>(32, n - begin);
        std::uint16_t scale_bits = static_cast<std::uint16_t>(scales[2 * blk]) |
                                   (static_cast<std::uint16_t>(scales[2 * blk + 1]) << 8);
        const float scale = f16_bits_to_f32(scale_bits);
        for (std::size_t i = 0; i < cnt; ++i) {
            std::uint8_t byte = nibbles[(begin + i) / 2];
            std::uint8_t nib = ((begin + i) % 2 == 0) ? (byte & 0x0Fu) : (byte >> 4);
            dst[begin + i] = (static_cast<int>(nib) - 8) * scale;
        }
    }
}

}  // namespace stell::models

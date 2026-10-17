#include "stell/kernels/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace stell::kernels {

// --- FP4 E2M1 ---------------------------------------------------------------
float e2m1_value(std::uint8_t nibble) {
    static const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool neg = nibble >= 8;
    const std::uint8_t mag = static_cast<std::uint8_t>(neg ? nibble - 8 : nibble);
    const float v = table[mag & 0x7];
    return neg ? -v : v;
}

// --- E4M3 (OCP, bias 7; NaN = 0x7F/0xFF; max norm = 448 = 0x7E) --------------
float e4m3_value(std::uint8_t code) {
    const bool neg = code & 0x80;
    const std::uint8_t exp = (code >> 3) & 0xF;
    const std::uint8_t man = code & 0x7;
    if (exp == 15 && man == 7)
        return std::numeric_limits<float>::quiet_NaN();
    float v;
    if (exp == 0)
        v = static_cast<float>(man) * 0x1p-9f;   // subnormal: man * 2^-9
    else
        v = std::ldexp(1.0f + static_cast<float>(man) / 8.0f,
                       static_cast<int>(exp) - 7);
    return neg ? -v : v;
}

// RNE (round-to-nearest-even) целочисленного значения x >= 0 (по дробной части).
static int rne_int(float x) {
    int i = static_cast<int>(x);
    const float frac = x - static_cast<float>(i);
    if (frac > 0.5f || (frac == 0.5f && (i & 1))) ++i;
    return i;
}

std::uint8_t e4m3_encode(float v) {
    if (std::isnan(v)) return 0x7F;
    const std::uint8_t sign = (v < 0.0f) ? 0x80 : 0x00;
    const float a = std::fabs(v);
    if (a == 0.0f) return 0x00;

    int ee;                       // a = m * 2^ee, m in [0.5, 1)
    std::frexp(a, &ee);

    // нормальная форма E4M3: a = (1 + man/8) * 2^(E), man ∈ [0, 8)
    // E = ee - 1 (поле экспоненты = E + 7 = ee + 6, bias 7)
    const int exp = ee + 6;
    if (exp < 1) {
        // subnormal: man = RNE(a * 2^9); переполнение сабнормали = min normal
        const int man = rne_int(std::ldexp(a, 9));
        if (man >= 8) return static_cast<std::uint8_t>(sign | 0x08);
        return static_cast<std::uint8_t>(sign | man);
    }

    int man = rne_int((std::ldexp(a, 1 - ee) - 1.0f) * 8.0f);
    int expc = exp;
    if (man == 8) { man = 0; ++expc; }

    if (expc > 15)
        // переполнение: кламп на максимум нормальный (софт-квант политика ADR-04)
        return static_cast<std::uint8_t>(sign | 0x7E);
    return static_cast<std::uint8_t>(sign | (expc << 3) | man);
}

// --- half → float (IEEE 754 binary16 → binary32, без F16-хардваря) -----------
float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h >> 15) << 31;
    const std::uint32_t exp16 = (h >> 10) & 0x1F;
    std::uint32_t man = h & 0x3FF;

    if (exp16 == 0x1F) {
        // inf / nan half → inf / nan binary32
        const std::uint32_t bits = sign | (0xFFu << 23) | (man ? 0x400000u : 0u);
        float out; std::memcpy(&out, &bits, 4); return out;
    }
    if (exp16 == 0) {
        if (man == 0) {
            // ноль (включая -0)
            const std::uint32_t bits = sign;
            float out; std::memcpy(&out, &bits, 4); return out;
        }
        // denormal half: значение = man * 2^-24, нормализуем:
        // man << k → [0x400, 0x800), e = -14 - k, E = 127 + e, мантисса = f << 13
        int e = -14;
        while (!(man & 0x400)) { man <<= 1; --e; }
        const std::uint32_t f = man & 0x3FF;
        const std::uint32_t exp32d = static_cast<std::uint32_t>(127 + e);
        const std::uint32_t bits = sign | (exp32d << 23) | (f << 13);
        float out; std::memcpy(&out, &bits, 4); return out;
    }

    const std::uint32_t exp32 = exp16 - 15 + 127;
    const std::uint32_t bits = sign | (exp32 << 23) | (man << 13);  // 10-бит man → 23-бит поле
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// --- NVFP4 ------------------------------------------------------------------
void dequant_nvfp4(const std::uint8_t* packed, const std::uint8_t* scales,
                   float global_scale, std::size_t n, float* out) {
    const std::size_t n_groups = (n + 15) / 16;
    for (std::size_t g = 0; g < n_groups; ++g) {
        const float s = e4m3_value(scales[g]) * global_scale;
        const std::size_t begin = g * 16;
        const std::size_t end = std::min(n, begin + 16);
        for (std::size_t i = begin; i < end; ++i) {
            const std::uint8_t byte = packed[i / 2];
            const std::uint8_t nibble =
                static_cast<std::uint8_t>((i % 2 == 0) ? (byte & 0x0F) : (byte >> 4));
            out[i] = e2m1_value(nibble) * s;
        }
    }
}

// --- W4A16 ------------------------------------------------------------------
void dequant_w4a16(const std::uint8_t* packed, const std::uint16_t* scales_f16,
                   std::size_t n, std::uint32_t group_size, float* out) {
    const std::size_t gs = group_size ? group_size : 1u;
    const std::size_t n_groups = (n + gs - 1) / gs;
    for (std::size_t g = 0; g < n_groups; ++g) {
        const float s = half_to_float(scales_f16[g]);
        const std::size_t begin = g * gs;
        const std::size_t end = std::min(n, begin + gs);
        for (std::size_t i = begin; i < end; ++i) {
            const std::uint8_t byte = packed[i / 2];
            const int nib = static_cast<int>((i % 2 == 0) ? (byte & 0x0F) : (byte >> 4));
            const int signed4 = (nib & 0x8) ? (nib - 16) : nib;   // two's complement
            out[i] = s * static_cast<float>(signed4);
        }
    }
}

} // namespace stell::kernels

// stell-llm — parallel: кодек CommDType <-> float.
//
// Нужен CPU-эмулятору для детерминированной редукции (ADR-10 §2: порядок
// ранков, dtype редукции) и для E4M3 KV-кванта (ADR-04). Конвертации
// round-to-nearest-even; E4M3 насыщается к max finite (448, OCP E4M3), NaN
// сохраняется как NaN.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include "stell/parallel/comm.hpp"

namespace stell::parallel {

namespace detail {

template <class T>
inline std::uint32_t bit_cast_u32(T v) {
    static_assert(sizeof(T) == sizeof(std::uint32_t));
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}
inline float u32_as_float(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

} // namespace detail

// --- binary16 <-> binary32 (round-to-nearest-even) --------------------------
inline float half_to_float(std::uint16_t h) {
    const std::uint32_t s = std::uint32_t(h & 0x8000u) << 16;
    const std::uint32_t e = (h >> 10) & 0x1Fu;
    const std::uint32_t m = h & 0x03FFu;
    if (e == 0x1F)                       // inf / nan
        return detail::u32_as_float(s | 0x7F800000u | (m ? (0x40000u | (m << 13)) : 0));
    if (e == 0) {                        // zero / subnormal (точно, без округления)
        if (m == 0) return detail::u32_as_float(s);
        const float v = std::ldexp(float(m), -24);
        return s ? -v : v;
    }
    return detail::u32_as_float(s | (std::uint32_t(e - 15 + 127) << 23) | (m << 13));
}

inline std::uint16_t float_to_half(float f) {
    const std::uint32_t b = detail::bit_cast_u32(f);
    const std::uint16_t sign = std::uint16_t((b >> 16) & 0x8000u);
    const std::uint32_t x = b & 0x7FFFFFFFu;
    if (x >= 0x7F800000u)                // inf / nan
        return std::uint16_t(sign | (x > 0x7F800000u ? 0x7E00u : 0x7C00u));
    if ((x >> 23) == 0)                  // denormal float (< 2^-126) -> 0 в half
        return sign;
    std::int32_t e = std::int32_t((x >> 23) & 0xFF) - 127 + 15; // half-bias 15
    const std::uint32_t M = (x & 0x007FFFFFu) | 0x00800000u;    // 24-битный значащий
    // Нормалы half: сдвиг 13; субнормалы: m10 = M >> (14 - e), e <= 0.
    std::uint32_t shift = (e <= 0) ? std::uint32_t(14 - e) : 13u;
    if (shift > 24) return sign;         // < 2^-25 -> 0 (tie к even тоже 0)
    std::uint32_t m10 = M >> shift;
    const std::uint32_t rem = M & ((1u << shift) - 1u);
    const std::uint32_t half_pt = 1u << (shift - 1u);
    if (rem > half_pt || (rem == half_pt && (m10 & 1u))) {
        ++m10;
        if (e > 0) {
            if (m10 == 0x800u) { m10 = 0x400u; ++e; }         // перенос порядка
        } else if (m10 == 1024u) { m10 = 0x400u; e = 1; }     // субнормал -> нормал
    }
    if (e >= 31) return std::uint16_t(sign | 0x7C00u);        // overflow -> inf
    if (e <= 0)  return std::uint16_t(sign | std::uint16_t(m10 & 0x03FFu));
    return std::uint16_t(sign | (std::uint16_t(e) << 10) | std::uint16_t(m10 & 0x03FFu));
}

// --- E4M3 (1s - 4e - 3m, bias 7; OCP FP8: e=15 — конечные значения,
// NaN только 0x7F/0xFF; max finite 448; overflow -> saturate к 448) ----------
inline float e4m3_to_float(std::uint8_t v) {
    const std::uint32_t s = std::uint32_t(v & 0x80u) << 24;
    const std::uint32_t e = (v >> 3) & 0x0Fu;
    const std::uint32_t m = v & 0x07u;
    if (e == 0x0F && m == 0x07)          // NaN (0x7F/0xFF)
        return detail::u32_as_float(s | 0x7FC00000u);
    if (e == 0) {                        // zero / subnormal: m * 2^-9
        if (m == 0) return detail::u32_as_float(s);
        const float x = std::ldexp(float(m), -9);
        return s ? -x : x;
    }
    // нормал: 2^(e-7) * (1 + m/8) = (8|8+m) * 2^(e-10); e=15 — тоже конечные.
    const float x = std::ldexp(float(std::uint32_t(8u) + m), int(e) - 10);
    return s ? -x : x;
}

inline std::uint8_t float_to_e4m3(float f) {
    const std::uint32_t b = detail::bit_cast_u32(f);
    const std::uint8_t sign = std::uint8_t((b >> 24) & 0x80u);
    const std::uint32_t x = b & 0x7FFFFFFFu;
    if (x > 0x7F800000u) return std::uint8_t(sign | 0x7Fu); // NaN
    if ((x >> 23) == 0) return sign;                        // denormal float -> 0
    std::int32_t e = std::int32_t((x >> 23) & 0xFF) - 127 + 7; // bias 7
    const std::uint32_t M = (x & 0x007FFFFFu) | 0x00800000u;
    // Нормалы: сдвиг 20; субнормалы: m3 = M >> (21 - e) — аналогично half.
    std::uint32_t shift = (e <= 0) ? std::uint32_t(21 - e) : 20u;
    if (shift > 24) return sign;         // < 2^-10.5 -> 0
    std::uint32_t m3 = M >> shift;
    const std::uint32_t rem = M & ((1u << shift) - 1u);
    const std::uint32_t half_pt = 1u << (shift - 1u);
    if (rem > half_pt || (rem == half_pt && (m3 & 1u))) {
        ++m3;
        if (e > 0) {
            if (m3 == 16u) { m3 = 8u; ++e; }             // перенос порядка
        } else if (m3 == 8u) { m3 = 8u; e = 1; }         // субнормал -> нормал
    }
    if (e > 15 || (e == 15 && m3 == 15))                 // overflow / NaN-регион
        return std::uint8_t(sign | 0x7Eu);               // saturate к 448
    if (e <= 0) return std::uint8_t(sign | std::uint8_t(m3 & 0x07u));
    return std::uint8_t(sign | (std::uint8_t(e) << 3) | std::uint8_t(m3 & 0x07u));
}

// --- типизированные конвертации для редукции/копий ---------------------------
inline float comm_to_float(CommDType t, const void* p) {
    switch (t) {
        case CommDType::F16: return half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
        case CommDType::F32: return *reinterpret_cast<const float*>(p);
        case CommDType::E4M3: return e4m3_to_float(*reinterpret_cast<const std::uint8_t*>(p));
    }
    return 0.f;
}

inline void comm_store_float(CommDType t, void* p, float v) {
    switch (t) {
        case CommDType::F16:
            *reinterpret_cast<std::uint16_t*>(p) = float_to_half(v);
            break;
        case CommDType::F32:
            *reinterpret_cast<float*>(p) = v;
            break;
        case CommDType::E4M3:
            *reinterpret_cast<std::uint8_t*>(p) = float_to_e4m3(v);
            break;
    }
}

} // namespace stell::parallel

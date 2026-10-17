// tests/parallel: dtype-кодек F16 / E4M3 (ADR-04) — конвертации и редукция.
#include "test_harness.hpp"
#include "stell/parallel/dtype_codec.hpp"

#include <cmath>
#include <vector>

using namespace stell::parallel;

TEST(codec_half_roundtrip_exact_values) {
    // Значения, точно представимые в half, конвертируются без потерь.
    for (float v : {0.f, 1.f, 2.f, 0.5f, 0.25f, -1.5f, 1024.f, 3.f, 8192.f}) {
        const std::uint16_t h = float_to_half(v);
        const float back = half_to_float(h);
        CHECK(std::fabs(back - v) < 1e-6f * (std::fabs(v) + 1));
        if (v == 0.f) CHECK(h == 0x0000u || h == 0x8000u);
    }
    CHECK(float_to_half(2.f) == 0x4000u);           // 2.0
    CHECK(float_to_half(-0.5f) == 0xB800u);         // -0.5
    CHECK(float_to_half(70000.f) == 0x7C00u);       // overflow → inf
    CHECK(float_to_half(1e-9f) == 0x0000u);         // < 2^-25 → 0
    // Субнормал half: 2^-20 точно представима.
    const float sub = std::ldexp(1.f, -20);
    CHECK(half_to_float(float_to_half(sub)) == sub);
    // Граница RN-even: 65520 → inf (tie к чётной мантиссе), 65504 → 65504.
    CHECK(float_to_half(65520.f) == 0x7C00u);
    CHECK(float_to_half(65504.f) == 0x7BFFu);
}

TEST(codec_half_to_float_known_bits) {
    CHECK(half_to_float(0x4000u) == 2.0f);
    CHECK(half_to_float(0xC000u) == -2.0f);
    CHECK(half_to_float(0x0000u) == 0.0f);
    CHECK(half_to_float(0x8000u) == 0.0f);           // -0
    CHECK(std::isnan(half_to_float(0x7E00u)));       // NaN
    CHECK(half_to_float(0x3D60u) == 1.34375f);       // 1 + 352/1024
    CHECK(half_to_float(0x3C00u) == 1.0f);
    // Субнормал: 0x0001 = 2^-24.
    CHECK(half_to_float(0x0001u) == std::ldexp(1.f, -24));
    CHECK(half_to_float(0x7BFFu) == 65504.0f);       // max finite
}

TEST(codec_e4m3_known_values) {
    // OCP E4M3 (bias 7): субнормалы m*2^-9; e=15 — конечные; NaN = 0x7F/0xFF;
    // max finite 448 = 2^8*(1+6/8); overflow → saturate к 448.
    CHECK(e4m3_to_float(0x00u) == 0.0f);
    CHECK(e4m3_to_float(0x08u) == std::ldexp(1.f, -6));   // e=1,m=0
    CHECK(e4m3_to_float(0x77u) == 240.0f);           // e=14,m=7 → 128*1.875
    CHECK(e4m3_to_float(0x7Eu) == 448.0f);           // max finite (e=15,m=6)
    CHECK(std::isnan(e4m3_to_float(0x7Fu)));         // NaN
    CHECK(e4m3_to_float(0x01u) == std::ldexp(1.f, -9));   // субнормал
    CHECK(e4m3_to_float(0x07u) == std::ldexp(7.f, -9));
    // Точные обратные конвертации для dyadic-значений внутри диапазона.
    for (float v : {0.f, 1.f, 2.f, 4.f, 0.5f, 0.25f, 12.f, -12.f, 448.f, 240.f}) {
        const std::uint8_t q = float_to_e4m3(v);
        const float back = e4m3_to_float(q);
        CHECK(std::fabs(back - v) <= 0.0625f * std::fabs(v));   // 3 бита мантиссы
    }
    CHECK(float_to_e4m3(448.f) == 0x7Eu);
    CHECK(e4m3_to_float(float_to_e4m3(1000.f)) == 448.0f);      // saturate
    CHECK(float_to_e4m3(1e-9f) == 0x00u);
    // Субнормальная граница RN-even: 2^-10 — точный tie между 0 и 2^-9 → 0;
    // чуть выше — округление к 2^-9 (0x01).
    CHECK(float_to_e4m3(std::ldexp(1.f, -10)) == 0x00u);
    CHECK(float_to_e4m3(1.5f * std::ldexp(1.f, -10)) == 0x01u);
    CHECK(float_to_e4m3(0.9f * std::ldexp(1.f, -10)) == 0x00u);
}

TEST(codec_roundtrip_f16_e4m3_symmetry) {
    // Кодек согласован в обе стороны на всех half-значениях.
    for (std::uint16_t h = 0; h < 0x8000u; ++h) {
        const float f = half_to_float(h);
        if (std::isnan(f)) continue;
        const std::uint16_t h2 = float_to_half(f);
        if (h == 0x7C00u) {                          // +inf → +inf
            CHECK(h2 == 0x7C00u);
            continue;
        }
        if (std::isinf(f)) continue;
        CHECK(h2 == h);                              // точный roundtrip
    }
    // E4M3: roundtrip для всех 256 кодов (включая NaN-регион 0x7F/0xFF).
    for (int code = 0; code < 256; ++code) {
        const std::uint8_t q = std::uint8_t(code);
        const float f = e4m3_to_float(q);
        const std::uint8_t q2 = float_to_e4m3(f);
        CHECK(q2 == q);                              // точный roundtrip (NaN тоже)
    }
}

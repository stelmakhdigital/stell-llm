// Replay-тест dequant NVFP4 / W4A16: таблицы значений, кодирование E4M3,
// half→float, эталонные буферы.
#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "replay_fixture.hpp"
#include "stell/kernels/dequant.hpp"

using namespace stell::kernels;
using stell::kernels::test::close;
using stell::kernels::test::fill_lcg;

void run() {
    // --- E2M1: полная таблица 16 nibble -------------------------------------
    const float e2m1_expect[16] = {0, -0.5, -1, -1.5, -2, -3, -4, -6,
                                   0, -0.5, -1, -1.5, -2, -3, -4, -6};
    // ВНИМАНИЕ: таблица ниже — nibbles 0..15 c битом знака в старшем бите
    const float e2m1_ref[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    for (int i = 0; i < 16; ++i) {
        STELL_CHECK_MSG(e2m1_value(static_cast<std::uint8_t>(i)) == e2m1_ref[i],
                        "E2M1 таблица");
        (void)e2m1_expect;
    }

    // --- E4M3 decode: известные коды ----------------------------------------
    STELL_CHECK(e4m3_value(0x00) == 0.0f);
    STELL_CHECK(close(e4m3_value(0x01), 0x1p-9f));   // subnormal man=1
    STELL_CHECK(close(e4m3_value(0x08), 0x1p-6f));   // min normal 2^-6
    STELL_CHECK(close(e4m3_value(0x38), 1.0f));
    STELL_CHECK(close(e4m3_value(0x30), 0.5f));
    STELL_CHECK(close(e4m3_value(0x40), 2.0f));
    STELL_CHECK(close(e4m3_value(0x44), 3.0f));
    STELL_CHECK(close(e4m3_value(0x48), 4.0f));
    STELL_CHECK(close(e4m3_value(0x70), 128.0f));
    STELL_CHECK(close(e4m3_value(0x7E), 448.0f));    // максимум нормы
    STELL_CHECK(std::isnan(e4m3_value(0x7F)));       // NaN
    STELL_CHECK(close(e4m3_value(0xB8), -1.0f));     // знак
    STELL_CHECK(e4m3_value(0x80) == 0.0f);

    // --- E4M3 encode: RNE, кламп на 448 -------------------------------------
    STELL_CHECK(e4m3_encode(0.0f) == 0x00);
    STELL_CHECK(e4m3_encode(1.0f) == 0x38);
    STELL_CHECK(e4m3_encode(-1.0f) == 0xB8);
    STELL_CHECK(e4m3_encode(0.5f) == 0x30);
    STELL_CHECK(e4m3_encode(2.0f) == 0x40);
    STELL_CHECK(e4m3_encode(1.0625f) == 0x38);       // тай-тайник RNE → чётный man=0
    STELL_CHECK(e4m3_encode(1.5f) == 0x3C);          // man=4: 1+4/8
    STELL_CHECK(e4m3_encode(0.25f) == 0x28);
    STELL_CHECK(e4m3_encode(0x1p-9f) == 0x01);       // 2^-9 — min subnormal
    STELL_CHECK(e4m3_encode(448.0f) == 0x7E);
    STELL_CHECK(e4m3_encode(500.0f) == 0x7E);        // кламп переполнения
    STELL_CHECK(e4m3_encode(-500.0f) == 0xFE);       // кламп с знаком
    STELL_CHECK(std::isnan(e4m3_value(e4m3_encode(std::numeric_limits<float>::quiet_NaN()))));

    // roundtrip: decode(encode(x)) близко к x (в пределах шага E4M3)
    {
        std::vector<float> vals;
        for (float x = 0.01f; x < 500.0f; x *= 1.7f) vals.push_back(x);
        for (float x : vals) {
            const float dq = e4m3_value(e4m3_encode(x));
            // погрешность RNE ≤ 6.25% (3 мантисс-бита) + субнормальная зона
            const float rel = std::fabs(dq - x) / std::fabs(x);
            STELL_CHECK_MSG(rel <= 0.07f || x < 0x1p-6f * 1.01f,
                            "E4M3 roundtrip в пределах шага");
        }
    }

    // --- half → float -------------------------------------------------------
    STELL_CHECK(half_to_float(0x0000) == 0.0f);
    STELL_CHECK(half_to_float(0x8000) == 0.0f);       // -0
    STELL_CHECK(close(half_to_float(0x3C00), 1.0f));
    STELL_CHECK(close(half_to_float(0x0001), 0x1p-24f));  // min denormal
    STELL_CHECK(close(half_to_float(0x3555), (1.0f + 341.0f / 1024.0f) * 0.25f));
    STELL_CHECK(std::isinf(half_to_float(0x7C00)));
    STELL_CHECK(std::isnan(half_to_float(0x7E00)));   // NaN half (man != 0)

    // --- NVFP4 dequant: hand-checked буфер ----------------------------------
    {
        // n=8: два nibble/байт, 4 байта; группа масштабов — ceil(8/16)=1 байт
        // nibbles: 0,1,2,3 (байт 0x21), 4,5,6,7 (байт 0x76... соберём вручную)
        std::uint8_t packed[4];
        packed[0] = 0x21;   // nibble0=1(0.5), nibble1=2(1.0)
        packed[1] = 0x43;   // nibble2=3(1.5), nibble3=4(2.0)
        packed[2] = 0x65;   // nibble4=5(3.0), nibble5=6(4.0)
        packed[3] = 0x87;   // nibble6=7(6.0), nibble7=8(-0)
        std::uint8_t scale = 0x38;                  // 1.0
        const float gscale = 1.0f;
        float out[8];
        dequant_nvfp4(packed, &scale, gscale, 8, out);
        const float ref[8] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 0.0f};
        for (int i = 0; i < 8; ++i)
            STELL_CHECK_MSG(close(out[i], ref[i]), "NVFP4 hand-checked buffer");
    }

    // --- NVFP4 dequant: LCG-фикстура против независимого декода -------------
    {
        const std::size_t n = 64;
        std::uint8_t packed[n / 2];
        std::uint8_t scales[n / 16];
        std::uint8_t* p = packed;
        test::Lcg g(7);
        for (std::size_t i = 0; i < n / 2; ++i) p[i] = g.next_u8();
        for (std::size_t i = 0; i < n / 16; ++i) scales[i] = g.next_u8();
        const float gscale = 1.25f;

        std::vector<float> out(n);
        dequant_nvfp4(packed, scales, gscale, n, out.data());

        // независимый декод (double, свои таблицы)
        const double e2m1_tab[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        auto e4m3_ref = [](std::uint8_t code) -> double {
            const bool neg = code & 0x80;
            const int e = (code >> 3) & 0xF;
            const int m = code & 0x7;
            if (e == 15 && m == 7) return std::nan("");
            double v = e == 0 ? m / 512.0 : (1.0 + m / 8.0) * std::pow(2.0, e - 7);
            return neg ? -v : v;
        };
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint8_t byte = packed[i / 2];
            const int nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            const double mag = e2m1_tab[nib & 7];
            const double val = (nib & 8 ? -mag : mag) *
                               e4m3_ref(scales[i / 16]) * gscale;
            STELL_CHECK_MSG(std::fabs(out[i] - val) <= 1e-6 + 1e-6 * std::fabs(val),
                            "NVFP4 replay против независимого декода");
        }
    }

    // --- W4A16 dequant: hand-checked ----------------------------------------
    {
        // nibbles: 0(-8→0?), int4 two's complement: 0..7 → 0..7, 8..15 → -8..-1
        std::uint8_t packed[3];
        packed[0] = 0x21;    // nibble0=1, nibble1=2
        packed[1] = 0xF3;    // nibble2=3, nibble3=15(-1)
        packed[2] = 0x89;    // nibble4=9(-7), nibble5=8(-8)
        std::uint16_t scale_f16 = 0x3C00;   // 1.0 (half)
        float out[6];
        dequant_w4a16(packed, &scale_f16, 6, 6, out);
        const float ref[6] = {1.0f, 2.0f, 3.0f, -1.0f, -7.0f, -8.0f};
        for (int i = 0; i < 6; ++i)
            STELL_CHECK_MSG(close(out[i], ref[i]), "W4A16 hand-checked");
    }

    // --- W4A16: group scales -------------------------------------------------
    {
        const std::size_t n = 8, gs = 4;
        std::uint8_t packed[4];
        test::Lcg g(9);
        for (std::size_t i = 0; i < 4; ++i) packed[i] = g.next_u8();
        std::uint16_t scales[2] = {0x3C00 /*1.0*/, 0x4700 /*7.0? → 0x47 = exp16=8?*/};
        // half 0x4700: exp16=0x11(17)? (0x4700>>10)=0x11=17, man=0x300=768
        // значение = (1 + 768/1024) * 2^(17-15) = 1.75 * 4 = 7.0
        float out[n];
        dequant_w4a16(packed, scales, n, gs, out);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint8_t byte = packed[i / 2];
            const int nib = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            const int s4 = (nib & 8) ? (nib - 16) : nib;
            const float scale = (i < 4) ? 1.0f : 7.0f;
            STELL_CHECK_MSG(close(out[i], scale * s4, 1e-6f), "W4A16 группы");
        }
    }
}

STELL_TEST_MAIN(run)

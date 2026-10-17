// Replay-тест KV-квант E4M3 (ADR-04, эмуляция): hand-checked блоки + LCG
// фиксура с оценкой погрешности.
#include "test_util.hpp"

#include <cmath>
#include <vector>

#include "replay_fixture.hpp"
#include "stell/kernels/kv_quant.hpp"

using namespace stell::kernels;
using stell::kernels::test::close;
using stell::kernels::test::fill_lcg;

void run() {
    // --- hand-checked: блок [1, 2, 4, 8] ------------------------------------
    {
        const float kv[4] = {1.0f, 2.0f, 4.0f, 8.0f};
        std::uint8_t codes[4];
        float scales[1];
        quantize_kv_e4m3(kv, 4, 4, codes, scales);
        // amax=8 → scale = 8/448; значения/масштаб ∈ {0.0625..0.5} — коды E4M3
        STELL_CHECK(close(scales[0], 8.0f / 448.0f));
        float out[4];
        dequantize_kv_e4m3(codes, scales, 4, 4, out);
        for (int i = 0; i < 4; ++i)
            STELL_CHECK_MSG(std::fabs(out[i] - kv[i]) / kv[i] <= 0.07f,
                            "квант-deквант в пределах шага E4M3 (3 мантисс-бита)");
    }

    // --- нулевой блок -------------------------------------------------------
    {
        const float kv[3] = {0.0f, 0.0f, 0.0f};
        std::uint8_t codes[3];
        float scales[1];
        quantize_kv_e4m3(kv, 3, 3, codes, scales);
        STELL_CHECK(scales[0] == 0.0f);
        for (int i = 0; i < 3; ++i) STELL_CHECK(codes[i] == 0x00);
    }

    // --- LCG-фикстура: относительная погрешность ≤ ~7% ----------------------
    {
        const std::size_t n = 256;
        std::vector<float> kv(n);
        fill_lcg(kv, 5);
        std::uint8_t codes[n];
        std::vector<float> scales((n + 31) / 32);
        quantize_kv_e4m3(kv.data(), n, 32, codes, scales.data());
        std::vector<float> out(n);
        dequantize_kv_e4m3(codes, scales.data(), n, 32, out.data());

        double max_rel = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (kv[i] == 0.0f) { STELL_CHECK(out[i] == 0.0f); continue; }
            const double rel = std::fabs(out[i] - kv[i]) / std::fabs(kv[i]);
            max_rel = std::max(max_rel, rel);
        }
        // E4M3: 3 бита мантиссы → max rel error RNE = 1/16 = 6.25% (в зоне
        // сабнормали выше, масштаб блока уменьшает шаг)
        STELL_CHECK_MSG(max_rel <= 0.10, "KV E4M3 софт-квант: max rel err <= 10%");
    }

    // --- несколько блоков: scales независимы --------------------------------
    {
        const float kv[6] = {4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        std::uint8_t codes[6];
        float scales[2];
        quantize_kv_e4m3(kv, 6, 3, codes, scales);
        // блок 0: amax=4 → scale=4/448; блок 1: amax=1 → scale=1/448
        STELL_CHECK(close(scales[0], 4.0f / 448.0f));
        STELL_CHECK(close(scales[1], 1.0f / 448.0f));
    }
}

STELL_TEST_MAIN(run)

// Общие фикстуры операторных replay-тестов (tests/core/kernels):
// детерминированный LCG + независимые наивные реализации-референсы.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>

namespace stell::kernels::test {

// Детерминированный генератор (fixed seed): одни и те же данные на каждом
// прогоне и на CPU/CUDA (replay — сравнение с эталонным буфером).
class Lcg {
public:
    explicit Lcg(std::uint32_t seed = 42u) : state_(seed ? seed : 1u) {}
    std::uint32_t next_u32() {
        // xorshift32 — детерминирован, портабелен
        state_ ^= state_ << 13;
        state_ ^= state_ >> 17;
        state_ ^= state_ << 5;
        return state_;
    }
    float next_float() {  // [-1, 1)
        return static_cast<float>(static_cast<std::int32_t>(next_u32() & 0xFFFF))
             / 8192.0f - 1.0f + 0.0625f;
    }
    std::uint8_t next_u8() { return static_cast<std::uint8_t>(next_u32() & 0xFF); }
private:
    std::uint32_t state_;
};

inline void fill_lcg(std::vector<float>& v, std::uint32_t seed) {
    Lcg g(seed);
    for (float& x : v) x = g.next_float();
}

inline bool close(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps * (1.0f + std::fabs(b));
}

// --- независимые наивные референсы (другая структура кода, чем в ядрах) ------

void naive_attention_f32(const float* q, const float* k, const float* v,
                         std::uint32_t heads, std::uint32_t T, std::uint32_t d,
                         float* out) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    for (std::uint32_t h = 0; h < heads; ++h)
        for (std::uint32_t i = 0; i < T; ++i) {
            // softmax по j (double-аккумуляция — заведомо другой путь)
            double ssum = 0.0;
            std::vector<double> p(T, 0.0);
            for (std::uint32_t j = 0; j < T; ++j) {
                double dot = 0.0;
                for (std::uint32_t dd = 0; dd < d; ++dd)
                    dot += static_cast<double>(q[(h * T + i) * d + dd]) *
                           static_cast<double>(k[(h * T + j) * d + dd]);
                p[j] = dot * scale;
                ssum = j == 0 ? p[j] : std::max(ssum, p[j]);
            }
            double denom = 0.0;
            for (std::uint32_t j = 0; j < T; ++j) {
                p[j] = std::exp(p[j] - ssum);
                denom += p[j];
            }
            for (std::uint32_t dd = 0; dd < d; ++dd) {
                double o = 0.0;
                for (std::uint32_t j = 0; j < T; ++j)
                    o += p[j] * static_cast<double>(v[(h * T + j) * d + dd]);
                out[(h * T + i) * d + dd] = static_cast<float>(o / denom);
            }
        }
}

} // namespace stell::kernels::test

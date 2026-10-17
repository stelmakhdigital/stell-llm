// Replay-тест SM70-attention FP32 QK/PV: эталонные буферы (hand-checked) +
// независимый наивный референс + layout-реплей (перестановка голов).
#include "test_util.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "replay_fixture.hpp"
#include "stell/kernels/attention.hpp"

using namespace stell::kernels;
using stell::kernels::test::Lcg;
using stell::kernels::test::close;
using stell::kernels::test::fill_lcg;
using stell::kernels::test::naive_attention_f32;

namespace {
// эталонный буфер T=2, d=2, heads=1: значения подобраны под ручную проверку
const float q_ref[4] = {1.0f, 0.0f, 0.0f, 1.0f};
const float k_ref[4] = {1.0f, 0.0f, 0.0f, 1.0f};
const float v_ref[4] = {2.0f, 0.0f, 0.0f, 3.0f};
} // namespace

void run() {
    // --- hand-checked: T=1, d=1, heads=1 → out = v --------------------------
    {
        const float q1[1] = {1.0f}, k1[1] = {1.0f}, v1[1] = {5.0f};
        float out1[1] = {0.0f};
        attention_forward_f32(q1, k1, v1, 1, 1, 1, out1);
        STELL_CHECK(close(out1[0], 5.0f));
    }

    // --- hand-checked: T=2, d=2 --------------------------------------------
    {
        float scores[4] = {0, 0, 0, 0};
        qk_scores_f32(q_ref, k_ref, 1, 2, 2, scores);
        // sqrt(d)=sqrt(2): q0·k0=1, q0·k1=0, q1·k0=0, q1·k1=1
        STELL_CHECK(close(scores[0], 1.0f / std::sqrt(2.0f)));
        STELL_CHECK(close(scores[1], 0.0f));
        STELL_CHECK(close(scores[2], 0.0f));
        STELL_CHECK(close(scores[3], 1.0f / std::sqrt(2.0f)));

        softmax_rows_f32(scores, 2, 2);
        // каждая строка softmax: [a, b], a+b=1, a = softmax(1/sqrt2) и т.д.
        STELL_CHECK(close(scores[0] + scores[1], 1.0f));
        STELL_CHECK(close(scores[2] + scores[3], 1.0f));
        // a > b (первый элемент больше): exp(1/sqrt2) > exp(0)
        // строка 0: q0·k0=1/sqrt2 > q0·k1=0 → scores[0] > scores[1]
        // строка 1: q1·k0=0 < q1·k1=1/sqrt2 → scores[3] > scores[2]
        STELL_CHECK(scores[0] > scores[1]);
        STELL_CHECK(scores[3] > scores[2]);

        float out[4] = {0, 0, 0, 0};
        pv_output_f32(scores, v_ref, 1, 2, 2, out);
        // строка 0: p00*v0 + p01*v1 (v1 = [0,3]) → [2*p00, 3*p01]
        // строка 1: p10*v0 + p11*v1 → [2*p10, 3*p11]
        float s2[4] = {0, 0, 0, 0};
        qk_scores_f32(q_ref, k_ref, 1, 2, 2, s2);
        softmax_rows_f32(s2, 2, 2);
        float expect[4];
        for (int i = 0; i < 2; ++i)
            for (int d = 0; d < 2; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < 2; ++j)
                    acc += s2[i * 2 + j] * v_ref[j * 2 + d];
                expect[i * 2 + d] = acc;
            }
        STELL_CHECK(close(out[0], expect[0]));
        STELL_CHECK(close(out[1], expect[1]));
        STELL_CHECK(close(out[2], expect[2]));
        STELL_CHECK(close(out[3], expect[3]));
    }

    // --- replay против независимого наивного референса (LCG-данные) ---------
    {
        const std::uint32_t heads = 4, T = 6, d = 8;
        std::vector<float> q(heads * T * d), k(heads * T * d), v(heads * T * d);
        fill_lcg(q, 101);
        fill_lcg(k, 202);
        fill_lcg(v, 303);
        std::vector<float> out(heads * T * d), ref(heads * T * d);
        attention_forward_f32(q.data(), k.data(), v.data(), heads, T, d, out.data());
        naive_attention_f32(q.data(), k.data(), v.data(), heads, T, d, ref.data());
        float max_err = 0.0f;
        for (std::size_t i = 0; i < out.size(); ++i)
            max_err = std::max(max_err, std::fabs(out[i] - ref[i]));
        STELL_CHECK_MSG(max_err < 1e-4f, "replay: ядро совпадает с независимым референсом");
    }

    // --- layout-реплей: перестановка голов не меняет результат per-head -----
    {
        const std::uint32_t heads = 3, T = 4, d = 4;
        std::vector<float> q(heads * T * d), k(heads * T * d), v(heads * T * d);
        fill_lcg(q, 11);
        fill_lcg(k, 22);
        fill_lcg(v, 33);
        // перестановка голов: [0,1,2] → [2,0,1]
        const int perm[3] = {2, 0, 1};
        std::vector<float> qp(heads * T * d), kp(heads * T * d), vp(heads * T * d);
        for (int h = 0; h < 3; ++h) {
            std::memcpy(qp.data() + perm[h] * T * d, q.data() + h * T * d, T * d * sizeof(float));
            std::memcpy(kp.data() + perm[h] * T * d, k.data() + h * T * d, T * d * sizeof(float));
            std::memcpy(vp.data() + perm[h] * T * d, v.data() + h * T * d, T * d * sizeof(float));
        }
        std::vector<float> out(heads * T * d), outp(heads * T * d);
        attention_forward_f32(q.data(), k.data(), v.data(), heads, T, d, out.data());
        attention_forward_f32(qp.data(), kp.data(), vp.data(), heads, T, d, outp.data());
        // outp[perm[h]] == out[h]
        for (int h = 0; h < 3; ++h)
            for (std::size_t i = 0; i < T * d; ++i)
                STELL_CHECK_MSG(close(outp[perm[h] * T * d + i], out[h * T * d + i]),
                                "layout-реплей: результат per-head инвариантен к позиции головы");
    }

    // --- decode-форма (round 2): одна query-строка против кеша (ADR-04) ------
    {
        const std::uint32_t heads = 4, kv_heads = 2, T = 5, d = 4;
        const float scale = 1.0f / std::sqrt(static_cast<float>(d));
        std::vector<float> kc(static_cast<std::size_t>(kv_heads) * T * d),
                           vc(kc.size()), q(heads * d), out(heads * d);
        fill_lcg(kc, 404);
        fill_lcg(vc, 505);
        fill_lcg(q, 606);
        attention_decode_f32(q.data(), kc.data(), vc.data(), heads, kv_heads, T, d,
                             out.data());
        // независимый референс: GQA-мапинг kvh = h / (heads/kv_heads)
        const std::uint32_t per_kv = heads / kv_heads;
        float max_err = 0.0f;
        for (std::uint32_t h = 0; h < heads; ++h) {
            const std::uint32_t kvh = h / per_kv;
            float probs[8];
            float mx = -1e30f;
            for (std::uint32_t j = 0; j < T; ++j) {
                float acc = 0.0f;
                for (std::uint32_t dd = 0; dd < d; ++dd)
                    acc += q[h * d + dd] * kc[kvh * T * d + j * d + dd];
                probs[j] = acc * scale;
                mx = std::max(mx, probs[j]);
            }
            float sum = 0.0f;
            for (std::uint32_t j = 0; j < T; ++j) { probs[j] = std::exp(probs[j] - mx); sum += probs[j]; }
            for (std::uint32_t dd = 0; dd < d; ++dd) {
                float acc = 0.0f;
                for (std::uint32_t j = 0; j < T; ++j)
                    acc += (probs[j] / sum) * vc[kvh * T * d + j * d + dd];
                max_err = std::max(max_err, std::fabs(out[h * d + dd] - acc));
            }
        }
        STELL_CHECK_MSG(max_err < 1e-5f, "decode-форма: совпадение с референсом GQA");
    }
    // вырожденный случай: kv_heads == heads (без GQA) — decode == forward-строка
    {
        const std::uint32_t heads = 2, T = 3, d = 2;
        std::vector<float> kc(heads * T * d), vc(kc.size());
        std::vector<float> q(heads * d), out(heads * d);
        fill_lcg(kc, 707);
        fill_lcg(vc, 808);
        fill_lcg(q, 909);
        attention_decode_f32(q.data(), kc.data(), vc.data(), heads, heads, T, d,
                             out.data());
        // полная форма: q-строки все равны текущей → последняя строка out == decode
        // (раскладка kernels: [heads][T][d])
        std::vector<float> qf(heads * T * d, 0.0f), ff(heads * T * d, 0.0f);
        for (std::uint32_t h = 0; h < heads; ++h)
            for (std::uint32_t t = 0; t < T; ++t)
                std::memcpy(qf.data() + (h * T + t) * d, q.data() + h * d,
                            d * sizeof(float));
        attention_forward_f32(qf.data(), kc.data(), vc.data(), heads, T, d, ff.data());
        for (std::uint32_t h = 0; h < heads; ++h)
            for (std::uint32_t dd = 0; dd < d; ++dd)
                STELL_CHECK_MSG(close(out[h * d + dd],
                                      ff[h * T * d + (T - 1) * d + dd]),
                                "decode == последняя строка полной формы");
    }

    // --- softmax rows: сумма строк = 1 --------------------------------------
    {
        float s[6];
        for (int i = 0; i < 6; ++i) s[i] = static_cast<float>(i) * 0.7f - 1.5f;
        softmax_rows_f32(s, 2, 3);
        STELL_CHECK(close(s[0] + s[1] + s[2], 1.0f));
        STELL_CHECK(close(s[3] + s[4] + s[5], 1.0f));
        // максимум строки — самый большой элемент (row0: -1.5,-0.8,-0.1; row1: 0.6,1.3,2.0)
        STELL_CHECK(s[2] > s[0] && s[2] > s[1]);
        STELL_CHECK(s[5] > s[3] && s[5] > s[4]);
    }
}

STELL_TEST_MAIN(run)

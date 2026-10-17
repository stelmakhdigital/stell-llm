// Replay-тест драфтер-ядер: argmax + top-k (softmax считается в тесте
// независимо, в double).
#include "test_util.hpp"

#include <cmath>
#include <vector>

#include "replay_fixture.hpp"
#include "stell/kernels/draft.hpp"

using namespace stell::kernels;
using stell::kernels::test::close;

void run() {
    // --- argmax --------------------------------------------------------------
    {
        const float l[5] = {0.1f, 3.0f, 2.0f, 3.0f, -1.0f};
        STELL_CHECK(argmax_f32(l, 5) == 1);       // первый максимум при равенстве
    }

    // --- top-k: softmax и порядок независимым расчётом -----------------------
    {
        const float l[4] = {1.0f, 2.0f, 0.5f, -1.0f};
        std::int32_t ids[4];
        float probs[4];
        draft_topk_f32(l, 4, 4, ids, probs);

        double ex[4];
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) { ex[i] = std::exp(l[i]); sum += ex[i]; }
        // порядок: 1 (e^2), 0 (e^1), 2 (e^0.5), 3 (e^-1)
        const int order[4] = {1, 0, 2, 3};
        for (int j = 0; j < 4; ++j) {
            STELL_CHECK(ids[j] == order[j]);
            STELL_CHECK(close(probs[j], static_cast<float>(ex[order[j]] / sum), 1e-6f));
        }
    }

    // --- top-k k < n: отбор трёх из пяти -------------------------------------
    {
        const float l[5] = {0.1f, 3.0f, 2.5f, 3.0f, -1.0f};
        std::int32_t ids[3];
        float probs[3];
        draft_topk_f32(l, 5, 3, ids, probs);
        // порядок: 1 (3.0), 3 (3.0 — позже индекс при равенстве), 2 (2.5)
        STELL_CHECK(ids[0] == 1);
        STELL_CHECK(ids[1] == 3);
        STELL_CHECK(ids[2] == 2);
        double sum = 0.0;
        for (int i = 0; i < 5; ++i) sum += std::exp(l[i]);
        STELL_CHECK(close(probs[0], static_cast<float>(std::exp(3.0) / sum), 1e-6f));
    }
}

STELL_TEST_MAIN(run)

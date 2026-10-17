// Replay-тест MoE gather/scatter: hand-checked фикстура + LCG-сравнение
// с независимым наивным референсом.
#include "test_util.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "replay_fixture.hpp"
#include "stell/kernels/moe_gather_scatter.hpp"

using namespace stell::kernels;
using stell::kernels::test::close;
using stell::kernels::test::fill_lcg;

void run() {
    // --- hand-checked: 2 токена, k=2, d=2 -----------------------------------
    {
        // x: два токена
        const float x[4] = {1.0f, 2.0f, 10.0f, 20.0f};
        // роутер: токен 0 → эксперты (3, 7), токен 1 → (1, 5)
        const std::int32_t ids[4] = {3, 7, 1, 5};
        const float w[4] = {0.6f, 0.4f, 0.9f, 0.1f};
        const std::uint32_t kt = 2, dm = 2, nt = 2;

        float gathered[nt * kt * dm];
        moe_gather(x, ids, nt, kt, dm, gathered);
        // строки t*kt+j — копии x[t]
        STELL_CHECK(close(gathered[0], 1.0f));
        STELL_CHECK(close(gathered[1], 2.0f));
        STELL_CHECK(close(gathered[2], 1.0f));
        STELL_CHECK(close(gathered[3], 2.0f));
        STELL_CHECK(close(gathered[4], 10.0f));
        STELL_CHECK(close(gathered[5], 20.0f));
        STELL_CHECK(close(gathered[6], 10.0f));
        STELL_CHECK(close(gathered[7], 20.0f));

        // expert_out: произвольные значения (после FFN)
        float eo[nt * kt * dm];
        for (std::size_t i = 0; i < nt * kt * dm; ++i) eo[i] = 1.0f + 0.1f * i;

        float y[nt * dm] = {0, 0, 0, 0};
        moe_scatter_add(eo, ids, w, nt, kt, dm, y);
        // y[t] = sum_j w[t,j] * eo[t*kt+j]
        float ref[nt * dm] = {0};
        for (std::uint32_t t = 0; t < nt; ++t)
            for (std::uint32_t j = 0; j < kt; ++j)
                for (std::uint32_t d = 0; d < dm; ++d)
                    ref[t * dm + d] += w[t * kt + j] * eo[(t * kt + j) * dm + d];
        for (std::size_t i = 0; i < nt * dm; ++i)
            STELL_CHECK_MSG(close(y[i], ref[i]), "scatter-add соответствует референсу");

        // аккумуляция: повторный scatter-add удваивает
        moe_scatter_add(eo, ids, w, nt, kt, dm, y);
        for (std::size_t i = 0; i < nt * dm; ++i)
            STELL_CHECK(close(y[i], 2.0f * ref[i]));
    }

    // --- LCG-фикстура против независимого референса --------------------------
    {
        const std::uint32_t nt = 7, kt = 3, dm = 5;
        std::vector<float> x(nt * dm), eo(nt * kt * dm), w(nt * kt);
        std::vector<std::int32_t> ids(nt * kt);
        fill_lcg(x, 1);
        fill_lcg(eo, 2);
        fill_lcg(w, 3);
        // expert_ids: детерминированные значения в допустимом диапазоне
        {
            test::Lcg g(4);
            for (auto& id : ids) id = static_cast<std::int32_t>(g.next_u32() % 128);
        }

        std::vector<float> gathered(nt * kt * dm);
        moe_gather(x.data(), ids.data(), nt, kt, dm, gathered.data());
        for (std::uint32_t t = 0; t < nt; ++t)
            for (std::uint32_t j = 0; j < kt; ++j)
                for (std::uint32_t d = 0; d < dm; ++d)
                    STELL_CHECK(close(gathered[(t * kt + j) * dm + d], x[t * dm + d]));

        std::vector<float> y(nt * dm, 0.0f);
        moe_scatter_add(eo.data(), ids.data(), w.data(), nt, kt, dm, y.data());
        float max_err = 0.0f;
        for (std::uint32_t t = 0; t < nt; ++t)
            for (std::uint32_t d = 0; d < dm; ++d) {
                float ref = 0.0f;
                for (std::uint32_t j = 0; j < kt; ++j)
                    ref += w[t * kt + j] * eo[(t * kt + j) * dm + d];
                max_err = std::max(max_err, std::fabs(y[t * dm + d] - ref));
            }
        STELL_CHECK_MSG(max_err < 1e-5f, "LCG-replay scatter-add");
    }
}

STELL_TEST_MAIN(run)

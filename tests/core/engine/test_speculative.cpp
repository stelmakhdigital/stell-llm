// Спекулятивный контур: драфт + FP32-верификация + автоотключение по калибровке.
#include "test_util.hpp"

#include <vector>

#include "stell/engine/speculative.hpp"

using namespace stell::engine::spec;

namespace {

class FakeDraft : public IDraftModel {
public:
    std::vector<std::int32_t>
    propose(std::uint32_t slot, std::int32_t last_token, std::uint32_t n) override {
        (void)slot; (void)last_token;
        std::vector<std::int32_t> out;
        for (std::uint32_t i = 0; i < n; ++i) out.push_back(100 + i);
        return out;
    }
};

// Верификатор принимает первые `accept` драфт-токенов (совпадение с FP32
// argmax цели), возвращает собственный следующий токен цели.
class FakeVerifier : public ITargetVerifier {
public:
    VerifyResult verify(std::uint32_t slot,
                        const std::vector<std::int32_t>& draft) override {
        (void)slot;
        VerifyResult r{};
        std::size_t n = std::min(accept, draft.size());
        r.n_accepted = n;
        r.next_token = static_cast<std::int32_t>(n) + 900;
        return r;
    }
    std::size_t accept = 0;
};

} // namespace

void run() {
    SpecConfig cfg;
    cfg.enabled = true;
    cfg.n_draft = 4;
    cfg.accept_target = 0.5;
    cfg.window = 32;
    cfg.warmup_steps = 4;

    FakeDraft draft;
    FakeVerifier verifier;
    verifier.accept = 2;                      // 2/4 = 50% — ровно на пороге

    SpeculativeController ctrl(cfg, draft, verifier);
    STELL_CHECK(ctrl.enabled());

    // шаг спекуляции: принятый префикс + токен цели (FP32)
    auto out = ctrl.step(0, 7);
    STELL_CHECK(out.spec_ran);
    STELL_CHECK(out.accepted == 2);
    STELL_CHECK(out.token == 902);

    // включённый конфиг, но accept=0 → автоотключение после warmup
    FakeVerifier verifier2;
    verifier2.accept = 0;
    SpeculativeController ctrl2(cfg, draft, verifier2);
    for (int i = 0; i < 8; ++i) ctrl2.step(0, 1);
    STELL_CHECK_MSG(!ctrl2.enabled(), "accept-rate 0 ниже порога — контур обязан выключиться");
    // после отключения шаг не спекулируется
    auto out2 = ctrl2.step(0, 1);
    STELL_CHECK(!out2.spec_ran);

    // accept-rate до порога в пределах warmup — не карается
    SpecConfig cfg3 = cfg;
    cfg3.warmup_steps = 100;
    SpeculativeController ctrl3(cfg3, draft, verifier2);
    for (int i = 0; i < 8; ++i) ctrl3.step(0, 1);
    STELL_CHECK(ctrl3.enabled());

    // --- калибровка: арбитр включается/выключает (PLAN.md §2.4) -------------
    SpeculativeController ctrl4(cfg, draft, verifier2);
    ctrl4.on_calibration(0.4);                // ниже порога
    STELL_CHECK(!ctrl4.enabled());
    ctrl4.on_calibration(0.65);               // выше порога+гистерезис
    STELL_CHECK(ctrl4.enabled());
    ctrl4.on_calibration(0.55);               // зона неопределённости — статус не меняем
    STELL_CHECK(ctrl4.enabled());
    ctrl4.on_calibration(0.1);
    STELL_CHECK(!ctrl4.enabled());

    // acceptance_rate считается по окну
    SpecConfig cfg5 = cfg;
    cfg5.enabled = true;
    SpeculativeController ctrl5(cfg5, draft, verifier);
    // verifier.accept = 2 из 4 → 0.5
    (void)ctrl5.step(0, 1);
    STELL_CHECK(std::abs(ctrl5.acceptance_rate() - 0.5) < 1e-9);
}

STELL_TEST_MAIN(run)

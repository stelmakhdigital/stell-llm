// DecodeExecutor: CUDA Graph по константной разметке + eager-фолббек (ADR-05).
// CPU-исполнитель ReplayDecodeExecutor фиксирует семантику: replay — только
// при совпадении тега разметки; расхождение → eager + сброс графа.
#include "test_util.hpp"

#include <vector>

#include "stell/engine/decode_executor.hpp"
#include "stell/engine/scheduler.hpp"

using namespace stell::engine;
using DecodeInput = IModelRunner::DecodeInput;
using DecodeOutput = IModelRunner::DecodeOutput;

namespace {

class CountingRunner : public IModelRunner {
public:
    std::vector<DecodeOutput> decode_step(const std::vector<DecodeInput>& in) override {
        ++calls;
        std::vector<DecodeOutput> out;
        for (const auto& i : in) out.push_back({i.slot, 42});
        return out;
    }
    int calls = 0;
};

} // namespace

void run() {
    CountingRunner runner;
    ReplayDecodeExecutor exec;

    // граф ещё не снят → первый вызов = eager
    exec.set_current_layout(7, 4096);
    std::vector<DecodeInput> batch{{0, 1}, {1, 2}, {2, 3}};
    exec.execute(batch, runner);
    STELL_CHECK(exec.eager_fallbacks() == 1);
    STELL_CHECK(exec.replays() == 0);
    STELL_CHECK(!exec.graph_active());

    // capture под константную разметку (ADR-05): слоты фиксированного размера
    GraphLayoutTag tag{};
    tag.layout_version = 7;
    tag.slot_bytes = 4096;
    tag.max_batch = 8;
    STELL_CHECK(exec.try_capture(tag));
    STELL_CHECK(exec.graph_active());

    // replay: разметка совпадает
    exec.set_current_layout(7, 4096);
    exec.execute(batch, runner);
    STELL_CHECK(exec.replays() == 1);
    STELL_CHECK(exec.eager_fallbacks() == 1);

    // батч больше max_batch графа → eager-фолббек
    std::vector<DecodeInput> big{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
                                 {5, 6}, {6, 7}, {7, 8}, {8, 9}};
    exec.execute(big, runner);
    STELL_CHECK(exec.eager_fallbacks() == 2);
    STELL_CHECK(!exec.graph_active());        // разметка «уехала» — граф сброшен

    // повторный capture — пере-разметка = переснятие (ADR-05/ADR-08)
    tag.max_batch = 16;
    STELL_CHECK(exec.try_capture(tag));
    exec.set_current_layout(7, 4096);
    exec.execute(big, runner);
    STELL_CHECK(exec.replays() == 2);

    // смена layout_version → eager + сброс
    exec.set_current_layout(8, 4096);
    exec.execute(batch, runner);
    STELL_CHECK(exec.eager_fallbacks() == 3);
    STELL_CHECK(!exec.graph_active());

    exec.release_graph();
    STELL_CHECK(!exec.graph_active());
}

STELL_TEST_MAIN(run)

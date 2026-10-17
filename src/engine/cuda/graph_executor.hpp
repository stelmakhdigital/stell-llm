// CUDA-исходники пишутся и проверяются ТОЛЬКО на целевом сервере
// (ssh -p 2214 192.168.1.114, SM70, cuda 12.8) — как и cuda/ ядра kernels.
#pragma once

#include <cstddef>
#include <cstdint>

#include "stell/engine/decode_executor.hpp"

namespace stell::engine::cuda {

// Исполнитель шага декода поверх cudaGraph (ADR-05): capture на константной
// разметке (stream capture), replay через cudaGraphLaunch, eager-фолббек —
// прямые запуски ядер на stream при расхождении разметки.
class GraphDecodeExecutor : public IDecodeExecutor {
public:
    GraphDecodeExecutor();
    ~GraphDecodeExecutor() override;

    bool try_capture(const GraphLayoutTag& tag) override;
    bool graph_active() const override;
    void execute(const std::vector<IModelRunner::DecodeInput>& batch,
                 IModelRunner& runner) override;
    void release_graph() override;

    // Демо-оп графа (round 1): масштабирование слот-скрэтча на константу —
    // ядро-плейсхолдер до появления ops-каталога моделей (ADR-09 §1 ops).
    void set_demo_op(float scale);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    GraphLayoutTag captured_{};
    bool active_ = false;
    float demo_scale_ = 1.0f;
};

} // namespace stell::engine::cuda

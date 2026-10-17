// CUDA Graph для decode: константная разметка + eager-фолббек (ADR-05).
// Stream capture → cudaGraph → instantiate → replay; расхождение тега
// разметки (slot_bytes / layout_version / max_batch) → eager-путь.
// ВАЖНО: файл не компилировался локально (нет nvcc) — верификация на сервере.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "graph_executor.hpp"

namespace stell::engine::cuda {

namespace {

// Плейсхолдер-ядро демо-операции графа (масштабирование слот-скрэтча);
// заменяется ops-каталогом моделей (ADR-09 §1) на следующей итерации.
__global__ void demo_scale_kernel(float* buf, int n, float k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] *= k;
}

} // namespace

struct GraphDecodeExecutor::Impl {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t stream = nullptr;
    float* scratch = nullptr;      // device-буфер слота-скрэтча (демо-оп)
    std::size_t scratch_elems = 0;
    float scale = 1.0f;

    ~Impl() {
        if (scratch) cudaFree(scratch);
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
    }
};

GraphDecodeExecutor::GraphDecodeExecutor() = default;

GraphDecodeExecutor::~GraphDecodeExecutor() {
    delete impl_;
}

void GraphDecodeExecutor::set_demo_op(float scale) { demo_scale_ = scale; }

bool GraphDecodeExecutor::try_capture(const GraphLayoutTag& tag) {
    delete impl_;
    impl_ = new Impl{};
    captured_ = tag;

    if (cudaStreamCreate(&impl_->stream) != cudaSuccess) { release_graph(); return false; }

    // слот-скрэтч фиксированного размера (ADR-05: slot_bytes ДО графа)
    impl_->scratch_elems = tag.slot_bytes / sizeof(float);
    if (cudaMalloc((void**)&impl_->scratch, tag.slot_bytes) != cudaSuccess) {
        release_graph();
        return false;
    }
    if (cudaMemset(impl_->scratch, 0, tag.slot_bytes) != cudaSuccess) {
        release_graph();
        return false;
    }

    // stream capture: демо-оп над слот-скрэтчем (константная разметка)
    if (cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
        release_graph();
        return false;
    }

    const int bs = 256;
    const int blocks = (int)((impl_->scratch_elems + bs - 1) / bs);
    const int n = (int)impl_->scratch_elems;
    const float k = demo_scale_;
    float* buf = impl_->scratch;
    demo_scale_kernel<<<blocks, bs, 0, impl_->stream>>>(buf, n, k);

    if (cudaStreamEndCapture(impl_->stream, &impl_->graph) != cudaSuccess) {
        release_graph();
        return false;
    }
    if (cudaGraphInstantiate(&impl_->graph_exec, impl_->graph, 0) != cudaSuccess) {
        release_graph();
        return false;
    }

    active_ = true;
    return true;
}

bool GraphDecodeExecutor::graph_active() const { return active_; }

void GraphDecodeExecutor::execute(const std::vector<IModelRunner::DecodeInput>& batch,
                                  IModelRunner& runner) {
    // Разметка шага: batch в пределах max_batch графа. Слот-скрэтч CUDA
    // версии резервируется под max_batch слотов графа (константная разметка);
    // расхождение → eager-фолббек (ADR-05).
    const bool layout_ok = active_
        && batch.size() <= static_cast<std::size_t>(captured_.max_batch);

    if (layout_ok) {
        // replay: cudaGraphLaunch на stream (ADR-05)
        cudaGraphLaunch(impl_->graph_exec, impl_->stream);
        cudaError_t err = cudaStreamSynchronize(impl_->stream);
        if (err == cudaSuccess) {
            ++replays_;
            (void)runner;
            return;
        }
        // сбой исполнения графа — деградация в eager (не тишина)
        std::fprintf(stderr, "[engine.cuda] graph launch failed: %d\n", (int)err);
    }

    // eager-фолббек: прямые запуски на stream (или CPU-путь через runner
    // в смешанной конфигурации) — счётчик для калибровки/дашборда.
    ++eager_fallbacks_;
    active_ = false;
    release_graph();
    (void)runner.decode_step(batch);
}

void GraphDecodeExecutor::release_graph() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
    active_ = false;
}

} // namespace stell::engine::cuda

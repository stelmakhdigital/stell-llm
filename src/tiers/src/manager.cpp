// Реализация TierManager (ADR-08): VRAM-slots → pinned RAM → NVMe(io_uring),
// политика LRU/LCU, router-lookahead prefetch, метрики.
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "nvme_file.hpp"

#if defined(STELL_CUDA)
#include <cuda_runtime.h>
#endif

namespace stell::tiers {

namespace {
double now_ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
}

// Буферы RAM-тира: при STELL_CUDA — pinned (cudaHostAlloc) как источник
// cudaMemcpyAsync DMA; CPU-эмуляция — обычная heap-память.
#if defined(STELL_CUDA)
std::uint8_t* alloc_ram_buffer(std::size_t n) {
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, n, cudaHostAllocDefault);
    if (e != cudaSuccess)
        throw TierError(TierError::Code::IoFailure,
                        std::string("tiers: cudaHostAlloc RAM-тира failed: ") +
                            cudaGetErrorString(e));
    return static_cast<std::uint8_t*>(p);
}
void free_ram_buffer(std::uint8_t* p) { if (p) cudaFreeHost(p); }
#else
// [R2a] round 4: RAM-буферы менеджера выровнены на 4096 (posix_memalign):
// O_DIRECT-чтение слота идёт ПРЯМО в RAM-буфер (без staging-копии).
// cudaHostAlloc (ветка выше) page-aligned — то же свойство, зафиксировано.
std::uint8_t* alloc_ram_buffer(std::size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 4096, n) != 0)
        throw TierError(TierError::Code::IoFailure,
                        "tiers: posix_memalign RAM-тира failed");
    return static_cast<std::uint8_t*>(p);
}
void free_ram_buffer(std::uint8_t* p) { std::free(p); }
#endif

// R1 (ADR-12 §2.3, round 3): env-флаг поверх конфига — итог = config || env.
bool env_verify_crc() {
    const char* v = std::getenv("STELL_TIER_VERIFY_CRC");
    return v && std::strcmp(v, "1") == 0; // явное включение только "1"
}

// R7: чтение size-t параметра окружения; нечисловое/пустое → базовое значение.
std::size_t env_size_or(const char* name, std::size_t dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    const unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v) return dflt;
    return static_cast<std::size_t>(x);
}

// [R2a] round 4: env-флаг O_DIRECT поверх конфига — итог = config || env
// (та же OR-семантика, что у verify_crc).
bool env_odirect() {
    const char* v = std::getenv("STELL_TIER_ODIRECT");
    return v && std::strcmp(v, "1") == 0; // явное включение только "1"
}
} // namespace

// --- TierAwaiter -------------------------------------------------------------
TierAwaiter::TierAwaiter() : state_(std::make_shared<State>()) {}

bool TierAwaiter::ready() const { return state_->done.load(std::memory_order_acquire); }

void TierAwaiter::wait() {
    std::unique_lock<std::mutex> lk(state_->mu);
    state_->cv.wait(lk, [&] { return state_->done.load(std::memory_order_acquire); });
    if (state_->err) std::rethrow_exception(state_->err);
}

bool TierAwaiter::wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(state_->mu);
    const bool ok = state_->cv.wait_for(lk, timeout,
        [&] { return state_->done.load(std::memory_order_acquire); });
    if (!ok) return false;
    if (state_->err) std::rethrow_exception(state_->err);
    return true;
}

void TierAwaiter::complete() {
    std::lock_guard<std::mutex> lk(state_->mu);
    state_->done.store(true, std::memory_order_release);
    state_->cv.notify_all();
}

void TierAwaiter::fail(std::exception_ptr err) {
    std::lock_guard<std::mutex> lk(state_->mu);
    state_->err = std::move(err);
    state_->done.store(true, std::memory_order_release);
    state_->cv.notify_all();
}

// Fan-in слоя: готов, когда готовы все части; wait() меряет паузу слоя.
class TierCompositeAwaiter final : public Awaiter {
public:
    TierCompositeAwaiter(std::vector<std::shared_ptr<TierAwaiter>> parts,
                         std::function<void(double)> pause_sink)
        : parts_(std::move(parts)), pause_sink_(std::move(pause_sink)) {}

    bool ready() const override {
        for (const auto& p : parts_)
            if (!p->ready()) return false;
        return true;
    }

    void wait() override {
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& p : parts_) p->wait();
        if (pause_sink_) pause_sink_(now_ms_since(t0));
    }

private:
    std::vector<std::shared_ptr<TierAwaiter>> parts_;
    std::function<void(double)> pause_sink_;
};

std::unique_ptr<Awaiter> TierAwaiter::all(
    std::vector<std::shared_ptr<TierAwaiter>> parts, std::function<void(double)> pause_sink) {
    return std::make_unique<TierCompositeAwaiter>(std::move(parts), std::move(pause_sink));
}

// --- TierManager -------------------------------------------------------------
TierManager::TierManager(TierConfig cfg)
    : cfg_(cfg),
      verify_crc_(cfg.verify_crc || env_verify_crc()), // R1: config || env
      policy_(cfg.policy), predictor_(cfg.history_cap), lookahead_(cfg.lookahead),
      batch_depth_(cfg.deep_prefetch_depth ? cfg.deep_prefetch_depth : 1), // [R2b]: k в полёте, мин. 1
      odirect_(cfg.odirect || env_odirect()) { // [R2a]: config || env (OR-семантика R1)
    if (cfg_.vram_slots == 0 || cfg_.ram_capacity == 0)
        throw TierError(TierError::Code::PlanMismatch, "tiers: vram_slots/ram_capacity must be > 0");
#if defined(STELL_CUDA)
    // Выделенный DMA-стрим: RAM(pinned)→VRAM(device) копирования не
    // смешиваются с legacy-стримом (ADR-08: Awaiter — fan-in io_uring/DMA).
    {
        cudaStream_t s = nullptr;
        const cudaError_t e = cudaStreamCreate(&s);
        if (e != cudaSuccess)
            throw TierError(TierError::Code::IoFailure,
                            std::string("tiers: cudaStreamCreate failed: ") +
                                cudaGetErrorString(e));
        dma_stream_ = s;
    }
#endif
    workers_.emplace_back([this] { worker_loop(); });
}

TierManager::~TierManager() {
    {
        std::lock_guard<std::mutex> lk(qmu_);
        shutdown_ = true;
    }
    qcv_.notify_all();
    for (auto& w : workers_) if (w.joinable()) w.join();
    // Устройство освобождаем только после join воркеров (DMA мог быть в полёте).
    for (auto& kv : ram_) free_ram_buffer(kv.second);
    ram_.clear();
#if defined(STELL_CUDA)
    if (dev_storage_) { cudaFree(dev_storage_); dev_storage_ = nullptr; }
    if (dma_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(dma_stream_));
        dma_stream_ = nullptr;
    }
#endif
}

GroupState* TierManager::find(GroupId id) {
    auto it = states_.find(id.value);
    return it == states_.end() ? nullptr : &it->second;
}
const GroupState* TierManager::find(GroupId id) const {
    auto it = states_.find(id.value);
    return it == states_.end() ? nullptr : &it->second;
}
const TensorGroup* TierManager::plan(GroupId id) const {
    for (const auto& g : plan_)
        if (g.id == id) return &g;
    return nullptr;
}

bool TierManager::resident_in_vram(GroupId id) const {
    std::lock_guard<std::mutex> lk(mu_);
    const GroupState* st = find(id);
    return st && st->in_vram;
}

// Расширение R8 (см. tier_manager.hpp): чтение слота группы из RAM-тира.
void TierManager::read_ram_slot(GroupId id, void* dst, std::size_t dst_bytes) const {
    std::lock_guard<std::mutex> lk(mu_);
    const GroupState* st = find(id);
    if (!st)
        throw TierError(TierError::Code::PlanMismatch,
                        "read_ram_slot: unknown GroupId");
    if (dst_bytes < slot_size_)
        throw TierError(TierError::Code::PlanMismatch,
                        "read_ram_slot: dst_bytes < slot_bytes плана");
    const auto it = ram_.find(id.value);
    if (it == ram_.end() || it->second == nullptr)
        throw TierError(TierError::Code::IoFailure,
                        "read_ram_slot: RAM-копии нет (pin→wait до чтения; "
                        "либо RAM-копия вытеснена при VRAM-resident)");
    std::memcpy(dst, it->second, slot_size_);
}

// [R5] round 4: чтение содержимого VRAM-слота группы (побитовая сверка с
// эталоном прямого лоадера — инвариант «одна группа = один слот», класс
// aliasing-багов ba42267). CPU-эмуляция: слот в vram_storage_ (тот же кодовый
// путь, что и VRAM в CPU-конфигурации); STELL_CUDA: cudaMemcpy D2H из
// device-слота.
void TierManager::read_vram_slot(GroupId id, void* dst, std::size_t dst_bytes) const {
    std::int32_t slot = -1;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const GroupState* st = find(id);
        if (!st)
            throw TierError(TierError::Code::PlanMismatch,
                            "read_vram_slot: unknown GroupId");
        if (dst_bytes < slot_size_)
            throw TierError(TierError::Code::PlanMismatch,
                            "read_vram_slot: dst_bytes < slot_bytes плана");
        if (!st->in_vram || st->slot < 0)
            throw TierError(TierError::Code::IoFailure,
                            "read_vram_slot: группа не resident в VRAM (pin→wait до чтения)");
        slot = st->slot;
#if !defined(STELL_CUDA)
        // CPU-эмуляция: слот в host-памяти (тот же кодовый путь, что и VRAM
        // в CPU-конфигурации) — копия под mu_ (защита от гонки с eviction).
        std::memcpy(dst, vram_slots_[(std::size_t)slot], slot_size_);
#endif
    }
#if defined(STELL_CUDA)
    // D2H вне mu_ (синхронный cudaMemcpy — не горячий путь).
    const cudaError_t ce = cudaMemcpy(dst, vram_slots_[(std::size_t)slot],
                                      slot_size_, cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess)
        throw TierError(TierError::Code::IoFailure,
                        std::string("read_vram_slot: cudaMemcpy D2H failed: ") +
                            cudaGetErrorString(ce));
#else
    (void)slot;
#endif
}

void TierManager::set_plan(const std::vector<TensorGroup>& groups) {
    // Безопасный ResetPlan: воркер мог взять job из очереди (queue пуста,
    // а process_job ещё идёт) — дожидаемся дренажа in-flight job'ов ДО mu_
    // (воркеру нужен mu_ на завершении job'а; см. ADR-08 §4 «холодная»
    // дисциплина — одновременный вызов с рабочими методами недопустим).
    for (int spin = 0; active_jobs_.load(std::memory_order_acquire) != 0; ++spin) {
        if (spin > 30000) // ~30 с: протокол-нарушение/залипший job
            throw TierError(TierError::Code::PlanMismatch,
                            "set_plan: reset while load jobs in-flight");
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (plan_active_) {
        // Повторный вызов = ResetPlan; одновременный вызов с рабочими методами
        // недопустим по контракту (ADR-08 §4) — проверяем по возможности.
        for (const auto& kv : states_)
            if (kv.second.pinned > 0 || !kv.second.waiters.empty())
                throw TierError(TierError::Code::PlanMismatch,
                                "set_plan: reset while groups pinned/in-flight");
        {
            std::lock_guard<std::mutex> qk(qmu_);
            if (!queue_.empty())
                throw TierError(TierError::Code::PlanMismatch,
                                "set_plan: reset while load jobs in-flight");
        }
    }
    if (groups.empty())
        throw TierError(TierError::Code::PlanMismatch, "set_plan: empty group plan");
    for (const auto& g : groups) {
        if (g.slot_bytes == 0)
            throw TierError(TierError::Code::PlanMismatch, "set_plan: slot_bytes == 0");
        for (const auto& other : groups)
            if (&other != &g && other.id == g.id)
                throw TierError(TierError::Code::PlanMismatch, "set_plan: duplicate GroupId");
    }

    std::size_t max_slot = cfg_.vram_slot_size;
    if (max_slot == 0) {
        max_slot = 0;
        for (const auto& g : groups) max_slot = std::max(max_slot, g.slot_bytes);
    }
    for (const auto& g : groups)
        if (g.slot_bytes > max_slot)
            throw TierError(TierError::Code::PlanMismatch,
                            "set_plan: slot_bytes exceeds vram_slot_size");

    plan_ = groups;
    std::sort(plan_.begin(), plan_.end(),
              [](const TensorGroup& a, const TensorGroup& b) { return a.id.value < b.id.value; });
    // NVMe-тир: открытие layout и сверка с планом (ADR-12 §2.2/§4).
    nvme_.reset();
    if (!cfg_.nvme_layout.empty()) {
        // R1: флаг verify_crc_ пробрасывается в NVMe-тир (payload-CRC при
        // pin/read — только при явном флаге; ADR-12 §2.3, round 3).
        // [R2a] round 4: флаг odirect_ пробрасывается вниз (O_DIRECT-тракт
        // данных; при неудаче открытия — честный откат на buffered,
        // TierNvmeFile::odirect_active() == false).
        auto file = std::make_unique<TierNvmeFile>(cfg_.nvme_layout, verify_crc_, odirect_);
        for (const auto& g : plan_) {
            const auto* e = file->find(g.id.value);
            if (!e)
                throw TierError(TierError::Code::PlanMismatch,
                                "set_plan: group missing in stell-tier layout");
            if (e->slot_bytes != g.slot_bytes)
                throw TierError(TierError::Code::PlanMismatch,
                                "set_plan: slot_bytes mismatch vs layout (ADR-12 §2.2)");
            if (e->dtype != dtype_code(g.dtype))
                throw TierError(TierError::Code::PlanMismatch,
                                "set_plan: dtype mismatch vs layout");
            if (e->role != role_code(g.role))
                throw TierError(TierError::Code::PlanMismatch,
                                "set_plan: role mismatch vs layout");
        }
        nvme_ = std::move(file);
    }

    slot_size_ = max_slot;
    states_.clear();
    for (auto& kv : ram_) free_ram_buffer(kv.second);
    ram_.clear();
    layers_.clear();
    tick_ = 0;
    last_accessed_ = GroupId{0};
    for (const auto& g : plan_) states_[g.id.value] = GroupState{};

    // VRAM-тир: N слотов фиксированного размера.
#if defined(STELL_CUDA)
    // CUDA/DMA-путь: VRAM-слоты — device-буферы (cudaMalloc-пул); DMA
    // RAM→VRAM — cudaMemcpyAsync на выделенном стриме (см. process_job).
    if (dev_storage_) { cudaFree(dev_storage_); dev_storage_ = nullptr; }
    {
        void* dev = nullptr;
        const cudaError_t e = cudaMalloc(&dev, cfg_.vram_slots * slot_size_);
        if (e != cudaSuccess)
            throw TierError(TierError::Code::IoFailure,
                            std::string("tiers: cudaMalloc VRAM-слотов failed: ") +
                                cudaGetErrorString(e));
        dev_storage_ = static_cast<std::uint8_t*>(dev);
    }
    vram_storage_.clear(); // host-эмуляция не используется (device ptr ниже)
    vram_slots_.assign(cfg_.vram_slots, nullptr);
    for (std::size_t i = 0; i < cfg_.vram_slots; ++i)
        vram_slots_[i] = dev_storage_ + i * slot_size_;
#else
    // CPU-эмуляция VRAM-тира: host-память.
    vram_storage_.assign(cfg_.vram_slots * slot_size_, 0);
    vram_slots_.assign(cfg_.vram_slots, nullptr);
    for (std::size_t i = 0; i < cfg_.vram_slots; ++i)
        vram_slots_[i] = vram_storage_.data() + i * slot_size_;
#endif
    free_slots_.clear();
    for (std::int32_t i = static_cast<std::int32_t>(cfg_.vram_slots) - 1; i >= 0; --i)
        free_slots_.push_back(i);

    plan_active_.store(true, std::memory_order_release);
}

void TierManager::set_layer_map(const std::vector<std::vector<GroupId>>& layers) {
    std::lock_guard<std::mutex> lk(mu_);
    layers_.clear();
    for (std::uint32_t l = 0; l < layers.size(); ++l)
        layers_[l] = layers[l];
}

void TierManager::report_router_counts(
    const std::unordered_map<std::uint32_t, std::uint64_t>& counts) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& kv : counts) {
        auto it = states_.find(kv.first);
        if (it != states_.end()) it->second.router_count = kv.second;
    }
}

void TierManager::record_layer_pause(double ms) {
    std::lock_guard<std::mutex> lk(pauses_mu_);
    pauses_.push_back(ms);
    if (pauses_.size() > 4096) pauses_.pop_front();
}

// --- pin/unpin/pin_layer ----------------------------------------------------
void TierManager::internal_prefetch(GroupId id) {
    GroupState* st = find(id);
    if (!st) return;
    st->prefetch_flag = true; // идемпотентно: повторный prefetch не удваивает работу
    // Bugfix (раунд 3): префетченная группа — «недавно использованная» для
    // LRU/LCU (иначе last_use = 0 → она первая жертва следующей eviction:
    // prefetch(l+1) второй загрузкой вытеснял первую, свежезагруженную,
    // → гарантированные промахи prefetch при churn, срез 3 MoE > VRAM).
    st->last_use = ++tick_;
    if (!st->in_vram && !st->loading) {
        st->loading = true;
        std::lock_guard<std::mutex> qk(qmu_);
        queue_.push_back(Job{id, nullptr});
        qcv_.notify_one();
    }
}

void TierManager::run_predictor(GroupId id) {
    if (last_accessed_.value != id.value) {
        predictor_.observe(last_accessed_, id);
        last_accessed_ = id;
    }
    if (lookahead_ == 0) return;
    for (GroupId p : predictor_.predict(id, lookahead_))
        internal_prefetch(p);
}

std::unique_ptr<Awaiter> TierManager::pin(const GroupKey& key) {
    auto a = pin_internal(key);
    return std::unique_ptr<Awaiter>(new TierAwaiter(*a)); // копия с общим State
}

std::shared_ptr<TierAwaiter> TierManager::pin_internal(const GroupKey& key) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!plan_active_.load(std::memory_order_acquire))
        throw TierError(TierError::Code::PlanMismatch, "pin: plan not set");
    const GroupState* stc = find(key.id);
    if (!stc)
        throw TierError(TierError::Code::PlanMismatch, "pin: unknown GroupId");
    const TensorGroup* g = plan(key.id);
    if (g->role != key.role)
        throw TierError(TierError::Code::PlanMismatch, "pin: role mismatch vs plan");

    GroupState* st = find(key.id); // non-const
    st->use_count++;
    st->last_use = ++tick_;
    run_predictor(key.id);

    auto a = std::make_shared<TierAwaiter>();
    if (st->in_vram) {
        if (st->prefetch_flag) {
            hits_.fetch_add(1); // prefetch попал: группа уже в VRAM к использованию
            st->prefetch_flag = false;
        }
        st->pinned++;
        a->complete(); // группа уже resident: handle готов немедленно
        return a;
    }
    // Промах: группа не в VRAM — загрузка; pinned++ запрещает eviction слота.
    misses_.fetch_add(1);
    st->pinned++;
    if (!st->loading) {
        st->loading = true;
        std::lock_guard<std::mutex> qk(qmu_);
        queue_.push_back(Job{key.id, nullptr});
        qcv_.notify_one();
    }
    st->waiters.push_back(a);
    return a;
}

void TierManager::unpin(const GroupKey& key) {
    std::lock_guard<std::mutex> lk(mu_);
    GroupState* st = find(key.id);
    if (!st || st->pinned == 0)
        throw TierError(TierError::Code::NotPinned, "unpin: group is not pinned");
    st->pinned--;
}

std::unique_ptr<Awaiter> TierManager::pin_layer(std::uint32_t layer) {
    std::vector<std::shared_ptr<TierAwaiter>> parts;
    std::vector<GroupKey> keys;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!plan_active_.load(std::memory_order_acquire))
            throw TierError(TierError::Code::PlanMismatch, "pin_layer: plan not set");
        auto it = layers_.find(layer);
        if (it == layers_.end())
            throw TierError(TierError::Code::PlanMismatch,
                            "pin_layer: layer map not set for this layer (см. set_layer_map)");
        for (GroupId gid : it->second) {
            const TensorGroup* g = plan(gid);
            if (!g)
                throw TierError(TierError::Code::PlanMismatch, "pin_layer: group not in plan");
            keys.push_back(GroupKey{gid, g->role});
        }
    }
    for (const auto& k : keys) parts.push_back(pin_internal(k));
    auto sink = [this](double ms) { record_layer_pause(ms); };
    return TierAwaiter::all(std::move(parts), sink);
}

// --- prefetch ----------------------------------------------------------------
void TierManager::prefetch(const std::vector<GroupKey>& keys) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!plan_active_.load(std::memory_order_acquire))
        throw TierError(TierError::Code::PlanMismatch, "prefetch: plan not set");
    for (const auto& k : keys) {
        if (!find(k.id))
            throw TierError(TierError::Code::PlanMismatch, "prefetch: unknown GroupId");
        internal_prefetch(k.id);
    }
}

// --- worker -----------------------------------------------------------------
void TierManager::enqueue_load(GroupId id, const std::shared_ptr<TierAwaiter>&) {
    // (внутренняя, для будущих задач; загрузка ставится в очередь в pin/prefetch)
    std::lock_guard<std::mutex> qk(qmu_);
    queue_.push_back(Job{id, nullptr});
    qcv_.notify_one();
}

void TierManager::worker_loop() {
    for (;;) {
        std::vector<Job> batch;
        {
            std::unique_lock<std::mutex> qk(qmu_);
            qcv_.wait(qk, [&] { return shutdown_ || !queue_.empty(); });
            if (shutdown_ && queue_.empty()) return;
            // [R2b] deep-prefetch: дренаж очереди БАТЧЕЙ до k групп (k в
            // полёте): k SQE io_uring — один submit, k чтений NVMe параллельно.
            // k=1 → прежнее поведение (группа за раз, batch_depth_ ≥ 1 из ctor).
            batch.push_back(queue_.front());
            queue_.pop_front();
            while (batch.size() < batch_depth_ && !queue_.empty()) {
                batch.push_back(queue_.front());
                queue_.pop_front();
            }
        }
        // Дренаж-счётчик для безопасного ResetPlan (set_plan ждёт 0; см. hpp)
        // — на весь батч (каждый job батча считается in-flight).
        active_jobs_.fetch_add(static_cast<int>(batch.size()),
                               std::memory_order_acq_rel);
        process_batch(batch);
        active_jobs_.fetch_sub(static_cast<int>(batch.size()),
                               std::memory_order_release);
    }
}

void TierManager::evict_ram_if_needed(std::size_t need) {
    // Вызывается под mu_. RAM-тир: LRU по last_use. Кандидаты — группы
    // с RAM-копией вне текущей загрузки, которую можно снять безопасно:
    //  - resident в VRAM (копия в VRAM сохранна),
    //  - не resident и не закреплена (pinned == 0).
    std::size_t used = ram_.size() * slot_size_;
    if (used + need <= cfg_.ram_capacity) return;
    std::vector<std::pair<GroupId, std::uint64_t>> cand;
    for (const auto& kv : states_) {
        const GroupState& st = kv.second;
        if (st.in_ram && !st.loading && (st.in_vram || st.pinned == 0))
            cand.push_back({GroupId{kv.first}, st.last_use});
    }
    std::sort(cand.begin(), cand.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& c : cand) {
        GroupState& st = states_[c.first.value];
        auto it = ram_.find(c.first.value);
        if (it == ram_.end()) continue;
        free_ram_buffer(it->second);
        ram_.erase(it);
        st.in_ram = false;
        used -= slot_size_;
        if (used + need <= cfg_.ram_capacity) break;
    }
}

void TierManager::update_ram_tier(GroupId id, std::size_t need) {
    evict_ram_if_needed(need);
    std::uint8_t*& buf = ram_[id.value];
    if (!buf)
        buf = alloc_ram_buffer(slot_size_); // pinned при STELL_CUDA (cudaHostAlloc)
    (void)need; // слот RAM-тира = размер VRAM-слота
}

std::int32_t TierManager::acquire_slot(GroupId) {
    // Под mu_. Если свободных слотов нет — eviction по политике LRU/LCU
    // среди resident unpinned (pinned-группы — неприкосновенны).
    if (!free_slots_.empty()) {
        std::int32_t s = free_slots_.back();
        free_slots_.pop_back();
        return s;
    }
    std::vector<ResidentCandidate> cand;
    for (const auto& kv : states_) {
        const GroupState& st = kv.second;
        if (st.in_vram && st.pinned == 0)
            cand.push_back(ResidentCandidate{GroupId{kv.first}, st.last_use,
                                             st.use_count, st.router_count});
    }
    if (cand.empty())
        throw TierError(TierError::Code::SlotExhausted,
                        "vram: no free slots and nothing evictable (all pinned)");
    const GroupId victim = choose_victim(cand, policy_);
    // Bugfix (раунд 3): слот жертвы нужно ЗАБРАТЬ себе — release_slot клалит
    // его в free_slots_, но возвращаемый слот обязан быть извлечён (pop_back).
    // Раньше free_slots_.back() возвращался без pop → слот оставался в списке
    // свободных → следующий acquire_slot выдавал ТОТ ЖЕ слот другой группе →
    // двукратная броня одного слота (aliasing данных VRAM, resident > slots;
    // проявлялось при любом churn — сценарий MoE > VRAM, срез 3).
    const std::int32_t victim_slot = states_[victim.value].slot;
    release_slot(victim_slot);
    states_[victim.value].in_vram = false;
    states_[victim.value].slot = -1;
    free_slots_.pop_back(); // забираем слот жертвы под загружаемую группу
    return victim_slot;
}

void TierManager::release_slot(std::int32_t slot) {
    free_slots_.push_back(slot);
}

// [R2b] round 4: батч-обработка до k групп (k в полёте). Фазы прежние
// (A: RAM-тир; B: NVMe→RAM; C: RAM→VRAM), но фаза B — ОДИН батч-чтение k
// групп (k SQE io_uring одним submit — параллельный I/O). Промах/ошибка
// одной группы НЕ роняет соседние по батчу (изоляция исходов).
void TierManager::process_batch(const std::vector<Job>& batch) {
    struct Entry {
        GroupId gid;
        std::vector<std::shared_ptr<TierAwaiter>> waiters;
        bool need_read = false;
        std::uint8_t* ram_buf = nullptr;
        std::exception_ptr err;   // ошибка фаз A/B этой группы
    };
    std::vector<Entry> ents;
    ents.reserve(batch.size());

    // Фаза A (под mu_, по группам): готовим RAM-тир.
    for (const Job& job : batch) {
        Entry e;
        e.gid = job.id;
        std::lock_guard<std::mutex> lk(mu_);
        GroupState* st = find(job.id);
        if (!st) continue; // ResetPlan — job устарел
        e.waiters = st->waiters;
        st->waiters.clear();
        e.need_read = !st->in_ram;
        try {
            if (e.need_read) update_ram_tier(job.id, slot_size_);
            auto it = ram_.find(job.id.value);
            e.ram_buf = it->second;
            if (!e.ram_buf)
                e.err = std::make_exception_ptr(TierError(
                    TierError::Code::IoFailure,
                    "tiers: RAM-буфер группы отсутствует"));
        } catch (...) {
            e.err = std::current_exception();
        }
        ents.push_back(std::move(e));
    }

    // Фаза B: NVMe → RAM. ОДИН батч-чтение всех нуждающихся групп ([R2b]):
    // k SQE io_uring — один submit (k чтений в полёте, параллельный I/O).
    // Диагностический крюк io_delay_ms: латентность ОДНОЙ группы; батч из k
    // в полёте перекрывает её (детерминизм тестов deep-prefetch; прод: 0).
    std::vector<std::size_t> to_read;
    for (std::size_t i = 0; i < ents.size(); ++i)
        if (!ents[i].err && ents[i].need_read) to_read.push_back(i);
    if (!to_read.empty()) {
        if (!nvme_) {
            for (std::size_t i : to_read)
                ents[i].err = std::make_exception_ptr(TierError(
                    TierError::Code::IoFailure,
                    "nvme: tier not configured (no stell-tier layout)"));
        } else {
            if (cfg_.io_delay_ms)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(cfg_.io_delay_ms));
            std::vector<std::pair<std::uint32_t, void*>> reqs;
            reqs.reserve(to_read.size());
            for (std::size_t i : to_read)
                reqs.emplace_back(ents[i].gid.value, ents[i].ram_buf);
            // read_groups НИЧЕГО не бросает по отдельным группам — исходы в
            // BatchOutcome.err (изоляция промаха одной группы в батче);
            // TierError возможен только на отказ кольца/файла — тогда падают
            // ВСЕ группы батча (честная семантика отказа носителя).
            const std::vector<TierNvmeFile::BatchOutcome> out =
                nvme_->read_groups(reqs);
            for (std::size_t j = 0; j < to_read.size(); ++j) {
                Entry& e = ents[to_read[j]];
                if (!out[j].err.empty()) {
                    e.err = std::make_exception_ptr(TierError(
                        TierError::Code::IoFailure, out[j].err));
                } else {
                    io_ops_.fetch_add(1);
                    io_bytes_.fetch_add(slot_size_);
                }
            }
        }
    }

    // Фаза C: RAM → VRAM-слот (DMA при STELL_CUDA; memcpy в CPU-эмуляции),
    // по группам батча; ошибка одной группы (SlotExhausted/DMA) — изолирована.
    for (Entry& e : ents) {
        if (e.err) {
            // Завершить всех ждущих с ошибкой (включая прибывших позже).
            std::vector<std::shared_ptr<TierAwaiter>> more;
            {
                std::lock_guard<std::mutex> lk(mu_);
                GroupState* st = find(e.gid);
                if (st) {
                    st->loading = false;
                    more = st->waiters;
                    st->waiters.clear();
                }
            }
            for (auto& a : e.waiters) a->fail(e.err);
            for (auto& a : more) a->fail(e.err);
            continue;
        }
        std::int32_t slot = -1;
        std::exception_ptr slot_err = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            try {
                slot = acquire_slot(e.gid); // TierError(SlotExhausted) при отсутствии жертв
            } catch (...) {
                slot_err = std::current_exception();
            }
        }
        if (slot_err) {
            std::vector<std::shared_ptr<TierAwaiter>> more;
            {
                std::lock_guard<std::mutex> lk(mu_);
                GroupState* st = find(e.gid);
                if (st) {
                    st->loading = false;
                    more = st->waiters;
                    st->waiters.clear();
                }
            }
            for (auto& a : e.waiters) a->fail(slot_err);
            for (auto& a : more) a->fail(slot_err);
            continue;
        }
        // Копирование вне mu_ (может быть тяжёлым).
#if defined(STELL_CUDA)
        // DMA RAM(pinned)→VRAM(device): cudaMemcpyAsync на выделенном стриме;
        // Awaiter группы завершается после синхронизации стрима (fan-in по
        // событиям CUDA — развитие следующего раунда, см. отчёт о честности).
        {
            cudaStream_t s = static_cast<cudaStream_t>(dma_stream_);
            cudaError_t ce = cudaMemcpyAsync(vram_slots_[(size_t)slot], e.ram_buf,
                                             slot_size_, cudaMemcpyHostToDevice, s);
            if (ce != cudaSuccess) {
                slot_err = std::make_exception_ptr(TierError(
                    TierError::Code::IoFailure,
                    std::string("tiers: DMA RAM→VRAM failed: ") + cudaGetErrorString(ce)));
            } else {
                ce = cudaStreamSynchronize(s);
                if (ce != cudaSuccess) {
                    slot_err = std::make_exception_ptr(TierError(
                        TierError::Code::IoFailure,
                        std::string("tiers: DMA stream sync failed: ") + cudaGetErrorString(ce)));
                }
            }
        }
#else
        std::memcpy(vram_slots_[(size_t)slot], e.ram_buf, slot_size_);
#endif
        if (slot_err) {
            std::vector<std::shared_ptr<TierAwaiter>> more;
            {
                std::lock_guard<std::mutex> lk(mu_);
                GroupState* st = find(e.gid);
                if (st) {
                    st->loading = false;
                    more = st->waiters;
                    st->waiters.clear();
                }
            }
            for (auto& a : e.waiters) a->fail(slot_err);
            for (auto& a : more) a->fail(slot_err);
            continue;
        }
        std::vector<std::shared_ptr<TierAwaiter>> all_waiters;
        {
            std::lock_guard<std::mutex> lk(mu_);
            GroupState* st = find(e.gid);
            if (!st) { release_slot(slot); continue; } // ResetPlan
            st->in_vram = true;
            st->slot = slot;
            st->in_ram = true;
            st->loading = false;
            io_ops_.fetch_add(1);
            io_bytes_.fetch_add(slot_size_);
            auto ws = st->waiters;
            st->waiters.clear();
            all_waiters = e.waiters;
            all_waiters.insert(all_waiters.end(), ws.begin(), ws.end());
        }
        for (auto& a : all_waiters) a->complete();
    }
}

// --- расширения R1/R7 (раунд 3) ----------------------------------------------

// R1: публичный verify-проход слотов NVMe-тира (ADR-12 §2.3, round 3).
std::size_t TierManager::verify_slots() {
    TierNvmeFile* f = nvme_.get();
    if (!f)
        throw TierError(TierError::Code::PlanMismatch,
                        "verify_slots: NVMe-тир не настроен (set_plan с nvme_layout)");
    std::size_t n = 0;
    // Дисциплина «холодной» операции (как set_plan, ADR-08 §4): вызывающий
    // не должен иметь параллельных pin/prefetch — состояние verified_ NVMe-
    // файла не синхронизировано с горячим путём.
    for (const auto& e : f->index()) {
        f->verify_group(e.group_id);
        ++n;
    }
    return n;
}

// R1: метрика пересчётов payload-CRC (0 при verify=off и без verify-прохода).
std::uint64_t TierManager::crc_checks() const {
    const TierNvmeFile* f = nvme_.get();
    return f ? f->crc_checks() : 0;
}

// [R2a] round 4: честная фиксация режима I/O данных (для bench/отчётов).
bool TierManager::nvme_odirect_active() const {
    const TierNvmeFile* f = nvme_.get();
    return f && f->odirect_active();
}

// --- R7: конфигурируемые границы тиров ---------------------------------------
TierConfig tier_config_from_env(TierConfig base) {
    base.vram_slots     = env_size_or("STELL_TIER_VRAM_SLOTS", base.vram_slots);
    base.vram_slot_size = env_size_or("STELL_TIER_VRAM_SLOT_BYTES", base.vram_slot_size);
    base.lookahead      = env_size_or("STELL_TIER_LOOKAHEAD", base.lookahead);
    base.history_cap    = env_size_or("STELL_TIER_HISTORY_CAP", base.history_cap);
    if (const std::size_t mb = env_size_or("STELL_TIER_RAM_MB", 0))
        base.ram_capacity = mb << 20;
    if (const char* v = std::getenv("STELL_TIER_VERIFY_CRC"))
        base.verify_crc = std::strcmp(v, "1") == 0; // "1" → on, что-либо другое → off
    if (const char* v = std::getenv("STELL_TIER_POLICY")) {
        if (std::strcmp(v, "lcu") == 0)      base.policy = EvictionPolicy::Lcu;
        else if (std::strcmp(v, "lru") == 0) base.policy = EvictionPolicy::Lru;
        // неизвестное значение → базовая политика (консервативно, без throw —
        // env — средство настройки, а не контракт; фиксация в докстринге).
    }
    if (const char* v = std::getenv("STELL_TIER_NVME_LAYOUT"))
        base.nvme_layout = v;
    // [R2a] round 4: O_DIRECT-бэкенд NVMe-тира ("1" → on, иное → off,
    // консервативно — buffered-дефолт).
    if (const char* v = std::getenv("STELL_TIER_ODIRECT"))
        base.odirect = std::strcmp(v, "1") == 0;
    // [R2b] round 4: k групп в полёте (deep-prefetch батч-глубина; 0/нечисловое
    // → базовое;TierManager клампит в мин. 1).
    base.deep_prefetch_depth = env_size_or("STELL_TIER_PREFETCH_DEPTH",
                                           base.deep_prefetch_depth);
    // Диагностический крюк латентности загрузки (тесты deep-prefetch; прод: 0).
    base.io_delay_ms = env_size_or("STELL_TIER_IO_DELAY_MS", base.io_delay_ms);
    return base;
}

TierConfig tier_config_co_tenancy(std::size_t vram_budget_bytes,
                                  std::size_t plan_bytes,
                                  std::size_t slot_bytes) {
    if (slot_bytes == 0 || slot_bytes % kTierSlotAlign != 0)
        throw TierError(TierError::Code::PlanMismatch,
                        "co-tenancy: slot_bytes must be a positive multiple of 4096 (ADR-12 §2.4)");
    TierConfig cfg; // дефолты: Lru, lookahead 2
    cfg.vram_slot_size = slot_bytes;
    // VRAM-тир МАЛ: слоты по бюджету; минимум 1 слот; больше плана не выделяем
    // (при co-tenancy VRAM-тир меньше плана — вытеснение/churn штатны).
    cfg.vram_slots = std::max<std::size_t>(
        1, std::min(vram_budget_bytes / slot_bytes, plan_bytes / slot_bytes));
    // RAM-тир ОСНОВНОЙ: вмещает план целиком (pinned; host RAM сервера —
    // сотни ГБ, допущение зафиксировано в докстринге tier_manager.hpp).
    cfg.ram_capacity = plan_bytes;
    return cfg;
}

// --- stats -------------------------------------------------------------------
TierStats TierManager::stats() const {
    TierStats ts{};
    ts.prefetch.hits = hits_.load(std::memory_order_relaxed);
    ts.prefetch.misses = misses_.load(std::memory_order_relaxed);
    const std::uint64_t total = ts.prefetch.hits + ts.prefetch.misses;
    ts.prefetch.hit_rate = total ? double(ts.prefetch.hits) / double(total) : 0.0;

    // Паузы слоёв: p50/p95/p99 по скользящему окну (ADR-08 §4: асимптотическая
    // статистика, не консистентный срез «в момент t»).
    std::deque<double> snap;
    {
        std::lock_guard<std::mutex> lk(pauses_mu_);
        snap = pauses_;
    }
    if (!snap.empty()) {
        auto q = [&](double f) -> double {
            const std::size_t idx = std::min(snap.size() - 1,
                static_cast<std::size_t>(f * double(snap.size() - 1)));
            return snap[idx];
        };
        std::sort(snap.begin(), snap.end());
        ts.layer_pause.p50_ms = q(0.50);
        ts.layer_pause.p95_ms = q(0.95);
        ts.layer_pause.p99_ms = q(0.99);
    }

    std::lock_guard<std::mutex> lk(mu_);
    std::uint64_t vram_bytes = 0;
    for (const auto& kv : states_)
        if (kv.second.in_vram) vram_bytes += slot_size_;
    std::uint64_t ram_bytes = ram_.size() * slot_size_;
    ts.tier_bytes[0] = vram_bytes;
    ts.tier_bytes[1] = ram_bytes;
    ts.tier_bytes[2] = nvme_ ? [&] {
        std::uint64_t s = 0;
        for (const auto& e : nvme_->index()) s += e.slot_bytes;
        return s;
    }() : 0;
    ts.io_bytes = io_bytes_.load(std::memory_order_relaxed);
    ts.io_ops = io_ops_.load(std::memory_order_relaxed);
    return ts;
}

} // namespace stell::tiers

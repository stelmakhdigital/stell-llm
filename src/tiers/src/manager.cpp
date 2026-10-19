// Реализация TierManager (ADR-08): VRAM-slots → pinned RAM → NVMe(io_uring),
// политика LRU/LCU, router-lookahead prefetch, метрики.
#include <stell/tiers/tier_manager.hpp>

#include <algorithm>
#include <cassert>
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
std::uint8_t* alloc_ram_buffer(std::size_t n) { return new std::uint8_t[n]; }
void free_ram_buffer(std::uint8_t* p) { delete[] p; }
#endif
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
    : cfg_(cfg), policy_(cfg.policy), predictor_(cfg.history_cap), lookahead_(cfg.lookahead) {
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
        auto file = std::make_unique<TierNvmeFile>(cfg_.nvme_layout);
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
        Job job;
        {
            std::unique_lock<std::mutex> qk(qmu_);
            qcv_.wait(qk, [&] { return shutdown_ || !queue_.empty(); });
            if (shutdown_ && queue_.empty()) return;
            job = queue_.front();
            queue_.pop_front();
        }
        // Дренаж-счётчик для безопасного ResetPlan (set_plan ждёт 0; см. hpp).
        active_jobs_.fetch_add(1, std::memory_order_acq_rel);
        process_job(job);
        active_jobs_.fetch_sub(1, std::memory_order_release);
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

void TierManager::process_job(const Job& job) {
    const GroupId gid = job.id;
    std::vector<std::shared_ptr<TierAwaiter>> waiters;

    // Завершить всех ждущих с ошибкой (включая прибывших позже).
    auto fail_all = [&](std::exception_ptr err) {
        std::vector<std::shared_ptr<TierAwaiter>> more;
        {
            std::lock_guard<std::mutex> lk(mu_);
            GroupState* st = find(gid);
            if (st) {
                st->loading = false;
                more = st->waiters;
                st->waiters.clear();
            }
        }
        for (auto& a : waiters) a->fail(err);
        for (auto& a : more) a->fail(err);
    };

    // Фаза A: готовим RAM-тир (под mu_).
    bool need_read = false;
    std::uint8_t* ram_buf = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        GroupState* st = find(gid);
        if (!st) return; // ResetPlan — job устарел
        waiters = st->waiters;
        st->waiters.clear();
        need_read = !st->in_ram;
        if (need_read) update_ram_tier(gid, slot_size_);
        auto it = ram_.find(gid.value);
        ram_buf = it->second;
        if (!ram_buf)
            throw TierError(TierError::Code::IoFailure, "tiers: RAM-буфер группы отсутствует");
    }

    // Фаза B: NVMe → RAM (io_uring, вне mu_).
    if (need_read) {
        if (!nvme_) {
            fail_all(std::make_exception_ptr(
                TierError(TierError::Code::IoFailure,
                          "nvme: tier not configured (no stell-tier layout)")));
            return;
        }
        try {
            nvme_->read_group(gid.value, ram_buf);
            io_ops_.fetch_add(1);
            io_bytes_.fetch_add(slot_size_);
        } catch (...) {
            fail_all(std::current_exception());
            return;
        }
    }

    // Фаза C: RAM → VRAM-слот (DMA при STELL_CUDA; memcpy в CPU-эмуляции).
    std::int32_t slot = -1;
    std::exception_ptr slot_err = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        try {
            slot = acquire_slot(gid); // TierError(SlotExhausted) при отсутствии жертв
        } catch (...) {
            slot_err = std::current_exception();
        }
    }
    if (slot_err) {
        fail_all(slot_err);
        return;
    }
    // Копирование вне mu_ (может быть тяжёлым).
#if defined(STELL_CUDA)
    // DMA RAM(pinned)→VRAM(device): cudaMemcpyAsync на выделенном стриме;
    // Awaiter группы завершается после синхронизации стрима (fan-in по
    // событиям CUDA — развитие следующего раунда, см. отчёт о честности).
    {
        cudaStream_t s = static_cast<cudaStream_t>(dma_stream_);
        cudaError_t e = cudaMemcpyAsync(vram_slots_[(size_t)slot], ram_buf,
                                        slot_size_, cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) {
            fail_all(std::make_exception_ptr(TierError(
                TierError::Code::IoFailure,
                std::string("tiers: DMA RAM→VRAM failed: ") + cudaGetErrorString(e))));
            return;
        }
        e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) {
            fail_all(std::make_exception_ptr(TierError(
                TierError::Code::IoFailure,
                std::string("tiers: DMA stream sync failed: ") + cudaGetErrorString(e))));
            return;
        }
    }
#else
    std::memcpy(vram_slots_[(size_t)slot], ram_buf, slot_size_);
#endif

    {
        std::lock_guard<std::mutex> lk(mu_);
        GroupState* st = find(gid);
        if (!st) { release_slot(slot); return; } // ResetPlan
        st->in_vram = true;
        st->slot = slot;
        st->in_ram = true;
        st->loading = false;
        io_ops_.fetch_add(1);
        io_bytes_.fetch_add(slot_size_);
        auto ws = st->waiters;
        st->waiters.clear();
        waiters.insert(waiters.end(), ws.begin(), ws.end());
    }
    for (auto& a : waiters) a->complete();
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

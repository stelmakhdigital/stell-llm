#include "stell/engine/scheduler.hpp"

#include <algorithm>
#include <cstring>

namespace stell::engine {

Scheduler::Scheduler(IModelRunner& runner, ISlotPool& pool)
    : Scheduler(runner, pool, nullptr, Config{}) {}

Scheduler::Scheduler(IModelRunner& runner, ISlotPool& pool,
                     parallel::ICommunicator* comm, Config cfg)
    : runner_(&runner), pool_(&pool), comm_(comm), cfg_(cfg) {}

void Scheduler::set_tokenizer(const ITokenizer* tok) { tokenizer_ = tok; }

SessionId Scheduler::submit(const SubmitRequest& req) {
    SlotId slot = 0;
    if (!pool_->acquire(slot)) {
        // Нет слота: отказ (ADR-11 §2: сессия получает слот сразу; error-событие
        // для C ABI — на стыке stell_submit). C++-сторона возвращает 0.
        return 0;
    }
    Session s{};
    s.id = next_session_id_++;
    s.slot = slot;
    s.req = req;
    s.last_token = req.prompt_tokens.empty() ? -1 : req.prompt_tokens.back();
    s.completion = 0;
    s.seq = 0;
    s.finished = false;

    SessionId id = s.id;
    sessions_.emplace(id, std::move(s));
    order_.push_back(id);
    return id;
}

bool Scheduler::cancel(SessionId id) {
    auto it = sessions_.find(id);
    if (it == sessions_.end() || it->second.finished) return false;
    Session& s = it->second;
    pool_->release(s.slot);
    Event ev{};
    ev.type = Event::Type::Finish;
    ev.finish_reason = "cancelled";
    ev.prompt_tokens = static_cast<std::uint32_t>(s.req.prompt_tokens.size());
    ev.completion_tokens = s.completion;
    emit(s, ev);
    s.finished = true;
    release_slot(s.slot);
    return true;
}

void Scheduler::emit(Session& s, Event ev) {
    ev.seq = ++s.seq;
    inbox_[s.id].push_back(std::move(ev));
}

void Scheduler::finish_session(Session& s, const char* reason) {
    Event ev{};
    ev.type = Event::Type::Finish;
    ev.finish_reason = reason;
    ev.prompt_tokens = static_cast<std::uint32_t>(s.req.prompt_tokens.size());
    ev.completion_tokens = s.completion;
    emit(s, ev);
    s.finished = true;
    pool_->release(s.slot);
    release_slot(s.slot);
}

void Scheduler::step() {
    // Сбор активных сессий в батч (continuous batching: новые сессии входят
    // в ближайший шаг — ADR-11 §2).
    std::vector<Session*> batch;
    batch.reserve(sessions_.size());
    for (SessionId id : order_) {
        auto it = sessions_.find(id);
        if (it == sessions_.end()) continue;
        Session& s = it->second;
        if (!s.finished && s.completion < static_cast<std::uint32_t>(s.req.sampling.max_tokens))
            batch.push_back(&s);
        if (batch.size() == cfg_.batch_step_max) break;
    }

    // ADR-10 §3 + уточнение round 1: мета шага = фиксированный F32-пейлоад
    // (8 слотов = 32 байта) через allgather; ранг 0 — источник (слоты
    // [0]=шаг, [1]=размер батча, 6 зарезервированных); остальные ранки отдают
    // заготовленные слоты и исполняют полученную мету. recv-ёмкость = 8 × world
    // (семантика allgather — сбор по ранкам; см. ADR-10 §2).
    if (comm_ && cfg_.broadcast_step_meta) {
        const parallel::rank_t world = comm_->world_size();
        float send[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (comm_->rank() == 0) {
            send[0] = static_cast<float>(steps_);
            send[1] = static_cast<float>(batch.size());
        }
        std::vector<float> recv(static_cast<std::size_t>(8) * world, 0.0f);
        comm_->allgather(send, recv.data(), 8, parallel::CommDType::F32);
        // Формат слотов — внутреннее дело engine (ADR-10 «Уточнение»: parallel
        // семантически нейтрален; решение по метe — детерминированное: ранг 0).
    }

    if (batch.empty()) { ++steps_; return; }

    std::vector<IModelRunner::DecodeInput> inputs;
    inputs.reserve(batch.size());
    for (Session* s : batch)
        inputs.push_back({s->slot, s->last_token, s->req.prompt_tokens, s->req.sampling});

    std::vector<IModelRunner::DecodeOutput> outputs = runner_->decode_step(inputs);

    for (const IModelRunner::DecodeOutput& out : outputs) {
        Session* s = nullptr;
        for (Session* cand : batch)
            if (cand->slot == out.slot) { s = cand; break; }
        if (!s) continue;

        Event ev{};
        ev.type = Event::Type::Token;
        ev.token_id = out.token;
        // round 2: text из токенизатора (byte-fallback; прод-токенизатор —
        // отдельная задача, честная пометка в tokenizer.hpp).
        if (tokenizer_) ev.text = tokenizer_->decode(out.token);
        emit(*s, ev);
        s->last_token = out.token;
        ++s->completion;

        const std::int32_t eos = s->req.eos_token;
        if (out.token == eos)
            finish_session(*s, "stop");
        else if (s->completion >= static_cast<std::uint32_t>(s->req.sampling.max_tokens))
            finish_session(*s, "length");
    }

    ++steps_;
}

void Scheduler::release_slot(SlotId slot) {
    // Слот вернулся в пул (finish/cancel): runner сбрасывает KV-кеш слота —
    // слот пула переиспользуется новой сессией (ADR-05/ADR-11 §3).
    runner_->release_slot(slot);
}

std::vector<Event> Scheduler::poll(SessionId id) {
    std::vector<Event> out;
    auto ib = inbox_.find(id);
    if (ib != inbox_.end()) {
        out.assign(ib->second.begin(), ib->second.end());
        inbox_.erase(ib);
    }
    auto it = sessions_.find(id);
    if (it != sessions_.end() && it->second.finished) {
        // Терминальное finish уже выдано и отдачно — запись удаляется.
        sessions_.erase(it);
        // убрать из order_
        order_.erase(std::remove(order_.begin(), order_.end(), id), order_.end());
    }
    return out;
}

std::size_t Scheduler::active() const {
    std::size_t n = 0;
    for (const auto& [id, s] : sessions_)
        if (!s.finished) ++n;
    return n;
}

std::uint64_t Scheduler::steps_done() const { return steps_; }

} // namespace stell::engine

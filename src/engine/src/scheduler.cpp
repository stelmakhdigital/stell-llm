#include "stell/engine/scheduler.hpp"

#include <algorithm>
#include <cstring>

namespace stell::engine {

Scheduler::Scheduler(IModelRunner& runner, ISlotPool& pool)
    : Scheduler(runner, pool, nullptr, Config{}) {}

Scheduler::Scheduler(IModelRunner& runner, ISlotPool& pool,
                     parallel::ICommunicator* comm, Config cfg)
    : runner_(&runner), pool_(&pool), comm_(comm), cfg_(cfg) {}

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

    // ADR-10 §3: планировщик на ранке 0 рассылает мету шага (маленькая,
    // фиксированная) через allgather; остальные ранки исполняют ту же логику.
    // Раунд 1: мета = [шаг, размер батча] как float-слоты (F32), документировано
    // в отчёте как упрощение до появления comm-реализации (поток C).
    if (comm_ && cfg_.broadcast_step_meta) {
        float meta[8] = {static_cast<float>(steps_),
                         static_cast<float>(batch.size()), 0, 0, 0, 0, 0, 0};
        float recv[8] = {};
        comm_->allgather(meta, recv, 8, parallel::CommDType::F32);
    }

    if (batch.empty()) { ++steps_; return; }

    std::vector<IModelRunner::DecodeInput> inputs;
    inputs.reserve(batch.size());
    for (Session* s : batch) inputs.push_back({s->slot, s->last_token});

    std::vector<IModelRunner::DecodeOutput> outputs = runner_->decode_step(inputs);

    for (const IModelRunner::DecodeOutput& out : outputs) {
        Session* s = nullptr;
        for (Session* cand : batch)
            if (cand->slot == out.slot) { s = cand; break; }
        if (!s) continue;

        Event ev{};
        ev.type = Event::Type::Token;
        ev.token_id = out.token;
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

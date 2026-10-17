// Планировщик continuous batching (PLAN.md §4 поток A, ADR-11 §2-§3).
// ≥8 одновременных сессий; слот — ADR-05/ADR-08; шаг батча — ранг 0 + broadcast
// мета-структуры через parallel::ICommunicator::allgather (ADR-10 §3 + уточнение
// round 1: 8×F32 слотов = 32 байта; ранг 0 — источник меты).
//
// Round 2 (интеграция): real ICommunicator (CPU-эмулятор parallel), ITokenizer
// (text в событиях, ADR-11 §3), runner получает полный контекст сессии
// (промпт + Sampling — сэмплинг принадлежит runner, ADR-11 «Уточнение»).
// Tiering (ADR-08) вклеен в декодовый контур engine ниже scheduler'а
// (ops_runner.cpp: set_plan/set_layer_map/pin_layer/prefetch); scheduler
// остаётся модельно-агностичным — ADR-08 §5: «engine отвечает за pin», а
// слои знает только исполняющий шаг модель runner.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "slot_pool.hpp"
#include "stell/parallel/comm.hpp"
#include "tokenizer.hpp"

namespace stell::engine {

using SessionId = std::uint64_t;

// Параметры сэмплинга — контракты bench-протокола (PLAN.md §5, ADR-11 §2):
// бенч-профиль T=1, top-p=0.95, top-k=20, seed=0.
struct SamplingParams {
    std::int32_t  max_tokens = 256;
    float         temperature = 1.0f;
    float         top_p = 0.95f;
    std::int32_t  top_k = 20;
    std::uint32_t seed = 0;
};

struct SubmitRequest {
    // serve применяет chat-шаблон и передаёт готовые токены (ADR-11 §2).
    std::vector<std::int32_t> prompt_tokens;
    SamplingParams sampling;
    // eos: id конца последовательности модели; -1 — отключить стоп по eos.
    std::int32_t eos_token = 2;
};

// Stream-события (ADR-11 §3): seq монотонен в рамках сессии.
struct Event {
    enum class Type { Token, Finish, Error };
    std::uint64_t seq = 0;
    Type type = Type::Token;
    std::int32_t token_id = -1;
    std::string text;             // text заполняется ITokenizer (round 2; byte-fallback)
    std::string finish_reason;    // "stop" | "length" | "cancelled"
    std::string error_code;
    std::string error_message;
    std::uint32_t prompt_tokens = 0;      // usage на finish (ADR-11 §3, bench-контракт)
    std::uint32_t completion_tokens = 0;
};

// Абстракция исполнения модели: engine ведёт bookkeeping, модель исполняет шаг.
// CUDA-конфигурация: runner поверх CUDA Graph (ADR-05) + eager-фолббек;
// CPU-интеграция round 2: CpuModelRunner (ops_runner.hpp) — реальный ops-каталог
// поверх kernels CPU-референс; CPU-тесты round 1: фейковый runner.
class IModelRunner {
public:
    struct DecodeInput {
        SlotId slot = 0;
        std::int32_t token = -1;          // последний токен контекста слота
        // Полный промпт сессии: runner делает prefill на первом шаге декода
        // (кеш KV слота заполняется всеми токенами промпта, кроме последнего).
        std::vector<std::int32_t> prompt;
        // Сэмплинг принадлежит runner (ADR-11 «Уточнение»: scheduler передаёт
        // Sampling из request_json; temperature==0 → greedy argmax — режим
        // golden-контракта ADR-09 §4).
        SamplingParams sampling{};
    };
    struct DecodeOutput {
        SlotId slot = 0;
        std::int32_t token = -1;          // сэмплированный следующий токен
    };
    virtual ~IModelRunner() = default;
    virtual std::vector<DecodeOutput> decode_step(const std::vector<DecodeInput>& inputs) = 0;

    // Слот освобождён (finish/cancel, ADR-11 §3): runner сбрасывает
    // вычислительное состояние слота (KV-кеш) — слот пула переиспользуется.
    virtual void release_slot(SlotId /*slot*/) {}
};

class Scheduler {
public:
    struct Config {
        std::size_t batch_step_max = 16;      // максимум слотов в одном шаге декода
        bool broadcast_step_meta = false;     // ADR-10 §3: ранг 0 рассылает мету шага
    };

    Scheduler(IModelRunner& runner, ISlotPool& pool);
    Scheduler(IModelRunner& runner, ISlotPool& pool,
              parallel::ICommunicator* comm, Config cfg);

    // Токенизатор (round 2): text в token-событиях; nullptr — text пуст
    // (round-1 поведение, честно помечено в отчёте).
    void set_tokenizer(const ITokenizer* tok);

    // Постановка запроса: сессия сразу получает слот и входит в ближайший шаг
    // батча (ADR-11 §2). Возвращает id сессии (>0) или 0 при отсутствии слота.
    SessionId submit(const SubmitRequest& req);

    // Отмена: сессия выбрасывается из батча на ближайшем шаге (ADR-11 §3).
    // Возвращает true, если сессия была активна.
    bool cancel(SessionId id);

    // Один шаг декода над всеми активными сессиями (continuous batching).
    void step();

    // Неблокирующее выкачивание событий сессии (ADR-11 §3). После finish
    // событие отдаётся последним; следующая poll вернёт пустой список
    // и запись сессии будет удалена.
    std::vector<Event> poll(SessionId id);

    std::size_t active() const;                 // активные (не завершённые) сессии
    std::uint64_t steps_done() const;           // счётчик шагов (метрики TPOT/TTFT)

private:
    struct Session {
        SessionId id = 0;
        SlotId slot = 0;
        SubmitRequest req{};
        std::int32_t last_token = -1;
        std::uint32_t completion = 0;
        std::uint64_t seq = 0;
        bool finished = false;
    };

    void emit(Session& s, Event ev);
    void finish_session(Session& s, const char* reason);

    void release_slot(SlotId slot);

    IModelRunner* runner_;
    ISlotPool* pool_;
    parallel::ICommunicator* comm_;
    Config cfg_;
    const ITokenizer* tokenizer_ = nullptr;

    std::unordered_map<SessionId, Session> sessions_;
    std::vector<SessionId> order_;               // порядок включения в батч (стабильный)
    std::unordered_map<SessionId, std::deque<Event>> inbox_;  // события по ADR-11 §3
    std::uint64_t next_session_id_ = 1;
    std::uint64_t steps_ = 0;
};

} // namespace stell::engine

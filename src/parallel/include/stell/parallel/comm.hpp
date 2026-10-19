// stell-llm — parallel: контракт engine ↔ parallel (ADR-10).
//
// Этот заголовок — ЗАФИКСИРОВАННЫЙ контракт (ADR-10 §2); менять только через ADR.
// Поверх него реализуются: NCCL-backend (CUDA/NVLink) и CPU-эмулятор
// (make_communicator с mode = CommMode::CpuEmulation, ADR-10 §4).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace stell::parallel {

using rank_t = std::int32_t;

// Тип данных комм — только те, что реально гоняются по NVLink (ADR-04).
enum class CommDType { F16, F32, E4M3 };

// Размер элемента в байтах: F16=2, F32=4, E4M3=1.
inline std::size_t dtype_size(CommDType t) noexcept {
    switch (t) {
        case CommDType::F16: return 2;
        case CommDType::F32: return 4;
        case CommDType::E4M3: return 1;
    }
    return 0;
}

// Метаданные токена для expert_alltoall (ADR-10 §2: expert_id + слот ADR-08).
// Буферы send_meta/recv_meta интерпретируются как TokenMeta[count].
struct TokenMeta {
    std::int32_t expert_id = -1;   // -1 => слот пуст (recv-заполнитель)
    std::int32_t slot_id = -1;     // слот сессии (ADR-08), одинаков по всем ранкам
};

// Ошибки — исключениями (единая политика с ADR-08).
class CommError : public std::runtime_error {
public:
    enum class Code { NcclFailure, RankMismatch, BufferTooSmall, Timeout };

    explicit CommError(Code c, const std::string& what = "")
        : std::runtime_error(what.empty() ? code_to_string(c) : std::string(code_to_string(c)) + ": " + what),
          code_(c) {}
    Code code() const noexcept { return code_; }

private:
    static const char* code_to_string(Code c) noexcept {
        switch (c) {
            case Code::NcclFailure: return "NcclFailure";
            case Code::RankMismatch: return "RankMismatch";
            case Code::BufferTooSmall: return "BufferTooSmall";
            case Code::Timeout: return "Timeout";
        }
        return "CommError";
    }
    Code code_;
};

enum class CommMode { Nccl, CpuEmulation };

// Конфиг параллелизма — из конфига engine (ADR-10 §3: { tp, pp, ep, device_ids }).
// world = pp * tp * ep; ранок = pp_rank * (tp*ep) + ep_rank * tp + tp_rank
// (см. stell/parallel/plan.hpp — разбиение по ADR-10 §3, порядок: PP, затем TP/EP).
struct ParallelConfig {
    CommMode mode = CommMode::CpuEmulation;
    rank_t tp = 1;                    // TP-группа: головы attention / KV-головы / колонки FFN
    rank_t pp = 1;                    // PP-цепочка: слои
    rank_t ep = 1;                    // EP-группа: эксперты (expert_id % ep == ep_rank)
    std::vector<int> device_ids;      // NCCL: CUDA-устройства, размер == world
    rank_t rank = 0;                  // CPU-эмуляция: ранок данного инстанса (0..world-1)
    int group_id = 0;                 // CPU-эмуляция: id процессной группы (для тестов)
    int timeout_ms = 5000;            // семантика тайм-аутов (ADR-10 §4, ошибка Timeout)
    std::size_t kv_ring_capacity = 16; // блоков в кольце KvStream (на пару ранков)
    void* cuda_stream = nullptr;      // NCCL: CUDA-stream, на котором обязаны исполняться
                                      // операции (ADR-10 §2); CPU-эмуляция игнорирует.

    rank_t world() const noexcept { return pp * tp * ep; }
};

class ICommunicator {
public:
    virtual ~ICommunicator() = default;
    virtual rank_t world_size() const = 0;
    virtual rank_t rank() const = 0;

    // Все операции синхронны по семантике (блокируют поток до завершения
    // коллективной операции) и обязаны запускаться на переданном CUDA-stream
    // (ParallelConfig::cuda_stream; CPU-эмуляция игнорирует stream).
    // Асинхронные хендлы — по мере надобности отдельным ADR, пока потребность
    // не доказана.
    virtual void allreduce(void* buf, std::size_t count, CommDType t) = 0;
    virtual void allgather(void* send, void* recv, std::size_t count, CommDType t) = 0;

    // EP: токены (индексы экспертов + эмбеддинги) уезжают к владельцам экспертов
    // и возвращаются после FFN. Семантика: send-токен i уходит на ранок
    // owner(expert_id[i]); recv_tokens возвращают локальные токены других ранков.
    //
    // Ширина токена (уточнение ADR-10, round 2): token_elems — количество
    // элементов dtype на ОДИН токен (для hidden = d_model). Токен i занимает
    // элементы send_tokens[i*token_elems .. (i+1)*token_elems); token_elems = 1
    // — совместимость со скалярной семантикой раунда 1. Мета-структура
    // (expert_id + слот ADR-08) — одна на токен, без изменений.
    //
    // Буферы: send_tokens — max_tokens * token_elems элементов; send_meta —
    // ровно max_tokens записей; записи с meta.expert_id < 0 не отправляются
    // (пустые слоты). recv_tokens — ёмкость max_tokens * world_size ТОКЕНОВ
    // (т.е. max_tokens * world_size * token_elems элементов), recv_meta —
    // max_tokens * world_size записей: входящие токены складываются
    // в детерминированном порядке (по возрастанию ранка-источника, затем по
    // индексу), остаток ёмкости заполняется нулями и TokenMeta{expert_id = -1}.
    // Переполнение ёмкости приёма — CommError(BufferTooSmall); молчаливое
    // сокращение пересылки запрещено (ADR-10, round 2).
    // При TP внутри EP-группы (tp > 1) токен уезжает на ВСЕ tp-ранки
    // группы-владельца эксперта (fan-out, см. plan.hpp::expert_alltoall_route).
    virtual void expert_alltoall(
        void* send_tokens, void* send_meta,   // meta: expert_id, слот (ADR-08)
        void* recv_tokens, void* recv_meta,
        std::size_t max_tokens, std::size_t token_elems, CommDType t) = 0;

    // Подгруппа (уточнение ADR-10, round 2): коллективы (allreduce/allgather/
    // expert_alltoall) и p2p (chain_*, kv_stream_*) внутри возвращаемого
    // комм-объекта исполняются ТОЛЬКО по ранкам из ranks — чужие ранки родителя
    // не участвуют и не затрагиваются. Ранки в подгруппе ПЕРЕНУМЕРОВАНЫ:
    // ранок i подгруппы — i-й (по возрастанию) глобальный ранок из ranks;
    // world_size() подгруппы = ranks.size().
    //
    // ranks: глобальные ранки родителя, уникальные, из [0, world_size());
    // ranks ОБЯЗАН содержать собственный ранок (иначе CommError(RankMismatch);
    // не-члены подгруппу не создают). Все члены обязаны вызвать subgroup с
    // ОДИНАКОВЫМ ranks (как ncclCommSplit по цвету) — иначе члены не сойдутся
    // в одном состоянии и получат Timeout.
    // Подмножество допустимо: subgroup можно вызывать и на подгруппе (вложенно).
    //
    // ВЛАДЕНИЕ У РОДИТЕЛЯ: возвращённый объект принадлежит родительскому
    // комм-объекту, живёт до его уничтожения и НЕ удаляется вызывающим
    // (удаление родителем). Повторный вызов с тем же ranks возвращает тот же
    // под-комм (состояние коллективов у подгруппы общее между членами).
    virtual ICommunicator* subgroup(const std::vector<rank_t>& ranks) = 0;

    // PP: point-to-point по цепочке (peer = rank±1). send кладёт данные в канал
    // (буферизуется), recv блокирует до появления (Timeout по ParallelConfig).
    virtual void chain_send(void* buf, std::size_t count, CommDType t, rank_t peer) = 0;
    virtual void chain_recv(void* buf, std::size_t count, CommDType t, rank_t peer) = 0;

    // KV-streaming: неблокирующий кольцевой канал KV-блоков (ADR-04 dtype).
    // push кладёт блок в кольцо к следующему ранку (rank+1)%world и блокирует
    // при переполнении кольца (Timeout); pop снимает блок из кольца предыдущего
    // ранка, false — кольцо пусто. bytes обязан совпадать с размером блока.
    virtual void kv_stream_push(const void* kv_block, std::size_t bytes) = 0;
    virtual bool  kv_stream_pop(void* dst, std::size_t bytes) = 0;
};

// Фабрика: NCCL-реализация (CUDA) либо CPU-эмулятор (ADR-10 §4).
std::unique_ptr<ICommunicator> make_communicator(const ParallelConfig& cfg);

} // namespace stell::parallel

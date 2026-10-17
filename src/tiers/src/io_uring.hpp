// RAII-обёртка io_uring: очередь, SQE/CQE, mmap колец по офсетам из
// io_uring_params. Полный путь NVMe-тира (ADR-08/PLAN.md §4). Если ядро
// не поддерживает io_uring — конструктор бросает TierError(IoFailure),
// вызывающий переключается на fallback-бэкенд (pread/pwrite).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <stell/tiers/errors.hpp>

#include "io_uring_abi.hpp"

namespace stell::tiers {

// Один подготовленный и поданный запрос + результат.
struct IoResult {
    std::uint64_t user_data = 0;
    std::int64_t  res = 0;      // байтов прочитано/записано или -errno
};

class IoUring {
public:
    explicit IoUring(std::uint32_t depth = 32); // бросает TierError(IoFailure)
    ~IoUring();
    IoUring(const IoUring&) = delete;
    IoUring& operator=(const IoUring&) = delete;

    // Готовит SQE (opcode READ/WRITE), user_data — тег завершения.
    void prep_read(int fd, std::uint64_t off, void* buf, std::uint32_t len,
                   std::uint64_t user_data);
    void prep_write(int fd, std::uint64_t off, const void* buf, std::uint32_t len,
                    std::uint64_t user_data);

    // Подает все готовые SQE, дожидается завершения запроса user_data
    // (блокирующе через io_uring_enter GETEVENTS). Возвращает res.
    std::int64_t submit_and_wait(std::uint64_t user_data);

    // Неблокирующий сбор завершений (для потокобезопасных снапшотов).
    std::vector<IoResult> poll();

    bool available() const { return fd_ >= 0; }

private:
    void* sq_ring_ = nullptr;
    void* cq_ring_ = nullptr;
    void* sqes_ = nullptr;
    std::uint64_t sq_sz_ = 0, cq_sz_ = 0;
    volatile std::uint32_t* sq_head_ = nullptr;
    volatile std::uint32_t* sq_tail_ = nullptr;
    volatile std::uint32_t* sq_flags_ = nullptr;
    volatile std::uint32_t* sq_array_ = nullptr;
    volatile std::uint32_t* cq_head_ = nullptr;
    volatile std::uint32_t* cq_tail_ = nullptr;
    abi::IoUringCqe* cqes_ = nullptr;
    abi::IoUringSqe* sqes_ptr_ = nullptr;
    std::uint32_t sq_mask_ = 0, cq_mask_ = 0;
    std::uint32_t sq_entries_ = 0, cq_entries_ = 0;
    std::uint64_t pending_ = 0; // счётчик поданных, не собранных
    int fd_ = -1;

    void submit(std::uint32_t flags);
    bool reap(volatile std::uint32_t* head, volatile std::uint32_t* tail,
              std::uint32_t mask, std::vector<IoResult>& out, bool block);
};

// Fallback-бэкенд: blocking pread/pwrite (для ядер без io_uring).
class FallbackIo {
public:
    static std::int64_t read(int fd, std::uint64_t off, void* buf, std::uint32_t len);
    static std::int64_t write(int fd, std::uint64_t off, const void* buf, std::uint32_t len);
};

} // namespace stell::tiers

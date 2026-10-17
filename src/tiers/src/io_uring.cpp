// Реализация RAII-обёртки io_uring и fallback-бэкенда.
#include "io_uring.hpp"

#include <cerrno>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425  // x86_64/aarch64 (uapi asm unistd_64)
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#endif

namespace stell::tiers {

#if defined(__linux__)

static long sys_uring_setup(unsigned depth, abi::IoUringParams* p) {
    return syscall(__NR_io_uring_setup, depth, p);
}
static long sys_uring_enter(int fd, unsigned to_submit, unsigned nr_complete,
                            unsigned flags) {
    return syscall(__NR_io_uring_enter, fd, to_submit, nr_complete, flags,
                   static_cast<void*>(nullptr));
}

IoUring::IoUring(std::uint32_t depth) {
    abi::IoUringParams params;
    std::memset(&params, 0, sizeof params);
    long fd = sys_uring_setup(depth, &params);
    if (fd < 0) {
        throw TierError(TierError::Code::IoFailure,
                        std::string("io_uring_setup failed: ") + std::strerror(errno));
    }
    fd_ = static_cast<int>(fd);
    sq_entries_ = params.sq_entries;
    cq_entries_ = params.cq_entries ? params.cq_entries : params.sq_entries;
    sq_mask_ = sq_entries_ - 1;
    cq_mask_ = cq_entries_ - 1;

    sq_sz_ = params.sq_off.array + sq_entries_ * sizeof(std::uint32_t);
    cq_sz_ = params.cq_off.cqes + cq_entries_ * sizeof(abi::IoUringCqe);

    sq_ring_ = mmap(nullptr, sq_sz_, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, fd_, abi::IORING_OFF_SQ_RING);
    cq_ring_ = mmap(nullptr, cq_sz_, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, fd_, abi::IORING_OFF_CQ_RING);
    sqes_ = mmap(nullptr, static_cast<std::uint64_t>(sq_entries_) * sizeof(abi::IoUringSqe),
                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_,
                 abi::IORING_OFF_SQES);
    if (sq_ring_ == MAP_FAILED || cq_ring_ == MAP_FAILED || sqes_ == MAP_FAILED) {
        throw TierError(TierError::Code::IoFailure, "io_uring: ring mmap failed");
    }
    sq_head_ = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<char*>(sq_ring_) + params.sq_off.head);
    sq_tail_ = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<char*>(sq_ring_) + params.sq_off.tail);
    sq_flags_ = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<char*>(sq_ring_) + params.sq_off.flags);
    sq_array_ = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<char*>(sq_ring_) + params.sq_off.array);
    cq_head_ = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<char*>(cq_ring_) + params.cq_off.head);
    cq_tail_ = reinterpret_cast<volatile std::uint32_t*>(
        reinterpret_cast<char*>(cq_ring_) + params.cq_off.tail);
    cqes_ = reinterpret_cast<abi::IoUringCqe*>(
        reinterpret_cast<char*>(cq_ring_) + params.cq_off.cqes);
    sqes_ptr_ = reinterpret_cast<abi::IoUringSqe*>(sqes_);
}

IoUring::~IoUring() {
    if (fd_ < 0) return;
    if (sq_ring_) munmap(const_cast<void*>(sq_ring_), sq_sz_);
    if (cq_ring_) munmap(const_cast<void*>(cq_ring_), cq_sz_);
    if (sqes_) munmap(const_cast<void*>(sqes_),
                      static_cast<std::uint64_t>(sq_entries_) * sizeof(abi::IoUringSqe));
    close(fd_);
}

void IoUring::prep_read(int fd, std::uint64_t off, void* buf, std::uint32_t len,
                        std::uint64_t user_data) {
    const std::uint32_t idx = (*sq_tail_) & sq_mask_;
    abi::IoUringSqe* sqe = &sqes_ptr_[idx];
    std::memset(sqe, 0, sizeof(abi::IoUringSqe));
    sqe->opcode = abi::IORING_OP_READ;
    sqe->fd = fd;
    sqe->off = off;
    sqe->addr = reinterpret_cast<std::uint64_t>(buf);
    sqe->len = len;
    sqe->user_data = user_data;
    sq_array_[idx] = idx;
    const std::uint32_t t = *sq_tail_;
    *sq_tail_ = t + 1;
    pending_++;
}

void IoUring::prep_write(int fd, std::uint64_t off, const void* buf, std::uint32_t len,
                         std::uint64_t user_data) {
    const std::uint32_t idx = (*sq_tail_) & sq_mask_;
    abi::IoUringSqe* sqe = &sqes_ptr_[idx];
    std::memset(sqe, 0, sizeof(abi::IoUringSqe));
    sqe->opcode = abi::IORING_OP_WRITE;
    sqe->fd = fd;
    sqe->off = off;
    sqe->addr = reinterpret_cast<std::uint64_t>(buf);
    sqe->len = len;
    sqe->user_data = user_data;
    sq_array_[idx] = idx;
    const std::uint32_t t2 = *sq_tail_;
    *sq_tail_ = t2 + 1;
    pending_++;
}

void IoUring::submit(std::uint32_t flags) {
    if (pending_ == 0) return;
    unsigned need_wakeup = (*sq_flags_) & abi::IORING_SQ_NEED_WAKEUP
                               ? abi::IORING_ENTER_SQ_WAKEUP : 0;
    long r = sys_uring_enter(fd_, static_cast<unsigned>(pending_), 1,
                             flags | need_wakeup);
    if (r < 0 && errno != EAGAIN && errno != EINTR) {
        pending_ = 0;
        *sq_head_ = *sq_tail_;
        throw TierError(TierError::Code::IoFailure,
                        std::string("io_uring_enter failed: ") + std::strerror(errno));
    }
    if (r > 0) {
        pending_ -= static_cast<std::uint64_t>(r);
    }
    *sq_head_ = *sq_tail_;
}

std::int64_t IoUring::submit_and_wait(std::uint64_t user_data) {
    submit(abi::IORING_ENTER_GETEVENTS);
    for (;;) {
        // Неблокирующий дренаж CQE.
        std::vector<IoResult> done = poll();
        for (const auto& r : done) {
            if (r.user_data == user_data) return r.res;
        }
        // Блокирующее ожидание одного завершения.
        long r = sys_uring_enter(fd_, 0, 1, abi::IORING_ENTER_GETEVENTS);
        if (r < 0 && errno != EINTR) {
            throw TierError(TierError::Code::IoFailure,
                            std::string("io_uring_enter(wait) failed: ") +
                                std::strerror(errno));
        }
    }
}

std::vector<IoResult> IoUring::poll() {
    std::vector<IoResult> out;
    reap(cq_head_, cq_tail_, cq_mask_, out, /*block=*/false);
    return out;
}

bool IoUring::reap(volatile std::uint32_t* head, volatile std::uint32_t* tail,
                   std::uint32_t mask, std::vector<IoResult>& out, bool block) {
    (void)block;
    std::uint32_t h = *head, t = *tail;
    std::uint32_t avail = t - h; // арифметика по модулю 2^32
    for (std::uint32_t i = 0; i < avail; ++i) {
        const std::uint32_t idx = (h + i) & mask;
        const abi::IoUringCqe* cqe = &cqes_[idx];
        out.push_back(IoResult{cqe->user_data, cqe->res});
    }
    *head = t; // потребляем всё, что собрали
    return avail > 0;
}

#endif // __linux__

std::int64_t FallbackIo::read(int fd, std::uint64_t off, void* buf, std::uint32_t len) {
    ssize_t r = pread(fd, buf, len, static_cast<off_t>(off));
    return r;
}

std::int64_t FallbackIo::write(int fd, std::uint64_t off, const void* buf, std::uint32_t len) {
    ssize_t r = pwrite(fd, buf, len, static_cast<off_t>(off));
    return r;
}

} // namespace stell::tiers

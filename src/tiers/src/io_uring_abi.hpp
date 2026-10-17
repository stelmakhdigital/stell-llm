// Минимальный ABI-подмножество io_uring (ядро Linux ≥ 5.1, x86_64/aarch64),
// воспроизведённое по include/uapi/linux/io_uring.h ядра Linux
// (SPDX: (GPL-2.0 WITH Linux-syscall-note) OR MIT; выравнивание полей и
// константы соответствуют uapi, офсеты колец получаются из io_uring_params,
// возвращаемого io_uring_setup — см. io_uring.cpp).
// Локальные /usr/include не содержат linux/io_uring.h, поэтому определения
// держим здесь. Системные номера системных вызовов берём из <sys/syscall.h>.
#pragma once

#include <cstdint>

#if defined(__linux__)

namespace stell::tiers::abi {

// enum io_uring_op (uapi, по порядку — см. io_uring.h ядра).
enum IoUringOp : std::uint8_t {
    IORING_OP_NOP = 0,
    IORING_OP_READV = 1,
    IORING_OP_WRITEV = 2,
    IORING_OP_FSYNC = 3,
    IORING_OP_READ_FIXED = 4,
    IORING_OP_WRITE_FIXED = 5,
    IORING_OP_POLL_ADD = 6,
    IORING_OP_POLL_REMOVE = 7,
    IORING_OP_SYNC_FILE_RANGE = 8,
    IORING_OP_SENDMSG = 9,
    IORING_OP_RECVMSG = 10,
    IORING_OP_TIMEOUT = 11,
    IORING_OP_TIMEOUT_REMOVE = 12,
    IORING_OP_ACCEPT = 13,
    IORING_OP_ASYNC_CANCEL = 14,
    IORING_OP_LINK_TIMEOUT = 15,
    IORING_OP_CONNECT = 16,
    IORING_OP_FALLOCATE = 17,
    IORING_OP_OPENAT = 18,
    IORING_OP_CLOSE = 19,
    IORING_OP_FILES_UPDATE = 20,
    IORING_OP_STATX = 21,
    IORING_OP_READ = 22,
    IORING_OP_WRITE = 23,
    IORING_OP_FADVISE = 24,
    IORING_OP_MADVISE = 25,
    IORING_OP_SEND = 26,
    IORING_OP_RECV = 27,
    IORING_OP_OPENAT2 = 28,
    IORING_OP_EPOLL_CTL = 29,
    IORING_OP_SPLICE = 30,
    IORING_OP_PROVIDE_BUFFERS = 31,
    IORING_OP_REMOVE_BUFFERS = 32,
    IORING_OP_TEE = 33,
    IORING_OP_SHUTDOWN = 34,
    IORING_OP_RENAMEAT = 35,
    IORING_OP_UNLINKAT = 36,
    IORING_OP_MKDIRAT = 37,
    IORING_OP_SYMLINKAT = 38,
    IORING_OP_LINKAT = 39,
    IORING_OP_MSG_RING = 40,
    IORING_OP_FSETXATTR = 41,
    IORING_OP_SETXATTR = 42,
    IORING_OP_FGETXATTR = 43,
    IORING_OP_GETXATTR = 44,
    IORING_OP_SOCKET = 45,
    IORING_OP_URING_CMD = 46,
    IORING_OP_SEND_ZC = 47,
    IORING_OP_SENDMSG_ZC = 48,
    IORING_OP_READ_MULTISHOT = 49,
    IORING_OP_WAITID = 50,
    IORING_OP_FUTEX_WAIT = 51,
    IORING_OP_FUTEX_WAKE = 52,
    IORING_OP_FUTEX_WAITV = 53,
    IORING_OP_FIXED_FD_INSTALL = 54,
    IORING_OP_FTRUNCATE = 55,
    IORING_OP_BIND = 56,
    IORING_OP_LISTEN = 57,
    IORING_OP_RECV_ZC = 58,
    IORING_OP_EPOLL_WAIT = 59,
    IORING_OP_READV_FIXED = 60,
    IORING_OP_WRITEV_FIXED = 61,
    IORING_OP_PIPE = 62,
    IORING_OP_NOP128 = 63,
    IORING_OP_URING_CMD128 = 64,
};

// mmap-офсеты (IORING_OFF_*).
constexpr std::uint64_t IORING_OFF_SQ_RING = 0ULL;
constexpr std::uint64_t IORING_OFF_CQ_RING = 0x8000000ULL;
constexpr std::uint64_t IORING_OFF_SQES    = 0x10000000ULL;

// io_uring_params->features.
constexpr std::uint32_t IORING_FEAT_SINGLE_MMAP = 1U << 0;

// sq_ring->flags.
constexpr std::uint32_t IORING_SQ_NEED_WAKEUP  = 1U << 0;
constexpr std::uint32_t IORING_SQ_CQ_OVERFLOW  = 1U << 1;

// io_uring_enter flags.
constexpr std::uint32_t IORING_ENTER_GETEVENTS = 1U << 0;
constexpr std::uint32_t IORING_ENTER_SQ_WAKEUP = 1U << 1;

// io_uring_setup flags (передаём только нужные).
constexpr std::uint32_t IORING_SETUP_CQSIZE = 1U << 3;

// struct io_uring_sqe (uapi; sizeof == 64 при обычной настройке).
struct IoUringSqe {
    std::uint8_t  opcode;
    std::uint8_t  flags;
    std::uint16_t ioprio;
    std::int32_t  fd;
    std::uint64_t off;
    std::uint64_t addr;
    std::uint32_t len;
    std::uint32_t rw_flags;
    std::uint64_t user_data;
    std::uint16_t buf_index;
    std::uint16_t personality;
    std::int32_t  splice_fd_in; // union splice_fd_in/file_index (u32)
    std::uint64_t addr3;
    std::uint64_t pad2;
};
static_assert(sizeof(IoUringSqe) == 64, "io_uring sqe ABI");

// struct io_uring_cqe (uapi; sizeof == 16).
struct IoUringCqe {
    std::uint64_t user_data;
    std::int32_t  res;
    std::uint32_t flags;
};
static_assert(sizeof(IoUringCqe) == 16, "io_uring cqe ABI");

// struct io_sqring_offsets / io_cqring_offsets / io_uring_params (uapi).
struct IoSqringOffsets {
    std::uint32_t head, tail, ring_mask, ring_entries, flags, dropped, array, resv1;
    std::uint64_t user_addr;
};
struct IoCqringOffsets {
    std::uint32_t head, tail, ring_mask, ring_entries, overflow, cqes, flags, resv1;
    std::uint64_t user_addr;
};
struct IoUringParams {
    std::uint32_t sq_entries, cq_entries, flags, sq_thread_cpu, sq_thread_idle,
                  features, wq_fd;
    std::uint32_t resv[3];
    IoSqringOffsets sq_off;
    IoCqringOffsets cq_off;
};

static_assert(sizeof(IoUringParams) == 40 + 40 + 40, "io_uring params ABI");

} // namespace stell::tiers::abi

#endif // __linux__

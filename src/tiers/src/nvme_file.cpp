#include "nvme_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace stell::tiers {

namespace {
std::string g_backend = "io_uring";
}

std::string nvme_io_backend() { return g_backend; }

TierNvmeFile::TierNvmeFile(std::string path)
    : reader_(std::make_unique<TierLayoutReader>(std::move(path))) {
    try {
        ring_ = std::make_unique<IoUring>(16);
    } catch (const TierError&) {
        // Ядро без io_uring (или запрет): fallback pread, тот же контракт.
        ring_.reset();
        g_backend = "fallback";
    }
}

TierNvmeFile::~TierNvmeFile() = default;

void TierNvmeFile::read_group(std::uint32_t group_id, void* dst) {
    const TierGroupIndexEntry* e = find(group_id);
    if (!e)
        throw TierError(TierError::Code::IoFailure, "nvme: group not found in layout");

    const std::uint64_t tag = next_tag_++;
    if (ring_) {
        // Полный путь io_uring: SQE(READ) → submit → CQE(res = байты).
        ring_->prep_read(reader_->fd(), e->data_offset, dst,
                         static_cast<std::uint32_t>(e->slot_bytes), tag);
        std::int64_t res = ring_->submit_and_wait(tag);
        if (res != static_cast<std::int64_t>(e->slot_bytes)) {
            if (res < 0)
                throw TierError(TierError::Code::IoFailure,
                                std::string("nvme: io_uring read failed: ") +
                                    std::strerror(static_cast<int>(-res)));
            throw TierError(TierError::Code::IoFailure, "nvme: short read via io_uring");
        }
    } else {
        reader_->read_group(group_id, dst); // pread + CRC-once
        return;
    }
    ++io_ops_;
    io_bytes_ += e->slot_bytes;

    // CRC — при первом чтении группы в кеше tiers (ADR-12 §2.3).
    if (std::find(verified_.begin(), verified_.end(), group_id) == verified_.end()) {
        if (crc32(dst, e->slot_bytes) != e->crc32)
            throw TierError(TierError::Code::IoFailure,
                            "nvme: group CRC mismatch (payload corrupted)");
        verified_.push_back(group_id);
    }
}

} // namespace stell::tiers

#include "nvme_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace stell::tiers {

namespace {
std::string g_backend = "io_uring";
}

std::string nvme_io_backend() { return g_backend; }

TierNvmeFile::TierNvmeFile(std::string path, bool verify_payload_crc)
    : verify_crc_(verify_payload_crc),
      reader_(std::make_unique<TierLayoutReader>(std::move(path), verify_payload_crc)) {
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
        reader_->read_group(group_id, dst); // pread + CRC-once (при verify=on)
        return;
    }
    ++io_ops_;
    io_bytes_ += e->slot_bytes;

    // Payload-CRC — при первом чтении группы и только при явном флаге
    // (ADR-12 §2.3, round 3: hot path по умолчанию без CRC).
    if (verify_crc_ &&
        std::find(verified_.begin(), verified_.end(), group_id) == verified_.end()) {
        ++crc_checks_;
        if (crc32(dst, e->slot_bytes) != e->crc32)
            throw TierError(TierError::Code::IoFailure,
                            "layout: payload CRC mismatch");
        verified_.push_back(group_id);
    }
}

void TierNvmeFile::verify_group(std::uint32_t group_id) {
    const TierGroupIndexEntry* e = find(group_id);
    if (!e)
        throw TierError(TierError::Code::IoFailure, "nvme: group not found in layout");
    // Внепутевая сверка: plain pread в scratch-буфер (не трогаем счётчики
    // горячего пути io_bytes/io_ops и очередь io_uring), пересчёт CRC.
    std::vector<std::uint8_t> buf(e->slot_bytes);
    const std::int64_t r = FallbackIo::read(reader_->fd(), e->data_offset, buf.data(),
                                            static_cast<std::uint32_t>(e->slot_bytes));
    if (r != static_cast<std::int64_t>(e->slot_bytes))
        throw TierError(TierError::Code::IoFailure,
                        "nvme: short read during verify pass");
    ++crc_checks_;
    if (crc32(buf.data(), buf.size()) != e->crc32)
        throw TierError(TierError::Code::IoFailure, "layout: payload CRC mismatch");
    // Успех: группа проверена — горячий путь (даже при verify=on) не будет
    // пересчитывать CRC для неё повторно.
    if (std::find(verified_.begin(), verified_.end(), group_id) == verified_.end())
        verified_.push_back(group_id);
}

} // namespace stell::tiers

#include "nvme_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

namespace stell::tiers {

namespace {
std::string g_backend = "io_uring";

// [R2] round 4, O_DIRECT: требование выравнивания буфера/офсета/длины.
// Консервативная фиксация 4096 = kTierSlotAlign (ADR-12 гарантирует офсет/
// длину слотов кратно 4096; покрывает логический блок 512/2048/4096
// распространённых FS — см. докстринг nvme_file.hpp; lsb кодом НЕ запрашивается).
constexpr std::size_t kDirectAlign = kTierSlotAlign;

inline std::uintptr_t ptr_addr(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p);
}
inline bool is_aligned(const void* p) {
    return ptr_addr(p) % kDirectAlign == 0;
}

// RAII выровненный staging-буфер (posix_memalign): для чтения с O_DIRECT,
// когда dst не выровнен (контракт read_group допускает произвольный dst).
struct AlignedStage {
    std::uint8_t* p = nullptr;
    explicit AlignedStage(std::size_t n) {
        void* raw = nullptr;
        if (posix_memalign(&raw, kDirectAlign, n) != 0)
            throw TierError(TierError::Code::IoFailure,
                            "nvme: posix_memalign staging failed");
        p = static_cast<std::uint8_t*>(raw);
    }
    ~AlignedStage() { std::free(p); }
    AlignedStage(const AlignedStage&) = delete;
    AlignedStage& operator=(const AlignedStage&) = delete;
};
} // namespace

std::string nvme_io_backend() { return g_backend; }

TierNvmeFile::TierNvmeFile(std::string path, bool verify_payload_crc, bool odirect)
    : reader_(std::make_unique<TierLayoutReader>(path, verify_payload_crc)),
      path_(path),
      verify_crc_(verify_payload_crc) {
    if (odirect) {
        // O_DIRECT-тракт данных: отдельный fd (заголовок/индекс читает reader
        // через buffered fd — маленькие невыровненные чтения). Не удалось
        // открыть (FS/ядро не поддерживают O_DIRECT, напр. tmpfs) → честный
        // откат на buffered: odirect_active() == false (проверяется тестами).
        odirect_fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECT);
        odirect_active_ = odirect_fd_ >= 0;
    }
    try {
        ring_ = std::make_unique<IoUring>(16);
    } catch (const TierError&) {
        // Ядро без io_uring (или запрет): fallback pread, тот же контракт.
        ring_.reset();
        g_backend = "fallback";
    }
}

TierNvmeFile::~TierNvmeFile() {
    if (odirect_fd_ >= 0) ::close(odirect_fd_);
}

int TierNvmeFile::data_fd() const {
    return odirect_active_ ? odirect_fd_ : reader_->fd();
}

std::int64_t TierNvmeFile::finish_read(std::uint32_t group_id,
                                       const TierGroupIndexEntry* e,
                                       std::int64_t res, void* dst) {
    if (res != static_cast<std::int64_t>(e->slot_bytes)) {
        if (res < 0)
            throw TierError(TierError::Code::IoFailure,
                            std::string("nvme: read failed: ") +
                                std::strerror(static_cast<int>(-res)));
        throw TierError(TierError::Code::IoFailure, "nvme: short read");
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
    return res;
}

void TierNvmeFile::read_group(std::uint32_t group_id, void* dst) {
    const TierGroupIndexEntry* e = find(group_id);
    if (!e)
        throw TierError(TierError::Code::IoFailure, "nvme: group not found in layout");

    const std::uint64_t tag = next_tag_++;
    if (ring_) {
        // Полный путь io_uring: SQE(READ) → submit → CQE(res = байты).
        // O_DIRECT: если dst не выровнен на 4096 — минимальное выровненное
        // staging-чтение + memcpy (контракт read_group не меняется; RAM-буферы
        // менеджера выровнены — читается прямо в них, без лишней копии).
        void* io_dst = dst;
        std::unique_ptr<AlignedStage> stage;
        if (!is_aligned(dst)) {
            stage = std::make_unique<AlignedStage>(e->slot_bytes);
            io_dst = stage->p;
        }
        ring_->prep_read(data_fd(), e->data_offset, io_dst,
                         static_cast<std::uint32_t>(e->slot_bytes), tag);
        const std::int64_t res = ring_->submit_and_wait(tag);
        finish_read(group_id, e, res, io_dst);
        if (stage) std::memcpy(dst, stage->p, e->slot_bytes);
    } else {
        if (!odirect_active_) {
            reader_->read_group(group_id, dst); // pread + CRC-once (при verify=on)
            return;
        }
        // Fallback + O_DIRECT: pread на O_DIRECT-фд требует того же
        // выравнивания — staging-буфер обязателен (dst произвольный).
        AlignedStage stage(e->slot_bytes);
        const std::int64_t res = FallbackIo::read(data_fd(), e->data_offset,
                                                  stage.p,
                                                  static_cast<std::uint32_t>(e->slot_bytes));
        finish_read(group_id, e, res, stage.p);
        std::memcpy(dst, stage.p, e->slot_bytes);
    }
}

std::vector<TierNvmeFile::BatchOutcome> TierNvmeFile::read_groups(
        const std::vector<std::pair<std::uint32_t, void*>>& reqs) {
    std::vector<BatchOutcome> out(reqs.size());

    // --- fallback (нет кольца): честно последовательно (параллельности I/O
    // нет — фиксировано в докстринге; deep-prefetch батча сводится к k
    // последовательным pread'ам). Ошибки — в исходах, без бросков.
    if (!ring_) {
        for (std::size_t i = 0; i < reqs.size(); ++i) {
            const TierGroupIndexEntry* e = find(reqs[i].first);
            if (!e) { out[i].err = "nvme: group not found in layout"; continue; }
            try {
                if (!odirect_active_) {
                    // Buffered fallback: делегат reader (pread + CRC-once).
                    reader_->read_group(reqs[i].first, reqs[i].second);
                    out[i].bytes = static_cast<std::int64_t>(e->slot_bytes);
                    ++io_ops_;
                    io_bytes_ += e->slot_bytes;
                } else {
                    AlignedStage stage(e->slot_bytes);
                    const std::int64_t res = FallbackIo::read(
                        data_fd(), e->data_offset, stage.p,
                        static_cast<std::uint32_t>(e->slot_bytes));
                    out[i].bytes = finish_read(reqs[i].first, e, res, stage.p);
                    std::memcpy(reqs[i].second, stage.p, e->slot_bytes);
                }
            } catch (const TierError& te) {
                out[i].err = te.what();
            } catch (...) {
                out[i].err = "nvme: unknown error (fallback batch)";
            }
        }
        return out;
    }

    // --- полный путь io_uring: k SQE одним submit (k групп в полёте,
    // [R2b]); чужие CQE не дропаются (submit_and_wait_all собирает все).
    std::vector<std::uint64_t> tags;
    std::vector<std::size_t> tag_idx;   // tags[j] → позиция в reqs/out
    std::vector<const TierGroupIndexEntry*> entries(reqs.size(), nullptr);
    std::vector<std::unique_ptr<AlignedStage>> stages(reqs.size());
    tags.reserve(reqs.size());
    tag_idx.reserve(reqs.size());
    try {
        for (std::size_t i = 0; i < reqs.size(); ++i) {
            const TierGroupIndexEntry* e = find(reqs[i].first);
            if (!e) { out[i].err = "nvme: group not found in layout"; continue; }
            entries[i] = e;
            void* io_dst = reqs[i].second;
            if (!is_aligned(reqs[i].second)) {
                stages[i] = std::make_unique<AlignedStage>(e->slot_bytes);
                io_dst = stages[i]->p;
            }
            // Тег только для РЕАЛЬНО поданных SQE (ненайденные группы — в
            // исходах, без тегов: submit_and_wait_all не ждёт несуществующие).
            tag_idx.push_back(i);
            tags.push_back(next_tag_++);
            ring_->prep_read(data_fd(), e->data_offset, io_dst,
                             static_cast<std::uint32_t>(e->slot_bytes), tags.back());
        }
        if (!tags.empty()) {
            const std::vector<std::int64_t> res = ring_->submit_and_wait_all(tags);
            for (std::size_t j = 0; j < tags.size(); ++j) {
                const std::size_t i = tag_idx[j];
                try {
                    out[i].bytes = finish_read(reqs[i].first, entries[i],
                                               res[j], reqs[i].second);
                    if (stages[i])
                        std::memcpy(reqs[i].second, stages[i]->p,
                                    entries[i]->slot_bytes);
                } catch (const TierError& te) {
                    out[i].err = te.what();
                }
            }
        }
    } catch (const TierError& te) {
        // Отказ кольца/подачи: весь батч — честная ошибка (менеджер разомкнёт
        // waiters всех групп батча).
        for (auto& o : out) if (o.err.empty()) o.err = te.what();
    }
    return out;
}

void TierNvmeFile::verify_group(std::uint32_t group_id) {
    const TierGroupIndexEntry* e = find(group_id);
    if (!e)
        throw TierError(TierError::Code::IoFailure, "nvme: group not found in layout");
    // Внепутевая сверка: plain pread в scratch-буфер по buffered-фд (не трогаем
    // счётчики горячего пути io_bytes/io_ops, кольцо и O_DIRECT-тракт),
    // пересчёт CRC.
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

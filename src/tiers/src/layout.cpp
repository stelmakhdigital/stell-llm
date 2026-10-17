// Реализация stell-tier layout v1 (ADR-12 §2): писатель, читатель, CRC32.
#include <stell/tiers/layout.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "io_uring.hpp"

namespace stell::tiers {

namespace {

constexpr std::uint32_t kCrc32Poly          = 0xEDB88320u; // IEEE reflected
constexpr std::size_t   kHeaderModelNameOff = 0x40;
constexpr std::size_t   kHeaderModelNameLen = 96;

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) {
    return (v + a - 1) / a * a;
}

// -- little-endian кодеки -----------------------------------------------------
void put32(std::uint8_t* p, std::uint32_t v) { std::memcpy(p, &v, 4); }
void put64(std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); }
std::uint32_t get32(const std::uint8_t* p) { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
std::uint64_t get64(const std::uint8_t* p) { std::uint64_t v; std::memcpy(&v, p, 8); return v; }

// Запись индекса: ровно kTierIndexEntrySize байт (о несоответствии name[20]
// в ADR-12 §2.2 — см. отчёт раунда; имя обрезается до 16 байт).
void encode_entry(const TierGroupIndexEntry& e, std::uint8_t* out /*64*/) {
    std::memset(out, 0, kTierIndexEntrySize);
    put32(out + 0x00, e.group_id);
    out[0x04] = e.role;
    out[0x05] = e.dtype;
    out[0x06] = e.tensor_count;
    out[0x07] = 0; // reserved
    put64(out + 0x08, e.data_offset);
    put64(out + 0x10, e.slot_bytes);
    put32(out + 0x18, e.shape_rank);
    for (int i = 0; i < 4; ++i) put32(out + 0x1C + 4 * i, e.shape[i]);
    put32(out + 0x2C, e.crc32);
    if (!e.name.empty()) {
        std::size_t n = std::min(e.name.size(), kTierGroupNameMax);
        std::memcpy(out + 0x30, e.name.data(), n);
    }
}

TierGroupIndexEntry decode_entry(const std::uint8_t* in /*64*/) {
    TierGroupIndexEntry e;
    e.group_id = get32(in + 0x00);
    e.role = in[0x04];
    e.dtype = in[0x05];
    e.tensor_count = in[0x06];
    e.data_offset = get64(in + 0x08);
    e.slot_bytes = get64(in + 0x10);
    e.shape_rank = get32(in + 0x18);
    for (int i = 0; i < 4; ++i) e.shape[i] = get32(in + 0x1C + 4 * i);
    e.crc32 = get32(in + 0x2C);
    std::size_t n = 0;
    while (n < kTierGroupNameMax && in[0x30 + n] != 0) ++n;
    e.name.assign(reinterpret_cast<const char*>(in + 0x30), n);
    return e;
}

} // namespace

// --- CRC32 -------------------------------------------------------------------
std::uint32_t crc32_update(std::uint32_t crc, const void* data, std::size_t len) noexcept {
    static const std::uint32_t* table = [] {
        static std::uint32_t t[256];
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? (c >> 1) ^ kCrc32Poly : c >> 1;
            t[i] = c;
        }
        return &t[0];
    }();
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    std::uint32_t c = crc;
    for (std::size_t i = 0; i < len; ++i)
        c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}

std::uint32_t crc32(const void* data, std::size_t len) noexcept {
    return crc32_update(0xFFFFFFFFu, data, len) ^ 0xFFFFFFFFu;
}

// --- писатель ----------------------------------------------------------------
TierLayoutWriter::TierLayoutWriter(std::string path, TierLayoutHeader header)
    : path_(std::move(path)), header_(std::move(header)) {
    if (header_.version != kTierVersion)
        throw std::invalid_argument("layout: unsupported version");
    if (header_.n_groups == 0)
        throw std::invalid_argument("layout: n_groups must be known before writing "
                                    "(конвертер передаёт размер плана, ADR-12 §2)");
}

TierLayoutWriter::~TierLayoutWriter() {
    if (fd_ >= 0) ::close(fd_);
    if (!finalized_) {
        // finalize() обязателен: незавершённый файл не оставляем.
        ::unlink(path_.c_str());
    }
}

void TierLayoutWriter::add_group(const TensorGroup& g, std::uint8_t tensor_count,
                                 const void* payload, std::size_t payload_len) {
    if (finalized_)
        throw std::invalid_argument("layout: writer already finalized");
    if (entries_.size() >= header_.n_groups)
        throw std::invalid_argument("layout: too many groups (n_groups exceeded)");
    if (payload_len > g.slot_bytes)
        throw std::invalid_argument("layout: payload exceeds slot_bytes");
    if (g.slot_bytes == 0 || g.slot_bytes % kTierSlotAlign != 0)
        throw std::invalid_argument("layout: slot_bytes must be a positive multiple of 4096 (ADR-12 §2.4)");
    if (g.shape.size() > 4)
        throw std::invalid_argument("layout: shape rank > 4 (ADR-12 §2.2, use tensor_count)");

    if (entries_.empty()) {
        // Индекс после заголовка, data-секция — выровнена на slot_align (§2.2/§2.4).
        next_offset_ = align_up(kTierHeaderSize
                                + static_cast<std::uint64_t>(header_.n_groups) * kTierIndexEntrySize,
                                kTierSlotAlign);
        if (fd_ < 0) {
            fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (fd_ < 0)
                throw TierError(TierError::Code::IoFailure,
                                std::string("layout: cannot create file: ") + std::strerror(errno));
        }
    }

    TierGroupIndexEntry e;
    e.group_id = g.id.value;
    e.role = role_code(g.role);
    e.dtype = dtype_code(g.dtype);
    e.tensor_count = tensor_count;
    e.data_offset = next_offset_;
    e.slot_bytes = g.slot_bytes;
    e.shape_rank = static_cast<std::uint32_t>(g.shape.size());
    for (std::size_t i = 0; i < g.shape.size(); ++i)
        e.shape[i] = static_cast<std::uint32_t>(g.shape[i]);
    e.name = g.name.substr(0, kTierGroupNameMax);

    // Слот целиком: payload + нулевое padding; CRC по слоту (см. hpp).
    std::vector<std::uint8_t> slot(g.slot_bytes, 0);
    if (payload && payload_len) std::memcpy(slot.data(), payload, payload_len);
    e.crc32 = crc32(slot.data(), slot.size());

    if (::ftruncate(fd_, static_cast<off_t>(next_offset_ + g.slot_bytes)) != 0)
        throw TierError(TierError::Code::IoFailure,
                        std::string("layout: ftruncate failed: ") + std::strerror(errno));
    std::int64_t w = FallbackIo::write(fd_, e.data_offset, slot.data(),
                                       static_cast<std::uint32_t>(slot.size()));
    if (w != static_cast<std::int64_t>(slot.size()))
        throw TierError(TierError::Code::IoFailure, "layout: short write of group slot");

    entries_.push_back(e);
    next_offset_ += g.slot_bytes;
    header_.n_tensors += tensor_count;
}

void TierLayoutWriter::finalize() {
    if (finalized_) return;
    if (entries_.size() != header_.n_groups)
        throw std::invalid_argument("layout: fewer groups than declared n_groups");
    // Индекс: выровнен на 512, сразу после заголовка (§2.2).
    header_.index_offset = kTierHeaderSize;
    header_.data_offset = align_up(kTierHeaderSize
                                   + static_cast<std::uint64_t>(header_.n_groups) * kTierIndexEntrySize,
                                   kTierSlotAlign);

    // Заголовок: ровно 512 байт.
    std::uint8_t hdr[kTierHeaderSize];
    std::memset(hdr, 0, sizeof(hdr));
    std::memcpy(hdr + 0x00, kTierMagic, 8);
    put32(hdr + 0x08, header_.version);
    put32(hdr + 0x0C, header_.flags);
    put64(hdr + 0x10, header_.created_ts);
    put32(hdr + 0x18, kTierSlotAlign);
    put32(hdr + 0x1C, kTierIndexAlign);
    put32(hdr + 0x20, header_.dtype_profile);
    put32(hdr + 0x24, header_.n_groups);
    put32(hdr + 0x28, header_.n_tensors);
    std::size_t n = std::min(header_.model_name.size(), kHeaderModelNameLen);
    std::memcpy(hdr + kHeaderModelNameOff, header_.model_name.data(), n);
    // ADR-12 §2: reserved-поле char[176] на 0xA0 оставлено нулевым; до конца
    // 512 байт — нулевой padding (несоответствие размеров в ADR-12 см. в отчёте).
    const std::uint32_t crc = crc32(hdr, 0x28); // поля 0x00..0x28 (ADR-12 §2)
    put32(hdr + 0x2C, crc);
    put64(hdr + 0x30, header_.index_offset);
    put64(hdr + 0x38, header_.data_offset);

    std::vector<std::uint8_t> index(entries_.size() * kTierIndexEntrySize, 0);
    for (std::size_t i = 0; i < entries_.size(); ++i)
        encode_entry(entries_[i], index.data() + i * kTierIndexEntrySize);

    // Порядок: слоты данных и индекс уже в файле; заголовок пишем в начало файла.
    std::int64_t w1 = FallbackIo::write(fd_, 0, hdr, kTierHeaderSize);
    std::int64_t w2 = FallbackIo::write(fd_, header_.index_offset, index.data(),
                                        static_cast<std::uint32_t>(index.size()));
    if (w1 != kTierHeaderSize || w2 != static_cast<std::int64_t>(index.size()))
        throw TierError(TierError::Code::IoFailure, "layout: short write of header/index");
    if (::fsync(fd_) != 0)
        throw TierError(TierError::Code::IoFailure, "layout: fsync failed");
    ::close(fd_);
    fd_ = -1;
    finalized_ = true;
}

// --- читатель ----------------------------------------------------------------
TierLayoutReader::TierLayoutReader(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw TierError(TierError::Code::IoFailure,
                        std::string("layout: cannot open: ") + std::strerror(errno));
    try {
        std::uint8_t hdr[kTierHeaderSize];
        std::int64_t r = FallbackIo::read(fd_, 0, hdr, kTierHeaderSize);
        if (r != kTierHeaderSize)
            throw TierError(TierError::Code::IoFailure, "layout: short header read");
        if (std::memcmp(hdr, kTierMagic, 8) != 0)
            throw TierError(TierError::Code::IoFailure, "layout: bad magic");
        header_.version = get32(hdr + 0x08);
        header_.flags = get32(hdr + 0x0C);
        header_.created_ts = get64(hdr + 0x10);
        const std::uint32_t slot_align = get32(hdr + 0x18);
        const std::uint32_t index_align = get32(hdr + 0x1C);
        header_.dtype_profile = get32(hdr + 0x20);
        header_.n_groups = get32(hdr + 0x24);
        header_.n_tensors = get32(hdr + 0x28);
        const std::uint32_t hdr_crc_stored = get32(hdr + 0x2C);
        header_.index_offset = get64(hdr + 0x30);
        header_.data_offset = get64(hdr + 0x38);
        (void)slot_align; (void)index_align;
        if (header_.version != kTierVersion)
            throw TierError(TierError::Code::IoFailure, "layout: unsupported version");
        if (slot_align != kTierSlotAlign || index_align != kTierIndexAlign)
            throw TierError(TierError::Code::IoFailure, "layout: unsupported alignment");
        const std::uint32_t crc = crc32(hdr, 0x28);
        if (crc != hdr_crc_stored)
            throw TierError(TierError::Code::IoFailure, "layout: header CRC mismatch");

        if (header_.n_groups > (1u << 24))
            throw TierError(TierError::Code::IoFailure, "layout: implausible n_groups");
        std::vector<std::uint8_t> index(
            static_cast<std::size_t>(header_.n_groups) * kTierIndexEntrySize);
        std::int64_t ri = FallbackIo::read(fd_, header_.index_offset, index.data(),
                                           static_cast<std::uint32_t>(index.size()));
        if (ri != static_cast<std::int64_t>(index.size()))
            throw TierError(TierError::Code::IoFailure, "layout: short index read");
        entries_.reserve(header_.n_groups);
        std::uint64_t prev_end = header_.data_offset;
        for (std::uint32_t i = 0; i < header_.n_groups; ++i) {
            TierGroupIndexEntry e = decode_entry(index.data() + i * kTierIndexEntrySize);
            // §2.2: data_offset монотонно растёт, слоты без дыр (кроме выравнивания).
            if (i == 0 && e.data_offset != header_.data_offset)
                throw TierError(TierError::Code::IoFailure, "layout: first group offset != data_offset");
            if (i > 0 && e.data_offset != prev_end)
                throw TierError(TierError::Code::IoFailure, "layout: hole/non-monotonic data_offset");
            if (e.slot_bytes == 0 || e.slot_bytes % kTierSlotAlign != 0)
                throw TierError(TierError::Code::IoFailure, "layout: bad slot_bytes");
            entries_.push_back(e);
            prev_end = e.data_offset + e.slot_bytes;
        }
    } catch (...) {
        ::close(fd_);
        fd_ = -1;
        throw;
    }
}

TierLayoutReader::~TierLayoutReader() {
    if (fd_ >= 0) ::close(fd_);
}

const TierGroupIndexEntry* TierLayoutReader::find(std::uint32_t group_id) const {
    for (const auto& e : entries_)
        if (e.group_id == group_id) return &e;
    return nullptr;
}

void TierLayoutReader::read_group(std::uint32_t group_id, void* dst) {
    const TierGroupIndexEntry* e = find(group_id);
    if (!e)
        throw TierError(TierError::Code::IoFailure, "layout: group not found");
    std::int64_t r = FallbackIo::read(fd_, e->data_offset, dst,
                                      static_cast<std::uint32_t>(e->slot_bytes));
    io_bytes_ += static_cast<std::uint64_t>(r > 0 ? r : 0);
    ++io_ops_;
    if (r != static_cast<std::int64_t>(e->slot_bytes))
        throw TierError(TierError::Code::IoFailure, "layout: short group read");
    // CRC — при первом чтении группы (ADR-12 §2.3).
    if (std::find(verified_.begin(), verified_.end(), group_id) == verified_.end()) {
        if (crc32(dst, e->slot_bytes) != e->crc32)
            throw TierError(TierError::Code::IoFailure,
                            "layout: group CRC mismatch (payload corrupted)");
        verified_.push_back(group_id);
    }
}

} // namespace stell::tiers

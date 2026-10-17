#include "stelltier_format.hpp"

#include <cstdio>
#include <cstring>
#include <ctime>

#include "crc32.hpp"
#include "stell/models/weights.hpp"

namespace stell::models {
namespace {

[[maybe_unused]] void put_u16(std::uint8_t* p, std::uint16_t v) {
    p[0] = static_cast<std::uint8_t>(v & 0xFF);
    p[1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
}

void put_u32(std::uint8_t* p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
}

void put_u64(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
}

[[maybe_unused]] std::uint16_t get_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

std::uint32_t get_u32(const std::uint8_t* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
    return v;
}

std::uint64_t get_u64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

std::uint64_t align_up64(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

}  // namespace

std::uint8_t stelltier_role_code(stell::tiers::Role r) {
    switch (r) {
        case stell::tiers::Role::Attn: return 0;
        case stell::tiers::Role::Expert: return 1;
        case stell::tiers::Role::Kv: return 2;
    }
    return 0xFF;
}

std::uint8_t stelltier_dtype_code(stell::tiers::DType d) {
    switch (d) {
        case stell::tiers::DType::F16: return 0;
        case stell::tiers::DType::F32: return 1;
        case stell::tiers::DType::E4M3: return 2;
        case stell::tiers::DType::NVFP4: return 3;
        case stell::tiers::DType::W4A16: return 4;
    }
    return 0xFF;
}

stell::tiers::Role stelltier_role_decode(std::uint8_t c) {
    switch (c) {
        case 0: return stell::tiers::Role::Attn;
        case 1: return stell::tiers::Role::Expert;
        case 2: return stell::tiers::Role::Kv;
    }
    throw LoaderError(LoaderError::Code::SchemaMismatch, "stelltier: unknown role code");
}

stell::tiers::DType stelltier_dtype_decode(std::uint8_t c) {
    switch (c) {
        case 0: return stell::tiers::DType::F16;
        case 1: return stell::tiers::DType::F32;
        case 2: return stell::tiers::DType::E4M3;
        case 3: return stell::tiers::DType::NVFP4;
        case 4: return stell::tiers::DType::W4A16;
    }
    throw LoaderError(LoaderError::Code::SchemaMismatch, "stelltier: unknown dtype code");
}

std::uint32_t stelltier_header_crc32(
    const std::array<std::uint8_t, kStellTierHeaderSize>& header) {
    // Поля 0x00..0x28 включительно (т.е. байты [0x00..0x2C) — до поля header_crc32).
    return crc32_ieee(header.data(), 0x2C);
}

void stelltier_build_header(std::uint32_t dtype_profile, const std::string& model_name,
                            std::uint32_t n_groups, std::uint64_t index_offset,
                            std::uint64_t data_offset,
                            std::array<std::uint8_t, kStellTierHeaderSize>* out) {
    std::array<std::uint8_t, kStellTierHeaderSize>& h = *out;
    h.fill(0);
    std::memcpy(h.data(), "STLTIER", 7);           // magic "STLTIER\0" (7 байт + нулевой)
    put_u32(h.data() + 0x08, 1);                    // version = 1
    put_u32(h.data() + 0x0C, 1);                    // flags: бит 0 — CRC-таблица в индексе
    put_u64(h.data() + 0x10, static_cast<std::uint64_t>(std::time(nullptr)));
    put_u32(h.data() + 0x18, kStellTierSlotAlign);   // slot_align = 4096
    put_u32(h.data() + 0x1C, kStellTierIndexAlign);  // index_align = 512
    put_u32(h.data() + 0x20, dtype_profile);
    put_u32(h.data() + 0x24, n_groups);
    put_u32(h.data() + 0x28, n_groups);             // n_tensors (по 1 тензору на группу)
    // header_crc32 (0x2C) — ниже
    put_u64(h.data() + 0x30, index_offset);
    put_u64(h.data() + 0x38, data_offset);
    std::memcpy(h.data() + 0x40, model_name.data(), std::min(model_name.size(), std::size_t{96}));
    put_u32(h.data() + 0x2C, stelltier_header_crc32(h));
}

// ВНИМАНИЕ (ADR-12 §2.2, неконсистентность контракта): запись = 64 байта,
// но name[20] на 0x30 даёт 0x44 — за пределами записи. Реализация: поле имени
// 16 байт (0x30..0x40), имя обрезается; требуется ADR-кларификация.
void stelltier_build_entry(const StellTierEntry& e,
                           std::array<std::uint8_t, kStellTierEntrySize>* out) {
    std::array<std::uint8_t, kStellTierEntrySize>& p = *out;
    p.fill(0);
    put_u32(p.data() + 0x00, e.group_id);
    p[0x04] = e.role;
    p[0x05] = e.dtype;
    p[0x06] = static_cast<std::uint8_t>(e.tensor_count);
    p[0x07] = 0;  // reserved
    put_u64(p.data() + 0x08, e.data_offset);
    put_u64(p.data() + 0x10, e.slot_bytes);
    put_u32(p.data() + 0x18, e.shape_rank);
    for (int i = 0; i < 4; ++i) put_u32(p.data() + 0x1C + 4 * i, e.shape[i]);
    put_u32(p.data() + 0x2C, e.crc32);
    std::memcpy(p.data() + 0x30, e.name.data(), std::min(e.name.size(), std::size_t{16}));
}

StellTierIndex stelltier_read_header_and_index(const std::string& path,
                                               std::uint64_t* file_size_out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw LoaderError(LoaderError::Code::IoFailure, "stelltier: cannot open " + path);
    struct FileCloser {
        std::FILE* f;
        ~FileCloser() { std::fclose(f); }
    } closer{f};

    std::array<std::uint8_t, kStellTierHeaderSize> h{};
    if (std::fread(h.data(), 1, h.size(), f) != h.size())
        throw LoaderError(LoaderError::Code::IoFailure, "stelltier: truncated header");
    if (std::memcmp(h.data(), "STLTIER", 7) != 0)
        throw LoaderError(LoaderError::Code::SchemaMismatch, "stelltier: bad magic");
    const std::uint32_t version = get_u32(h.data() + 0x08);
    if (version != 1)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "stelltier: unsupported version " + std::to_string(version));
    if (crc32_ieee(h.data(), 0x2C) != get_u32(h.data() + 0x2C))
        throw LoaderError(LoaderError::Code::ChecksumMismatch, "stelltier: header CRC mismatch");
    if (get_u32(h.data() + 0x18) != kStellTierSlotAlign ||
        get_u32(h.data() + 0x1C) != kStellTierIndexAlign)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "stelltier: unexpected slot/index alignment");

    StellTierIndex idx;
    idx.dtype_profile = get_u32(h.data() + 0x20);
    idx.n_groups = get_u32(h.data() + 0x24);
    idx.data_offset = get_u64(h.data() + 0x38);
    idx.model_name.assign(reinterpret_cast<const char*>(h.data() + 0x40), 96);
    while (!idx.model_name.empty() && idx.model_name.back() == '\0') idx.model_name.pop_back();

    // Индекс: сразу после заголовка, выровнен на 512 (конкретно — 0x200).
    const std::uint64_t index_offset = get_u64(h.data() + 0x30);
    if (index_offset % kStellTierIndexAlign != 0)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "stelltier: index_offset not aligned to index_align");
    std::fseek(f, static_cast<long>(index_offset), SEEK_SET);
    idx.entries.resize(idx.n_groups);
    std::uint64_t prev_end = idx.data_offset;
    std::uint64_t cursor = index_offset;
    for (std::uint32_t i = 0; i < idx.n_groups; ++i) {
        std::array<std::uint8_t, kStellTierEntrySize> e{};
        if (std::fread(e.data(), 1, e.size(), f) != e.size())
            throw LoaderError(LoaderError::Code::IoFailure, "stelltier: truncated index");
        cursor += e.size();
        StellTierEntry& ent = idx.entries[i];
        ent.group_id = get_u32(e.data() + 0x00);
        ent.role = e[0x04];
        ent.dtype = e[0x05];
        ent.tensor_count = e[0x06];
        ent.data_offset = get_u64(e.data() + 0x08);
        ent.slot_bytes = get_u64(e.data() + 0x10);
        ent.shape_rank = get_u32(e.data() + 0x18);
        for (int k = 0; k < 4; ++k) ent.shape[k] = get_u32(e.data() + 0x1C + 4 * k);
        ent.crc32 = get_u32(e.data() + 0x2C);
        ent.name.assign(reinterpret_cast<const char*>(e.data() + 0x30), 16);
        while (!ent.name.empty() && ent.name.back() == '\0') ent.name.pop_back();

        // ADR-12 §2.2: порядок = порядок data-секции, монотонный, без дыр.
        if (ent.data_offset != align_up64(prev_end, kStellTierSlotAlign))
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "stelltier: data_offset hole/misalignment in group " +
                                  std::to_string(i));
        if (ent.slot_bytes % kStellTierSlotAlign != 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "stelltier: slot_bytes not multiple of slot_align");
        prev_end = ent.data_offset + ent.slot_bytes;
    }
    (void)cursor;

    std::fseek(f, 0, SEEK_END);
    const long fsz = std::ftell(f);
    if (fsz < 0) throw LoaderError(LoaderError::Code::IoFailure, "stelltier: cannot size file");
    const std::uint64_t file_size = static_cast<std::uint64_t>(fsz);
    if (file_size_out) *file_size_out = file_size;
    if (idx.n_groups > 0 && file_size < prev_end)
        throw LoaderError(LoaderError::Code::IoFailure, "stelltier: data section truncated");
    return idx;
}

}  // namespace stell::models

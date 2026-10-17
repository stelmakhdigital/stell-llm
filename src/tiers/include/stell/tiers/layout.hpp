// stell-tier layout v1 (ADR-12 §2): читатель/писатель формата NVMe-тира.
// Один файл <uuid>.stelltier на модель/профиль; все поля little-endian.
// Разбиение модели на группы — ответственность models (ADR-12 §1); tiers
// принимает готовые группы и хранит ровно то, что нужно ему.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <stell/tiers/errors.hpp>
#include <stell/tiers/types.hpp>

namespace stell::tiers {

// --- константы формата (ADR-12 §2) -----------------------------------------
constexpr std::size_t   kTierSlotAlign    = 4096; // выравнивание данных групп
constexpr std::size_t   kTierIndexAlign   = 512;  // выравнивание записей индекса
constexpr std::size_t   kTierHeaderSize   = 512;  // ровно 512 байт, выровнен на 4096
constexpr std::size_t   kTierIndexEntrySize = 64; // каждая запись индекса — 64 байта
constexpr std::size_t   kTierModelNameMax = 96;   // char[96] в заголовке
constexpr std::size_t   kTierGroupNameMax = 16;   // см. примечание о несоответствии ADR-12 §2.2
constexpr std::uint32_t kTierVersion     = 1;

constexpr char kTierMagic[8] = {'S', 'T', 'L', 'T', 'I', 'E', 'R', '\0'};

// Коды dtype в записи индекса (ADR-12 §2.2): совпадают с порядком ADR-08 DType.
inline std::uint8_t dtype_code(DType d) {
    return static_cast<std::uint8_t>(static_cast<int>(d) + 0); // F16=0..W4A16=4
}
// Коды role в записи индекса (ADR-12 §2.2): 0=attn, 1=expert, 2=kv
// (порядок ADR-08 Role: Expert=0, Attn=1, Kv=2 — поэтому явное отображение).
inline std::uint8_t role_code(Role r) {
    switch (r) {
        case Role::Attn:   return 0;
        case Role::Expert: return 1;
        case Role::Kv:     return 2;
    }
    return 0;
}
inline Role role_from_code(std::uint8_t c) {
    switch (c) {
        case 0: return Role::Attn;
        case 1: return Role::Expert;
        case 2: return Role::Kv;
    }
    throw TierError(TierError::Code::IoFailure, "layout: unknown role code");
}
inline DType dtype_from_code(std::uint8_t c) {
    if (c > 4) throw TierError(TierError::Code::IoFailure, "layout: unknown dtype code");
    return static_cast<DType>(c);
}

// --- заголовок (512 байт, ADR-12 §2) ----------------------------------------
struct TierLayoutHeader {
    std::uint32_t version     = kTierVersion;
    std::uint32_t flags       = 0;          // бит 0: checksum-таблица в конце (не пишем)
    std::uint64_t created_ts  = 0;          // unix, метка конвертера
    std::uint32_t dtype_profile = 0;        // 1=F16, 2=NVFP4, 3=W4A16 (ADR-03)
    std::uint32_t n_groups    = 0;
    std::uint32_t n_tensors   = 0;          // суммарно логических тензоров
    std::uint64_t index_offset = kTierHeaderSize; // §2.2
    std::uint64_t data_offset = 0;          // смещение первой группы данных
    std::string   model_name;               // только для человека; tiers не читает
};

// --- запись индекса групп (64 байта, ADR-12 §2.2) ---------------------------
struct TierGroupIndexEntry {
    std::uint32_t group_id = 0;
    std::uint8_t  role     = 0;
    std::uint8_t  dtype    = 0;
    std::uint8_t  tensor_count = 0;
    std::uint64_t data_offset = 0;      // выровнен на slot_align, монотонно растёт
    std::uint64_t slot_bytes  = 0;      // кратен 4096
    std::uint32_t shape_rank  = 0;
    std::array<std::uint32_t, 4> shape{}; // до 4 измерений
    std::uint32_t crc32       = 0;      // CRC32 полезной нагрузки группы
    std::string   name;                 // человекочитаемое, обрезано
};

// --- CRC32 (IEEE, poly 0xEDB88320) ------------------------------------------
std::uint32_t crc32(const void* data, std::size_t len) noexcept;
std::uint32_t crc32_update(std::uint32_t crc, const void* data, std::size_t len) noexcept;

// --- писатель ---------------------------------------------------------------
// Гарантии: заголовок ровно 512 байт + CRC; индекс выровнен на 512; данные
// выровнены на 4096, data_offset монотонный без дыр; slot_bytes кратен 4096;
// CRC группы считается по слоту целиком (данные + нулевое padding).
class TierLayoutWriter {
public:
    // header.n_groups должен быть задан заранее (конвертер знает размер плана):
    // офсеты data-секции вычисляются до добавления групп (монотонный порядок §2.2).
    TierLayoutWriter(std::string path, TierLayoutHeader header);
    ~TierLayoutWriter(); // незавершённый файл удаляется (finalize обязателен)

    // Добавляет группу: payload — данные, записывается слотом (padding нулями).
    void add_group(const TensorGroup& g, std::uint8_t tensor_count,
                   const void* payload, std::size_t payload_len);

    void finalize(); // пишет заголовок+индекс с финальными офсетами, закрывает

private:
    std::string path_;
    TierLayoutHeader header_;
    std::vector<TierGroupIndexEntry> entries_;
    std::uint64_t next_offset_ = 0;   // курсор data-секции
    int fd_ = -1;
    bool finalized_ = false;
};

// --- читатель ---------------------------------------------------------------
class TierLayoutReader {
public:
    explicit TierLayoutReader(std::string path); // TierError(IoFailure) при порче
    ~TierLayoutReader();
    TierLayoutReader(const TierLayoutReader&) = delete;
    TierLayoutReader& operator=(const TierLayoutReader&) = delete;

    const TierLayoutHeader& header() const { return header_; }
    const std::vector<TierGroupIndexEntry>& index() const { return entries_; }
    const TierGroupIndexEntry* find(std::uint32_t group_id) const;

    // Читает слот группы целиком в dst (size ≥ slot_bytes).
    // CRC проверяется при первом чтении группы (ADR-12 §2.3) — один раз
    // за жизнь файла в кеше tiers; повторные чтения не перепроверяют.
    void read_group(std::uint32_t group_id, void* dst);

    int fd() const { return fd_; }
    std::uint64_t io_bytes() const { return io_bytes_; }
    std::uint64_t io_ops() const { return io_ops_; }

private:
    std::string path_;
    int fd_ = -1;
    TierLayoutHeader header_;
    std::vector<TierGroupIndexEntry> entries_;
    std::vector<std::uint32_t> verified_;   // группы с проверенной CRC
    std::uint64_t io_bytes_ = 0;
    std::uint64_t io_ops_ = 0;
};

} // namespace stell::tiers

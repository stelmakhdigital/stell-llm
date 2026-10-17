// ============================================================================
// stell-tier layout v1 (ADR-12 §2) — writer + reader на стороне models.
//
// Единый кодовый путь для конвертера (writer) и лоадера StellTier (reader).
// У tiers свой reader (Tier Manager, поток B) — код не дублируется, общая
// для сторон только спецификация ADR-12 §2 и CRC32.
//
// Все поля little-endian. Заголовок ровно 512 байт (выравнивание файла 4096 —
// начало), индекс после заголовка (записи 64 байта, выровнен на 512),
// данные групп выровнены на 4096, монотонно, без дыр.
// header_crc32 — CRC32 байтов [0x00..0x2C) (поля до header_crc32 включительно
// n_groups на 0x28..0x2B); crc32 группы — CRC32 её слота (payload + padding
// выравнивания, т.е. ровно slot_bytes байт как записано в файле).
// ============================================================================
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "stell/models/tiers_compat.hpp"

namespace stell::models {

constexpr std::size_t kStellTierHeaderSize = 512;
constexpr std::size_t kStellTierEntrySize = 64;
constexpr std::size_t kStellTierSlotAlign = 4096;
constexpr std::size_t kStellTierIndexAlign = 512;

struct StellTierEntry {
    std::uint32_t group_id = 0;
    std::uint8_t role = 0;        // 0=attn, 1=expert, 2=kv (ADR-12 §2.2)
    std::uint8_t dtype = 0;       // 0=F16, 1=F32, 2=E4M3, 3=NVFP4, 4=W4A16
    std::uint32_t tensor_count = 1;
    std::uint64_t data_offset = 0;
    std::uint64_t slot_bytes = 0;
    std::uint32_t shape_rank = 0;
    std::uint32_t shape[4] = {0, 0, 0, 0};
    std::uint32_t crc32 = 0;      // CRC32 слота (payload + padding)
    std::string name;             // обрезается до 20 байт (ADR-12 §2.2)
};

struct StellTierIndex {
    std::uint32_t dtype_profile = 0;  // 1=F16, 2=NVFP4, 3=W4A16 (ADR-12 §2)
    std::uint32_t n_groups = 0;
    std::uint64_t data_offset = 0;
    std::string model_name;           // обрезано до 96 байт, tiers не читает
    std::vector<StellTierEntry> entries;
};

// Кодирование полей (ADR-12 §2.2).
std::uint8_t stelltier_role_code(stell::tiers::Role r);
std::uint8_t stelltier_dtype_code(stell::tiers::DType d);
stell::tiers::Role stelltier_role_decode(std::uint8_t c);
stell::tiers::DType stelltier_dtype_decode(std::uint8_t c);

// CRC32 заголовка: байты [0x00..0x2C) уже собранного заголовка.
std::uint32_t stelltier_header_crc32(const std::array<std::uint8_t, kStellTierHeaderSize>& header);

// Сборка заголовка (без CRC данных групп; header_crc32 проставляется здесь).
// dtype_profile/model_name/n_groups — из Index; data_offset обязан быть
// согласованным с записями (заполняет writer конвертера).
void stelltier_build_header(std::uint32_t dtype_profile, const std::string& model_name,
                            std::uint32_t n_groups, std::uint64_t index_offset,
                            std::uint64_t data_offset,
                            std::array<std::uint8_t, kStellTierHeaderSize>* out);

// Сборка записи индекса (64 байта, little-endian).
void stelltier_build_entry(const StellTierEntry& e,
                           std::array<std::uint8_t, kStellTierEntrySize>* out);

// Парсинг файла (reader лоадера StellTier): проверяет magic/version/header_crc32,
// выравнивание/монотонность data_offset, отсутствие дыр; ошибки — LoaderError
// (ChecksumMismatch / IoFailure / SchemaMismatch). Возвращает индекс.
// file_size — для проверки усечения. Если index_crc_table — CRC групп НЕ проверяет.
StellTierIndex stelltier_read_header_and_index(const std::string& path, std::uint64_t* file_size);

}  // namespace stell::models

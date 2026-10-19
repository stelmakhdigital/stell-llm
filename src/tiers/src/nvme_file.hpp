// NVMe-тир: stell-tier файл (ADR-12) поверх io_uring (полный путь:
// очередь, SQE/CQE, выравнивание 4096; ядро без io_uring → fallback pread).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <stell/tiers/layout.hpp>

#include "io_uring.hpp"

namespace stell::tiers {

// Бэкенд I/O одного открытого stell-tier файла.
//  - io_uring: prep_read → submit_and_wait → CQE (основной путь, Linux);
//  - fallback: pread (ядра без io_uring).
// CRC-политика (ADR-12 §2.3, round 3): header_crc32 проверяется всегда;
// payload-CRC групп — при первом чтении и только при verify_payload_crc
// (горячий путь менеджера по умолчанию НЕ пересчитывает CRC; верификация —
// вне пути декода: verify_group/verify_slots или STELL_TIER_VERIFY_CRC=1).
class TierNvmeFile {
public:
    explicit TierNvmeFile(std::string path, bool verify_payload_crc = true);
    ~TierNvmeFile();
    TierNvmeFile(const TierNvmeFile&) = delete;
    TierNvmeFile& operator=(const TierNvmeFile&) = delete;

    const TierLayoutHeader& header() const { return reader_->header(); }
    const std::vector<TierGroupIndexEntry>& index() const { return reader_->index(); }
    const TierGroupIndexEntry* find(std::uint32_t group_id) const { return reader_->find(group_id); }

    // Читает слот группы целиком в dst (size ≥ slot_bytes) через io_uring.
    void read_group(std::uint32_t group_id, void* dst);

    // Публичный verify-проход (ADR-12 §2.3, round 3): внепутевая сверка
    // payload-CRC одного слота с индексом — читает слот в scratch-буфер и
    // пересчитывает CRC независимо от verified_; успех помечает группу
    // проверенной (последующий горячий путь с verify=on не перепроверяет).
    // «Холодная» операция: не вызывать параллельно с pin/prefetch
    // (внутреннее состояние verified_ не синхронизировано с горячим путём —
    // та же дисциплина, что set_plan, ADR-08 §4). Не учитывается в
    // io_bytes/io_ops горячей статистики тир↔тир.
    void verify_group(std::uint32_t group_id);

    std::uint64_t io_bytes() const { return io_bytes_; }
    std::uint64_t io_ops() const { return io_ops_; }
    // Метрика R1: сколько раз payload-CRC фактически пересчитывался
    // (0 при verify=off и без verify-прохода — фиксация «hot path без CRC»).
    std::uint64_t crc_checks() const { return crc_checks_; }

private:
    std::unique_ptr<TierLayoutReader> reader_;
    std::unique_ptr<IoUring> ring_;   // nullptr → fallback pread
    bool verify_crc_ = true;
    std::vector<std::uint32_t> verified_;
    std::uint64_t io_bytes_ = 0;
    std::uint64_t io_ops_ = 0;
    std::uint64_t crc_checks_ = 0;
    std::uint64_t next_tag_ = 1;
};

// Имя активного бэкенда (для тестов/дашборда): "io_uring" или "fallback".
std::string nvme_io_backend();

} // namespace stell::tiers

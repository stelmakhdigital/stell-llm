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
// CRC группы проверяется при первом чтении (ADR-12 §2.3) — один раз за жизнь
// файла в кеше tiers, повторные чтения не перепроверяют.
class TierNvmeFile {
public:
    explicit TierNvmeFile(std::string path);
    ~TierNvmeFile();
    TierNvmeFile(const TierNvmeFile&) = delete;
    TierNvmeFile& operator=(const TierNvmeFile&) = delete;

    const TierLayoutHeader& header() const { return reader_->header(); }
    const std::vector<TierGroupIndexEntry>& index() const { return reader_->index(); }
    const TierGroupIndexEntry* find(std::uint32_t group_id) const { return reader_->find(group_id); }

    // Читает слот группы целиком в dst (size ≥ slot_bytes) через io_uring.
    void read_group(std::uint32_t group_id, void* dst);

    std::uint64_t io_bytes() const { return io_bytes_; }
    std::uint64_t io_ops() const { return io_ops_; }

private:
    std::unique_ptr<TierLayoutReader> reader_;
    std::unique_ptr<IoUring> ring_;   // nullptr → fallback pread
    std::vector<std::uint32_t> verified_;
    std::uint64_t io_bytes_ = 0;
    std::uint64_t io_ops_ = 0;
    std::uint64_t next_tag_ = 1;
};

// Имя активного бэкенда (для тестов/дашборда): "io_uring" или "fallback".
std::string nvme_io_backend();

} // namespace stell::tiers

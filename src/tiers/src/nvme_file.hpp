// NVMe-тир: stell-tier файл (ADR-12) поверх io_uring (полный путь:
// очередь, SQE/CQE, выравнивание 4096; ядро без io_uring → fallback pread).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
//
// [R2] round 4 — O_DIRECT-бэкенд (planner [Q6]): при odirect=true файл для
// данных групп открывается ОТДЕЛЬНЫМ fd с O_DIRECT (заголовок/индекс читает
// TierLayoutReader через обычный buffered fd — маленькие невыровненные
// чтения; O_DIRECT-кон тракт только данные слотов). Честная семантика:
//  - выравнивание: O_DIRECT требует кратности буфера, офсета и длины
//    логическому блоку файловой системы. Layout v1 (ADR-12) гарантирует
//    data_offset кратно 4096 и slot_bytes кратно 4096 — офсет/длина всегда
//    валидны. Логический блок FS кодом НЕ запрашивается (ioctl на regular
//    файле не универсален) — консервативная фиксация 4096 (= kTierSlotAlign,
//    покрывает lsb 512/2048/4096 распространённых FS: ext4/xfs). Честно
//    зафиксировано; если реальный lsb больше — O_DIRECT вернёт EINVAL,
//    что будет честной ошибкой чтения, а не порчей данных.
//  - буфер назначения: если dst выровнен на 4096 (RAM-буферы менеджера
//    выделяются posix_memalign(4096); cudaHostAlloc — page-aligned) — чтение
//    ИДЁТ ПРЯМО в dst (без лишней копии — выигрыш O_DIRECT). Если dst не
//    выровнен (сырые тесты/внешние вызовы) — минимальное выровненное
//    staging-чтение: буфер posix_memalign(4096) + memcpy в dst. Контракт
//    read_group (произвольный dst) не меняется.
//  - отсутствие readahead: честное свойство O_DIRECT (детерминированный
//    cold); компенсируется deep-prefetch (k групп в полёте, [R2b]).
//  - fallback: если O_DIRECT открыть не удалось (FS/ядро не поддерживают,
//    напр. tmpfs) — чтение идёт через buffered fd, odirect_active() == false
//    (прогон обязан честно помечаться; тест проверяет odirect_active()).
class TierNvmeFile {
public:
    // odirect=false → buffered (дефолт, прецедент сохранён); true → попытка
    // O_DIRECT (при неудаче открытия — тихий откат на buffered с честным
    // odirect_active()==false).
    explicit TierNvmeFile(std::string path, bool verify_payload_crc = true,
                          bool odirect = false);
    ~TierNvmeFile();
    TierNvmeFile(const TierNvmeFile&) = delete;
    TierNvmeFile& operator=(const TierNvmeFile&) = delete;

    const TierLayoutHeader& header() const { return reader_->header(); }
    const std::vector<TierGroupIndexEntry>& index() const { return reader_->index(); }
    const TierGroupIndexEntry* find(std::uint32_t group_id) const { return reader_->find(group_id); }

    // [R2] round 4: O_DIRECT-тракт данных реально активен (fd открыт с
    // O_DIRECT). false → отката на buffered (FS не поддерживает O_DIRECT).
    bool odirect_active() const { return odirect_active_; }

    // Читает слот группы целиком в dst (size ≥ slot_bytes) через io_uring.
    void read_group(std::uint32_t group_id, void* dst);

    // [R2b] round 4 — батч-чтение k групп одним submit (k в полёте):
    // reqs — пары (group_id, dst); dst может быть любым (невыровненные буферы
    // читаются через выровненный staging). Возвращает исход по каждому
    // запросу (bytes ≥ 0 и err пусто — успех). НИЧЕГО не бросает по
    // отдельным группам: ошибки — в исходах (изоляция промахов батча);
    // TierError возможен только при отказе самого кольца/файла.
    struct BatchOutcome {
        std::int64_t bytes = 0;   // прочитано байтов (== slot_bytes при успехе)
        std::string  err;         // пусто — успех; иначе сообщение ошибки
    };
    std::vector<BatchOutcome> read_groups(
        const std::vector<std::pair<std::uint32_t, void*>>& reqs);

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
    // fd для чтения данных слотов (O_DIRECT-тракт или buffered reader).
    int data_fd() const;
    // Итог одного чтения: CRC-once при verify=on; err из исхода батча.
    // Возвращает байтов прочитано (≥ 0) — или бросает TierError(IoFailure).
    std::int64_t finish_read(std::uint32_t group_id, const TierGroupIndexEntry* e,
                             std::int64_t res, void* dst);

    std::unique_ptr<TierLayoutReader> reader_;
    std::unique_ptr<IoUring> ring_;   // nullptr → fallback pread
    std::string path_;                // для отдельного O_DIRECT-фд данных
    int odirect_fd_ = -1;             // O_RDONLY|O_DIRECT; -1 → не активен
    bool odirect_active_ = false;
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

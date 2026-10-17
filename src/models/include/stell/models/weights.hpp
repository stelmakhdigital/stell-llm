// ============================================================================
// Загрузчики весов — контракт ADR-09 §3.
//
// Форматы (ADR-03): Safetensors | GGUF | NVFP4W4A16 | StellTier.
//   - Safetensors/GGUF — входные форматы; лоадер резолвит source_pattern
//     каждой группы, проверяет dtype/shape против плана и заполняет буфер
//     группы (несовпадение = исключение, тишина запрещена).
//   - NVFP4W4A16 — профильный вариант лоадера (ADR-03: квантованное хранение
//     не является отдельным входным форматом): читает safetensors/GGUF
//     (F16/F32-тензор) и на read() квантует в целевой dtype группы
//     (NVFP4 | W4A16). Dequant инференсом выполняют ядра engine/kernels —
//     лоадер только упаковывает (референсный dequant для тестов — quant.hpp).
//   - StellTier — чтение layout ADR-12 §2 (writer — модуль models, ADR-12 §4).
//
// Упаковочные профили квантования (блок = 32 элемента), единый для лоадера
// NVFP4W4A16 и конвертера stell-tier (см. quant.hpp):
//   NVFP4: 16 байт нибблов E2M1n (nibble = sign<<3 | idx, idx 0..7 →
//          0,0.5,1,1.5,2,3,4,6) + 1 байт scale E4M3 на блок;
//   W4A16: 16 байт нибблов int4 (nibble = q + 8, q ∈ [-8,7]) +
//          2 байта scale F16 на блок; dequant: (nibble - 8) * scale.
// ============================================================================
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "stell/models/description.hpp"
#include "stell/models/tiers_compat.hpp"

namespace stell::models {

enum class WeightsFormat { Safetensors, GGUF, NVFP4W4A16, StellTier };

const char* weights_format_name(WeightsFormat f);

class IWeightsLoader;
class IWeightsSource;

// Единая политика ошибок лоадера — исключения (ADR-09 §3, ADR-08 §3).
class LoaderError : public std::runtime_error {
public:
    enum class Code {
        SchemaMismatch,     // dtype/shape/план не согласованы
        TensorNotFound,     // source_pattern не разрешился в тензора файла
        ChecksumMismatch,   // CRC-проверка провалена (stell-tier)
        UnsupportedQuant,   // dtype/профиль квантования не поддержан лоадером
        IoFailure           // I/O: файл усечён, чтение не удалось
    };

    LoaderError(Code c, const std::string& what)
        : std::runtime_error(what), code_(c) {}
    Code code() const noexcept { return code_; }

private:
    Code code_;
};

class IWeightsLoader {
public:
    virtual ~IWeightsLoader() = default;
    virtual WeightsFormat format() const = 0;

    // Открывает источник весов и возвращает план групп, согласованный с описанием
    // (резолюция source_pattern — энергичная, при open: fail-fast, ADR-09 §3).
    virtual std::unique_ptr<IWeightsSource> open(const std::string& uri,
                                                 const ModelDescription& desc) = 0;
};

// Источник весов: групповое чтение для tiers (CPU-путь обязан работать без GPU).
class IWeightsSource {
public:
    virtual ~IWeightsSource() = default;
    // Чтение группы по её GroupId. Контракт:
    //   dst_bytes == group.slot_bytes (иначе LoaderError::SchemaMismatch);
    //   буфер заполняется payload'ом группы в целевом dtype плана, хвост слота
    //   (выравнивание до slot_bytes) — нулями.
    // Ошибки: LoaderError{TensorNotFound, SchemaMismatch, UnsupportedQuant,
    //                    ChecksumMismatch, IoFailure}.
    // KV-группы (role=Kv) весов не имеют → LoaderError::IoFailure при read().
    virtual void read(const stell::tiers::GroupKey& key, void* dst, std::size_t dst_bytes) = 0;
    virtual void close() = 0;
};

// Фабрика лоадеров по формату.
std::unique_ptr<IWeightsLoader> make_loader(WeightsFormat f);

// Определение формата по URI: расширение (.safetensors/.gguf/.stelltier) либо
// магия файла ("GGUF", "STLTIER\0"), иначе — safetensors (8-байтовая длина заголовка).
WeightsFormat detect_format(const std::string& uri);

}  // namespace stell::models

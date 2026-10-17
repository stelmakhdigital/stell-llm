// Лоадер GGUF v3 (subset) — контракт ADR-09 §3.
//
// Поддержано: magic "GGUF", version 3, метаданные типов u32/i32/u64/i64/f32/
// f64/bool/string/array (массивы пропускаются), выравнивание из
// "general.alignment" (по умолчанию 32), dtype тензора F32(0)/F16(1) — прочие
// dtype (квантованные Q4_0 и т.п.) → UnsupportedQuant. Имена тензоров
// сравниваются с source_pattern описания буквально (GGUF-модели используют
// конвенции имён gguf/blk — для них описания генерируются конвертерами).
#include "tensor_file_formats.hpp"

#include <cstring>
#include <set>

namespace stell::models {
namespace {

constexpr std::uint32_t kGGUFMetaUint8 = 0, kGGUFMetaInt8 = 1;
constexpr std::uint32_t kGGUFMetaUint16 = 2, kGGUFMetaInt16 = 3;
constexpr std::uint32_t kGGUFMetaUint32 = 4, kGGUFMetaInt32 = 5;
constexpr std::uint32_t kGGUFMetaUint64 = 6, kGGUFMetaInt64 = 7;
constexpr std::uint32_t kGGUFMetaFloat32 = 8, kGGUFMetaFloat64 = 9;
constexpr std::uint32_t kGGUFMetaBool = 10, kGGUFMetaString = 11, kGGUFMetaArray = 12;

std::uint32_t read_u32_le(std::FILE* f) {
    std::uint8_t b[4];
    if (std::fread(b, 1, 4, f) != 4)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(b[i]) << (8 * i);
    return v;
}

std::uint64_t read_u64_le_stream(std::FILE* f) {
    std::uint8_t b[8];
    if (std::fread(b, 1, 8, f) != 8)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[i]) << (8 * i);
    return v;
}

std::string read_gguf_string(std::FILE* f) {
    const std::uint64_t len = read_u64_le_stream(f);
    std::string s(len, '\0');
    if (len && std::fread(s.data(), 1, len, f) != len)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated string");
    return s;
}

void skip_meta_value(std::FILE* f, std::uint32_t type) {
    switch (type) {
        case kGGUFMetaUint8: case kGGUFMetaInt8: case kGGUFMetaBool: {
            std::uint8_t b; if (std::fread(&b, 1, 1, f) != 1)
                throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
            return;
        }
        case kGGUFMetaUint16: case kGGUFMetaInt16: {
            std::uint8_t b[2]; if (std::fread(b, 1, 2, f) != 2)
                throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
            return;
        }
        case kGGUFMetaUint32: case kGGUFMetaInt32: case kGGUFMetaFloat32:
            read_u32_le(f);
            return;
        case kGGUFMetaUint64: case kGGUFMetaInt64: case kGGUFMetaFloat64:
            read_u64_le_stream(f);
            return;
        case kGGUFMetaString:
            read_gguf_string(f);
            return;
        case kGGUFMetaArray: {
            const std::uint32_t elem = read_u32_le(f);
            const std::uint64_t count = read_u64_le_stream(f);
            for (std::uint64_t i = 0; i < count; ++i) {
                if (elem == kGGUFMetaString) read_gguf_string(f);
                else if (elem == kGGUFMetaArray)
                    throw LoaderError(LoaderError::Code::IoFailure,
                                      "gguf: nested arrays не поддержаны");
                else skip_meta_value(f, elem);
            }
            return;
        }
        default:
            throw LoaderError(LoaderError::Code::IoFailure,
                              "gguf: unsupported metadata value type " + std::to_string(type));
    }
}

}  // namespace

// Разбор заголовка GGUF (общая функция: лоадер + конвертер stell-tier).
TensorFileMap parse_gguf_header(std::FILE* f) {        
        std::uint8_t magic[4];
        read_exact(f, 0, magic, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch, "gguf: bad magic");
        if (std::fseek(f, 4, SEEK_SET) != 0)
            throw LoaderError(LoaderError::Code::IoFailure, "gguf: seek failed");
        const std::uint32_t version = read_u32_le(f);
        if (version != 3)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: unsupported version " + std::to_string(version) +
                                  " (subset: v3)");
        const std::uint64_t n_tensors = read_u64_le_stream(f);
        const std::uint64_t n_kv = read_u64_le_stream(f);

        std::uint64_t alignment = 32;
        for (std::uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = read_gguf_string(f);
            const std::uint32_t type = read_u32_le(f);
            if (key == "general.alignment" && type == kGGUFMetaUint32)
                alignment = read_u32_le(f);
            else
                skip_meta_value(f, type);
        }

        TensorFileMap map;
        struct PendingTensor {
            std::string name;
            std::uint32_t n_dims;
            std::vector<std::uint64_t> dims;
            std::int64_t dtype;
            std::uint64_t offset;  // относительный к базе данных
        };
        std::vector<PendingTensor> pending;
        for (std::uint64_t i = 0; i < n_tensors; ++i) {
            PendingTensor t;
            t.name = read_gguf_string(f);
            t.n_dims = read_u32_le(f);
            t.dims.resize(t.n_dims);
            for (std::uint32_t d = 0; d < t.n_dims; ++d) t.dims[d] = read_u64_le_stream(f);
            t.dtype = static_cast<std::int64_t>(read_u32_le(f));
            t.offset = read_u64_le_stream(f);
            pending.push_back(std::move(t));
        }
        // База данных: конец тензорных описаний, выровнен на alignment (GGUF v3).
        const long base_pos = std::ftell(f);
        if (base_pos < 0)
            throw LoaderError(LoaderError::Code::IoFailure, "gguf: cannot position");
        map.data_base = static_cast<std::uint64_t>(base_pos);
        map.data_base = (map.data_base + alignment - 1) / alignment * alignment;

        for (auto& t : pending) {
            TensorInfo info;
            std::size_t n = 1;
            for (auto d : t.dims) {
                info.shape.push_back(static_cast<std::int64_t>(d));
                n *= static_cast<std::size_t>(d);
            }
            switch (t.dtype) {
                case 0: info.dtype = stell::tiers::DType::F32; info.dtype_str = "F32"; info.itemsize = 4; break;
                case 1: info.dtype = stell::tiers::DType::F16; info.dtype_str = "F16"; info.itemsize = 2; break;
                default:
                    throw LoaderError(
                        LoaderError::Code::UnsupportedQuant,
                        "gguf: tensor '" + t.name + "': unsupported data_type " +
                            std::to_string(t.dtype) +
                            " (subset: F32/F16; квантованные GGUF — через конвертер)");
            }
            info.offset = map.data_base + t.offset;
            info.bytes = info.itemsize * n;
            map.by_name[t.name] = info;
        }
        return map;
}

class GgufSource : public FileWeightsSource {
public:
    using FileWeightsSource::FileWeightsSource;

protected:
    TensorFileMap parse_header() override { return parse_gguf_header(file()); }

    void do_fill(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                 void* dst, std::size_t dst_bytes) override {
        check_shape(g, t);
        if (!compatible_plain(g.dtype, t))
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "group '" + g.name + "': gguf tensor dtype '" + t.dtype_str +
                                  "' не совместим; для квантованых профилей используйте "
                                  "лоадер NVFP4W4A16 (ADR-03)");
        fill_plain(g, t, dst, dst_bytes);
    }

};

class GgufLoader : public IWeightsLoader {
public:
    WeightsFormat format() const override { return WeightsFormat::GGUF; }
    std::unique_ptr<IWeightsSource> open(const std::string& uri,
                                         const ModelDescription& desc) override {
        auto src = std::make_unique<GgufSource>(uri, desc);
        src->resolve_all();
        return src;
    }
};

IWeightsLoader* make_gguf_loader() { return new GgufLoader(); }

}  // namespace stell::models

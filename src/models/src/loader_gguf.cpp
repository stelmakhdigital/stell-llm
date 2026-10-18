// Лоадер GGUF v3 (subset) — контракт ADR-09 §3.
//
// Поддержано: magic "GGUF", version 3, метаданные типов u32/i32/u64/i64/f32/
// f64/bool/string (массивы пропускаются), выравнивание из "general.alignment"
// (по умолчанию 32), dtype тензора F32(0)/F16(1); round 2: Q4_0(20)/Q8_0(36)
// — квантованные GGUF-блоки, dequant → f32 → целевой dtype группы.
// Прочие dtype → UnsupportedQuant.
//
// Имена тензоров сравниваются с source_pattern описания буквально (GGUF-модели
// используют конвенции имён gguf/blk — описания генерируются конвертером
// gguf_description.cpp, нативные имена llama.cpp-GGUF; ADR-09 «Уточнение»).
//
// SHAPE (round 2): GGUF ne = {in, out} для матриц ({n_embd, vocab} для
// token_embd); байты C-порядка == row-major [out, in] — parse_gguf_header
// возвращает РАЗВЕРНУТЫЙ dims (логическую форму, HF-конвенция), байты без
// изменений. См. tensor_file_formats.hpp.
#include "tensor_file_formats.hpp"
#include "quant.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <vector>

namespace stell::models {

// --- общие читатели GGUF v3 (объявлены в tensor_file_formats.hpp) ----------

std::uint32_t gguf_read_u32(std::FILE* f) {
    std::uint8_t b[4];
    if (std::fread(b, 1, 4, f) != 4)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(b[i]) << (8 * i);
    return v;
}

std::uint64_t gguf_read_u64(std::FILE* f) {
    std::uint8_t b[8];
    if (std::fread(b, 1, 8, f) != 8)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[i]) << (8 * i);
    return v;
}

std::string gguf_read_string(std::FILE* f) {
    const std::uint64_t len = gguf_read_u64(f);
    std::string s(len, '\0');
    if (len && std::fread(s.data(), 1, len, f) != len)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated string");
    return s;
}

GgufMetaValue gguf_read_meta_value(std::FILE* f, std::uint32_t type) {
    GgufMetaValue v;
    v.type = type;
    switch (type) {
        case kGgufMetaUint8:
        case kGgufMetaInt8:
        case kGgufMetaBool: {
            std::uint8_t b;
            if (std::fread(&b, 1, 1, f) != 1)
                throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
            v.u64 = b;
            v.i64 = static_cast<std::int64_t>(b);
            return v;
        }
        case kGgufMetaUint16:
        case kGgufMetaInt16: {
            std::uint8_t b[2];
            if (std::fread(b, 1, 2, f) != 2)
                throw LoaderError(LoaderError::Code::IoFailure, "gguf: truncated");
            std::uint16_t u = 0;
            for (int i = 0; i < 2; ++i) u |= static_cast<std::uint16_t>(b[i]) << (8 * i);
            v.u64 = u;
            v.i64 = static_cast<std::int64_t>(static_cast<std::int16_t>(u));
            return v;
        }
        case kGgufMetaUint32:
        case kGgufMetaInt32:
        case kGgufMetaFloat32: {
            const std::uint32_t u = gguf_read_u32(f);
            v.u64 = u;
            v.i64 = static_cast<std::int64_t>(static_cast<std::int32_t>(u));
            if (type == kGgufMetaFloat32) {
                float fv;
                std::memcpy(&fv, &u, 4);
                v.f64 = fv;
            }
            return v;
        }
        case kGgufMetaUint64:
        case kGgufMetaInt64:
        case kGgufMetaFloat64: {
            const std::uint64_t u = gguf_read_u64(f);
            v.u64 = u;
            if (type == kGgufMetaInt64) v.i64 = static_cast<std::int64_t>(u);
            if (type == kGgufMetaFloat64) {
                double dv;
                std::memcpy(&dv, &u, 8);
                v.f64 = dv;
            }
            return v;
        }
        case kGgufMetaString:
            v.str = gguf_read_string(f);
            return v;
        case kGgufMetaArray: {
            const std::uint32_t elem = gguf_read_u32(f);
            const std::uint64_t count = gguf_read_u64(f);
            v.u64 = count;
            // Массивы round 1 пропускаются (содержимое не материализуется);
            // вложенные массивы — ошибка.
            for (std::uint64_t i = 0; i < count; ++i) {
                if (elem == kGgufMetaString) gguf_read_string(f);
                else if (elem == kGgufMetaArray)
                    throw LoaderError(LoaderError::Code::IoFailure,
                                      "gguf: nested arrays не поддержаны");
                else gguf_read_meta_value(f, elem);
            }
            return v;
        }
        default:
            throw LoaderError(LoaderError::Code::IoFailure,
                              "gguf: unsupported metadata value type " +
                                  std::to_string(type));
    }
}

void gguf_dequant_tensor(std::FILE* f, const TensorInfo& t,
                         std::vector<float>& out) {
    std::size_t n = 1;
    for (auto d : t.shape) n *= static_cast<std::size_t>(d);
    out.assign(n, 0.0f);
    const std::size_t n_blocks = (n + 31) / 32;
    const std::size_t block_bytes = (t.dtype_str == "Q4_0") ? 18 : 34;
    std::vector<std::uint8_t> raw(n_blocks * block_bytes);
    if (std::fseek(f, static_cast<long>(t.offset), SEEK_SET) != 0 ||
        std::fread(raw.data(), 1, raw.size(), f) != raw.size())
        throw LoaderError(LoaderError::Code::IoFailure,
                          "gguf: truncated tensor '" + t.dtype_str + "'");
    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const std::size_t begin = blk * 32;
        const std::size_t cnt = std::min<std::size_t>(32, n - begin);
        const std::uint8_t* base = raw.data() + blk * block_bytes;
        float delta = 0.0f;
        if (t.dtype_str == "Q4_0") {
            const std::uint16_t h = static_cast<std::uint16_t>(base[0]) |
                                    (static_cast<std::uint16_t>(base[1]) << 8);
            delta = f16_bits_to_f32(h);  // delta в НАЧАЛЕ блока Q4_0
            for (std::size_t i = 0; i < cnt; ++i) {
                const std::uint8_t byte = base[2 + i / 2];
                const int nib = ((i % 2) == 0) ? (byte & 0x0F) : (byte >> 4);
                out[begin + i] = static_cast<float>(nib - 8) * delta;
            }
        } else {
            const std::size_t delta_pos = 32;  // delta в КОНЦЕ блока Q8_0
            const std::uint16_t h =
                static_cast<std::uint16_t>(base[delta_pos]) |
                (static_cast<std::uint16_t>(base[delta_pos + 1]) << 8);
            delta = f16_bits_to_f32(h);
            for (std::size_t i = 0; i < cnt; ++i) {
                const std::int8_t q = static_cast<std::int8_t>(base[i]);
                out[begin + i] = static_cast<float>(q) * delta;
            }
        }
    }
}

// Разбор заголовка GGUF (общая функция: лоадер + конвертер stell-tier).
TensorFileMap parse_gguf_header(std::FILE* f) {
    std::uint8_t magic[4];
    read_exact(f, 0, magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0)
        throw LoaderError(LoaderError::Code::SchemaMismatch, "gguf: bad magic");
    if (std::fseek(f, 4, SEEK_SET) != 0)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: seek failed");
    const std::uint32_t version = gguf_read_u32(f);
    if (version != 3)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "gguf: unsupported version " + std::to_string(version) +
                              " (subset: v3)");
    const std::uint64_t n_tensors = gguf_read_u64(f);
    const std::uint64_t n_kv = gguf_read_u64(f);

    std::uint64_t alignment = 32;
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = gguf_read_string(f);
        const std::uint32_t type = gguf_read_u32(f);
        if (key == "general.alignment" && type == kGgufMetaUint32)
            alignment = static_cast<std::uint32_t>(gguf_read_meta_value(f, type).u64);
        else
            gguf_read_meta_value(f, type);  // skip
    }

    struct PendingTensor {
        std::string name;
        std::uint32_t n_dims;
        std::vector<std::uint64_t> dims;   // как в файле (порядок GGUF)
        std::int64_t dtype;
        std::uint64_t offset;  // относительный к базе данных
    };
    std::vector<PendingTensor> pending;
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        PendingTensor t;
        t.name = gguf_read_string(f);
        t.n_dims = gguf_read_u32(f);
        t.dims.resize(t.n_dims);
        for (std::uint32_t d = 0; d < t.n_dims; ++d) t.dims[d] = gguf_read_u64(f);
        t.dtype = static_cast<std::int64_t>(gguf_read_u32(f));
        t.offset = gguf_read_u64(f);
        pending.push_back(std::move(t));
    }
    // База данных: конец тензорных описаний, выровнен на alignment (GGUF v3).
    const long base_pos = std::ftell(f);
    if (base_pos < 0)
        throw LoaderError(LoaderError::Code::IoFailure, "gguf: cannot position");
    TensorFileMap map;
    map.data_base = static_cast<std::uint64_t>(base_pos);
    map.data_base = (map.data_base + alignment - 1) / alignment * alignment;

    for (auto& t : pending) {
        TensorInfo info;
        std::size_t n = 1;
        // Round 2: РАЗВЕРНУТЫЙ порядок dims (см. комментарий в .hpp):
        // GGUF ne = {in, out} для матриц → логическая форма [out, in].
        for (std::size_t d = t.dims.size(); d-- > 0;) {
            info.shape.push_back(static_cast<std::int64_t>(t.dims[d]));
            n *= static_cast<std::size_t>(t.dims[d]);
        }
        info.dtype_str = "GGML_" + std::to_string(t.dtype);
        switch (t.dtype) {
            case 0:
                info.dtype = stell::tiers::DType::F32;
                info.dtype_str = "F32";
                info.itemsize = 4;
                break;
            case 1:
                info.dtype = stell::tiers::DType::F16;
                info.dtype_str = "F16";
                info.itemsize = 2;
                break;
            case kGgmlQ40:
                info.dtype = std::nullopt;  // квантованный GGML — не tiers-DType
                info.dtype_str = "Q4_0";
                info.itemsize = 0;  // блоки: см. gguf_dequant_tensor
                break;
            case kGgmlQ80:
                info.dtype = std::nullopt;
                info.dtype_str = "Q8_0";
                info.itemsize = 0;
                break;
            default:
                throw LoaderError(
                    LoaderError::Code::UnsupportedQuant,
                    "gguf: tensor '" + t.name + "': unsupported data_type " +
                        std::to_string(t.dtype) +
                        " (subset: F32/F16/Q4_0/Q8_0; прочие квантованные — "
                        "через конвертер, round 3)");
            }
        info.offset = map.data_base + t.offset;
        if (t.dtype == 0 || t.dtype == 1) info.bytes = info.itemsize * n;
        else if (t.dtype == kGgmlQ40) info.bytes = ((n + 31) / 32) * 18;
        else info.bytes = ((n + 31) / 32) * 34;
        map.by_name[t.name] = info;
    }
    return map;
}

namespace {

class GgufSource : public FileWeightsSource {
public:
    using FileWeightsSource::FileWeightsSource;

protected:
    TensorFileMap parse_header() override { return parse_gguf_header(file()); }

    void do_fill(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                 void* dst, std::size_t dst_bytes) override {
        check_shape(g, t);
        // Квантованные GGML-блоки: dequant → f32 → целевой dtype группы.
        if (is_quantized_ggml(t.dtype_str)) {
            std::size_t n = 1;
            for (auto d : g.shape) n *= static_cast<std::size_t>(d);
            std::vector<float> f32;
            gguf_dequant_tensor(file(), t, f32);
            fill_plain_f32(g, dst, dst_bytes, f32.data(), n);
            return;
        }
        if (!compatible_plain(g.dtype, t))
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "group '" + g.name + "': gguf tensor dtype '" + t.dtype_str +
                                  "' не совместим");
        fill_plain(g, t, dst, dst_bytes);
    }

private:
    // Плоский путь из f32-буфера (dequant'а): F32-группа — как есть,
    // F16-группа — конверсия F32→F16 (референс f32_to_f16_bits).
    void fill_plain_f32(const stell::tiers::TensorGroup& g, void* dst,
                        std::size_t dst_bytes, const float* src, std::size_t n) {
        std::uint8_t* out = static_cast<std::uint8_t*>(dst);
        std::size_t payload = n * 4;
        if (g.dtype == stell::tiers::DType::F16) {
            payload = n * 2;
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint16_t h = f32_to_f16_bits(src[i]);
                out[2 * i] = static_cast<std::uint8_t>(h & 0xFF);
                out[2 * i + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFF);
            }
        } else if (g.dtype == stell::tiers::DType::F32) {
            std::memcpy(out, src, payload);
        } else {
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "group '" + g.name + "': целевой dtype '" +
                                  t_dtype_str(g.dtype) +
                                  "' недоступен для GGML-блоков (round 2: F16/F32)");
        }
        if (payload != expected_payload(g))
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': payload size mismatch");
        std::memset(out + payload, 0, dst_bytes - payload);
    }

    static const char* t_dtype_str(stell::tiers::DType d) {
        switch (d) {
            case stell::tiers::DType::F16: return "F16";
            case stell::tiers::DType::F32: return "F32";
            case stell::tiers::DType::E4M3: return "E4M3";
            case stell::tiers::DType::NVFP4: return "NVFP4";
            case stell::tiers::DType::W4A16: return "W4A16";
        }
        return "?";
    }
};

}  // namespace

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

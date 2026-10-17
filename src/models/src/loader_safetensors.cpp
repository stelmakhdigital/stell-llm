// Лоадер safetensors (HF) — контракт ADR-09 §3.
#include "tensor_file_formats.hpp"
#include "quant.hpp"

#include <cstring>
#include <set>

#include "stell/models/json.hpp"

namespace stell::models {
namespace {

struct SafetensorsDtypeEntry {
    const char* name;
    std::optional<stell::tiers::DType> dtype;
    std::size_t itemsize;
};

const SafetensorsDtypeEntry kSafetensorsDtypes[] = {
    {"F16",      stell::tiers::DType::F16, 2},
    {"F32",      stell::tiers::DType::F32, 4},
    {"BF16",     std::nullopt,             2},  // конвертация BF16 вне раунда 1
    {"F64",      std::nullopt,             8},
    {"I8",       std::nullopt,             1},
    {"U8",       std::nullopt,             1},
    {"I16",      std::nullopt,             2},
    {"U16",      std::nullopt,             2},
    {"I32",      std::nullopt,             4},
    {"U32",      std::nullopt,             4},
    {"I64",      std::nullopt,             8},
    {"U64",      std::nullopt,             8},
    {"BOOL",     std::nullopt,             1},
    {"F8_E4M3",  stell::tiers::DType::E4M3, 1},
    {"F8_E5M2",  std::nullopt,             1},
};

const SafetensorsDtypeEntry* find_safetensors_dtype(const std::string& name) {
    for (const auto& e : kSafetensorsDtypes)
        if (name == e.name) return &e;
    return nullptr;
}

}  // namespace

// Разбор заголовка safetensors (общая функция: лоадер + конвертер stell-tier).
TensorFileMap parse_safetensors_header(std::FILE* f) {
    const std::uint64_t header_len = read_u64_le(f, 0);
    std::string header_json(header_len, '\0');
    read_exact(f, 8, header_json.data(), header_len);
    JsonValue root;
    try {
        root = json_parse(header_json);
    } catch (const JsonError& e) {
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          std::string("safetensors: bad header JSON: ") + e.what());
    }
    if (root.type != JsonValue::Type::Object)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "safetensors: header root must be an object");

    TensorFileMap map;
    map.data_base = 8 + header_len;
    for (const auto& [name, val] : root.object_v) {
        if (name == "__metadata__") continue;
        if (val.type != JsonValue::Type::Object)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "safetensors: tensor info must be an object");
        const std::string dtype_str = val.get_string("dtype");
        const SafetensorsDtypeEntry* de = find_safetensors_dtype(dtype_str);
        if (!de)
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "safetensors: unknown dtype '" + dtype_str + "'");
        TensorInfo info;
        info.dtype_str = dtype_str;
        info.dtype = de->dtype;
        info.itemsize = de->itemsize;
        for (const auto& dim : val.get_array("shape").array_v) {
            if (dim.type != JsonValue::Type::Int || dim.int_v < 0)
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "safetensors: shape must be non-negative ints");
            info.shape.push_back(dim.int_v);
        }
        const JsonValue& offs = val.get_array("data_offsets");
        if (offs.array_v.size() != 2)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "safetensors: data_offsets must be [begin, end]");
        const std::uint64_t begin = static_cast<std::uint64_t>(offs.array_v[0].int_v);
        const std::uint64_t end = static_cast<std::uint64_t>(offs.array_v[1].int_v);
        if (end < begin)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "safetensors: data_offsets end < begin");
        info.offset = map.data_base + begin;
        info.bytes = end - begin;
        std::size_t n = 1;
        for (auto d : info.shape) n *= static_cast<std::size_t>(d);
        if (info.bytes != de->itemsize * n)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "safetensors: tensor '" + name + "' size/shape mismatch");
        map.by_name[name] = info;
    }
    return map;
}

class SafetensorsSource : public FileWeightsSource {
public:
    using FileWeightsSource::FileWeightsSource;

protected:
    TensorFileMap parse_header() override { return parse_safetensors_header(file()); }

    void do_fill(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                 void* dst, std::size_t dst_bytes) override {
        check_shape(g, t);
        if (!compatible_plain(g.dtype, t))
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "group '" + g.name + "' (dtype " + dtype_name(g.dtype) +
                                  "): safetensors tensor dtype '" + t.dtype_str +
                                  "' не совместим; для квантованых профилей используйте "
                                  "лоадер NVFP4W4A16 (ADR-03)");
        fill_plain(g, t, dst, dst_bytes);
    }

private:
    static const char* dtype_name(stell::tiers::DType d) {
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

}  // namespace stell::models

// Фабрика (namespace stell::models; класс SafetensorsLoader определён ниже).
namespace stell::models {

class SafetensorsLoader : public IWeightsLoader {
public:
    WeightsFormat format() const override { return WeightsFormat::Safetensors; }
    std::unique_ptr<IWeightsSource> open(const std::string& uri,
                                         const ModelDescription& desc) override {
        auto src = std::make_unique<SafetensorsSource>(uri, desc);
        src->resolve_all();
        return src;
    }
};

IWeightsLoader* make_safetensors_loader() { return new SafetensorsLoader(); }

}  // namespace stell::models

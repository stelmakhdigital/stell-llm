// Общий каркас файловых лоадеров (safetensors/GGUF/NVFP4W4A16): контракт
// IWeightsSource::read (ADR-09 §3) реализован один раз; наследники задают
// разбор заголовка и заполнение payload.
#pragma once

#include <cstring>
#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "stell/models/description.hpp"
#include "stell/models/weights.hpp"
#include "quant.hpp"

namespace stell::models {

// Информация о тензоре входного файла.
struct TensorInfo {
    std::string dtype_str;                       // как в файле ("F16", "F32", ...)
    std::optional<stell::tiers::DType> dtype;    // маппится, если поддержан
    std::size_t itemsize = 0;
    std::vector<std::int64_t> shape;
    std::uint64_t offset = 0;                    // абсолютное смещение payload в файле
    std::uint64_t bytes = 0;                     // размер payload
};

struct TensorFileMap {
    std::map<std::string, TensorInfo> by_name;
    std::uint64_t data_base = 0;  // абсолютное смещение начала секции данных
};

// Чтение ровно count байт по абсолютному смещению (общая утилита).
inline void read_exact(std::FILE* f, std::uint64_t offset, void* dst, std::size_t count) {
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0)
        throw LoaderError(LoaderError::Code::IoFailure, "seek failed");
    const std::size_t got = std::fread(dst, 1, count, f);
    if (got != count)
        throw LoaderError(LoaderError::Code::IoFailure,
                          "truncated file (read " + std::to_string(got) + " of " +
                              std::to_string(count) + ")");
}

// Файл + резолюция source_pattern → TensorInfo по GroupId. KV-группы
// (role=Kv, ADR-09 §1) весов не имеют и не резолвятся.
class FileWeightsSource : public IWeightsSource {
public:
    FileWeightsSource(std::string path, ModelDescription desc)
        : path_(std::move(path)), desc_(std::move(desc)) {}

    void read(const stell::tiers::GroupKey& key, void* dst, std::size_t dst_bytes) override {
        const stell::tiers::TensorGroup* g = find_group(key);
        if (!g)
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              "read(): unknown GroupId " + std::to_string(key.id.value));
        if (g->role == stell::tiers::Role::Kv)
            throw LoaderError(LoaderError::Code::IoFailure,
                              "group '" + g->name + "': kv-группа не имеет весов в файле");
        if (dst_bytes != g->slot_bytes)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g->name + "': dst_bytes != slot_bytes (" +
                                  std::to_string(dst_bytes) + " != " +
                                  std::to_string(g->slot_bytes) + ")");
        auto it = tensors_.find(g->id.value);
        if (it == tensors_.end())
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              "group '" + g->name + "': не разрешён при open()");
        do_fill(*g, it->second, dst, dst_bytes);
    }

    void close() override {
        if (file_) { std::fclose(file_); file_ = nullptr; }
    }

    // Энергичная резолюция всех некешевых групп (fail-fast, ADR-09 §3).
    // Round 2 (R9): группы со срезом стекового тензора (source_slice ≥ 0,
    // стековные экспертные тензоры GGUF-mixtral) резолвятся в СРЕЗ тензора —
    // непрерывный блок по старшей размерности (row-major); форма среза =
    // форме группы (проверка — SchemaMismatch).
    void resolve_all() {
        file_ = std::fopen(path_.c_str(), "rb");
        if (!file_) throw LoaderError(LoaderError::Code::IoFailure, "cannot open " + path_);
        tfm_ = parse_header();
        for (std::size_t i = 0; i < desc_.groups.size(); ++i) {
            const auto& g = desc_.groups[i];
            if (g.role == stell::tiers::Role::Kv) continue;
            const std::string& pattern = desc_.group_source_patterns[i];
            auto it = tfm_.by_name.find(pattern);
            if (it == tfm_.by_name.end())
                throw LoaderError(LoaderError::Code::TensorNotFound,
                                  "tensor not found for group '" + g.name + "': pattern '" +
                                      pattern + "'");
            tensors_[g.id.value] = slice_tensor_info(
                it->second, g, group_slice(i));
        }
    }

    // Срез группы i (−1 = тензор целиком; R9: стековные тензоры).
    std::int64_t group_slice(std::size_t i) const {
        return i < desc_.group_source_slices.size() ? desc_.group_source_slices[i] : -1;
    }

    // Эффективная TensorInfo группы: без среза — как в файле; со срезом —
    // сдвиг/ограничение payload по старшей размерности (непрерывный блок,
    // row-major). Форма среза обязана совпадать с формой группы.
    static TensorInfo slice_tensor_info(const TensorInfo& t,
                                        const stell::tiers::TensorGroup& g,
                                        std::int64_t slice) {
        if (slice < 0) return t;
        if (t.shape.empty() || t.shape.size() - 1 != g.shape.size())
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': стековный тензор " +
                                  std::to_string(t.shape.size()) + "-D, группа " +
                                  std::to_string(g.shape.size()) + "-D");
        for (std::size_t d = 0; d < g.shape.size(); ++d)
            if (t.shape[d + 1] != g.shape[d])
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "group '" + g.name + "': shape mismatch (стековный тензор)");
        const std::int64_t stack = t.shape[0];
        if (slice >= stack)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': source_slice " +
                                  std::to_string(slice) + " >= стека " +
                                  std::to_string(stack));
        std::size_t n_slice = 1;
        for (auto d : g.shape) n_slice *= static_cast<std::size_t>(d);
        TensorInfo info = t;
        info.shape = g.shape;
        if (t.dtype) {
            info.offset += static_cast<std::uint64_t>(slice) * n_slice * t.itemsize;
            info.bytes = n_slice * t.itemsize;
        } else if (is_quantized_ggml(t.dtype_str)) {
            const std::size_t blocks = (n_slice + 31) / 32;
            info.offset += static_cast<std::uint64_t>(slice) * blocks * ggml_block_bytes(t.dtype_str);
            info.bytes = blocks * ggml_block_bytes(t.dtype_str);
        } else {
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "group '" + g.name + "': source_slice для dtype '" +
                                  t.dtype_str + "' не поддержан");
        }
        return info;
    }

protected:
    const std::string& path() const { return path_; }
    std::FILE* file() const { return file_; }

    virtual TensorFileMap parse_header() = 0;            // заполняет by_name + data_base
    virtual void do_fill(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                         void* dst, std::size_t dst_bytes) = 0;

    const ModelDescription& desc() const { return desc_; }

    // Чтение payload тензора в буфер (ровно t.bytes байт).
    void read_tensor_payload(const TensorInfo& t, void* dst) const {
        read_exact(file_, t.offset, dst, t.bytes);
    }

    // Заполнение слота группы plain-dtype: копия payload (с F32→F16 конвертом
    // при необходимости), хвост слота — нули. Общий путь safetensors/GGUF.
    void fill_plain(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                    void* dst, std::size_t dst_bytes) {
        std::vector<std::uint8_t> raw(t.bytes);
        read_tensor_payload(t, raw.data());
        std::uint8_t* out = static_cast<std::uint8_t*>(dst);
        std::size_t payload = t.bytes;
        if (t.dtype == stell::tiers::DType::F32 && g.dtype == stell::tiers::DType::F16) {
            // F32→F16 — допустимая конвертация лоадера (не dequant, ADR-09 §3).
            std::size_t n = 1;
            for (auto d : g.shape) n *= static_cast<std::size_t>(d);
            std::vector<std::uint8_t> half(n * 2);
            for (std::size_t i = 0; i < n; ++i) {
                float v;
                std::memcpy(&v, raw.data() + 4 * i, 4);
                const std::uint16_t h = f32_to_f16_bits(v);
                half[2 * i] = static_cast<std::uint8_t>(h & 0xFF);
                half[2 * i + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFF);
            }
            payload = half.size();
            std::memcpy(out, half.data(), payload);
        } else {
            std::memcpy(out, raw.data(), payload);
        }
        if (payload != expected_payload(g))
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': payload size mismatch");
        std::memset(out + payload, 0, dst_bytes - payload);
    }

    static std::size_t expected_payload(const stell::tiers::TensorGroup& g) {
        std::size_t n = 1;
        for (auto d : g.shape) n *= static_cast<std::size_t>(d);
        switch (g.dtype) {
            case stell::tiers::DType::F16: return n * 2;
            case stell::tiers::DType::F32: return n * 4;
            case stell::tiers::DType::E4M3: return n;
            case stell::tiers::DType::NVFP4:
            case stell::tiers::DType::W4A16:
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "квантованный payload — только лоадером NVFP4W4A16 (ADR-03)");
        }
        return 0;
    }

    // Проверка согласованности плана и тензора (ADR-09 §3: mismatch = исключение).
    void check_shape(const stell::tiers::TensorGroup& g, const TensorInfo& t) const {
        if (t.shape != g.shape)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': shape mismatch (" + shape_str(t.shape) +
                                  " != " + shape_str(g.shape) + ")");
    }

    // Разрешённые dtype источника для целевого dtype группы (без квантования).
    static bool compatible_plain(stell::tiers::DType target, const TensorInfo& t) {
        if (target == stell::tiers::DType::F16 &&
            (t.dtype == stell::tiers::DType::F16 || t.dtype == stell::tiers::DType::F32))
            return true;   // F32→F16 — допустимая конвертация лоадера (не dequant)
        if (target == stell::tiers::DType::F32 && t.dtype == stell::tiers::DType::F32)
            return true;
        if (target == stell::tiers::DType::E4M3 && t.dtype == stell::tiers::DType::E4M3)
            return true;
        return false;
    }

private:
    const stell::tiers::TensorGroup* find_group(const stell::tiers::GroupKey& key) const {
        for (const auto& g : desc_.groups)
            if (g.id == key.id) return &g;
        return nullptr;
    }

    static std::string shape_str(const std::vector<std::int64_t>& s) {
        std::string out = "[";
        for (std::size_t i = 0; i < s.size(); ++i)
            out += (i ? "," : "") + std::to_string(s[i]);
        return out + "]";
    }

    std::string path_;
    ModelDescription desc_;
    std::FILE* file_ = nullptr;
    TensorFileMap tfm_;
    std::map<std::uint32_t, TensorInfo> tensors_;            // GroupId → инфо

protected:
};

inline std::uint64_t read_u64_le(std::FILE* f, std::uint64_t offset) {
    std::uint8_t b[8];
    // Чтение ровно 8 байт по абсолютному смещению (файл уже открыт).
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0)
        throw LoaderError(LoaderError::Code::IoFailure, "seek failed");
    if (std::fread(b, 1, 8, f) != 8)
        throw LoaderError(LoaderError::Code::IoFailure, "truncated file");
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[i]) << (8 * i);
    return v;
}

}  // namespace stell::models

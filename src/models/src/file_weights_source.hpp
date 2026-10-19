// Общий каркас файловых лоадеров (safetensors/GGUF/NVFP4W4A16): контракт
// IWeightsSource::read (ADR-09 §3) реализован один раз; наследники задают
// разбор заголовка и заполнение payload.
#pragma once

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <system_error>

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
    // Round 4 (R3-подготовка): BF16-тензора HF-чекпоинтов (Qwen/Mixtral/OLMoE)
    // маппятся в F16 с флагом: значения декодируются bf16_bits_to_f32, конверсия
    // BF16→F16 (→f32→F16) выполняется лоадером/конвертером как допустимая
    // (та же семантика, что F32→F16; см. plain_fill_slot).
    bool bf16 = false;
    // Round 4 (R6): индекс шарда в каталоге HF-шардов (0 = одиночный файл);
    // FILE* шарда — files_[shard_index] источника.
    std::size_t shard_index = 0;
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

// Полезная нагрузка группы plain-dtype (байты, до выравнивания) — общий для
// лоадеров и конвертера (ADR-03): F16=2Б/эл, F32=4Б/эл, E4M3=1Б/эл; квантованные
// профили (NVFP4/W4A16) — UnsupportedQuant (квантование — QuantizingSource/конвертер).
inline std::size_t expected_payload(const stell::tiers::TensorGroup& g) {
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

// Общая (лоадер + конвертер) конверсия plain-payload источника в целевой dtype
// группы с записью в слот (хвост до slot_bytes — нули, ADR-09 §3):
//   F16→F16, F32→F32, E4M3→E4M3  — копия байтов;
//   F32→F16, BF16→F16            — допустимая конвертация лоадера (не dequant);
//                                  R5: clamp-сатурация к ±65504 (ADR-03 round 5),
//                                  счётчик — f16_saturation_count() (quant.hpp);
// прочие комбинации — SchemaMismatch/UnsupportedQuant (разрешения — в
// compatible_plain вызовой стороны; здесь — честная проверка размеров).
// Побитовая идентичность путей loader/converter — контракт round 4 (R4/R6):
// слоты layout == слотам прямого лоадера. raw — ровно t.bytes байт тензора.
inline void plain_fill_slot(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                            const std::vector<std::uint8_t>& raw, void* dst,
                            std::size_t dst_bytes) {
    std::size_t n = 1;
    for (auto d : g.shape) n *= static_cast<std::size_t>(d);
    if (raw.size() != t.bytes)
        throw LoaderError(LoaderError::Code::IoFailure,
                          "group '" + g.name + "': payload source буфер != t.bytes");
    std::uint8_t* out = static_cast<std::uint8_t*>(dst);
    std::size_t payload = 0;
    const bool src_f32 = t.dtype == stell::tiers::DType::F32;
    const bool src_bf16 = t.dtype == stell::tiers::DType::F16 && t.bf16;
    if (g.dtype == stell::tiers::DType::F16 && (src_f32 || src_bf16)) {
        // F32/BF16 → F16 (допустимая конвертация лоадера, ADR-09 §3).
        // Источник обязан содержать itemsize*n байт (round 4 fixup: честная
        // проверка до индексации — усечённый/несогласованный тензор →
        // SchemaMismatch, не выход за границы буфера).
        if (raw.size() != n * t.itemsize)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': payload источника ('" +
                                  t.dtype_str + "', " + std::to_string(raw.size()) +
                                  " Б) не согласован с формой группы (" +
                                  std::to_string(n) + " × " +
                                  std::to_string(t.itemsize) + " Б)");
        payload = n * 2;
        if (payload > dst_bytes)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': payload > dst_bytes");
        for (std::size_t i = 0; i < n; ++i) {
            const float v0 = src_f32
                                 ? [&] { float f; std::memcpy(&f, raw.data() + 4 * i, 4); return f; }()
                                 : bf16_bits_to_f32(static_cast<std::uint16_t>(raw[2 * i]) |
                                                    (static_cast<std::uint16_t>(raw[2 * i + 1]) << 8));
            // R5 ([Q6], ADR-03 round 5): clamp-сатурация к ±65504 — inf в
            // весах не порождается; обычные значения — побитово == прежней
            // конверсии; счётчик сатураций — в логе конвертера (R5).
            const std::uint16_t h = f32_to_f16_bits(saturate_f16(v0));
            out[2 * i] = static_cast<std::uint8_t>(h & 0xFF);
            out[2 * i + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFF);
        }
    } else {
        // Копия байтов (F16→F16, F32→F32, E4M3→E4M3).
        payload = raw.size();
        if (payload > dst_bytes)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': payload > dst_bytes");
        std::memcpy(out, raw.data(), payload);
    }
    if (payload != expected_payload(g))
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "group '" + g.name + "': payload size mismatch");
    std::memset(out + payload, 0, dst_bytes - payload);
}

// Разбор заголовка safetensors — определение в loader_safetensors.cpp
// (объявление в tensor_file_formats.hpp; здесь — форвард для
// merge_safetensors_shards: прямой include циклится с TensorInfo).
TensorFileMap parse_safetensors_header(std::FILE* f);

// Round 4 (R6/R3-подготовка): каталог safetensors-шардов HF (model-0000X-of-
// 0000Y.safetensors) — единый источник плана со смешанными dtype-группами.
// Открывает все шард-файлы (FILE* остаются открытыми — владелец закрывает
// close()), возвращает объединённую карту тензоров (TensorInfo.shard_index —
// индекс в *files_out, offset — абсолютный внутри шарда). Порядок шардов —
// отсортированный по имени (канонический порядок HF). Ошибки: IoFailure (в
// каталоге нет .safetensors-шардов / шард не открылся), SchemaMismatch (тензор
// в нескольких шардах).
inline TensorFileMap merge_safetensors_shards(const std::string& dir,
                                                      std::vector<std::FILE*>* files_out) {
    std::error_code ec;
    std::vector<std::string> shards;
    for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
        if (de.path().extension() == ".safetensors")
            shards.push_back(de.path().string());
    }
    if (ec)
        throw LoaderError(LoaderError::Code::IoFailure,
                          "safetensors shards: cannot list '" + dir + "'");
    std::sort(shards.begin(), shards.end());  // канонический порядок шардов HF
    if (shards.empty())
        throw LoaderError(LoaderError::Code::IoFailure,
                          "safetensors shards: в каталоге '" + dir +
                              "' нет .safetensors-шардов");

    TensorFileMap merged;
    for (std::size_t i = 0; i < shards.size(); ++i) {
        std::FILE* f = std::fopen(shards[i].c_str(), "rb");
        if (!f)
            throw LoaderError(LoaderError::Code::IoFailure,
                              "safetensors shards: cannot open " + shards[i]);
        files_out->push_back(f);
        TensorFileMap tfm = parse_safetensors_header(f);
        for (auto& [name, info] : tfm.by_name) {
            info.shard_index = i;
            if (!merged.by_name.emplace(name, info).second)
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "safetensors shards: тензор '" + name +
                                      "' присутствует в нескольких шардах каталога '" +
                                      dir + "'");
        }
    }
    return merged;
}

// Файл (или каталог шардов) + резолюция source_pattern → TensorInfo по GroupId.
// KV-группы (role=Kv, ADR-09 §1) весов не имеют и не резолвятся.
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
        for (std::FILE* f : files_) {
            if (f) std::fclose(f);
        }
        files_.clear();
        file_ = nullptr;
    }

    // Энергичная резолюция всех некешевых групп (fail-fast, ADR-09 §3).
    // Round 2 (R9): группы со срезом стекового тензора (source_slice ≥ 0,
    // стековные экспертные тензоры GGUF-mixtral) резолвятся в СРЕЗ тензора —
    // непрерывный блок по старшей размерности (row-major); форма среза =
    // форме группы (проверка — SchemaMismatch).
    void resolve_all() {
        std::error_code ec;
        if (std::filesystem::is_directory(path_, ec)) {
            // Round 4 (R6): каталог safetensors-шардов HF — один IWeightsSource
            // на весь план (смешанные dtype-группы — атрибут каждой группы,
            // ADR-09; ветвлений по «профилю» у источника нет).
            file_ = nullptr;
            tfm_ = merge_safetensors_shards(path_, &files_);
        } else {
            file_ = std::fopen(path_.c_str(), "rb");
            if (!file_) throw LoaderError(LoaderError::Code::IoFailure, "cannot open " + path_);
            files_.push_back(file_);
            tfm_ = parse_header();
        }
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

    // Чтение payload тензора в буфер (ровно t.bytes байт). Шард-aware (R6):
    // тензора каталога читаются из своего шарда.
    void read_tensor_payload(const TensorInfo& t, void* dst) const {
        read_exact(files_.at(t.shard_index), t.offset, dst, t.bytes);
    }

    // Заполнение слота группы plain-dtype: конверсия/копия payload (общий
    // путь с конвертером — plain_fill_slot, побитовая идентичность путей),
    // хвост слота — нули. Общий путь safetensors/GGUF.
    void fill_plain(const stell::tiers::TensorGroup& g, const TensorInfo& t,
                    void* dst, std::size_t dst_bytes) {
        std::vector<std::uint8_t> raw(t.bytes);
        read_tensor_payload(t, raw.data());
        plain_fill_slot(g, t, raw, dst, dst_bytes);
    }
    // Проверка согласованности плана и тензора (ADR-09 §3: mismatch = исключение).
    void check_shape(const stell::tiers::TensorGroup& g, const TensorInfo& t) const {
        if (t.shape != g.shape)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "group '" + g.name + "': shape mismatch (" + shape_str(t.shape) +
                                  " != " + shape_str(g.shape) + ")");
    }

    // Разрешённые dtype источника для целевого dtype группы (без квантования):
    // F16-цель ← F16/BF16 (конверсия bf16→f32→f16) и F32 (конверсия F32→F16) —
    // допустимые конвертации лоадера (не dequant), ADR-09 §3; F32/E4M3 — только
    // точный источник. BF16-флаг: TensorInfo.dtype маппится в F16, декодирование —
    // bf16_bits_to_f32 (см. plain_fill_slot).
    static bool compatible_plain(stell::tiers::DType target, const TensorInfo& t) {
        if (target == stell::tiers::DType::F16 &&
            (t.dtype == stell::tiers::DType::F16 || t.dtype == stell::tiers::DType::F32))
            return true;   // F32→F16 / BF16→F16 — допустимые конвертации лоадера
        if (target == stell::tiers::DType::F32 && t.dtype == stell::tiers::DType::F32)
            return true;
        if (target == stell::tiers::DType::E4M3 && t.dtype == stell::tiers::DType::E4M3)
            return true;
        return false;
    }

private:
    std::vector<std::FILE*> files_;  // [0] = file_ (одиночный) / шарды каталога
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
    std::FILE* file_ = nullptr;   // одиночный файл (файл() для parse_header impl)
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

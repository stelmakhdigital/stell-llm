#include "stell/models/convert.hpp"

#include <cstdio>
#include <filesystem>
#include <cstring>
#include <functional>
#include <set>
#include <vector>

#include "file_weights_source.hpp"
#include "crc32.hpp"
#include "tensor_file_formats.hpp"
#include "quant.hpp"
#include <array>
#include <cctype>
#include "stelltier_format.hpp"
#include "stell/models/weights.hpp"

namespace stell::models {
namespace {

std::uint64_t align_up64(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

std::string sanitize(std::string s) {
    for (char& c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) c = '_';
    return s;
}

// Слот группы = payload (при квантованном профиле — квантованный) + нулевой
// хвост до slot_bytes; CRC32 — по слоту как записан (ADR-12 §2.3).
std::vector<std::uint8_t> make_slot(const stell::tiers::TensorGroup& g,
                                    const TensorInfo& t, std::FILE* f) {
    std::vector<std::uint8_t> slot(static_cast<std::size_t>(g.slot_bytes), 0);
    std::size_t numel = 1;
    for (auto d : g.shape) numel *= static_cast<std::size_t>(d);

    if (g.dtype == stell::tiers::DType::NVFP4 || g.dtype == stell::tiers::DType::W4A16) {
        // Квантованный профиль: источник F16/F32, квантование — здесь (ADR-03).
        if (!t.dtype || (*t.dtype != stell::tiers::DType::F16 &&
                         *t.dtype != stell::tiers::DType::F32))
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "converter: источник должен быть F16/F32, получено '" +
                                  t.dtype_str + "'");
        std::vector<float> src(numel);
        if (*t.dtype == stell::tiers::DType::F16) {
            std::vector<std::uint8_t> raw(t.bytes);
            if (std::fseek(f, static_cast<long>(t.offset), SEEK_SET) != 0 ||
                std::fread(raw.data(), 1, raw.size(), f) != raw.size())
                throw LoaderError(LoaderError::Code::IoFailure, "converter: truncated tensor");
            for (std::size_t i = 0; i < numel; ++i) {
                std::uint16_t h = static_cast<std::uint16_t>(raw[2 * i]) |
                                  (static_cast<std::uint16_t>(raw[2 * i + 1]) << 8);
                src[i] = f16_bits_to_f32(h);
            }
        } else {
            if (std::fseek(f, static_cast<long>(t.offset), SEEK_SET) != 0 ||
                std::fread(src.data(), 4, numel, f) != numel)
                throw LoaderError(LoaderError::Code::IoFailure, "converter: truncated tensor");
        }
        const std::size_t payload =
            g.dtype == stell::tiers::DType::NVFP4
                ? (numel + 1) / 2 + (numel + 31) / 32
                : (numel + 1) / 2 + 2 * ((numel + 31) / 32);
        if (payload > slot.size())
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "converter: payload > slot_bytes for group '" + g.name + "'");
        if (g.dtype == stell::tiers::DType::NVFP4)
            quantize_nvfp4(src.data(), numel, slot.data());
        else
            quantize_w4a16(src.data(), numel, slot.data());
    } else {
        if (t.bytes > slot.size())
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "converter: tensor bytes > slot_bytes for group '" + g.name + "'");
        if (std::fseek(f, static_cast<long>(t.offset), SEEK_SET) != 0 ||
            std::fread(slot.data(), 1, t.bytes, f) != t.bytes)
            throw LoaderError(LoaderError::Code::IoFailure,
                              "converter: truncated tensor for group '" + g.name + "'");
    }
    return slot;
}

}  // namespace

void StellTierConverter::convert(const ModelDescription& desc, const std::string& source_uri,
                                 const std::string& out_dir) {
    // Выбор лоадера: квантованный профиль — NVFP4W4A16-вариант (ADR-03);
    // иначе — по формату файла.
    const WeightsFormat fmt = detect_format(source_uri);
    std::unique_ptr<IWeightsLoader> loader;
    if (desc.dtype_weights == "NVFP4" || desc.dtype_weights == "W4A16") {
        loader = make_loader(WeightsFormat::NVFP4W4A16);
    } else {
        loader = make_loader(fmt);
    }
    std::unique_ptr<IWeightsSource> src = loader->open(source_uri, desc);

    // Тензоры входного файла нужны напрямую (чтение payload при построении слота).
    // Прощего пути без дублирования разбора нет: конвертер работает поверх
    // TensorFileMap — поэтому для safetensors/GGUF разбираем заголовок заново
    // (файл малозатратен); StellTier-вход конвертером не поддержан (он уже layout).
    TensorFileMap tfm;
    std::FILE* fin = std::fopen(source_uri.c_str(), "rb");
    if (!fin) throw LoaderError(LoaderError::Code::IoFailure, "converter: cannot open " + source_uri);
    struct FileCloser {
        std::FILE* f;
        ~FileCloser() { std::fclose(f); }
    } fcloser{fin};
    if (fmt == WeightsFormat::Safetensors)
        tfm = parse_safetensors_header(fin);
    else if (fmt == WeightsFormat::GGUF)
        tfm = parse_gguf_header(fin);
    else
        throw LoaderError(LoaderError::Code::UnsupportedQuant,
                          "converter: вход должен быть safetensors/GGUF (ADR-03) — "
                          "stell-tier уже целевой layout");

    // Слоты групп (в порядке плана), CRC32 по слоту (ADR-12 §2.2/§2.3).
    std::vector<StellTierEntry> entries;
    std::vector<std::vector<std::uint8_t>> slots;
    for (std::size_t i = 0; i < desc.groups.size(); ++i) {
        const auto& g = desc.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;  // KV-кеш не хранится в layout
        const std::string& pattern = desc.group_source_patterns[i];
        auto it = tfm.by_name.find(pattern);
        if (it == tfm.by_name.end())
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              "converter: tensor not found for group '" + g.name + "': '" +
                                  pattern + "'");
        // Round 2 (R9): срез стековного тензора (GGUF-mixtral, source_slice) —
        // та же конвенция, что и в лоадерах (file_weights_source.hpp).
        const std::int64_t slice =
            i < desc.group_source_slices.size() ? desc.group_source_slices[i] : -1;
        const TensorInfo& ti = FileWeightsSource::slice_tensor_info(it->second, g, slice);
        if (ti.shape != g.shape)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "converter: shape mismatch for group '" + g.name + "'");
        slots.push_back(make_slot(g, ti, fin));

        StellTierEntry e;
        e.group_id = g.id.value;
        e.role = stelltier_role_code(g.role);
        e.dtype = stelltier_dtype_code(g.dtype);
        e.tensor_count = 1;
        e.shape_rank = static_cast<std::uint32_t>(g.shape.size());
        for (std::size_t k = 0; k < g.shape.size() && k < 4; ++k)
            e.shape[k] = static_cast<std::uint32_t>(g.shape[k]);
        e.slot_bytes = g.slot_bytes;
        e.name = g.name.substr(0, 16);  // ADR-12 §2.2: имя обрезано (16 байт, см. stelltier_format.cpp)
        entries.push_back(e);
    }

    // Раскладка (ADR-12 §2.2/§2.4): заголовок 512 (файл выровнен 4096 с 0),
    // индекс на 0x200, данные монотонно с выравниванием 4096, без дыр.
    const std::uint64_t index_offset = kStellTierHeaderSize;
    const std::uint64_t data_start =
        align_up64(index_offset + entries.size() * kStellTierEntrySize, kStellTierSlotAlign);
    std::uint64_t cursor = data_start;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        entries[i].data_offset = cursor;
        entries[i].crc32 = crc32_ieee(slots[i].data(), slots[i].size());
        cursor += entries[i].slot_bytes;
    }

    std::filesystem::create_directories(out_dir);  // конвертер создаёт каталог вывода
    const std::string out_path =
        out_dir + "/" + sanitize(desc.name) + ".stelltier";
    std::FILE* fout = std::fopen(out_path.c_str(), "wb");
    if (!fout) throw LoaderError(LoaderError::Code::IoFailure, "converter: cannot create " + out_path);
    struct OutCloser {
        std::FILE* f;
        ~OutCloser() { std::fclose(f); }
    } ocloser{fout};

    std::array<std::uint8_t, kStellTierHeaderSize> header{};
    stelltier_build_header(dtype_profile_code(desc.dtype_weights), desc.name, entries.size(),
                           index_offset, entries.empty() ? data_start : entries[0].data_offset,
                           &header);
    std::fwrite(header.data(), 1, header.size(), fout);

    std::vector<std::uint8_t> index_blob(entries.size() * kStellTierEntrySize, 0);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        std::array<std::uint8_t, kStellTierEntrySize> rec{};
        stelltier_build_entry(entries[i], &rec);
        std::memcpy(index_blob.data() + i * kStellTierEntrySize, rec.data(), rec.size());
    }
    std::fwrite(index_blob.data(), 1, index_blob.size(), fout);

    // Паддинг от конца индекса до data_start (секция данных выровнена на 4096).
    for (std::uint64_t p = index_offset + index_blob.size(); p < data_start; ++p)
        std::fputc(0, fout);

    for (const auto& slot : slots)
        std::fwrite(slot.data(), 1, slot.size(), fout);
    if (std::ferror(fout))
        throw LoaderError(LoaderError::Code::IoFailure, "converter: write failed");
}

}  // namespace stell::models

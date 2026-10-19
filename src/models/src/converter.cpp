#include "stell/models/convert.hpp"

#include <cstdio>
#include <algorithm>
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
// Plain-ветка использует общий с лоадерами plain_fill_slot (round 4: побитовая
// идентичность слотов конвертера и прямого лоадера — F32/BF16→F16 конверсии
// включены; ранее конвертер копировал байты без конверсии — расхождение с
// fill_plain было бы при F32/BF16-источнике F16-группы).
std::vector<std::uint8_t> make_slot(const stell::tiers::TensorGroup& g,
                                    const TensorInfo& t, std::FILE* f) {
    std::vector<std::uint8_t> slot(static_cast<std::size_t>(g.slot_bytes), 0);
    std::size_t numel = 1;
    for (auto d : g.shape) numel *= static_cast<std::size_t>(d);

    if (g.dtype == stell::tiers::DType::NVFP4 || g.dtype == stell::tiers::DType::W4A16) {
        // Квантованный профиль: источник F16/F32/BF16, квантование — здесь (ADR-03).
        if (!t.dtype || (*t.dtype != stell::tiers::DType::F16 &&
                         *t.dtype != stell::tiers::DType::F32))
            throw LoaderError(LoaderError::Code::UnsupportedQuant,
                              "converter: источник должен быть F16/F32/BF16, получено '" +
                                  t.dtype_str + "'");
        std::vector<float> src(numel);
        if (*t.dtype == stell::tiers::DType::F16) {
            std::vector<std::uint8_t> raw(t.bytes);
            if (std::fseek(f, static_cast<long>(t.offset), SEEK_SET) != 0 ||
                std::fread(raw.data(), 1, raw.size(), f) != raw.size())
                throw LoaderError(LoaderError::Code::IoFailure, "converter: truncated tensor");
            const bool bf16 = t.bf16;  // round 4: BF16-флаг (dtype маппится в F16)
            for (std::size_t i = 0; i < numel; ++i) {
                std::uint16_t h = static_cast<std::uint16_t>(raw[2 * i]) |
                                  (static_cast<std::uint16_t>(raw[2 * i + 1]) << 8);
                src[i] = bf16 ? bf16_bits_to_f32(h) : f16_bits_to_f32(h);
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
        // Plain-ветка — общий путь с лоадерами (plain_fill_slot): копия либо
        // допустимая конвертация F32/BF16→F16, хвост слота — нули.
        // Round 4 fixup (найдено прогоном [R3] на реальных весах,
        // tiny-random-Qwen2Moe): предохранитель «t.bytes > slot_bytes»
        // срабатывал ДО конверсии и ложно отвергал конвертируемые источники
        // (F32-тензор 2n→4n байт при F16-группе) — валидация размеров
        // честно выполняется в plain_fill_slot (payload == expected_payload(g)
        // и payload ≤ dst_bytes → SchemaMismatch; тиша запрещена).
        std::vector<std::uint8_t> raw(t.bytes);
        if (std::fseek(f, static_cast<long>(t.offset), SEEK_SET) != 0 ||
            std::fread(raw.data(), 1, t.bytes, f) != t.bytes)
            throw LoaderError(LoaderError::Code::IoFailure,
                              "converter: truncated tensor for group '" + g.name + "'");
        plain_fill_slot(g, t, raw, slot.data(), slot.size());
    }
    return slot;
}

}  // namespace

void StellTierConverter::convert(const ModelDescription& desc, const std::string& source_uri,
                                 const std::string& out_dir) {
    // Round 4: каталог шардов — материализующий convert() не поддержан
    // (работает с одним файлом); используйте convert_streaming (R7/R6).
    {
        std::error_code ec;
        if (std::filesystem::is_directory(source_uri, ec))
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "converter: вход — каталог шардов; используйте "
                              "convert_streaming (поддержка каталога — в потоковой конвертации)");
    }
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

// ============================================================================
// Потоковая конвертация (round 3, R7 — подготовка среза 3).
// Writer БЕЗ полной загрузки модели в RAM: источник читается по группам плана
// последовательно (шард за шардом для HF safetensors-шардов), каждая группа
// материализуется по одной (make_slot: слот + source-буфер — O(максимальной
// группы) байт), пишется в layout с flush'ем по группам. В RAM держатся только
// записи индекса (64 Б/группу, метаданные) — память НЕ O(размера модели).
// Раскладка файла побитово совпадает с convert() (проверено тестом:
// слоты/CRC/индекс/логиты интерпретатора == прямому пути).
// ============================================================================
namespace {

// Набор источников: один файл (safetensors/GGUF) ИЛИ каталог safetensors-
// шардов (HF-чекпоинт). Round 4 (R6): слияние шардов — общая функция с
// лоадерами (merge_safetensors_shards): один источник обслуживает весь план
// (смешанные dtype-группы — атрибут каждой группы, ветвлений по «профилю»
// нет); FILE* шардов открыты на время конвертации, payload читается по
// требованию последовательно шард за шардом в порядке групп плана (R7).
class ShardWeightsSource {
public:
    explicit ShardWeightsSource(const std::string& uri) {
        std::error_code ec;
        if (std::filesystem::is_directory(uri, ec)) {
            format_ = WeightsFormat::Safetensors;
            tfm_ = merge_safetensors_shards(uri, &files_);
        } else {
            std::FILE* f = std::fopen(uri.c_str(), "rb");
            if (!f) throw LoaderError(LoaderError::Code::IoFailure,
                                      "converter: cannot open " + uri);
            files_.push_back(f);
            format_ = detect_format(uri);
            if (format_ == WeightsFormat::StellTier)
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "converter: stell-tier уже целевой layout");
            if (format_ != WeightsFormat::Safetensors && format_ != WeightsFormat::GGUF)
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "converter: вход должен быть safetensors/GGUF (ADR-03)");
            tfm_ = (format_ == WeightsFormat::Safetensors)
                       ? parse_safetensors_header(f)
                       : parse_gguf_header(f);
        }
    }

    ~ShardWeightsSource() { close(); }
    void close() {
        for (std::FILE* f : files_) {
            if (f) std::fclose(f);
        }
        files_.clear();
    }

    const TensorInfo* find(const std::string& pattern) const {
        auto it = tfm_.by_name.find(pattern);
        return it == tfm_.by_name.end() ? nullptr : &it->second;
    }

    // FILE* шарда с тензором pattern (файлы уже открыты; чтение идёт
    // последовательно шард за шардом в порядке групп плана — R7).
    std::FILE* file_for(const std::string& pattern) {
        return files_.at(tfm_.by_name.at(pattern).shard_index);
    }

private:
    WeightsFormat format_ = WeightsFormat::Safetensors;
    std::vector<std::FILE*> files_;
    TensorFileMap tfm_;
};

}  // namespace

void StellTierConverter::convert_streaming(const ModelDescription& desc,
                                           const std::string& source_uri,
                                           const std::string& out_dir,
                                           const StellTierWriterConfig& cfg,
                                           Stats* stats) {
    // --- 1. Конфигурация: выравнивания фиксированы layout v1 (ADR-12 §2); ----
    //      параметры тиров — извне (tiers-конфиг), валидация совместимости плана.
    if (cfg.slot_align != kStellTierSlotAlign || cfg.index_align != kStellTierIndexAlign)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "converter: slot_align/index_align фиксированы layout v1 "
                          "(4096/512, ADR-12 §2/§2.4) — новая версия формата через ADR");

    // --- 2. Источник(и): метаданные шардов (payload в RAM не читается) ------
    ShardWeightsSource shards(source_uri);

    // --- 3. Резолюция плана (fail-fast) + раскладка (только метаданные) ------
    std::vector<StellTierEntry> entries;          // 64 Б/группу + поля — O(числа групп)
    struct PendingSlot {
        const stell::tiers::TensorGroup* g;
        TensorInfo ti;
        std::string pattern;   // индекс entries ≠ индекс групп плана (KV пропущены)
    };
    std::vector<PendingSlot> pending;
    std::size_t n_nonkv = 0;
    // R5: провенанс dtype_orig (ADR-03 round 5) — считается на этапе резолвции.
    if (stats) {
        stats->f16_saturations = 0;
        stats->groups_dtype_orig_bf16 = 0;
        stats->groups_dtype_orig_f32 = 0;
    }
    for (const auto& g : desc.groups)
        if (g.role != stell::tiers::Role::Kv) ++n_nonkv;

    const std::uint64_t index_offset = kStellTierHeaderSize;
    const std::uint64_t data_start =
        align_up64(index_offset + n_nonkv * kStellTierEntrySize, kStellTierSlotAlign);
    std::uint64_t cursor = data_start;

    for (std::size_t i = 0; i < desc.groups.size(); ++i) {
        const auto& g = desc.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;  // KV-кеш не хранится в layout
        const std::string& pattern = desc.group_source_patterns[i];
        const TensorInfo* ti = shards.find(pattern);
        if (!ti)
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              "converter: tensor not found for group '" + g.name +
                                  "': '" + pattern + "'");
        const std::int64_t slice =
            i < desc.group_source_slices.size() ? desc.group_source_slices[i] : -1;
        TensorInfo eff = FileWeightsSource::slice_tensor_info(*ti, g, slice);
        if (eff.shape != g.shape)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "converter: shape mismatch for group '" + g.name + "'");

        // Профиль тиров (из tiers-конфига): группа обязана помещаться в VRAM-слот,
        // иначе профиль «свободно 2–11 ГБ на V100 при co-tenancy» неисполним.
        if (cfg.vram_slot_size && g.slot_bytes > cfg.vram_slot_size)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "converter: группа '" + g.name + "' (slot_bytes " +
                                  std::to_string(g.slot_bytes) +
                                  ") не помещается в VRAM-слот tiers-профиля (" +
                                  std::to_string(cfg.vram_slot_size) + ")");

        StellTierEntry e;
        e.group_id = g.id.value;
        e.role = stelltier_role_code(g.role);
        e.dtype = stelltier_dtype_code(g.dtype);
        e.tensor_count = 1;
        e.shape_rank = static_cast<std::uint32_t>(g.shape.size());
        for (std::size_t k = 0; k < g.shape.size() && k < 4; ++k)
            e.shape[k] = static_cast<std::uint32_t>(g.shape[k]);
        e.slot_bytes = g.slot_bytes;
        e.name = g.name.substr(0, 16);  // ADR-12 §2.2: имя обрезано (16 байт)
        e.data_offset = cursor;
        e.crc32 = 0;  // проставляется после записи слота (fixup записи индекса)
        cursor += e.slot_bytes;
        entries.push_back(e);
        pending.push_back(PendingSlot{&g, std::move(eff), pattern});
        // R5 ([Q6], ADR-03 round 5): провенанс исходного dtype группы
        // (dtype_orig: bf16/f32 — честность профиля: точка регрессии знает,
        // что было сжатие из BF16/F32).
        if (stats) {
            const TensorInfo& ti = pending.back().ti;
            if (ti.bf16) ++stats->groups_dtype_orig_bf16;
            else if (ti.dtype == stell::tiers::DType::F32) ++stats->groups_dtype_orig_f32;
        }
    }

    // --- 4. Файл: header + индекс (CRC групп — placeholder) + padding --------
    std::filesystem::create_directories(out_dir);  // конвертер создаёт каталог вывода
    const std::string out_path = out_dir + "/" + sanitize(desc.name) + ".stelltier";
    const std::string tmp_path = out_path + ".tmp";  // атомарная финализация:
    // частичный файл не остаётся под финальным именем при ошибке в середине.
    std::FILE* fout = std::fopen(tmp_path.c_str(), "wb");
    if (!fout)
        throw LoaderError(LoaderError::Code::IoFailure, "converter: cannot create " + tmp_path);
    struct OutCloser {
        std::FILE* f;
        ~OutCloser() { if (f) std::fclose(f); }
    } ocloser{fout};

    std::array<std::uint8_t, kStellTierHeaderSize> header{};
    stelltier_build_header(dtype_profile_code(desc.dtype_weights), desc.name, entries.size(),
                           index_offset,
                           entries.empty() ? data_start : entries.front().data_offset, &header);
    std::fwrite(header.data(), 1, header.size(), fout);
    for (const auto& e : entries) {
        std::array<std::uint8_t, kStellTierEntrySize> rec{};
        stelltier_build_entry(e, &rec);
        std::fwrite(rec.data(), 1, rec.size(), fout);
    }
    // Паддинг от конца индекса до data_start (секция данных выровнена на 4096):
    // пишется блоками, не байт за байтом (могут быть мегабайты при сотнях групп).
    {
        std::vector<std::uint8_t> zeros(4096, 0);
        std::uint64_t pos = index_offset + entries.size() * kStellTierEntrySize;
        while (pos < data_start) {
            const std::uint64_t chunk = std::min<std::uint64_t>(zeros.size(), data_start - pos);
            std::fwrite(zeros.data(), 1, chunk, fout);
            pos += chunk;
        }
    }

    // --- 5. Поток по группам: слот → payload + CRC → fixup записи индекса ----
    std::uint64_t unflushed = 0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& g = *pending[i].g;
        const TensorInfo& ti = pending[i].ti;
        const std::string& pattern = pending[i].pattern;
        std::FILE* fin = shards.file_for(pattern);  // последовательный переход шард→шард
        // R5: дельта глобального счётчика сатураций по этой группе
        // (конверсии F32/BF16→F16 в plain_fill_slot — clamp к ±65504).
        const std::uint64_t sat0 = f16_saturation_count();
        const std::vector<std::uint8_t> slot = make_slot(g, ti, fin);  // O(группы)
        if (stats) stats->f16_saturations += f16_saturation_count() - sat0;

        std::fseek(fout, static_cast<long>(entries[i].data_offset), SEEK_SET);
        std::fwrite(slot.data(), 1, slot.size(), fout);
        if (std::ferror(fout))
            throw LoaderError(LoaderError::Code::IoFailure, "converter: write failed");

        // Fixup записи индекса: CRC слота известен только после его записи;
        // раскладка файла требует индекс ПЕРЕД данными (ADR-12 §2.2) — перезапись
        // своей 64-байтовой записи, курсор возвращается на следующий слот.
        entries[i].crc32 = crc32_ieee(slot.data(), slot.size());
        std::array<std::uint8_t, kStellTierEntrySize> rec{};
        stelltier_build_entry(entries[i], &rec);
        std::fseek(fout, static_cast<long>(index_offset + i * kStellTierEntrySize), SEEK_SET);
        std::fwrite(rec.data(), 1, rec.size(), fout);

        unflushed += slot.size();
        if (unflushed >= cfg.flush_group_bytes) {  // flush по группам (R7)
            std::fflush(fout);
            unflushed = 0;
        }
    }

    // --- 6. Финализация: self-check собственным reader'ом + атомарный rename -
    if (std::ferror(fout))
        throw LoaderError(LoaderError::Code::IoFailure, "converter: write failed");
    const std::uint64_t expected_size = entries.empty()
                                            ? data_start
                                            : entries.back().data_offset + entries.back().slot_bytes;
    if (std::fflush(fout) != 0)
        throw LoaderError(LoaderError::Code::IoFailure, "converter: flush failed");
    std::fseek(fout, 0, SEEK_END);  // курсор мог остаться на fixup'е индекса
    const std::int64_t written = ftello(fout) >= 0 ? ftello(fout) : -1;
    if (written >= 0 && static_cast<std::uint64_t>(written) != expected_size)
        throw LoaderError(LoaderError::Code::IoFailure,
                          "converter: размер файла не сошёлся (" +
                              std::to_string(written) + " != " +
                              std::to_string(expected_size) + ")");

    // Self-check: собственный reader по временному файлу (заголовок+индекс;
    // payload не читается — дёшево). Ловит порчу раскладки на месте.
    {
        std::uint64_t fsz = 0;
        const StellTierIndex idx = stelltier_read_header_and_index(tmp_path, &fsz);
        if (idx.entries.size() != entries.size() || fsz != expected_size)
            throw LoaderError(LoaderError::Code::IoFailure,
                              "converter: self-check layout failed");
    }

    ocloser.f = nullptr;  // fclose в деструкторе больше не нужен
    if (std::fclose(fout) != 0)
        throw LoaderError(LoaderError::Code::IoFailure, "converter: close failed");
    std::filesystem::rename(tmp_path, out_path);
    shards.close();
}

}  // namespace stell::models

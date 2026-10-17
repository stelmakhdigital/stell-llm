// Лоадер stell-tier layout (ADR-12 §2) — контракт ADR-09 §3 + ADR-12 §4.
// Сверка layout с планом групп (slot_bytes/dtype/role/shape) — обязанность
// models-лоадера; tiers при set_plan выполняет ту же сверку независимо
// (ADR-12 §2.2: mismatch = PlanMismatch на стороне tiers / SchemaMismatch здесь).
#include "stelltier_format.hpp"
#include "stell/models/description.hpp"
#include "stell/models/weights.hpp"
#include "crc32.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace stell::models {
namespace {

class StellTierSource : public IWeightsSource {
public:
    StellTierSource(std::string path, ModelDescription desc)
        : path_(std::move(path)), desc_(std::move(desc)) {
        file_ = std::fopen(path_.c_str(), "rb");
        if (!file_) throw LoaderError(LoaderError::Code::IoFailure, "cannot open " + path_);
        index_ = stelltier_read_header_and_index(path_, &file_size_);

        // Энергичная сверка плана и layout (fail-fast): каждая некешевая группа
        // описания обязана присутствовать в индексе с совпадающими полями.
        std::map<std::uint32_t, const StellTierEntry*> by_id;
        for (const auto& e : index_.entries) by_id[e.group_id] = &e;
        std::set<std::uint32_t> used;
        for (std::size_t i = 0; i < desc_.groups.size(); ++i) {
            const auto& g = desc_.groups[i];
            if (g.role == stell::tiers::Role::Kv) continue;
            auto it = by_id.find(g.id.value);
            if (it == by_id.end())
                throw LoaderError(LoaderError::Code::TensorNotFound,
                                  "stelltier: group '" + g.name + "' (id " +
                                      std::to_string(g.id.value) + ") отсутствует в layout");
            const StellTierEntry& e = *it->second;
            used.insert(g.id.value);
            check_entry(g, e);
        }
        // Лишние записи индекса (не из плана) — SchemaMismatch (ADR-09 §3:
        // план групп — единственный источник истины о составе).
        for (const auto& [id, e] : by_id)
            if (!used.count(id))
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "stelltier: layout содержит группу id " +
                                      std::to_string(id) + ", отсутствующую в плане");
    }

    ~StellTierSource() override { close(); }

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
                              "group '" + g->name + "': dst_bytes != slot_bytes");
        const StellTierEntry* e = entry_for(g->id.value);
        if (!e)
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              "group '" + g->name + "': отсутствует в layout");
        check_entry(*g, *e);

        std::vector<std::uint8_t> slot(static_cast<std::size_t>(e->slot_bytes));
        read_exact(file_, e->data_offset, slot.data(), slot.size());
        // CRC32 полезной нагрузки (payload + padding) — один раз за жизнь группы
        // в источнике (ADR-12 §2.3; повторные проверки — забота tiers).
        if (!crc_checked_.count(g->id.value)) {
            if (crc32_ieee(slot.data(), slot.size()) != e->crc32)
                throw LoaderError(LoaderError::Code::ChecksumMismatch,
                                  "stelltier: CRC mismatch for group '" + g->name + "'");
            crc_checked_.insert(g->id.value);
        }
        std::memcpy(dst, slot.data(), slot.size());
    }

    void close() override {
        if (file_) { std::fclose(file_); file_ = nullptr; }
    }

private:
    static void check_entry(const stell::tiers::TensorGroup& g, const StellTierEntry& e) {
        if (e.role != stelltier_role_code(g.role) ||
            e.dtype != stelltier_dtype_code(g.dtype))
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "stelltier: role/dtype mismatch for group '" + g.name + "'");
        if (e.slot_bytes != g.slot_bytes)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "stelltier: slot_bytes mismatch for group '" + g.name +
                                  "' (" + std::to_string(e.slot_bytes) + " != " +
                                  std::to_string(g.slot_bytes) + ")");
        if (e.shape_rank > 4)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "stelltier: shape_rank > 4 (ADR-12 §2.2)");
        for (std::uint32_t k = 0; k < e.shape_rank; ++k) {
            if (k < g.shape.size() &&
                g.shape[k] != static_cast<std::int64_t>(e.shape[k]))
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "stelltier: shape mismatch for group '" + g.name + "'");
        }
    }

    const StellTierEntry* entry_for(std::uint32_t id) const {
        for (const auto& e : index_.entries)
            if (e.group_id == id) return &e;
        return nullptr;
    }

    const stell::tiers::TensorGroup* find_group(const stell::tiers::GroupKey& key) const {
        for (const auto& g : desc_.groups)
            if (g.id == key.id) return &g;
        return nullptr;
    }

    static void read_exact(std::FILE* f, std::uint64_t offset, void* dst, std::size_t count) {
        if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0)
            throw LoaderError(LoaderError::Code::IoFailure, "seek failed");
        if (std::fread(dst, 1, count, f) != count)
            throw LoaderError(LoaderError::Code::IoFailure, "truncated group slot");
    }

    std::string path_;
    ModelDescription desc_;
    std::FILE* file_ = nullptr;
    std::uint64_t file_size_ = 0;
    StellTierIndex index_;
    std::set<std::uint32_t> crc_checked_;
};

}  // namespace

class StellTierLoader : public IWeightsLoader {
public:
    WeightsFormat format() const override { return WeightsFormat::StellTier; }
    std::unique_ptr<IWeightsSource> open(const std::string& uri,
                                         const ModelDescription& desc) override {
        auto src = std::make_unique<StellTierSource>(uri, desc);
        return src;
    }
};

IWeightsLoader* make_stelltier_loader() { return new StellTierLoader(); }

}  // namespace stell::models

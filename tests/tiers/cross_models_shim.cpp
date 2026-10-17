// cross_models_shim.cpp — models-сторона кросс-модульного теста tiers↔models.
//
// Отдельная единица трансляции (см. cross_models_abi.h — почему ABI):
// здесь живут заголовки models (конвертер StellTierConverter, реестр,
// синтетический writer safetensors). Наружу — плоский C-фасад.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "cross_models_abi.h"
#include "synth_writers.hpp"
#include "stell/models/convert.hpp"
#include "stell/models/registry.hpp"
#include "stelltier_format.hpp" // reader models (CRC-канон [0x00..0x2C))

using namespace stell::models;
using namespace stell_test;

namespace {

struct Built {
    std::string layout_path;
    std::vector<CMGroupInfo> plan; // некешевые группы, порядок = индекс layout
};

Built g_built[2];
std::string g_err;

[[noreturn]] void fail(const std::string& msg) {
    g_err = msg;
    throw std::runtime_error(msg);
}

// Синтетические тензоры: source_pattern → payload (F16/F32; квантованный
// профиль — источник F16, конвертер квантует сам, ADR-12 §1).
std::vector<SynthTensor> make_tensors(const ModelDescription& d) {
    std::vector<SynthTensor> out;
    for (std::size_t i = 0; i < d.group_source_patterns.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;
        SynthTensor t;
        t.name = d.group_source_patterns[i];
        const bool quant = g.dtype == stell::tiers::DType::NVFP4 ||
                           g.dtype == stell::tiers::DType::W4A16;
        t.dtype = quant ? "F16" : (g.dtype == stell::tiers::DType::F32 ? "F32" : "F16");
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto dim : g.shape) n *= static_cast<std::size_t>(dim);
        if (t.dtype == "F32") {
            std::vector<float> v(n);
            for (std::size_t k = 0; k < n; ++k) v[k] = 0.25f;
            t.data = f32_payload(v);
        } else {
            std::vector<std::uint16_t> v(n);
            for (std::size_t k = 0; k < n; ++k)
                v[k] = static_cast<std::uint16_t>(((k * 3 + 5) % 31) + 1);
            t.data = f16_payload(v);
        }
        out.push_back(std::move(t));
    }
    return out;
}

Built build_one(const std::string& fixtures_dir, const std::string& fixture,
                const std::string& out_dir, int which) {
    Built& b = g_built[which];
    Registry r;
    ModelDescription d;
    try {
        d = r.load(fixtures_dir + "/" + fixture);
    } catch (const std::exception& e) {
        fail(std::string("cm: registry load failed: ") + e.what());
    }

    const std::string in_path =
        out_dir + "/" + (which == 0 ? "dense.safetensors" : "moe.safetensors");
    try {
        write_safetensors(in_path, make_tensors(d));
    } catch (const std::exception& e) {
        fail(std::string("cm: synth safetensors failed: ") + e.what());
    }

    try {
        StellTierConverter conv;
        conv.convert(d, in_path, out_dir);
    } catch (const std::exception& e) {
        fail(std::string("cm: converter failed: ") + e.what());
    }
    b.layout_path = out_dir + "/" + d.name + ".stelltier";

    // Models-сторона: собственный reader — валидация layout (CRC-канон
    // [0x00..0x2C), выравнивания, монотонность, отсутствие усечения).
    try {
        const StellTierIndex idx = stelltier_read_header_and_index(b.layout_path, nullptr);
        std::size_t k = 0;
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            const auto& g = d.groups[i];
            if (g.role == stell::tiers::Role::Kv) continue;
            CMGroupInfo info{};
            info.id = g.id.value;
            info.role = static_cast<std::uint8_t>(stelltier_role_code(g.role));
            info.dtype = static_cast<std::uint8_t>(stelltier_dtype_code(g.dtype));
            info.slot_bytes = g.slot_bytes;
            info.shape_rank = static_cast<std::uint32_t>(g.shape.size());
            for (std::size_t s = 0; s < g.shape.size() && s < 4; ++s)
                info.shape[s] = static_cast<std::uint32_t>(g.shape[s]);
            std::snprintf(info.name, sizeof info.name, "%s", g.name.c_str());
            if (k >= idx.entries.size() || idx.entries[k].group_id != info.id ||
                idx.entries[k].slot_bytes != info.slot_bytes)
                fail("cm: layout index != plan order at " + std::to_string(k));
            ++k;
            b.plan.push_back(info);
        }
        if (idx.n_groups != b.plan.size())
            fail("cm: n_groups != non-kv plan size");
    } catch (const std::exception& e) {
        fail(std::string("cm: models reader validation failed: ") + e.what());
    }
    return b;
}

} // namespace

extern "C" {

int cm_build(const char* fixtures_dir, const char* out_dir,
             char* dense_path, int dense_cap, char* moe_path, int moe_cap) {
    try {
        // Каталог вывода создаём заранее (конвертер тоже создаёт его, но
        // чище иметь его до записи синтетики).
        std::filesystem::create_directories(out_dir);
        g_built[0] = build_one(fixtures_dir, "mini_dense.json", out_dir, 0);
        g_built[1] = build_one(fixtures_dir, "mini_moe.json", out_dir, 1);
    } catch (const std::exception& e) {
        g_err = e.what();
        std::fprintf(stderr, "cm_build FAILED: %s\n", g_err.c_str());
        return -1;
    }
    if ((int)g_built[0].layout_path.size() >= dense_cap ||
        (int)g_built[1].layout_path.size() >= moe_cap)
        return -1;
    std::snprintf(dense_path, (size_t)dense_cap, "%s", g_built[0].layout_path.c_str());
    std::snprintf(moe_path, (size_t)moe_cap, "%s", g_built[1].layout_path.c_str());
    return 0;
}

int cm_plan_size(int which) {
    if (which < 0 || which > 1) return -1;
    return (int)g_built[which].plan.size();
}

int cm_plan_group(int which, int idx, struct CMGroupInfo* out) {
    if (which < 0 || which > 1) return -1;
    const auto& v = g_built[which].plan;
    if (idx < 0 || idx >= (int)v.size()) return -1;
    *out = v[(size_t)idx];
    return 0;
}

uint32_t cm_stored_header_crc(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { g_err = "cm: cannot open " + std::string(path); return 0; }
    std::uint8_t h[512] = {};
    const int got = (int)std::fread(h, 1, sizeof h, f);
    std::fclose(f);
    if (got != 512) { g_err = "cm: short header read"; return 0; }
    std::uint32_t v = 0;
    for (int b = 0; b < 4; ++b)
        v |= static_cast<std::uint32_t>(h[0x2C + b]) << (8 * b);
    return v;
}

int cm_read_slot(const char* path, uint32_t group_id, uint8_t* dst, size_t cap) {
    try {
        const StellTierIndex idx = stelltier_read_header_and_index(path, nullptr);
        for (const auto& e : idx.entries) {
            if (e.group_id != group_id) continue;
            if (e.slot_bytes > cap) { g_err = "cm: cap < slot_bytes"; return -1; }
            std::FILE* f = std::fopen(path, "rb");
            if (!f) { g_err = "cm: cannot open " + std::string(path); return -1; }
            const int ok = std::fseek(f, (long)e.data_offset, SEEK_SET) == 0 &&
                           std::fread(dst, 1, (size_t)e.slot_bytes, f) == e.slot_bytes;
            std::fclose(f);
            if (!ok) { g_err = "cm: slot read failed"; return -1; }
            return (int)e.slot_bytes;
        }
        g_err = "cm: group not found";
        return -1;
    } catch (const std::exception& e) {
        g_err = std::string("cm: models reader rejected: ") + e.what();
        return -1;
    }
}

const char* cm_last_error(void) { return g_err.c_str(); }

} // extern "C"

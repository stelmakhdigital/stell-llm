// Синтетические writer'ы входных форматов для тестов engine round 2.
// Соглашение с tests/models/synth_writers.hpp (поток D): safetensors —
// u64 header_len LE + JSON + данные, выравнивание секции данных на 8 байт.
// Свою копию пишем здесь (engine не включает чужие тестовые заголовки;
// раскладка честно сверена с tests/models/synth_writers.hpp).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "stell/engine/plain_spec.hpp"

namespace stell_core_test {

struct STensor {
    std::string name;
    std::string dtype;                  // "F16" | "F32" (safetensors)
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> data;     // payload LE
};

// --- little-endian -----------------------------------------------------------
inline void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void put_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}

// F32 → байты payload (LE).
inline std::vector<std::uint8_t> f32_payload(const std::vector<float>& v) {
    std::vector<std::uint8_t> out;
    out.reserve(v.size() * 4);
    for (float x : v) {
        std::uint32_t b = 0;
        std::memcpy(&b, &x, 4);
        put_u32(out, b);
    }
    return out;
}

// F16 → байты payload (LE; половинки как u16).
inline std::vector<std::uint8_t> f16_payload(const std::vector<std::uint16_t>& halves) {
    std::vector<std::uint8_t> out;
    out.reserve(halves.size() * 2);
    for (std::uint16_t h : halves) {
        out.push_back(static_cast<std::uint8_t>(h & 0xFF));
        out.push_back(static_cast<std::uint8_t>((h >> 8) & 0xFF));
    }
    return out;
}

// f32 → f16 (RNE; для тестовых значений — кратных 2^-4, конверт точный).
inline std::uint16_t f32_to_f16(float v) {
    std::uint32_t b = 0;
    std::memcpy(&b, &v, 4);
    const std::uint16_t sign = static_cast<std::uint16_t>((b >> 16) & 0x8000u);
    const std::uint32_t exp32 = (b >> 23) & 0xFFu;
    if (exp32 == 0xFF) return static_cast<std::uint16_t>(sign | 0x7C00u);  // inf/nan
    std::int32_t e = static_cast<std::int32_t>(exp32) - 127 + 15;
    std::uint32_t m24 = (b & 0x7FFFFFu) | 0x800000u;   // с неявным битом
    if (e >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);      // кламп inf
    if (e <= 0) return sign;                           // denormal/0: тестовые значения > 2^-14
    const std::uint32_t mant10 = m24 >> 13;
    const std::uint32_t rem = m24 & 0x1FFFu;
    std::uint32_t m = mant10;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) ++m;
    if (m > 0x3FFu) { m = 0; ++e; }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10) | m);
}

// Детерминированное заполнение: LCG по сид-хэшу имени (одинаковое между
// прогонами; значения — кратные 2^-4 ∈ [-1, 1): точные в F16, без округления).
inline std::uint64_t name_hash(const std::string& s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char c : s) { h ^= static_cast<unsigned char>(c); h *= 1099511628211ULL; }
    return h;
}

inline std::vector<float> fill_weights(const std::string& name, std::size_t n) {
    std::uint64_t s = name_hash(name);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 144269504088888963ULL;
        const std::int64_t q = static_cast<std::int64_t>((s >> 17) % 33) - 16;
        out[i] = static_cast<float>(q) * 0.0625f;    // k/16, k ∈ [-16, 16] — точно в F16
    }
    return out;
}

// Хелперы round 5: честные slot_bytes группы (конвенция description.cpp
// models: payload dtype + выравнивание слота до 4096) и сборщик PlainGroup
// для синтетических описаний (F16/F32-профили тестов).
inline std::size_t test_payload_bytes(const std::string& dtype, std::size_t n) {
    if (dtype == "F16") return n * 2;
    if (dtype == "F32") return n * 4;
    throw std::runtime_error("synth: dtype '" + dtype + "' не поддержан хелпером");
}
inline std::size_t test_align_slot(std::size_t v) { return (v + 4095) / 4096 * 4096; }

inline stell::engine::PlainGroup make_group(
    std::uint32_t id, const std::string& name, const std::string& dtype,
    std::vector<std::int64_t> shape, const std::string& pattern,
    bool kv_block = false) {
    stell::engine::PlainGroup g{};
    g.id = id;
    g.name = name;
    g.role = kv_block ? stell::engine::PlainRoleKv : stell::engine::PlainRoleAttn;
    g.dtype = dtype == "F32" ? stell::engine::PlainDTypeF32
                             : stell::engine::PlainDTypeF16;
    g.shape = std::move(shape);
    std::size_t n = 1;
    for (auto d : g.shape) n *= static_cast<std::size_t>(d);
    g.slot_bytes = test_align_slot(test_payload_bytes(dtype, n));
    g.source_pattern = pattern;
    g.kv_block = kv_block;
    return g;
}

// ops-граф слоя для синтетических описаний (конвенция фикстур mini_*):
// rmsnorm(eps 1e-5), attention(sm70_fp32_qk_pv) + опционально ffn(silu_mul)
// и moe_gather_scatter(sm70_moe_gs) — per-layer маршрутизация [R3] решает
// применимость ffn/moe по составу групп слоя.
inline std::vector<stell::engine::PlainOp> base_ops(bool with_ffn, bool with_moe) {
    std::vector<stell::engine::PlainOp> ops;
    {
        stell::engine::PlainOp op{};
        op.op = "rmsnorm";
        op.params.push_back({"eps", 1e-5, "", true});
        ops.push_back(op);
    }
    {
        stell::engine::PlainOp op{};
        op.op = "attention";
        op.params.push_back({"kernel", 0.0, "sm70_fp32_qk_pv", false});
        ops.push_back(op);
    }
    if (with_ffn) {
        stell::engine::PlainOp op{};
        op.op = "ffn";
        op.params.push_back({"activation", 0.0, "silu_mul", false});
        ops.push_back(op);
    }
    if (with_moe) {
        stell::engine::PlainOp op{};
        op.op = "moe_gather_scatter";
        op.params.push_back({"kernel", 0.0, "sm70_moe_gs", false});
        ops.push_back(op);
    }
    return ops;
}

// safetensors writer (раскладка tests/models/synth_writers.hpp: u64 header_len +
// JSON + payload; заголовок добивается пробелами до кратности 8; data_offsets —
// смещения внутри секции данных).
inline void write_safetensors(const std::string& path,
                              const std::vector<STensor>& tensors) {
    std::string header = "{";
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const STensor& t = tensors[i];
        std::uint64_t begin = 0, end = 0;
        for (std::size_t j = 0; j < i; ++j) begin += tensors[j].data.size();
        end = begin + t.data.size();
        std::string shape = "[";
        for (std::size_t k = 0; k < t.shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(t.shape[k]);
        shape += "]";
        header += (i ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"" +
                  t.dtype + "\",\"shape\":" + shape + ",\"data_offsets\":[" +
                  std::to_string(begin) + "," + std::to_string(end) + "]}";
    }
    header += ",\"__metadata__\":{\"stell_core_test\":\"synthetic\"}}";
    std::size_t padded = header.size();
    while ((8 + padded) % 8 != 0) ++padded;
    header.resize(padded, ' ');

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot create " + path);
    std::vector<std::uint8_t> len;
    put_u64(len, header.size());
    std::fwrite(len.data(), 1, len.size(), f);
    std::fwrite(header.data(), 1, header.size(), f);
    for (const auto& t : tensors) std::fwrite(t.data.data(), 1, t.data.size(), f);
    std::fclose(f);
}

// Тензоры фикстуры: по source_pattern каждой некешевой группы; dtype тензора:
// F32-группы → F32-тензор, квантованные/неквантованные F16-группы → F16-тензор
// (соглашение tests/models/test_weights_quant.cpp: NVFP4-профиль — источник F16).
inline std::vector<STensor> fixture_tensors(
    const stell::engine::PlainSpec& spec) {
    using stell::engine::PlainGroup;
    std::vector<STensor> out;
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block) continue;
        STensor t;
        t.name = g.source_pattern;
        const bool is_f32 = (g.dtype == stell::engine::PlainDTypeF32);
        t.dtype = is_f32 ? "F32" : "F16";
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto d : g.shape) n *= static_cast<std::size_t>(d);
        if (is_f32) {
            t.data = f32_payload(fill_weights(g.source_pattern, n));
        } else {
            const std::vector<float> vals = fill_weights(g.source_pattern, n);
            std::vector<std::uint16_t> halves(n);
            for (std::size_t i = 0; i < n; ++i) halves[i] = f32_to_f16(vals[i]);
            t.data = f16_payload(halves);
        }
        out.push_back(std::move(t));
    }
    return out;
}

} // namespace stell_core_test

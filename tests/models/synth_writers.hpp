// Синтетические writer'ы входных форматов для тестов лоадеров:
//   write_safetensors — safetensors: u64 header_len LE + JSON + данные;
//   write_gguf        — GGUF v3 subset, согласован с loader_gguf.cpp.
// Оба писателя формируют little-endian байты (x86/ARM64 — LE, для тестов ок).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace stell_test {

// Прототипы file-хелперов (реализации ниже).
void put_u64_file(std::FILE* f, std::uint64_t v);
void put_u32_file(std::FILE* f, std::uint32_t v);

struct SynthTensor {
    std::string name;
    std::string dtype;              // "F16"/"F32" (safetensors), GGUF: dtype-код 0/1
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> data; // payload LE
    std::int64_t gguf_code = -1;    // GGUF: переопределить dtype-код (Q4_0=20, Q8_0=36)
};

// --- little-endian helpers ---------------------------------------------------
inline void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}

inline void put_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}

inline void put_f32(std::vector<std::uint8_t>& out, float v) {
    std::uint32_t b = 0;
    std::memcpy(&b, &v, 4);
    put_u32(out, b);
}

// --- safetensors -------------------------------------------------------------
// Данные начинаются после 8 + header_len; заголовок дополняется пробелами
// до кратности 8 (как в формате safetensors). data_offsets — смещения
// внутри секции данных.
void write_safetensors(const std::string& path, const std::vector<SynthTensor>& tensors) {
    std::string header = "{";
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const SynthTensor& t = tensors[i];
        std::uint64_t begin = 0, end = 0;
        // Смещения считаем заранее: payload подряд в порядке tensors.
        for (std::size_t j = 0; j < i; ++j) begin += tensors[j].data.size();
        end = begin + t.data.size();
        std::string shape = "[";
        for (std::size_t k = 0; k < t.shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(t.shape[k]);
        shape += "]";
        header += (i ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"" + t.dtype +
                    "\",\"shape\":" + shape + ",\"data_offsets\":[" +
                    std::to_string(begin) + "," + std::to_string(end) + "]}";
    }
    header += ",\"__metadata__\":{\"stell_test\":\"synthetic\"}}";

    // Выравнивание секции данных на 8 байт пробелами.
    std::size_t padded = header.size();
    while ((8 + padded) % 8 != 0) ++padded;
    header.resize(padded, ' ');

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot create " + path);
    put_u64_file(f, header.size());
    std::fwrite(header.data(), 1, header.size(), f);
    for (const auto& t : tensors) std::fwrite(t.data.data(), 1, t.data.size(), f);
    std::fclose(f);
}

inline void put_u64_file(std::FILE* f, std::uint64_t v) {
    std::uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
    std::fwrite(b, 1, 8, f);
}

// --- GGUF v3 (subset, согласован с loader_gguf.cpp) --------------------------
constexpr std::uint32_t kMetaUint32 = 4, kMetaString = 11;
constexpr std::int64_t kGGML_F32 = 0, kGGML_F16 = 1;

// GGML dtype коды (см. loader_gguf.cpp): 0=F32, 1=F16.
constexpr std::uint32_t kGGMLMetaUint32 = 4;
constexpr std::uint32_t kGGMLMetaFloat32 = 8;
constexpr std::uint32_t kGGMLMetaBool = 10;
constexpr std::uint32_t kGGMLMetaString = 11;

inline void put_string(std::vector<std::uint8_t>& out, const std::string& s) {
    put_u64(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

inline void put_string_file(std::FILE* f, const std::string& s) {
    std::uint8_t b[8];
    std::uint64_t len = s.size();
    for (int i = 0; i < 8; ++i) b[i] = static_cast<std::uint8_t>((len >> (8 * i)) & 0xFF);
    std::fwrite(b, 1, 8, f);
    std::fwrite(s.data(), 1, s.size(), f);
}

// Значение метаданных GGUF (subset writer): u32/строка/f32/bool.
struct GgufMetaKV {
    std::string key;
    std::uint32_t type = kGGMLMetaUint32;
    std::uint64_t u64 = 0;
    double f64 = 0.0;
    std::string str;
};

inline void put_meta_kv(std::vector<std::uint8_t>& out, const GgufMetaKV& kv) {
    put_string(out, kv.key);
    put_u32(out, kv.type);
    switch (kv.type) {
        case kGGMLMetaUint32: put_u32(out, static_cast<std::uint32_t>(kv.u64)); break;
        case kGGMLMetaBool: out.push_back(static_cast<std::uint8_t>(kv.u64 ? 1 : 0)); break;
        case kGGMLMetaFloat32: put_f32(out, static_cast<float>(kv.f64)); break;
        case kGGMLMetaString: put_string(out, kv.str); break;
    }
}

// Данные: база = конец тензорных описаний, выровнен на 32 (alignment из
// general.alignment, по умолчанию 32); offset — смещение внутри базы (GGUF v3).
// meta — произвольные метаданные (конвертер описаний читает их для arch/размеров).
// dims тензора пишутся из SynthTensor.shape КАК ЕСТЬ — для GGUF-нативного
// порядка тест передаёт dims {in, out} (см. loader_gguf.cpp: развернутый
// порядок == логической форме (конвенция safetensors/JSON-конфигов).
void write_gguf_meta(const std::string& path, const std::vector<SynthTensor>& tensors,
                     const std::vector<GgufMetaKV>& meta,
                     std::uint64_t alignment = 32) {
    // Метаданные: general.alignment (u32) + заданные пары.
    std::vector<std::uint8_t> meta_bytes;
    put_string(meta_bytes, "general.alignment");
    put_u32(meta_bytes, kGGMLMetaUint32);
    put_u32(meta_bytes, static_cast<std::uint32_t>(alignment));
    for (const auto& kv : meta) put_meta_kv(meta_bytes, kv);

    // Тензорные описания: name, n_dims u32, dims u64*, dtype i32, offset u64.
    std::vector<std::uint8_t> infos;
    std::uint64_t cursor = 0;
    for (const auto& t : tensors) {
        put_string(infos, t.name);
        put_u32(infos, static_cast<std::uint32_t>(t.shape.size()));
        for (auto d : t.shape) put_u64(infos, static_cast<std::uint64_t>(d));
        const std::int64_t dt =
            t.gguf_code >= 0
                ? t.gguf_code
                : ((t.dtype == "F32") ? kGGML_F32 : kGGML_F16);
        put_u32(infos, static_cast<std::uint32_t>(dt));
        put_u64(infos, cursor);
        cursor += t.data.size();
    }

    const std::uint64_t data_base_raw =
        4 + 4 + 8 + 8 + meta_bytes.size() + infos.size();
    const std::uint64_t data_base =
        (data_base_raw + alignment - 1) / alignment * alignment;

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot create " + path);
    std::fwrite("GGUF", 1, 4, f);
    put_u32_file(f, 3);            // version
    put_u64_file(f, tensors.size());
    put_u64_file(f, meta.size() + 1);  // n_kv = general.alignment + meta
    std::fwrite(meta_bytes.data(), 1, meta_bytes.size(), f);
    std::fwrite(infos.data(), 1, infos.size(), f);
    // Паддинг до data_base.
    for (std::uint64_t p = data_base_raw; p < data_base; ++p) std::fputc(0, f);
    for (const auto& t : tensors) std::fwrite(t.data.data(), 1, t.data.size(), f);
    std::fclose(f);
}

// Round-1 обёртка: без метаданных (кроме general.alignment).
void write_gguf(const std::string& path, const std::vector<SynthTensor>& tensors) {
    write_gguf_meta(path, tensors, {});
}

inline void put_u32_file(std::FILE* f, std::uint32_t v) {
    std::uint8_t b[4];
    for (int i = 0; i < 4; ++i) b[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
    std::fwrite(b, 1, 4, f);
}

inline void put_u64_file_stream(std::FILE* f, std::uint64_t v) { put_u64_file(f, v); }

// F32 → байты payload.
inline std::vector<std::uint8_t> f32_payload(const std::vector<float>& v) {
    std::vector<std::uint8_t> out;
    out.reserve(v.size() * 4);
    for (float x : v) put_f32(out, x);
    return out;
}

inline std::vector<std::uint8_t> f16_payload(const std::vector<std::uint16_t>& halves) {
    std::vector<std::uint8_t> out;
    out.reserve(halves.size() * 2);
    for (std::uint16_t h : halves) {
        out.push_back(static_cast<std::uint8_t>(h & 0xFF));
        out.push_back(static_cast<std::uint8_t>((h >> 8) & 0xFF));
    }
    return out;
}

}  // namespace stell_test

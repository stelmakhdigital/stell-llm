// ============================================================================
// Конвертер GGUF → описание модели (round 2, задача 5, ADR-09 «Уточнение»).
//
// GGUF не несёт JSON-описания — models генерирует его из метаданных и
// тензорных имён файла, используя НАТИВНЫЕ llama.cpp-GGUF имена:
//   token_embd.weight, output_norm.weight, output.weight,
//   blk.{i}.attn_norm.weight, blk.{i}.attn_q.weight, blk.{i}.attn_k.weight,
//   blk.{i}.attn_v.weight, blk.{i}.attn_output.weight,
//   blk.{i}.ffn_norm.weight, blk.{i}.ffn_gate.weight, blk.{i}.ffn_up.weight,
//   blk.{i}.ffn_down.weight.
// source_pattern в генерируемом описании — те же нативные имена; GGUF-лоадер
// сравнивает их буквально (ADR-09 §3). KV-группы — псевдо-паттерн
// blk.{layer}.kv_cache (в файле KV нет, лоадер роль=kv пропускает).
//
// Метаданные (subset):
//   general.architecture (обязателен; round 1: только "llama"),
//   llama.block_count, llama.attention.head_count, llama.embedding_length —
//   обязательны; llama.attention.head_count_kv, llama.attention.key_length,
//   llama.attention.layer_norm_rms_epsilon, llama.vocab_size, general.name —
//   опциональны (умолчания: n_kv_heads = n_heads; d_head = d_model/n_heads;
//   eps = 1e-5; vocab_size — из token_embd; name = "llama-gguf").
//
// SHAPE: тензорные dims GGUF разворачиваются при разборе (parse_gguf_header,
// tensor_file_formats.hpp) — логическая форма [out, in] (HF-конвенция);
// d_ff берётся из blk.0.ffn_up.weight (развернутый dims[0]).
// Dtype: матрицы — F16 (конверсия F32→F16 при загрузке); нормы/вложения —
// по dtype тензора (F32/F16); GGUF Q4_0/Q8_0 — квантованные блоки
// (dequant → f32 → целевой dtype, см. loader_gguf.cpp); прочие
// квантованные dtype — UnsupportedQuant (конвертер, round 3).
// ============================================================================
#include "stell/models/convert.hpp"

#include "quant.hpp"
#include "tensor_file_formats.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace stell::models {

namespace {

// --- JSON-сборка (парсер в json.hpp, сериализатора нет — пишем руками) ------

void json_escape(std::string& out, const std::string& s) {
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
}

std::string fmt_int(std::int64_t v) { return std::to_string(v); }

// Числа с плавающей точкой: "%.8g" → валидный JSON-литерал (1e-05 → "1e-05").
std::string fmt_f64(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.8g", v);
    return buf;
}

std::string fmt_shape(const std::vector<std::int64_t>& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) out += ",";
        out += fmt_int(s[i]);
    }
    return out + "]";
}

std::string make_group(const std::string& name, const std::string& role,
                       const std::vector<std::int64_t>& shape,
                       const std::string& dtype, const std::string& pattern,
                       std::int64_t layer_cardinality = -1, bool kv_block = false) {
    std::string out = "{\"name\":\"";
    json_escape(out, name);
    out += "\",\"role\":\"" + role + "\",\"shape\":" + fmt_shape(shape);
    out += ",\"dtype\":\"" + dtype + "\",\"source_pattern\":\"";
    json_escape(out, pattern);
    out += "\"";
    if (kv_block) out += ",\"kv_block\":true";
    if (layer_cardinality >= 0)
        out += ",\"cardinality\":{" + std::string("\"layer\":") + fmt_int(layer_cardinality) + "}";
    return out + "}";
}

// --- чтение GGUF --------------------------------------------------------------

struct GgufMetaTable {
    std::map<std::string, GgufMetaValue> kv;
};

// Целочисленная метаданная (u32/u64/i32/i64, а также f32/f64 — редкий случай).
std::uint64_t meta_u64(const GgufMetaTable& meta, const std::string& key,
                       std::uint64_t def, std::string* err) {
    auto it = meta.kv.find(key);
    if (it == meta.kv.end()) return def;
    const GgufMetaValue& v = it->second;
    switch (v.type) {
        case kGgufMetaUint8: case kGgufMetaUint16: case kGgufMetaUint32:
        case kGgufMetaUint64: return v.u64;
        case kGgufMetaInt8: case kGgufMetaInt16: case kGgufMetaInt32:
        case kGgufMetaInt64: return static_cast<std::uint64_t>(v.i64);
        case kGgufMetaFloat32: case kGgufMetaFloat64:
            return static_cast<std::uint64_t>(v.f64);
        default:
            *err = "gguf: metadata '" + key + "' is not an integer value";
            return def;
    }
}

double meta_f64(const GgufMetaTable& meta, const std::string& key, double def,
                std::string* err) {
    auto it = meta.kv.find(key);
    if (it == meta.kv.end()) return def;
    const GgufMetaValue& v = it->second;
    if (v.type == kGgufMetaFloat32 || v.type == kGgufMetaFloat64) return v.f64;
    *err = "gguf: metadata '" + key + "' is not a float value";
    return def;
}

std::string meta_str(const GgufMetaTable& meta, const std::string& key,
                     const std::string& def) {
    auto it = meta.kv.find(key);
    if (it == meta.kv.end()) return def;
    return it->second.str;
}

// Требование целочисленной метаданной (обязательная).
std::uint64_t meta_req(const GgufMetaTable& meta, const std::string& key,
                       std::string* err) {
    auto it = meta.kv.find(key);
    if (it == meta.kv.end()) {
        *err = "gguf: missing required metadata '" + key + "'";
        return 0;
    }
    return meta_u64(meta, key, 0, err);
}

// Поиск тензора по нативному имени (разбор через parse_gguf_header).
const TensorInfo* find_tensor(const TensorFileMap& tfm, const std::string& name,
                              std::string* err) {
    auto it = tfm.by_name.find(name);
    if (it == tfm.by_name.end()) {
        *err = "gguf: tensor '" + name + "' not found (GGUF llama: нативные имена " +
               "llama.cpp; см. ADR-09 «Уточнение»)";
        return nullptr;
    }
    return &it->second;
}

// Подстановка "{layer}" числом — для проверки формы конкретного слоя
// (в описании паттерны остаются с токеном {layer}, как в llama-class.json).
std::string with_layer(const std::string& pattern, std::uint64_t layer) {
    const std::string token = "{layer}";
    const std::string num = std::to_string(layer);
    std::string out = pattern;
    std::size_t pos = 0;
    while ((pos = out.find(token, pos)) != std::string::npos)
        out.replace(pos, token.size(), num);
    return out;
}

// Группа матриц: dtype всегда F16 (конверсия F32→F16 при загрузке; GGUF
// Q4_0/Q8_0 → dequant → f32 → F16). Нормы: dtype тензора (F32/F16).
std::string tensor_dtype(const TensorInfo& t, std::string* err) {
    if (!t.dtype) {
        // Квантованные GGML-блоки (Q4_0/Q8_0): группы F16 (dequant → f32 → F16).
        if (is_quantized_ggml(t.dtype_str)) return "F16";
        *err = "gguf: tensor dtype '" + t.dtype_str + "' unsupported " +
               "(конвертер round 2: F32/F16/Q4_0/Q8_0; прочие квантованные — " +
               "через конвертер round 3)";
        return "";
    }
    switch (*t.dtype) {
        case stell::tiers::DType::F32: return "F32";
        case stell::tiers::DType::F16: return "F16";
        default: break;
    }
    *err = "gguf: tensor dtype '" + t.dtype_str + "' unsupported";
    return "";
}

}  // namespace

std::string gguf_to_description(const std::string& gguf_uri) {
    std::FILE* f = std::fopen(gguf_uri.c_str(), "rb");
    if (!f)
        throw LoaderError(LoaderError::Code::TensorNotFound,
                          "gguf: cannot open " + gguf_uri);
    try {
        std::uint8_t magic[4];
        read_exact(f, 0, magic, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: bad magic (expected 'GGUF')");
        if (std::fseek(f, 4, SEEK_SET) != 0)
            throw LoaderError(LoaderError::Code::IoFailure, "gguf: seek failed");
        const std::uint32_t version = gguf_read_u32(f);
        if (version != 3)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: unsupported version " + std::to_string(version) +
                                  " (subset: v3)");
        const std::uint64_t n_tensors = gguf_read_u64(f);
        const std::uint64_t n_kv = gguf_read_u64(f);

        // 1) Метаданные: ключ, тип, значение.
        GgufMetaTable meta;
        for (std::uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = gguf_read_string(f);
            const std::uint32_t type = gguf_read_u32(f);
            meta.kv[key] = gguf_read_meta_value(f, type);
        }

        // 2) Тензорные описания: переразбор через общий parse_gguf_header
        // (dims разворачиваются в логическую форму [out, in]).
        if (std::fseek(f, 0, SEEK_SET) != 0)
            throw LoaderError(LoaderError::Code::IoFailure, "gguf: seek failed");
        const TensorFileMap tfm = parse_gguf_header(f);
        std::fclose(f);
        f = nullptr;

        std::string err;
        // --- архитектура -------------------------------------------------------
        const std::string arch = meta_str(meta, "general.architecture", "");
        if (arch != "llama")
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: unsupported architecture '" + arch +
                                  "' (round 1 конвертер: только llama; " +
                                  "прочие — отдельные классы конвенций)");
        const std::string name = meta_str(meta, "general.name", "llama-gguf");

        const std::uint64_t n_layers =
            meta_req(meta, "llama.block_count", &err);
        if (err.empty() && n_layers == 0)
            err = "gguf: llama.block_count must be > 0";
        const std::uint64_t n_heads =
            meta_req(meta, "llama.attention.head_count", &err);
        const std::uint64_t d_model =
            meta_req(meta, "llama.embedding_length", &err);
        if (!err.empty())
            throw LoaderError(LoaderError::Code::SchemaMismatch, err);

        const std::uint64_t n_kv_heads =
            meta_u64(meta, "llama.attention.head_count_kv", n_heads, &err);
        std::uint64_t d_head = meta_u64(meta, "llama.attention.key_length", 0, &err);
        if (err.empty() && d_head == 0) {
            if (d_model % n_heads != 0)
                throw LoaderError(
                    LoaderError::Code::SchemaMismatch,
                    "gguf: llama.attention.key_length отсутствует, а "
                    "llama.embedding_length не делится на head_count");
            d_head = d_model / n_heads;
        }
        if (err.empty() && n_kv_heads == 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: llama.attention.head_count_kv must be > 0");
        const double eps = meta_f64(meta, "llama.attention.layer_norm_rms_epsilon",
                                    1e-5, &err);
        if (!err.empty())
            throw LoaderError(LoaderError::Code::SchemaMismatch, err);

        // --- обязательные тензора ----------------------------------------------
        const TensorInfo* emb = find_tensor(tfm, "token_embd.weight", &err);
        const TensorInfo* out_norm = find_tensor(tfm, "output_norm.weight", &err);
        const TensorInfo* head = find_tensor(tfm, "output.weight", &err);
        if (!err.empty())
            throw LoaderError(LoaderError::Code::TensorNotFound, err);
        std::uint64_t vocab_size =
            meta_u64(meta, "llama.vocab_size", 0, &err);
        if (err.empty() && vocab_size == 0)
            vocab_size = static_cast<std::uint64_t>(emb->shape[0]);
        const std::vector<std::int64_t> emb_expected = {
            static_cast<std::int64_t>(vocab_size),
            static_cast<std::int64_t>(d_model)};
        if (!err.empty() || emb->shape != emb_expected)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              err.empty() ? "gguf: token_embd.weight: shape mismatch" : err);
        if (head->shape != std::vector<std::int64_t>{static_cast<std::int64_t>(vocab_size), static_cast<std::int64_t>(d_model)})
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: output.weight: shape mismatch");

        // d_ff — из blk.0.ffn_up.weight (развернутый dims[0] = out dim).
        std::uint64_t d_ff = 0;
        if (n_layers > 0) {
            const TensorInfo* ffn_up =
                find_tensor(tfm, "blk.0.ffn_up.weight", &err);
            if (!err.empty())
                throw LoaderError(LoaderError::Code::TensorNotFound, err);
            d_ff = static_cast<std::uint64_t>(ffn_up->shape[0]);
        }

        std::string json;
        json += "{\"schema\":\"1.0\",\"name\":\"";
        json_escape(json, name);
        json += "\",\"class\":\"dense\",\"dtype_weights\":\"F16\"";
        json += ",\"vocab_size\":" + fmt_int(static_cast<std::int64_t>(vocab_size));
        json += ",\"d_model\":" + fmt_int(static_cast<std::int64_t>(d_model));
        json += ",\"n_layers\":" + fmt_int(static_cast<std::int64_t>(n_layers));
        json += ",\"n_heads\":" + fmt_int(static_cast<std::int64_t>(n_heads));
        json += ",\"d_head\":" + fmt_int(static_cast<std::int64_t>(d_head));
        json += ",\"kv\":{\"dtype\":\"F16\",\"n_kv_heads\":";
        json += fmt_int(static_cast<std::int64_t>(n_kv_heads));
        json += ",\"d_kv\":" + fmt_int(static_cast<std::int64_t>(d_head));
        json += ",\"quant\":null}";
        json += ",\"ops\":[{\"op\":\"rmsnorm\",\"io\":[\"hidden\"],\"params\":{\"eps\":";
        json += fmt_f64(eps);
        json += "}}";
        json += ",{\"op\":\"attention\",\"io\":[\"hidden\",\"kv\"],\"params\":{\"kernel\":\"sm70_fp32_qk_pv\"}}";
        json += ",{\"op\":\"ffn\",\"io\":[\"hidden\"],\"params\":{\"activation\":\"silu_mul\"}}]";
        json += ",\"tensor_groups\":[";

        // --- валидация слоёв ---------------------------------------------------
        auto check_tensor = [&](const std::string& name,
                                const std::vector<std::int64_t>& expected) {
            const TensorInfo* t = find_tensor(tfm, name, &err);
            if (!err.empty())
                throw LoaderError(LoaderError::Code::TensorNotFound, err);
            std::string dt = tensor_dtype(*t, &err);
            if (!err.empty())
                throw LoaderError(LoaderError::Code::UnsupportedQuant, err);
            if (t->shape != expected)
                throw LoaderError(
                    LoaderError::Code::SchemaMismatch,
                    "gguf: tensor '" + name + "': shape mismatch (развернутый dims: [out, in])");
            return dt;
        };

        struct LayerTensor {
            const char* pattern;   // нативное GGUF-имено с токеном {layer}
            bool is_norm;          // форма {d_model} (1-D)
            std::uint64_t out_dim; // ожидаемый shape[0] матрицы
            std::uint64_t in_dim;  // ожидаемый shape[1] матрицы
        };
        const LayerTensor layer_tensors[] = {
            {"blk.{layer}.attn_norm.weight", true, 0, 0},
            {"blk.{layer}.ffn_norm.weight", true, 0, 0},
            {"blk.{layer}.attn_q.weight", false, n_heads * d_head, d_model},
            {"blk.{layer}.attn_k.weight", false, n_kv_heads * d_head, d_model},
            {"blk.{layer}.attn_v.weight", false, n_kv_heads * d_head, d_model},
            {"blk.{layer}.attn_output.weight", false, d_model, d_model},
            {"blk.{layer}.ffn_gate.weight", false, d_ff, d_model},
            {"blk.{layer}.ffn_up.weight", false, d_ff, d_model},
            {"blk.{layer}.ffn_down.weight", false, d_model, d_ff},
        };
        for (std::uint64_t layer = 0; layer < n_layers; ++layer) {
            for (const LayerTensor& lt : layer_tensors) {
                std::vector<std::int64_t> expected;
                if (lt.is_norm)
                    expected = {static_cast<std::int64_t>(d_model)};
                else
                    expected = {static_cast<std::int64_t>(lt.out_dim),
                                static_cast<std::int64_t>(lt.in_dim)};
                check_tensor(with_layer(lt.pattern, layer), expected);
            }
        }

        // --- шаблоны групп (один раз; cardinality слой размножит) --------------
        std::vector<std::string> groups;
        std::string dt_emb = check_tensor("token_embd.weight", emb->shape);
        std::string dt_on = check_tensor("output_norm.weight", out_norm->shape);
        groups.push_back(make_group("embed", "attn", emb->shape, dt_emb,
                                    "token_embd.weight"));
        groups.push_back(make_group("norm_final", "attn", out_norm->shape, dt_on,
                                    "output_norm.weight"));
        groups.push_back(make_group("head", "attn", head->shape, "F16",
                                    "output.weight"));
        // Нормы: dtype тензора слоя 0 (F32/F16 — GGUF хранит фактический);
        // матрицы: F16 (конверсия F32→F16 / dequant Q4_0/Q8_0 при загрузке).
        std::string dt_an = check_tensor(
            "blk.0.attn_norm.weight", {static_cast<std::int64_t>(d_model)});
        std::string dt_fn = check_tensor(
            "blk.0.ffn_norm.weight", {static_cast<std::int64_t>(d_model)});
        groups.push_back(make_group("l{layer}.attn.norm", "attn",
                                    {static_cast<std::int64_t>(d_model)}, dt_an,
                                    "blk.{layer}.attn_norm.weight",
                                    static_cast<std::int64_t>(n_layers)));
        groups.push_back(make_group("l{layer}.ffn.norm", "attn",
                                    {static_cast<std::int64_t>(d_model)}, dt_fn,
                                    "blk.{layer}.ffn_norm.weight",
                                    static_cast<std::int64_t>(n_layers)));
        struct MatTemplate {
            const char* tensor;
            const char* group;
            std::uint64_t out_dim;
            std::uint64_t in_dim;
        };
        const MatTemplate mats[] = {
            {"blk.{layer}.attn_q.weight", "l{layer}.attn.wq",
             n_heads * d_head, d_model},
            {"blk.{layer}.attn_k.weight", "l{layer}.attn.wk",
             n_kv_heads * d_head, d_model},
            {"blk.{layer}.attn_v.weight", "l{layer}.attn.wv",
             n_kv_heads * d_head, d_model},
            {"blk.{layer}.attn_output.weight", "l{layer}.attn.wo", d_model,
             d_model},
            {"blk.{layer}.ffn_gate.weight", "l{layer}.ffn.gate", d_ff,
             d_model},
            {"blk.{layer}.ffn_up.weight", "l{layer}.ffn.up", d_ff, d_model},
            {"blk.{layer}.ffn_down.weight", "l{layer}.ffn.down", d_model, d_ff},
        };
        for (const MatTemplate& m : mats) {
            const std::vector<std::int64_t> expected = {
                static_cast<std::int64_t>(m.out_dim),
                static_cast<std::int64_t>(m.in_dim)};
            check_tensor(with_layer(m.tensor, 0), expected);
            groups.push_back(make_group(m.group, "attn", expected, "F16",
                                        m.tensor,
                                        static_cast<std::int64_t>(n_layers)));
        }
        // KV-группа: в GGUF KV нет — псевдо-паттерн blk.{layer}.kv_cache
        // (лоадер роль=kv пропускает, ADR-09 §1).
        const std::vector<std::int64_t> kv_shape = {
            static_cast<std::int64_t>(d_model),
            static_cast<std::int64_t>(n_kv_heads * d_head)};
        groups.push_back(make_group("l{layer}.kv", "kv", kv_shape, "F16",
                                    "blk.{layer}.kv_cache",
                                    static_cast<std::int64_t>(n_layers),
                                    /*kv_block=*/true));

        for (std::size_t i = 0; i < groups.size(); ++i) {
            if (i) json += ",";
            json += groups[i];
        }
        json += "]}";
        return json;
    } catch (...) {
        if (f) std::fclose(f);
        throw;
    }
}

}  // namespace stell::models

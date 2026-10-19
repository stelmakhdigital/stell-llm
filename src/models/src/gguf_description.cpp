// ============================================================================
// Конвертер GGUF → описание модели (round 2, задача 5 + R9; ADR-09
// «Уточнение»).
//
// GGUF не несёт JSON-описания — models генерирует его из метаданных и
// тензорных имён файла, используя НАТИВНЫЕ имена тензоров формата GGUF
// (см. ADR-09 «Уточнение»):
//   token_embd.weight, output_norm.weight, output.weight,
//   blk.{i}.attn_norm.weight, blk.{i}.attn_q.weight, blk.{i}.attn_k.weight,
//   blk.{i}.attn_v.weight, blk.{i}.attn_output.weight,
//   blk.{i}.ffn_norm.weight, blk.{i}.ffn_gate.weight, blk.{i}.ffn_up.weight,
//   blk.{i}.ffn_down.weight.
// source_pattern в генерируемом описании — те же нативные имена; GGUF-лоадер
// сравнивает их буквально (ADR-09 §3). KV-группы — псевдо-паттерн
// blk.{layer}.kv_cache (в файле KV нет, лоадер роль=kv пропускает).
//
// Поддержанные архитектуры (subset):
//   - "llama" (dense): метаданные llama.block_count / llama.attention.
//     head_count / llama.embedding_length обязательны; llama.attention.
//     head_count_kv, llama.attention.key_length, llama.attention.
//     layer_norm_rms_epsilon, llama.vocab_size, general.name — опциональны
//     (умолчания: n_kv_heads = n_heads; d_head = d_model/n_heads; eps = 1e-5;
//     vocab_size — из token_embd; name = "llama-gguf").
//   - "mixtral" (MoE, R9): дополнительно обязательны llama.expert_count
//     (n_experts) и llama.expert_used_count (n_experts_active) — ИЗ МЕТАДАННЫХ
//     GGUF; нативные имена MoE-раскладки:
//       blk.{i}.ffn_gate_inp.weight      — роутер [n_experts, d_model];
//       blk.{i}.exp_probs_b.weight       — смещение роутера [n_experts]
//                                          (опционально: в исходной раскладке
//                                          mixtral смещения
//                                          нет — группа не генерируется);
//       blk.{i}.ffn_gate_exps.weight     — gate всех экспертов, стековный
//                                          тензор [n_experts, d_ff, d_model];
//       blk.{i}.ffn_up_exps.weight       — up, стековный [n_experts, d_ff,
//                                          d_model];
//       blk.{i}.ffn_down_exps.weight     — down, стековный [n_experts,
//                                          d_model, d_ff].
//     Стековные тензорные группы экспертов получают cardinality
//     {layer, expert} и source_slice = "{expert}" (срез по старшей
//     размерности — непрерывный блок, row-major; см. description.hpp).
//     d_ff берётся из blk.0.ffn_up_exps.weight (развернутый dims[1]).
//
// SHAPE: тензорные dims GGUF разворачиваются при разборе (parse_gguf_header,
// tensor_file_formats.hpp) — логическая форма [out, in] (конвенция
// раскладки JSON-конфигов/сafetensors).
// Dtype: матрицы — F16 (конверсия F32→F16 при загрузке); нормы/вложения/
// роутер — по dtype тензора (F32/F16); GGUF Q4_0/Q8_0 — квантованные блоки
// (dequant → f32 → целевой dtype, см. loader_gguf.cpp); прочие квантованные
// dtype — UnsupportedQuant.
// ============================================================================
#include "stell/models/convert.hpp"

#include "quant.hpp"
#include "tensor_file_formats.hpp"
#include "description_json.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace stell::models {

namespace {



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
        *err = "gguf: tensor '" + name + "' not found (нативные имена тензоров "
               "формата GGUF; см. ADR-09 «Уточнение»)";
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
// Q4_0/Q8_0 → dequant → f32 → F16). Нормы/роутер: dtype тензора (F32/F16).
std::string tensor_dtype(const TensorInfo& t, std::string* err) {
    if (!t.dtype) {
        // Квантованные GGML-блоки (Q4_0/Q8_0): группы F16 (dequant → f32 → F16).
        if (is_quantized_ggml(t.dtype_str)) return "F16";
        *err = "gguf: tensor dtype '" + t.dtype_str + "' unsupported " +
               "(конвертер round 2: F32/F16/Q4_0/Q8_0; прочие квантованные — " +
               "через отдельный класс конверсии)";
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

// Проверка тензора: dtype поддержан, форма == expected. Возвращает dtype
// группы (F32/F16/F16-для-квантованных). Ошибки — LoaderError с честным
// кодом: отсутствие тензора → TensorNotFound, недопустимый dtype →
// UnsupportedQuant, расхождение формы → SchemaMismatch (как в round 1).
std::string check_tensor(const TensorFileMap& tfm, const std::string& name,
                         const std::vector<std::int64_t>& expected) {
    std::string err;
    const TensorInfo* t = find_tensor(tfm, name, &err);
    const std::string dt = tensor_dtype(*t, &err);
    if (!err.empty())
        throw LoaderError(LoaderError::Code::UnsupportedQuant, err);
    if (t->shape != expected)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "gguf: tensor '" + name +
                              "': shape mismatch (развернутый dims: [out, in])");
    return dt;
}

// Общие метаданные attention (слои, головы, kv, eps) — одинаковы для llama
// (dense) и mixtral (MoE); возвращает false при ошибке (err заполнен).
struct AttnMeta {
    std::uint64_t n_layers = 0, n_heads = 0, d_model = 0, n_kv_heads = 0,
                   d_head = 0;
    double eps = 1e-5;
};

bool read_attn_meta(const GgufMetaTable& meta, AttnMeta* out, std::string* err) {
    out->n_layers = meta_req(meta, "llama.block_count", err);
    if (err->empty() && out->n_layers == 0)
        *err = "gguf: llama.block_count must be > 0";
    out->n_heads = meta_req(meta, "llama.attention.head_count", err);
    out->d_model = meta_req(meta, "llama.embedding_length", err);
    if (!err->empty()) return false;
    out->n_kv_heads = meta_u64(meta, "llama.attention.head_count_kv",
                               out->n_heads, err);
    std::uint64_t d_head = meta_u64(meta, "llama.attention.key_length", 0, err);
    if (err->empty() && d_head == 0) {
        if (out->d_model % out->n_heads != 0)
            *err = "gguf: llama.attention.key_length отсутствует, а "
                   "llama.embedding_length не делится на head_count";
        d_head = out->d_model / out->n_heads;
    }
    if (!err->empty()) return false;
    out->d_head = d_head;
    if (out->n_kv_heads == 0)
        *err = "gguf: llama.attention.head_count_kv must be > 0";
    if (err->empty())
        out->eps = meta_f64(meta, "llama.attention.layer_norm_rms_epsilon",
                            1e-5, err);
    return err->empty();
}

// Верхние тензора (token_embd/output_norm/output) и vocab_size — общие для
// llama (dense) и mixtral. Отсутствие тензора → TensorNotFound; расхождение
// формы → SchemaMismatch (тишина запрещена, ADR-09 §3). head-тензора может
// не быть в некоторых планах — здесь он обязателен (нативная конвенция).
const TensorInfo* read_common_tensors(const GgufMetaTable& meta,
                                      const TensorFileMap& tfm,
                                      const AttnMeta& attn,
                                      std::uint64_t* vocab_size) {
    std::string err;
    const TensorInfo* emb = find_tensor(tfm, "token_embd.weight", &err);
    if (err.empty()) (void)find_tensor(tfm, "output_norm.weight", &err);
    const TensorInfo* head = nullptr;
    if (err.empty()) head = find_tensor(tfm, "output.weight", &err);
    if (!err.empty())
        throw LoaderError(LoaderError::Code::TensorNotFound, err);
    *vocab_size = meta_u64(meta, "llama.vocab_size", 0, &err);
    if (!err.empty())
        throw LoaderError(LoaderError::Code::SchemaMismatch, err);
    if (*vocab_size == 0)
        *vocab_size = static_cast<std::uint64_t>(emb->shape[0]);
    const std::vector<std::int64_t> emb_expected = {
        static_cast<std::int64_t>(*vocab_size),
        static_cast<std::int64_t>(attn.d_model)};
    if (emb->shape != emb_expected)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "gguf: token_embd.weight: shape mismatch");
    if (head->shape != std::vector<std::int64_t>{
            static_cast<std::int64_t>(*vocab_size),
            static_cast<std::int64_t>(attn.d_model)})
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "gguf: output.weight: shape mismatch");
    return head;
}

// --- llama (dense) ------------------------------------------------------------

std::string llama_description(const GgufMetaTable& meta,
                              const TensorFileMap& tfm) {
    std::string err;
    const std::string name = meta_str(meta, "general.name", "llama-gguf");

    AttnMeta attn;
    if (!read_attn_meta(meta, &attn, &err))
        throw LoaderError(LoaderError::Code::SchemaMismatch, err);
    const std::uint64_t n_layers = attn.n_layers;

    std::uint64_t vocab_size = 0;
    const TensorInfo* head = read_common_tensors(meta, tfm, attn, &vocab_size);

    // d_ff — из blk.0.ffn_up.weight (развернутый dims[0] = out dim).
    std::uint64_t d_ff = 0;
    if (n_layers > 0) {
        std::string e2;
        const TensorInfo* ffn_up = find_tensor(tfm, "blk.0.ffn_up.weight", &e2);
        if (!e2.empty())
            throw LoaderError(LoaderError::Code::TensorNotFound, e2);
        d_ff = static_cast<std::uint64_t>(ffn_up->shape[0]);
    }

    std::string json;
    json += "{\"schema\":\"1.0\",\"name\":\"";
    json_escape(json, name);
    json += "\",\"class\":\"dense\",\"dtype_weights\":\"F16\"";
    json += ",\"vocab_size\":" + fmt_int(static_cast<std::int64_t>(vocab_size));
    json += ",\"d_model\":" + fmt_int(static_cast<std::int64_t>(attn.d_model));
    json += ",\"n_layers\":" + fmt_int(static_cast<std::int64_t>(n_layers));
    json += ",\"n_heads\":" + fmt_int(static_cast<std::int64_t>(attn.n_heads));
    json += ",\"d_head\":" + fmt_int(static_cast<std::int64_t>(attn.d_head));
    json += ",\"kv\":{\"dtype\":\"F16\",\"n_kv_heads\":";
    json += fmt_int(static_cast<std::int64_t>(attn.n_kv_heads));
    json += ",\"d_kv\":" + fmt_int(static_cast<std::int64_t>(attn.d_head));
    json += ",\"quant\":null}";
    json += ",\"ops\":[{\"op\":\"rmsnorm\",\"io\":[\"hidden\"],\"params\":{\"eps\":";
    json += fmt_f64(attn.eps);
    json += "}}";
    json += ",{\"op\":\"attention\",\"io\":[\"hidden\",\"kv\"],\"params\":{\"kernel\":\"sm70_fp32_qk_pv\"}}";
    json += ",{\"op\":\"ffn\",\"io\":[\"hidden\"],\"params\":{\"activation\":\"silu_mul\"}}]";
    json += ",\"tensor_groups\":[";

    std::vector<std::string> groups;
    // --- валидация слоёв ---------------------------------------------------
    {
        struct LayerTensor {
            const char* pattern;   // нативное GGUF-имя с токеном {layer}
            bool is_norm;          // форма {d_model} (1-D)
            std::uint64_t out_dim; // ожидаемый shape[0] матрицы
            std::uint64_t in_dim;  // ожидаемый shape[1] матрицы
        };
        const LayerTensor layer_tensors[] = {
            {"blk.{layer}.attn_norm.weight", true, 0, 0},
            {"blk.{layer}.ffn_norm.weight", true, 0, 0},
            {"blk.{layer}.attn_q.weight", false, attn.n_heads * attn.d_head, attn.d_model},
            {"blk.{layer}.attn_k.weight", false, attn.n_kv_heads * attn.d_head, attn.d_model},
            {"blk.{layer}.attn_v.weight", false, attn.n_kv_heads * attn.d_head, attn.d_model},
            {"blk.{layer}.attn_output.weight", false, attn.d_model, attn.d_model},
            {"blk.{layer}.ffn_gate.weight", false, d_ff, attn.d_model},
            {"blk.{layer}.ffn_up.weight", false, d_ff, attn.d_model},
            {"blk.{layer}.ffn_down.weight", false, attn.d_model, d_ff},
        };
        for (std::uint64_t layer = 0; layer < n_layers; ++layer) {
            for (const LayerTensor& lt : layer_tensors) {
                std::vector<std::int64_t> expected;
                if (lt.is_norm)
                    expected = {static_cast<std::int64_t>(attn.d_model)};
                else
                    expected = {static_cast<std::int64_t>(lt.out_dim),
                                static_cast<std::int64_t>(lt.in_dim)};
                check_tensor(tfm, with_layer(lt.pattern, layer), expected);
            }
        }
    }

    // --- шаблоны групп (один раз; cardinality слой размножит) --------------
    {
        std::string dt_emb = check_tensor(tfm, "token_embd.weight",
                                          {static_cast<std::int64_t>(vocab_size),
                                           static_cast<std::int64_t>(attn.d_model)});
        groups.push_back(make_group("embed", "attn", {static_cast<std::int64_t>(vocab_size),
                                                      static_cast<std::int64_t>(attn.d_model)},
                                    dt_emb, "token_embd.weight"));
        std::string dt_on = check_tensor(tfm, "output_norm.weight",
                                         {static_cast<std::int64_t>(attn.d_model)});
        groups.push_back(make_group("norm_final", "attn",
                                    {static_cast<std::int64_t>(attn.d_model)}, dt_on,
                                    "output_norm.weight"));
        if (head) {
            groups.push_back(make_group("head", "attn", head->shape, "F16",
                                        "output.weight"));
        }
        // Нормы: dtype тензора слоя 0 (F32/F16 — GGUF хранит фактический);
        // матрицы: F16 (конверсия F32→F16 / dequant Q4_0/Q8_0 при загрузке).
        std::string dt_an = check_tensor(tfm, "blk.0.attn_norm.weight",
                                         {static_cast<std::int64_t>(attn.d_model)});
        std::string dt_fn = check_tensor(tfm, "blk.0.ffn_norm.weight",
                                         {static_cast<std::int64_t>(attn.d_model)});
        groups.push_back(make_group("l{layer}.attn.norm", "attn",
                                    {static_cast<std::int64_t>(attn.d_model)}, dt_an,
                                    "blk.{layer}.attn_norm.weight",
                                    {{"layer", static_cast<std::int64_t>(n_layers)}}));
        groups.push_back(make_group("l{layer}.ffn.norm", "attn",
                                    {static_cast<std::int64_t>(attn.d_model)}, dt_fn,
                                    "blk.{layer}.ffn_norm.weight",
                                    {{"layer", static_cast<std::int64_t>(n_layers)}}));
        struct MatTemplate {
            const char* tensor;
            const char* group;
            std::uint64_t out_dim;
            std::uint64_t in_dim;
        };
        const MatTemplate mats[] = {
            {"blk.{layer}.attn_q.weight", "l{layer}.attn.wq",
             attn.n_heads * attn.d_head, attn.d_model},
            {"blk.{layer}.attn_k.weight", "l{layer}.attn.wk",
             attn.n_kv_heads * attn.d_head, attn.d_model},
            {"blk.{layer}.attn_v.weight", "l{layer}.attn.wv",
             attn.n_kv_heads * attn.d_head, attn.d_model},
            {"blk.{layer}.attn_output.weight", "l{layer}.attn.wo", attn.d_model,
             attn.d_model},
            {"blk.{layer}.ffn_gate.weight", "l{layer}.ffn.gate", d_ff,
             attn.d_model},
            {"blk.{layer}.ffn_up.weight", "l{layer}.ffn.up", d_ff, attn.d_model},
            {"blk.{layer}.ffn_down.weight", "l{layer}.ffn.down", attn.d_model, d_ff},
        };
        for (const MatTemplate& m : mats) {
            const std::vector<std::int64_t> expected = {
                static_cast<std::int64_t>(m.out_dim),
                static_cast<std::int64_t>(m.in_dim)};
            check_tensor(tfm, with_layer(m.tensor, 0), expected);
            groups.push_back(make_group(m.group, "attn", expected, "F16",
                                        m.tensor,
                                        {{"layer", static_cast<std::int64_t>(n_layers)}}));
        }
        // KV-группа: в GGUF KV нет — псевдо-паттерн blk.{layer}.kv_cache
        // (лоадер роль=kv пропускает, ADR-09 §1).
        const std::vector<std::int64_t> kv_shape = {
            static_cast<std::int64_t>(attn.d_model),
            static_cast<std::int64_t>(attn.n_kv_heads * attn.d_head)};
        groups.push_back(make_group("l{layer}.kv", "kv", kv_shape, "F16",
                                    "blk.{layer}.kv_cache",
                                    {{"layer", static_cast<std::int64_t>(n_layers)}},
                                    "", /*kv_block=*/true));
    }

    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (i) json += ",";
        json += groups[i];
    }
    json += "]}";
    return json;
}

// --- mixtral (MoE, R9) --------------------------------------------------------

std::string mixtral_description(const GgufMetaTable& meta,
                                const TensorFileMap& tfm) {
    std::string err;
    const std::string name = meta_str(meta, "general.name", "mixtral-gguf");

    AttnMeta attn;
    if (!read_attn_meta(meta, &attn, &err))
        throw LoaderError(LoaderError::Code::SchemaMismatch, err);
    const std::uint64_t n_layers = attn.n_layers;

    // MoE-метаданные — из GGUF (план Q8: n_experts/n_experts_used из метаданных).
    const std::uint64_t n_experts = meta_req(meta, "llama.expert_count", &err);
    const std::uint64_t n_active = meta_req(meta, "llama.expert_used_count", &err);
    if (!err.empty())
        throw LoaderError(LoaderError::Code::SchemaMismatch, err);
    if (n_experts == 0 || n_active == 0 || n_active > n_experts)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "gguf: llama.expert_count/expert_used_count must be "
                          "positive and expert_used_count <= expert_count");

    std::uint64_t vocab_size = 0;
    const TensorInfo* head = read_common_tensors(meta, tfm, attn, &vocab_size);

    // d_ff — из blk.0.ffn_up_exps.weight: логическая форма
    // [n_experts, d_ff, d_model] → d_ff = shape[1].
    std::uint64_t d_ff = 0;
    {
        std::string e2;
        const TensorInfo* up_exps =
            find_tensor(tfm, "blk.0.ffn_up_exps.weight", &e2);
        if (!e2.empty())
            throw LoaderError(LoaderError::Code::TensorNotFound, e2);
        if (up_exps->shape.size() != 3 ||
            up_exps->shape[0] != static_cast<std::int64_t>(n_experts) ||
            up_exps->shape[2] != static_cast<std::int64_t>(attn.d_model))
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: blk.0.ffn_up_exps.weight: shape mismatch "
                              "(expected [n_experts, d_ff, d_model])");
        d_ff = static_cast<std::uint64_t>(up_exps->shape[1]);
        if (d_ff == 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "gguf: d_ff must be positive");
    }

    // --- валидация слоёв ---------------------------------------------------
    const std::int64_t E = static_cast<std::int64_t>(n_experts);
    const std::int64_t dm = static_cast<std::int64_t>(attn.d_model);
    const std::int64_t dff = static_cast<std::int64_t>(d_ff);
    const std::int64_t dkv = static_cast<std::int64_t>(attn.n_kv_heads * attn.d_head);
    const std::int64_t dq = static_cast<std::int64_t>(attn.n_heads * attn.d_head);
    struct LayerTensor {
        const char* pattern;
        std::vector<std::int64_t> expected;
    };
    for (std::uint64_t layer = 0; layer < n_layers; ++layer) {
        const LayerTensor lts[] = {
            {"blk.{layer}.attn_norm.weight", {dm}},
            {"blk.{layer}.ffn_norm.weight", {dm}},
            {"blk.{layer}.attn_q.weight", {dq, dm}},
            {"blk.{layer}.attn_k.weight", {dkv, dm}},
            {"blk.{layer}.attn_v.weight", {dkv, dm}},
            {"blk.{layer}.attn_output.weight", {dm, dm}},
            {"blk.{layer}.ffn_gate_inp.weight", {E, dm}},
            {"blk.{layer}.ffn_gate_exps.weight", {E, dff, dm}},
            {"blk.{layer}.ffn_up_exps.weight", {E, dff, dm}},
            {"blk.{layer}.ffn_down_exps.weight", {E, dm, dff}},
        };
        for (const LayerTensor& lt : lts) {
            check_tensor(tfm, with_layer(lt.pattern, layer), lt.expected);
        }
    }

    // --- JSON описания ------------------------------------------------------
    std::string json;
    json += "{\"schema\":\"1.0\",\"name\":\"";
    json_escape(json, name);
    json += "\",\"class\":\"moe\",\"dtype_weights\":\"F16\"";
    json += ",\"vocab_size\":" + fmt_int(static_cast<std::int64_t>(vocab_size));
    json += ",\"d_model\":" + fmt_int(static_cast<std::int64_t>(attn.d_model));
    json += ",\"n_layers\":" + fmt_int(static_cast<std::int64_t>(n_layers));
    json += ",\"n_heads\":" + fmt_int(static_cast<std::int64_t>(attn.n_heads));
    json += ",\"d_head\":" + fmt_int(static_cast<std::int64_t>(attn.d_head));
    json += ",\"kv\":{\"dtype\":\"F16\",\"n_kv_heads\":";
    json += fmt_int(static_cast<std::int64_t>(attn.n_kv_heads));
    json += ",\"d_kv\":" + fmt_int(static_cast<std::int64_t>(attn.d_head));
    json += ",\"quant\":null}";
    json += ",\"moe\":{\"n_experts\":" + fmt_int(static_cast<std::int64_t>(n_experts));
    json += ",\"n_experts_active\":" + fmt_int(static_cast<std::int64_t>(n_active));
    json += ",\"expert_d_ff\":" + fmt_int(static_cast<std::int64_t>(d_ff));
    json += ",\"router\":\"softmax-topk\"}";
    json += ",\"ops\":[{\"op\":\"rmsnorm\",\"io\":[\"hidden\"],\"params\":{\"eps\":";
    json += fmt_f64(attn.eps);
    json += "}}";
    json += ",{\"op\":\"attention\",\"io\":[\"hidden\",\"kv\"],\"params\":{\"kernel\":\"sm70_fp32_qk_pv\"}}";
    json += ",{\"op\":\"moe_gather_scatter\",\"io\":[\"hidden\",\"experts\"],\"params\":{\"kernel\":\"sm70_moe_gs\"}}]";
    json += ",\"tensor_groups\":[";

    std::vector<std::string> groups;
    {
        std::string dt_emb = check_tensor(tfm, "token_embd.weight",
                                          {static_cast<std::int64_t>(vocab_size), dm});
        groups.push_back(make_group("embed", "attn",
                                    {static_cast<std::int64_t>(vocab_size), dm},
                                    dt_emb, "token_embd.weight"));
        std::string dt_on = check_tensor(tfm, "output_norm.weight", {dm});
        groups.push_back(make_group("norm_final", "attn", {dm}, dt_on,
                                    "output_norm.weight"));
        if (head)
            groups.push_back(make_group("head", "attn", head->shape, "F16",
                                        "output.weight"));

        const std::string layer_card_key = "layer";
        const std::int64_t layer_card = static_cast<std::int64_t>(n_layers);
        std::string dt_an = check_tensor(tfm, "blk.0.attn_norm.weight", {dm});
        std::string dt_fn = check_tensor(tfm, "blk.0.ffn_norm.weight", {dm});
        groups.push_back(make_group("l{layer}.attn.norm", "attn", {dm}, dt_an,
                                    "blk.{layer}.attn_norm.weight",
                                    {{layer_card_key, layer_card}}));
        groups.push_back(make_group("l{layer}.ffn.norm", "attn", {dm}, dt_fn,
                                    "blk.{layer}.ffn_norm.weight",
                                    {{layer_card_key, layer_card}}));

        struct MatTemplate {
            const char* tensor;
            const char* group;
            std::vector<std::int64_t> shape;
        };
        const MatTemplate mats[] = {
            {"blk.{layer}.attn_q.weight", "l{layer}.attn.wq", {dq, dm}},
            {"blk.{layer}.attn_k.weight", "l{layer}.attn.wk", {dkv, dm}},
            {"blk.{layer}.attn_v.weight", "l{layer}.attn.wv", {dkv, dm}},
            {"blk.{layer}.attn_output.weight", "l{layer}.attn.wo", {dm, dm}},
        };
        for (const MatTemplate& m : mats) {
            check_tensor(tfm, with_layer(m.tensor, 0), m.shape);
            groups.push_back(make_group(m.group, "attn", m.shape, "F16",
                                        m.tensor,
                                        {{layer_card_key, layer_card}}));
        }
        // KV-группа: в GGUF KV нет — псевдо-паттерн blk.{layer}.kv_cache
        // (лоадер роль=kv пропускает, ADR-09 §1).
        groups.push_back(make_group("l{layer}.kv", "kv", {dm, dkv}, "F16",
                                    "blk.{layer}.kv_cache",
                                    {{layer_card_key, layer_card}}, "",
                                    /*kv_block=*/true));

        // Роутер: dtype тензора (F32/F16); смещение роутера — опциональный
        // нативный тензор exp_probs_b (в части исходных раскладок mixtral
        // отсутствует).
        std::string dt_rt = check_tensor(tfm, "blk.0.ffn_gate_inp.weight",
                                         {E, dm});
        groups.push_back(make_group("l{layer}.mlp.router", "expert", {E, dm},
                                    dt_rt, "blk.{layer}.ffn_gate_inp.weight",
                                    {{layer_card_key, layer_card}}));
        std::string e3;
        if (find_tensor(tfm, "blk.0.exp_probs_b.weight", &e3)) {
            std::string dt_b = check_tensor(tfm, "blk.0.exp_probs_b.weight", {E});
            groups.push_back(make_group("l{layer}.mlp.router_bias", "expert",
                                        {E}, dt_b,
                                        "blk.{layer}.exp_probs_b.weight",
                                        {{layer_card_key, layer_card}}));
        }

        // Стековные экспертные тензоры: cardinality {layer, expert} (порядок
        // документа: слой — внешний, эксперт — внутренний, быстрый),
        // source_slice = "{expert}" — срез по старшей размерности.
        const std::vector<std::pair<std::string, std::int64_t>> card = {
            {layer_card_key, layer_card}, {"expert", E}};
        struct StackTemplate {
            const char* tensor;
            const char* group;
            std::vector<std::int64_t> shape;
        };
        const StackTemplate stacks[] = {
            {"blk.{layer}.ffn_gate_exps.weight", "l{layer}.e{expert}.gate",
             {dff, dm}},
            {"blk.{layer}.ffn_up_exps.weight", "l{layer}.e{expert}.up",
             {dff, dm}},
            {"blk.{layer}.ffn_down_exps.weight", "l{layer}.e{expert}.down",
             {dm, dff}},
        };
        for (const StackTemplate& s : stacks) {
            check_tensor(tfm, with_layer(s.tensor, 0),
                         {E, s.shape[0], s.shape[1]});
            groups.push_back(make_group(s.group, "expert", s.shape, "F16",
                                        s.tensor, card, "{expert}"));
        }
    }

    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (i) json += ",";
        json += groups[i];
    }
    json += "]}";
    return json;
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
        (void)n_tensors;  // описания тензоров перечитываются parse_gguf_header
        (void)n_kv;       // (общий разбор; здесь читаются только метаданные)

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

        // 3) Архитектура: llama (dense) / mixtral (MoE, R9) — по метаданным.
        const std::string arch = meta_str(meta, "general.architecture", "");
        if (arch == "llama") return llama_description(meta, tfm);
        if (arch == "mixtral") return mixtral_description(meta, tfm);
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "gguf: unsupported architecture '" + arch +
                              "' (subset: llama, mixtral; прочие — отдельные "
                              "классы конвенций)");
    } catch (...) {
        if (f) std::fclose(f);
        throw;
    }
}

}  // namespace stell::models

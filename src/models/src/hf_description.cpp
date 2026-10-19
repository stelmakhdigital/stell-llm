// ============================================================================
// Конвертер HF-чекпоинта (config.json + safetensors-шарды) → описание модели
// (round 4, [R4]; ADR-09, по прецеденту gguf_to_description).
//
// Вход: каталог модели HF (config.json + model-0000X-of-0000Y.safetensors;
// канонический порядок шардов — отсортированный) ИЛИ одиночный .safetensors.
// Имена тензоров — НАТИВНЫЕ имена HF (model.embed_tokens.weight,
// model.layers.{layer}.self_attn.q_proj.weight, …); source_pattern в
// генерируемом описании — те же нативные имена, лоадер/конвертер сравнивают
// их буквально (ADR-09 §3).
//
// config.json (subset):
//   architectures (обяз.)  — диспетчер архитектуры (см. ниже);
//   hidden_size (обяз.)    → d_model;
//   num_hidden_layers (обяз.) → n_layers;
//   num_attention_heads (обяз.) → n_heads;
//   num_key_value_heads (опц., умолчание n_heads) → kv.n_kv_heads;
//   head_dim (опц., умолчание hidden_size/num_attention_heads) → d_head;
//   vocab_size (опц., умолчание — dim0 тензора model.embed_tokens.weight);
//   rms_norm_eps (опц., умолчание 1e-5) → ops rmsnorm eps;
//   rope_theta (опц.)      → параметр attention (rope-параметр, [R4]);
//   intermediate_size (обяз. dense) → d_ff (валидация по формам тензоров);
//   num_experts|num_local_experts + num_experts_per_tok (обяз. MoE).
//
// Поддержанные архитектуры (subset):
//   - LlamaForCausalLM / Qwen2ForCausalLM (dense): llama-нейминг тензоров HF.
//   - Qwen2MoeForCausalLM / Qwen1.5MoeForCausalLM / OlmoeForCausalLM (MoE,
//     поэертельные тензоры экспертов): роутер — mlp.router.weight (Qwen2Moe)
//     или mlp.gate.weight (Qwen1.5Moe/Olmoe — выбирается по факту в шардах);
//     эксперты — mlp.experts.{expert}.{gate,up,down}_proj.weight;
//     shared-expert тензоры плана не касаются (не группируются — честная
//     пометка в REPORT-ROUND4.md; движковый MoE-контур shared expert не
//     содержит, интерпретатор плана такие группы не требует).
//   - MixtralForCausalLM (MoE, стековные тензоры): block_sparse_moe.gate
//     (роутер), block_sparse_moe.w1/w3 ([n_experts, d_ff, d_model] — gate/up)
//     и w2 ([n_experts, d_model, d_ff] — down); срезы экспертов —
//     source_slice = "{expert}" (канон R9/ADR-09 round 2).
//
// Dtype групп (канон ADR-09 block32, [R4] «смешанный план»): dtype_weights —
// аргумент profile ("F16" по умолчанию; "NVFP4"/"W4A16" — квантование
// выполняет отдельный проход: конвертер/лоадер NVFP4W4A16); линейные веса
// (q/k/v/o, gate/up/down, эксперты) — dtype профиля; нормы/роутер — dtype
// тензора источника (F32/F16, BF16 → F16); embed/head — F16 (канон
// registry-классов); kv — F16.
//
// Ошибки — LoaderError: нет config.json → SchemaMismatch; шард не
// соответствует плану (нет тензора для группы) → TensorNotFound; формы/
// dtype-несовместимость → SchemaMismatch/UnsupportedQuant; неизвестная
// архитектура/профиль → SchemaMismatch. Валидация энергичная: каждая
// некешевая группа расширенного плана обязана разрешиться в тензора шардов
// (fail-fast, ADR-09 §3).
// ============================================================================
#include "stell/models/convert.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

#include "file_weights_source.hpp"
#include "description_json.hpp"
#include "tensor_file_formats.hpp"

namespace stell::models {

std::string hf_to_description(const std::string& hf_dir,
                              const std::string& profile) {
    // --- 0. Профиль dtype_weights (канон ADR-09 block32; "E4M3" — не профиль) --
    std::string dtype_weights;
    if (profile == "F16") dtype_weights = "F16";
    else if (profile == "NVFP4") dtype_weights = "NVFP4";
    else if (profile == "W4A16") dtype_weights = "W4A16";
    else
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "hf: профиль '" + profile +
                              "' не поддержан (F16|NVFP4|W4A16; квантование — "
                              "отдельный проход конвертера/лоадера, ADR-09 block32)");

    // --- 1. config.json (обязателен: нет → SchemaMismatch) --------------------
    const std::string cfg_path = hf_dir + "/config.json";
    std::FILE* fc = std::fopen(cfg_path.c_str(), "rb");
    if (!fc)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "hf: config.json не найден в '" + hf_dir +
                              "' (вход — каталог HF-чекпоинта)");
    std::string cfg_text;
    {
        char buf[65536];
        std::size_t got;
        while ((got = std::fread(buf, 1, sizeof buf, fc)) > 0) cfg_text.append(buf, got);
        std::fclose(fc);
    }
    JsonValue cfg;
    try {
        cfg = json_parse(cfg_text);
    } catch (const JsonError& e) {
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          std::string("hf: bad config.json: ") + e.what());
    }
    if (cfg.type != JsonValue::Type::Object)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "hf: config.json root must be an object");

    // Чтение целочисленного поля (обязательное/опциональное).
    auto cfg_u64 = [&](const char* key, bool required, std::uint64_t def,
                       bool* ok = nullptr) -> std::uint64_t {
        if (ok) *ok = true;
        const JsonValue* v = cfg.find(key);
        if (!v || v->type != JsonValue::Type::Int) {
            if (!required) return def;
            if (ok) { *ok = false; return 0; }
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              std::string("hf: config.json: отсутствует/не целое поле '") +
                                  key + "'");
        }
        if (v->int_v < 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              std::string("hf: config.json: поле '") + key +
                                  "' отрицательно");
        return static_cast<std::uint64_t>(v->int_v);
    };
    auto cfg_f64 = [&](const char* key, bool required, double def) -> double {
        const JsonValue* v = cfg.find(key);
        if (!v) {
            if (required)
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  std::string("hf: config.json: отсутствует поле '") +
                                      key + "'");
            return def;
        }
        if (v->type == JsonValue::Type::Int) return static_cast<double>(v->int_v);
        if (v->type == JsonValue::Type::Double) return v->double_v;
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          std::string("hf: config.json: поле '") + key +
                              "' не число");
    };

    const std::uint64_t d_model = cfg_u64("hidden_size", true, 0);
    const std::uint64_t n_layers = cfg_u64("num_hidden_layers", true, 0);
    const std::uint64_t n_heads = cfg_u64("num_attention_heads", true, 0);
    const std::uint64_t n_kv_heads = cfg_u64("num_key_value_heads", false, n_heads);
    if (d_model == 0 || n_layers == 0 || n_heads == 0 || n_kv_heads == 0)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "hf: config.json: hidden_size/num_hidden_layers/"
                          "num_attention_heads/num_key_value_heads обязаны быть > 0");
    std::uint64_t d_head = cfg_u64("head_dim", false, 0);
    if (d_head == 0) {
        if (d_model % n_heads != 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: config.json: head_dim отсутствует, а hidden_size "
                              "не делится на num_attention_heads");
        d_head = d_model / n_heads;
    }
    const double eps = cfg_f64("rms_norm_eps", false, 1e-5);

    // --- 2. Шарды: объединённая карта тензоров (payload не читается) ----------
    std::vector<std::FILE*> shard_files;
    TensorFileMap tfm;
    {
        std::error_code ec;
        const bool is_dir = std::filesystem::is_directory(hf_dir, ec);
        if (is_dir) {
            tfm = merge_safetensors_shards(hf_dir, &shard_files);
        } else {
            // Одиночный .safetensors (прогон с одним файлом; каталог — канон).
            std::FILE* f = std::fopen(hf_dir.c_str(), "rb");
            if (!f)
                throw LoaderError(LoaderError::Code::IoFailure,
                                  "hf: cannot open " + hf_dir);
            shard_files.push_back(f);
            if (detect_format(hf_dir) != WeightsFormat::Safetensors)
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "hf: вход должен быть каталогом HF-чекпоинта "
                                  "(config.json + safetensors-шарды) или .safetensors");
            tfm = parse_safetensors_header(f);
        }
    }
    struct ShardsCloser {
        std::vector<std::FILE*>* v;
        ~ShardsCloser() {
            for (std::FILE* f : *v) {
                if (f) std::fclose(f);
            }
        }
    } shards_closer{&shard_files};

    auto find_tensor = [&](const std::string& name) -> const TensorInfo* {
        auto it = tfm.by_name.find(name);
        return it == tfm.by_name.end() ? nullptr : &it->second;
    };
    auto require_tensor = [&](const std::string& name, const char* what) -> const TensorInfo& {
        const TensorInfo* t = find_tensor(name);
        if (!t)
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              std::string("hf: ") + what + ": тензор '" + name +
                                  "' отсутствует в шардах '" + hf_dir + "'");
        return *t;
    };
    // Проверка формы тензора против ожидаемой (шард не соответствует плану).
    auto check_shape = [&](const std::string& name, const std::vector<std::int64_t>& exp) {
        const TensorInfo& t = require_tensor(name, "hf");
        if (t.shape != exp)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: тензор '" + name + "': shape mismatch (" +
                                  fmt_shape(t.shape) + " != " + fmt_shape(exp) + ")");
        return t.dtype_str;
    };

    // --- 3. Верхние тензора + vocab_size --------------------------------------
    const TensorInfo& emb = require_tensor("model.embed_tokens.weight",
                                           "hf: верхний уровень");
    std::uint64_t vocab_size = cfg_u64("vocab_size", false, 0);
    if (vocab_size == 0) {
        if (emb.shape.empty())
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: model.embed_tokens.weight: пустая форма");
        vocab_size = static_cast<std::uint64_t>(emb.shape[0]);  // умолчание — из тензора
    }
    check_shape("model.embed_tokens.weight",
                {static_cast<std::int64_t>(vocab_size),
                 static_cast<std::int64_t>(d_model)});
    check_shape("model.norm.weight",
                {static_cast<std::int64_t>(d_model)});

    // dtype нормы/роутера — по dtype тензора источника (F32/F16; BF16 → F16).
    auto source_group_dtype = [&](const std::string& name) -> std::string {
        const TensorInfo& t = require_tensor(name, "hf");
        if (t.dtype == stell::tiers::DType::F32) return "F32";
        if (t.dtype == stell::tiers::DType::F16) return "F16";  // bf16-флаг → F16-конверсия
        throw LoaderError(LoaderError::Code::UnsupportedQuant,
                          "hf: тензор '" + name + "' (dtype '" + t.dtype_str +
                              "'): источник нормы/роутера должен быть F32/F16/BF16");
    };
    const std::string norm0_dtype = source_group_dtype(
        "model.layers.0.input_layernorm.weight");

    // Имя модели: _name_or_path (basename) либо basename каталога.
    std::string name = "hf-model";
    if (const JsonValue* n = cfg.find("_name_or_path"); n && n->type == JsonValue::Type::String)
        name = n->string_v;
    else {
        std::string base = hf_dir;
        while (!base.empty() && base.back() == '/') base.pop_back();
        const std::size_t pos = base.find_last_of('/');
        if (pos != std::string::npos) base = base.substr(pos + 1);
        if (!base.empty()) name = base;
    }

    // --- 4. Архитектура (диспетчер по architectures[0]) ------------------------
    std::string arch;
    if (const JsonValue* a = cfg.find("architectures"); a && a->type == JsonValue::Type::Array &&
                                                          !a->array_v.empty())
        arch = a->array_v[0].string_v;
    const bool dense = (arch == "LlamaForCausalLM" || arch == "Qwen2ForCausalLM");
    const bool moe_per_expert = (arch == "Qwen2MoeForCausalLM" ||
                                 arch == "Qwen1.5MoeForCausalLM" ||
                                 arch == "OlmoeForCausalLM");
    const bool moe_stacked = (arch == "MixtralForCausalLM");
    if (!dense && !moe_per_expert && !moe_stacked)
        throw LoaderError(LoaderError::Code::SchemaMismatch,
                          "hf: архитектура '" + arch +
                              "' не поддержана (subset: LlamaForCausalLM, "
                              "Qwen2ForCausalLM, Qwen2MoeForCausalLM, "
                              "Qwen1.5MoeForCausalLM, OlmoeForCausalLM, "
                              "MixtralForCausalLM)");

    const std::int64_t dm = static_cast<std::int64_t>(d_model);
    const std::int64_t dqq = static_cast<std::int64_t>(n_heads * d_head);
    const std::int64_t dkv = static_cast<std::int64_t>(n_kv_heads * d_head);
    const std::int64_t L = static_cast<std::int64_t>(n_layers);
    const std::string lp = "model.layers.{layer}";
    const std::string l0 = "model.layers.0";  // конкретные проверки слоя 0
    const std::string card_key = "layer";

    std::vector<std::string> groups;

    // --- общие группы ----------------------------------------------------------
    groups.push_back(make_group("embed", "attn",
                                {static_cast<std::int64_t>(vocab_size), dm}, "F16",
                                "model.embed_tokens.weight"));
    groups.push_back(make_group("norm_final", "attn", {dm},
                                source_group_dtype("model.norm.weight"),
                                "model.norm.weight"));
    if (find_tensor("lm_head.weight")) {
        check_shape("lm_head.weight",
                    {static_cast<std::int64_t>(vocab_size), dm});
        groups.push_back(make_group("head", "attn",
                                    {static_cast<std::int64_t>(vocab_size), dm},
                                    "F16", "lm_head.weight"));
    }
    // KV-группа: псевдо-паттерн layers.{layer}.kv_cache (в шардах KV нет —
    // лоадер роль=kv пропускает, ADR-09 §1).
    groups.push_back(make_group("l{layer}.kv", "kv", {dm, dkv}, "F16",
                                "layers.{layer}.kv_cache", {{card_key, L}},
                                "", /*kv_block=*/true));

    std::string moe_json;
    std::string ffn_op_json;

    if (dense) {
        // --- 4a. Dense (llama/qwen2-нейминг) -----------------------------------
        const std::uint64_t d_ff_u = cfg_u64("intermediate_size", true, 0);
        if (d_ff_u == 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: config.json: intermediate_size обязана быть > 0");
        const std::int64_t dff = static_cast<std::int64_t>(d_ff_u);
        // Проверка d_ff по факту в шардах (слой 0 обязателен).
        check_shape(l0 + ".mlp.gate_proj.weight", {dff, dm});
        check_shape(l0 + ".mlp.up_proj.weight", {dff, dm});
        check_shape(l0 + ".mlp.down_proj.weight", {dm, dff});

        struct LayerTensor {
            const char* pattern;   // с префиксом model.layers.{layer}.
            const char* group;
            bool is_norm;
            std::int64_t out_dim;
            std::int64_t in_dim;
        };
        const LayerTensor lts[] = {
            {".input_layernorm.weight",         "l{layer}.attn.norm", true, dm, 0},
            {".post_attention_layernorm.weight", "l{layer}.ffn.norm", true, dm, 0},
            {".self_attn.q_proj.weight", "l{layer}.attn.wq", false, dqq, dm},
            {".self_attn.k_proj.weight", "l{layer}.attn.wk", false, dkv, dm},
            {".self_attn.v_proj.weight", "l{layer}.attn.wv", false, dkv, dm},
            {".self_attn.o_proj.weight", "l{layer}.attn.wo", false, dm, dm},
            {".mlp.gate_proj.weight",    "l{layer}.ffn.gate", false, dff, dm},
            {".mlp.up_proj.weight",      "l{layer}.ffn.up", false, dff, dm},
            {".mlp.down_proj.weight",    "l{layer}.ffn.down", false, dm, dff},
        };
        for (const auto& lt : lts) {
            const std::vector<std::int64_t> shape =
                lt.is_norm ? std::vector<std::int64_t>{dm}
                           : std::vector<std::int64_t>{lt.out_dim, lt.in_dim};
            // Нормы: dtype тензора СВОЕГО источника (F32/F16, BF16 → F16);
            // матрицы: dtype профиля (источник обязан быть F16/F32/BF16 —
            // квантование выполняет отдельный проход конвертера/лоадера,
            // ADR-09 block32).
            std::string dtype;
            if (lt.is_norm) {
                dtype = source_group_dtype(l0 + lt.pattern);
            } else {
                const TensorInfo& t = require_tensor(l0 + lt.pattern, "hf");
                if (!t.dtype || (*t.dtype != stell::tiers::DType::F16 &&
                                 *t.dtype != stell::tiers::DType::F32))
                    throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                      "hf: тензор '" + l0 + lt.pattern +
                                          "' (dtype '" + t.dtype_str +
                                          "'): источник матрицы должен быть "
                                          "F16/F32/BF16");
                dtype = dtype_weights;
            }
            groups.push_back(make_group(lt.group, "attn", shape, dtype,
                                        lp + lt.pattern, {{card_key, L}}));
        }
        ffn_op_json =
            ",{\"op\":\"ffn\",\"io\":[\"hidden\"],\"params\":{\"activation\":\"silu_mul\"}}";
    } else if (moe_per_expert) {
        // --- 4b. MoE, поэертельные тензоры (Qwen2Moe/Qwen1.5Moe/Olmoe) --------
        const std::uint64_t n_experts = cfg_u64("num_experts", false, 0)
                                            ? cfg_u64("num_experts", false, 0)
                                            : cfg_u64("num_local_experts", false, 0);
        const std::uint64_t n_active = cfg_u64("num_experts_per_tok", true, 0);
        if (n_experts == 0 || n_active == 0 || n_active > n_experts)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: config.json: num_experts/num_local_experts и "
                              "num_experts_per_tok обязаны быть > 0 и "
                              "num_experts_per_tok <= num_experts");
        // Роутер: mlp.router.weight (Qwen2Moe) или mlp.gate.weight
        // (Qwen1.5Moe/Olmoe) — выбирается по факту в шардах.
        std::string router_suffix;
        if (find_tensor(l0 + ".mlp.router.weight"))
            router_suffix = ".router.weight";
        else if (find_tensor(l0 + ".mlp.gate.weight"))
            router_suffix = ".gate.weight";
        else
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              "hf: роутер MoE не найден (mlp.router.weight / "
                              "mlp.gate.weight) в '" + hf_dir + "'");
        // Эксперты: {gate,up,down}_proj.weight (Qwen/Olmoe) или
        // {gate,up,down}.weight — выбирается по факту.
        std::string proj_suffix[3];  // gate/up/down
        {
            const char* names[3] = {"gate", "up", "down"};
            for (int k = 0; k < 3; ++k) {
                if (find_tensor(l0 + ".mlp.experts.0." + names[k] + "_proj.weight"))
                    proj_suffix[k] = std::string("_proj.weight");
                else if (find_tensor(l0 + ".mlp.experts.0." + names[k] + ".weight"))
                    proj_suffix[k] = std::string(".weight");
                else
                    throw LoaderError(LoaderError::Code::TensorNotFound,
                                      "hf: тензор эксперта не найден (mlp.experts.0." +
                                          std::string(names[k]) + "_proj.weight / " +
                                          names[k] + ".weight) в '" + hf_dir + "'");
            }
        }
        const std::int64_t dff = [&] {
            const TensorInfo& t = require_tensor(l0 + ".mlp.experts.0.gate" +
                                                     proj_suffix[0], "hf");
            if (t.shape.size() != 2 || t.shape[0] <= 0)
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "hf: тензора экспертов должны быть матрицами "
                                  "[d_ff, d_model]");
            return t.shape[0];
        }();
        if (dff <= 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: expert_d_ff обязана быть > 0");
        check_shape(l0 + ".mlp" + router_suffix,
                    {static_cast<std::int64_t>(n_experts), dm});
        const std::int64_t E = static_cast<std::int64_t>(n_experts);
        check_shape(l0 + ".mlp.experts.0.gate" + proj_suffix[0], {dff, dm});
        check_shape(l0 + ".mlp.experts.0.up" + proj_suffix[1], {dff, dm});
        check_shape(l0 + ".mlp.experts.0.down" + proj_suffix[2], {dm, dff});

        groups.push_back(make_group("l{layer}.attn.norm", "attn", {dm},
                                    norm0_dtype,
                                    lp + ".input_layernorm.weight", {{card_key, L}}));
        groups.push_back(make_group("l{layer}.ffn.norm", "attn", {dm},
                                    source_group_dtype(l0 + ".post_attention_layernorm.weight"),
                                    lp + ".post_attention_layernorm.weight", {{card_key, L}}));
        struct MatTemplate {
            const char* suffix;
            const char* group;
            std::int64_t out_dim;
            std::int64_t in_dim;
        };
        const MatTemplate mats[] = {
            {".self_attn.q_proj.weight", "l{layer}.attn.wq", dqq, dm},
            {".self_attn.k_proj.weight", "l{layer}.attn.wk", dkv, dm},
            {".self_attn.v_proj.weight", "l{layer}.attn.wv", dkv, dm},
            {".self_attn.o_proj.weight", "l{layer}.attn.wo", dm, dm},
        };
        for (const auto& m : mats) {
            const TensorInfo& t = require_tensor(l0 + m.suffix, "hf");
            if (!t.dtype || (*t.dtype != stell::tiers::DType::F16 &&
                             *t.dtype != stell::tiers::DType::F32))
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "hf: тензор '" + l0 + m.suffix +
                                      "' (dtype '" + t.dtype_str +
                                      "'): источник матрицы должен быть F16/F32/BF16");
            groups.push_back(make_group(m.group, "attn", {m.out_dim, m.in_dim},
                                        dtype_weights, lp + m.suffix,
                                        {{card_key, L}}));
        }
        groups.push_back(make_group("l{layer}.mlp.router", "expert",
                                    {static_cast<std::int64_t>(n_experts), dm},
                                    source_group_dtype(l0 + ".mlp" + router_suffix),
                                    lp + ".mlp" + router_suffix,
                                    {{card_key, L}}));
        const std::vector<std::pair<std::string, std::int64_t>> card = {
            {card_key, L}, {"expert", E}};
        groups.push_back(make_group("l{layer}.e{expert}.gate", "expert",
                                    {dff, dm}, dtype_weights,
                                    lp + ".mlp.experts.{expert}.gate" + proj_suffix[0],
                                    card));
        groups.push_back(make_group("l{layer}.e{expert}.up", "expert",
                                    {dff, dm}, dtype_weights,
                                    lp + ".mlp.experts.{expert}.up" + proj_suffix[1],
                                    card));
        groups.push_back(make_group("l{layer}.e{expert}.down", "expert",
                                    {dm, dff}, dtype_weights,
                                    lp + ".mlp.experts.{expert}.down" + proj_suffix[2],
                                    card));
        moe_json = ",\"moe\":{\"n_experts\":" + fmt_int(static_cast<std::int64_t>(n_experts)) +
                   ",\"n_experts_active\":" + fmt_int(static_cast<std::int64_t>(n_active)) +
                   ",\"expert_d_ff\":" + fmt_int(dff) + ",\"router\":\"softmax-topk\"}";
        ffn_op_json = "";
    } else {
        // --- 4c. MoE, стековные тензоры (Mixtral) ------------------------------
        const std::uint64_t n_experts = cfg_u64("num_local_experts", false, 0)
                                            ? cfg_u64("num_local_experts", false, 0)
                                            : cfg_u64("num_experts", false, 0);
        const std::uint64_t n_active = cfg_u64("num_experts_per_tok", true, 0);
        if (n_experts == 0 || n_active == 0 || n_active > n_experts)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: config.json: num_local_experts и "
                              "num_experts_per_tok обязаны быть > 0 и "
                              "num_experts_per_tok <= num_local_experts");
        const std::string mlp = "block_sparse_moe.";
        check_shape(l0 + "." + mlp + "gate.weight",
                    {static_cast<std::int64_t>(n_experts), dm});
        const std::int64_t dff = [&] {
            const TensorInfo& t =
                require_tensor(l0 + "." + mlp + "w1.weight", "hf");
            if (t.shape.size() != 3 ||
                t.shape[0] != static_cast<std::int64_t>(n_experts) ||
                t.shape[2] != dm)
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "hf: " + mlp + "w1.weight: shape mismatch "
                                  "(expected [n_experts, d_ff, d_model])");
            return t.shape[1];
        }();
        if (dff <= 0)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: expert_d_ff обязана быть > 0");
        check_shape(l0 + "." + mlp + "w3.weight",
                    {static_cast<std::int64_t>(n_experts), dff, dm});
        check_shape(l0 + "." + mlp + "w2.weight",
                    {static_cast<std::int64_t>(n_experts), dm, dff});
        // Источники матриц: F16/F32/BF16.
        for (const char* w : {"gate.weight", "w1.weight", "w3.weight", "w2.weight"}) {
            const TensorInfo& t = require_tensor(l0 + "." + mlp + w, "hf");
            if (w[0] == 'w' &&
                (!t.dtype || (*t.dtype != stell::tiers::DType::F16 &&
                              *t.dtype != stell::tiers::DType::F32)))
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "hf: тензор '" + l0 + "." + mlp + w +
                                      "' (dtype '" + t.dtype_str +
                                      "'): источник матрицы должен быть F16/F32/BF16");
        }

        groups.push_back(make_group("l{layer}.attn.norm", "attn", {dm},
                                    norm0_dtype,
                                    lp + ".input_layernorm.weight", {{card_key, L}}));
        groups.push_back(make_group("l{layer}.ffn.norm", "attn", {dm},
                                    source_group_dtype(l0 + ".post_attention_layernorm.weight"),
                                    lp + ".post_attention_layernorm.weight", {{card_key, L}}));
        struct MatTemplate {
            const char* suffix;
            const char* group;
            std::int64_t out_dim;
            std::int64_t in_dim;
        };
        const MatTemplate mats[] = {
            {".self_attn.q_proj.weight", "l{layer}.attn.wq", dqq, dm},
            {".self_attn.k_proj.weight", "l{layer}.attn.wk", dkv, dm},
            {".self_attn.v_proj.weight", "l{layer}.attn.wv", dkv, dm},
            {".self_attn.o_proj.weight", "l{layer}.attn.wo", dm, dm},
        };
        for (const auto& m : mats) {
            const TensorInfo& t = require_tensor(l0 + m.suffix, "hf");
            if (!t.dtype || (*t.dtype != stell::tiers::DType::F16 &&
                             *t.dtype != stell::tiers::DType::F32))
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "hf: тензор '" + l0 + m.suffix +
                                      "' (dtype '" + t.dtype_str +
                                      "'): источник матрицы должен быть F16/F32/BF16");
            groups.push_back(make_group(m.group, "attn", {m.out_dim, m.in_dim},
                                        dtype_weights, lp + m.suffix,
                                        {{card_key, L}}));
        }
        groups.push_back(make_group("l{layer}.mlp.router", "expert",
                                    {static_cast<std::int64_t>(n_experts), dm},
                                    source_group_dtype(l0 + "." + mlp + "gate.weight"),
                                    lp + "." + mlp + "gate.weight",
                                    {{card_key, L}}));
        // Стековные экспертные тензоры: cardinality {layer, expert},
        // source_slice = "{expert}" (срез по старшей размерности, R9).
        const std::vector<std::pair<std::string, std::int64_t>> card = {
            {card_key, L}, {"expert", static_cast<std::int64_t>(n_experts)}};
        groups.push_back(make_group("l{layer}.e{expert}.gate", "expert",
                                    {dff, dm}, dtype_weights,
                                    lp + "." + mlp + "w1.weight", card, "{expert}"));
        groups.push_back(make_group("l{layer}.e{expert}.up", "expert",
                                    {dff, dm}, dtype_weights,
                                    lp + "." + mlp + "w3.weight", card, "{expert}"));
        groups.push_back(make_group("l{layer}.e{expert}.down", "expert",
                                    {dm, dff}, dtype_weights,
                                    lp + "." + mlp + "w2.weight", card, "{expert}"));
        moe_json = ",\"moe\":{\"n_experts\":" + fmt_int(static_cast<std::int64_t>(n_experts)) +
                   ",\"n_experts_active\":" + fmt_int(static_cast<std::int64_t>(n_active)) +
                   ",\"expert_d_ff\":" + fmt_int(dff) + ",\"router\":\"softmax-topk\"}";
        ffn_op_json = "";
    }

    // --- 5. JSON описания (схема 1.0) ------------------------------------------
    std::string json;
    json += "{\"schema\":\"1.0\",\"name\":\"";
    json_escape(json, name);
    json += "\",\"class\":\"" + std::string(moe_stacked || moe_per_expert ? "moe" : "dense") +
            "\",\"dtype_weights\":\"" + dtype_weights + "\"";
    json += ",\"vocab_size\":" + fmt_int(static_cast<std::int64_t>(vocab_size));
    json += ",\"d_model\":" + fmt_int(dm);
    json += ",\"n_layers\":" + fmt_int(L);
    json += ",\"n_heads\":" + fmt_int(static_cast<std::int64_t>(n_heads));
    json += ",\"d_head\":" + fmt_int(static_cast<std::int64_t>(d_head));
    json += ",\"kv\":{\"dtype\":\"F16\",\"n_kv_heads\":" +
            fmt_int(static_cast<std::int64_t>(n_kv_heads)) +
            ",\"d_kv\":" + fmt_int(static_cast<std::int64_t>(d_head)) +
            ",\"quant\":null}";
    json += moe_json;
    json += ",\"ops\":[{\"op\":\"rmsnorm\",\"io\":[\"hidden\"],\"params\":{\"eps\":";
    json += fmt_f64(eps);
    json += "}}";
    json += ",{\"op\":\"attention\",\"io\":[\"hidden\",\"kv\"],\"params\":{\"kernel\":\"sm70_fp32_qk_pv\"";
    const double rope_theta = cfg_f64("rope_theta", false, 0.0);
    if (rope_theta > 0.0)  // rope-параметр ([R4]: из config.json)
        json += ",\"rope_theta\":" + fmt_f64(rope_theta);
    json += "}}";
    if (!ffn_op_json.empty()) json += ffn_op_json;
    if (moe_stacked || moe_per_expert)
        json += ",{\"op\":\"moe_gather_scatter\",\"io\":[\"hidden\",\"experts\"],\"params\":{\"kernel\":\"sm70_moe_gs\"}}";
    json += "],\"tensor_groups\":[";

    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (i) json += ",";
        json += groups[i];
    }
    json += "]}";

    // --- 6. Валидация: разбор + энергичная резолюция ВСЕХ некешевых групп ------
    // (fail-fast, ADR-09 §3): каждая группа обязана разрешиться в тензора
    // шардов с согласованной формой; шард не соответствует плану →
    // TensorNotFound/SchemaMismatch.
    const ModelDescription d = parse_model_description(json, "hf-generated");
    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;
        const std::string& pattern = d.group_source_patterns[i];
        auto it = tfm.by_name.find(pattern);
        if (it == tfm.by_name.end())
            throw LoaderError(LoaderError::Code::TensorNotFound,
                              "hf: шард не соответствует плану — нет тензора '" +
                                  pattern + "' для группы '" + g.name + "'");
        const std::int64_t slice =
            i < d.group_source_slices.size() ? d.group_source_slices[i] : -1;
        const TensorInfo& eff = FileWeightsSource::slice_tensor_info(it->second, g, slice);
        if (eff.shape != g.shape)
            throw LoaderError(LoaderError::Code::SchemaMismatch,
                              "hf: группа '" + g.name + "': shape mismatch против "
                              "тензора '" + pattern + "'");
        // Dtype-совместимость источника и группы (plain-пути; квантованный
        // профиль — F16/F32/BF16-источник, квантование — отдельный проход).
        const bool plain_target = g.dtype == stell::tiers::DType::F16 ||
                                  g.dtype == stell::tiers::DType::F32;
        if (plain_target) {
            const TensorInfo& ti = it->second;
            const bool ok =
                (g.dtype == stell::tiers::DType::F16 &&
                 ti.dtype && (*ti.dtype == stell::tiers::DType::F16 ||
                              *ti.dtype == stell::tiers::DType::F32)) ||
                (g.dtype == stell::tiers::DType::F32 &&
                 ti.dtype == stell::tiers::DType::F32 && !ti.bf16);
            if (!ok)
                throw LoaderError(LoaderError::Code::SchemaMismatch,
                                  "hf: группа '" + g.name + "' (dtype " +
                                      std::string(g.dtype == stell::tiers::DType::F16
                                                      ? "F16" : "F32") +
                                      "): источник '" + ti.dtype_str +
                                      "' не совместим");
        } else {
            const TensorInfo& ti = it->second;
            if (!ti.dtype || (*ti.dtype != stell::tiers::DType::F16 &&
                              *ti.dtype != stell::tiers::DType::F32))
                throw LoaderError(LoaderError::Code::UnsupportedQuant,
                                  "hf: квантованная группа '" + g.name +
                                      "': источник должен быть F16/F32/BF16, получено '" +
                                      ti.dtype_str + "'");
        }
    }

    return json;
}

}  // namespace stell::models

// ============================================================================
// Тесты hf_to_description (round 4, [R4]; ADR-09; по прецеденту
// test_gguf_description): синтетический каталог HF-чекпоинта (config.json +
// safetensors-шарды — fixture от generate-скрипта, здесь — синтетический
// писатель) → описание (нативные имена тензоров HF) → запись/чтение →
// логиты интерпретатора по stell-tier пути == референсу плана (прямой путь
// того же плана) — ПОБИТОВО. Error-кейсы: нет config.json →
// LoaderError(SchemaMismatch); шард не соответствует плану → TensorNotFound;
// форма — SchemaMismatch; неизвестная архитектура/профиль — SchemaMismatch.
// Смешанный план (канон ADR-09 block32): квант-профиль — линейные веса
// NVFP4, нормы/роутер — F32/F16 источника, embed/head — F16.
// BF16 (R3-подготовка): конверсия BF16→F16 — побайтово == референсу.
// ============================================================================
#include "stell/models/convert.hpp"
#include "stell/models/registry.hpp"
#include "stell/models/weights.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "quant.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"

// engine-интерпретатор (референс того же плана; мост ADR-09 §4).
#include "stell/engine/plain_spec.hpp"
#include "stell/engine/ops_runner.hpp"

using namespace stell::models;
using namespace stell_test;
namespace se = stell::engine;

namespace {

std::string out_dir(const char* tag) {
    std::filesystem::remove_all(tag);
    std::filesystem::create_directories(tag);
    return tag;
}

SynthTensor tensor(const std::string& name, const std::string& dtype,
                   std::vector<std::int64_t> shape, std::vector<std::uint8_t> data) {
    SynthTensor t;
    t.name = name;
    t.dtype = dtype;
    t.shape = std::move(shape);
    t.data = std::move(data);
    return t;
}

// Детерминированные F16/F32/BF16-payload'ы (конвенция test_convert_streaming).
std::vector<std::uint8_t> f16_det(std::size_t n, int seed) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint16_t h =
            static_cast<std::uint16_t>(((k * 3 + seed + 5) % 31) + 1);
        out[2 * k] = static_cast<std::uint8_t>(h & 0xFF);
        out[2 * k + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFF);
    }
    return out;
}

std::vector<std::uint8_t> f32_det(std::size_t n, float seed) {
    std::vector<std::uint8_t> out(n * 4);
    for (std::size_t k = 0; k < n; ++k) {
        const float v = seed + 0.0625f * static_cast<float>((k * 7) % 13);
        std::memcpy(out.data() + 4 * k, &v, 4);
    }
    return out;
}

// BF16: биты = старшая половина F32 (конверсия точная по построению).
// f32 — n*4 байт (n элементов); out — n*2 байт BF16.
std::vector<std::uint8_t> bf16_from_f32(const std::vector<std::uint8_t>& f32) {
    std::vector<std::uint8_t> out(f32.size() / 2);
    for (std::size_t k = 0; k < out.size() / 2; ++k)
        out[2 * k] = f32[4 * k + 2], out[2 * k + 1] = f32[4 * k + 3];
    return out;
}

// --- синтетический HF-чекпоинт: dense llama/qwen2-нейминг ---------------------
// d_model=8, n_heads=2, d_head=4, n_kv_heads=1, vocab=16, d_ff=16, layers=2.
std::vector<SynthTensor> dense_hf_tensors(bool bf16_norms = false) {
    std::vector<SynthTensor> out;
    const std::string norm_dt = bf16_norms ? "BF16" : "F32";
    auto push_f16 = [&](const std::string& name, std::vector<std::int64_t> sh, int seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F16", std::move(sh), f16_det(n, seed)));
    };
    auto push_f32 = [&](const std::string& name, std::vector<std::int64_t> sh, float seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F32", std::move(sh), f32_det(n, seed)));
    };
    auto push_bf16 = [&](const std::string& name, std::vector<std::int64_t> sh, float seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "BF16", std::move(sh),
                             bf16_from_f32(f32_det(n, seed))));
    };
    auto push_norm = [&](const std::string& name, std::vector<std::int64_t> sh, float seed) {
        if (bf16_norms) push_bf16(name, std::move(sh), seed);
        else push_f32(name, std::move(sh), seed);
    };
    push_f16("model.embed_tokens.weight", {16, 8}, 3);
    push_f32("model.norm.weight", {8}, 0.25f);
    push_f16("lm_head.weight", {16, 8}, 1000);
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        push_norm(p + "input_layernorm.weight", {8}, 0.5f + layer);
        push_norm(p + "post_attention_layernorm.weight", {8}, 0.75f + layer);
        push_f16(p + "self_attn.q_proj.weight", {8, 8}, layer * 13 + 7);
        push_f16(p + "self_attn.k_proj.weight", {4, 8}, layer * 21 + 11);
        push_f16(p + "self_attn.v_proj.weight", {4, 8}, layer * 29 + 13);
        push_f16(p + "self_attn.o_proj.weight", {8, 8}, layer * 31 + 17);
        push_f16(p + "mlp.gate_proj.weight", {16, 8}, layer * 17 + 19);
        push_f16(p + "mlp.up_proj.weight", {16, 8}, layer * 7 + 23);
        push_f16(p + "mlp.down_proj.weight", {8, 16}, layer * 11 + 27);
    }
    return out;
}

// --- синтетический HF-чекпоинт: MoE (Qwen2Moe-нейминг, поэертельные эксперты) -
// d_model=8, n_heads=2, d_head=4, n_kv_heads=1, vocab=16, layers=2,
// n_experts=4, n_active=2, expert_d_ff=8.
std::vector<SynthTensor> moe_hf_tensors() {
    std::vector<SynthTensor> out;
    auto push_f16 = [&](const std::string& name, std::vector<std::int64_t> sh, int seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F16", std::move(sh), f16_det(n, seed)));
    };
    auto push_f32 = [&](const std::string& name, std::vector<std::int64_t> sh, float seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F32", std::move(sh), f32_det(n, seed)));
    };
    push_f16("model.embed_tokens.weight", {16, 8}, 5);
    push_f32("model.norm.weight", {8}, 0.5f);
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        push_f32(p + "input_layernorm.weight", {8}, 0.25f + layer);
        push_f32(p + "post_attention_layernorm.weight", {8}, 0.75f + layer);
        push_f16(p + "self_attn.q_proj.weight", {8, 8}, layer * 13 + 3);
        push_f16(p + "self_attn.k_proj.weight", {4, 8}, layer * 21 + 5);
        push_f16(p + "self_attn.v_proj.weight", {4, 8}, layer * 29 + 9);
        push_f16(p + "self_attn.o_proj.weight", {8, 8}, layer * 31 + 17);
        push_f32(p + "mlp.router.weight", {4, 8}, 0.125f + layer);
        for (int e = 0; e < 4; ++e) {
            const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            push_f16(ep + "gate_proj.weight", {8, 8}, layer * 53 + e * 7 + 19);
            push_f16(ep + "up_proj.weight", {8, 8}, layer * 41 + e * 11 + 23);
            push_f16(ep + "down_proj.weight", {8, 8}, layer * 67 + e * 13 + 27);
        }
    }
    return out;
}

// --- синтетический HF-чекпоинт: MoE (Mixtral-нейминг, стековные тензоры) ------
std::vector<SynthTensor> mixtral_hf_tensors() {
    std::vector<SynthTensor> out;
    auto push_f16 = [&](const std::string& name, std::vector<std::int64_t> sh, int seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F16", std::move(sh), f16_det(n, seed)));
    };
    auto push_f32 = [&](const std::string& name, std::vector<std::int64_t> sh, float seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F32", std::move(sh), f32_det(n, seed)));
    };
    push_f16("model.embed_tokens.weight", {16, 8}, 7);
    push_f32("model.norm.weight", {8}, 0.75f);
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        push_f32(p + "input_layernorm.weight", {8}, 0.5f + layer);
        push_f32(p + "post_attention_layernorm.weight", {8}, 0.125f + layer);
        push_f16(p + "self_attn.q_proj.weight", {8, 8}, layer * 13 + 3);
        push_f16(p + "self_attn.k_proj.weight", {4, 8}, layer * 21 + 5);
        push_f16(p + "self_attn.v_proj.weight", {4, 8}, layer * 29 + 9);
        push_f16(p + "self_attn.o_proj.weight", {8, 8}, layer * 31 + 17);
        push_f32(p + "block_sparse_moe.gate.weight", {4, 8}, 0.25f + layer);
        // Стековные: [n_experts, d_ff, d_model] / [n_experts, d_model, d_ff].
        push_f16(p + "block_sparse_moe.w1.weight", {4, 8, 8}, layer * 53 + 19);
        push_f16(p + "block_sparse_moe.w3.weight", {4, 8, 8}, layer * 41 + 23);
        push_f16(p + "block_sparse_moe.w2.weight", {4, 8, 8}, layer * 67 + 27);
    }
    return out;
}

void write_config(const std::string& dir, const char* arch, bool moe) {
    std::filesystem::create_directories(dir);
    std::string cfg = std::string(R"({
  "architectures": [")") + arch + R"("],
  "hidden_size": 8,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "vocab_size": 16,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "intermediate_size": 16)";
    if (moe)
        cfg += R"(,
  "num_experts": 4,
  "num_local_experts": 4,
  "num_experts_per_tok": 2)";
    cfg += "\n}";
    std::FILE* f = std::fopen((dir + "/config.json").c_str(), "wb");
    if (!f) throw std::runtime_error("config.json");
    std::fwrite(cfg.data(), 1, cfg.size(), f);
    std::fclose(f);
}

// Шарды: тензоры делятся на два файла (канонический порядок HF — сортировка
// имён внутри merge_safetensors_shards; разбиение здесь — по списку).
void write_shards(const std::string& dir, const std::vector<SynthTensor>& ts) {
    std::filesystem::create_directories(dir);
    const std::size_t half = ts.size() / 2 + (ts.size() % 2);
    stell_test::write_safetensors(dir + "/model-00001-of-00002.safetensors",
                                  std::vector<SynthTensor>(ts.begin(), ts.begin() + half));
    stell_test::write_safetensors(dir + "/model-00002-of-00002.safetensors",
                                  std::vector<SynthTensor>(ts.begin() + half, ts.end()));
}

// Одиночный .safetensors с теми же тензора́ми (референс прямого пути).
std::string write_single(const std::string& path, const std::vector<SynthTensor>& ts) {
    stell_test::write_safetensors(path, ts);
    return path;
}

// Roundtrip: логиты интерпретатора по stell-tier layout (конвертация каталога
// шардов) == прямому пути того же плана — ПОБИТОВО.
void roundtrip_logits(const ModelDescription& d, const std::string& hf_dir,
                      const std::string& single_file, const char* out_tag,
                      se::PlainWeightsFormat direct_fmt) {
    StellTierConverter conv;
    conv.convert_streaming(d, hf_dir, out_dir(out_tag));
    const std::string layout =
        std::string(out_tag) + "/" + d.name + ".stelltier";

    const se::PlainSpec spec = se::plain_spec_from_description(d);
    const se::ExecutionPlan plan = se::build_execution_plan(spec);
    // Прямой путь: лоадер models по ИСХОДНОМУ описанию d (source_slice
    // стековных тензоров сохраняется; engine-мост load_plain_weights
    // пересобирает описание из PlainSpec без срезов — поэтому лоадер открывается
    // здесь; см. REPORT-ROUND4.md, «открытые вопросы»).
    const WeightsFormat direct_fmt_models = direct_fmt == se::PlainWeightsNVFP4W4A16
                                                ? WeightsFormat::NVFP4W4A16
                                                : WeightsFormat::Safetensors;
    se::LoadedWeights direct;
    {
        auto loader = make_loader(direct_fmt_models);
        auto src = loader->open(single_file, d);
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            if (d.groups[i].role == stell::tiers::Role::Kv) continue;
            std::vector<std::uint8_t> buf(d.groups[i].slot_bytes, 0);
            src->read({d.groups[i].id, d.groups[i].role}, buf.data(), buf.size());
            direct.slots[d.groups[i].id.value] = std::move(buf);
        }
        src->close();
    }
    const se::LoadedWeights via_layout =
        se::load_plain_weights(layout, spec, se::PlainWeightsStellTier);

    CHECK(direct.slots.size() == via_layout.slots.size());
    for (const auto& [gid, buf] : via_layout.slots) {
        const auto it = direct.slots.find(gid);
        CHECK_MSG(it != direct.slots.end(), "группы нет в прямом пути");
        CHECK_MSG(it->second.size() == buf.size() &&
                      std::memcmp(it->second.data(), buf.data(), buf.size()) == 0,
                  "слот группы (id " + std::to_string(gid) +
                      ") layout != прямому лоадеру (побитно)");
    }

    se::OpsInterpreter ref_interp(spec, plan, direct, /*tiers=*/nullptr);
    se::OpsInterpreter tier_interp(spec, plan, via_layout, /*tiers=*/nullptr);
    const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5};
    const std::vector<float> ref = ref_interp.run_prompt_logits(prompt);
    const std::vector<float> got = tier_interp.run_prompt_logits(prompt);
    CHECK(ref.size() == static_cast<std::size_t>(d.vocab_size));
    CHECK(got.size() == ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i)
        CHECK_MSG(ref[i] == got[i],
                  "логиты интерпретатора по layout != референсу плана (побитно)");
}

int find_group(const ModelDescription& d, const std::string& name) {
    for (std::size_t i = 0; i < d.groups.size(); ++i)
        if (d.groups[i].name == name) return static_cast<int>(i);
    return -1;
}

}  // namespace

void body() {
    // =========================================================================
    // 1. Dense (LlamaForCausalLM): happy path + roundtrip логитов (F16-профиль).
    // =========================================================================
    const std::string dense_dir = out_dir("/tmp/stell_hf_dense");
    {
        const auto ts = dense_hf_tensors();
        write_config(dense_dir, "LlamaForCausalLM", false);
        write_shards(dense_dir, ts);
        const std::string json = hf_to_description(dense_dir);

        const JsonValue root = json_parse(json);
        CHECK(root.get_string("schema") == "1.0");
        CHECK(root.get_string("name") == "stell_hf_dense");   // basename каталога
        CHECK(root.get_string("class") == "dense");
        CHECK(root.get_string("dtype_weights") == "F16");
        CHECK(root.get_uint("vocab_size") == 16);
        CHECK(root.get_uint("d_model") == 8);
        CHECK(root.get_uint("n_layers") == 2);
        CHECK(root.get_uint("n_heads") == 2);
        CHECK(root.get_uint("d_head") == 4);
        const JsonValue* kvj = root.find("kv");
        CHECK(kvj->get_uint("n_kv_heads") == 1);
        CHECK(kvj->get_uint("d_kv") == 4);
        // rope-параметр config.json → параметр ops attention ([R4]).
        const std::vector<JsonValue>& ops = root.get_array("ops").array_v;
        CHECK(ops.size() == 3);
        CHECK(ops[1].get_string("op") == "attention");
        CHECK(ops[1].find("params")->find("kernel")->string_v == "sm70_fp32_qk_pv");
        const JsonValue* rt = ops[1].find("params")->find("rope_theta");
        CHECK(rt != nullptr);
        const double rope_theta_val = rt->type == JsonValue::Type::Double
                                          ? rt->double_v
                                          : static_cast<double>(rt->int_v);
        CHECK(std::fabs(rope_theta_val - 10000.0) < 1e-9);
        CHECK(std::fabs(ops[0].find("params")->find("eps")->double_v - 1e-5) < 1e-12);
        CHECK(ops[2].get_string("op") == "ffn");

        const ModelDescription d = parse_model_description(json, "hf-generated");
        // Группы: embed + norm_final + head + 2×(2 нормы + 7 матриц) + 2 kv = 23.
        CHECK(d.groups.size() == 23);
        const int embed = find_group(d, "embed");
        CHECK((embed >= 0) &&
              d.group_source_patterns[embed] == "model.embed_tokens.weight");
        const int wq0 = find_group(d, "l000.attn.wq");
        CHECK((wq0 >= 0) && d.group_source_patterns[wq0] ==
                                "model.layers.0.self_attn.q_proj.weight");
        CHECK((d.groups[wq0].shape == std::vector<std::int64_t>{8, 8}));
        const int ffn_norm1 = find_group(d, "l001.ffn.norm");
        CHECK((ffn_norm1 >= 0) &&
              d.group_source_patterns[ffn_norm1] ==
                  "model.layers.1.post_attention_layernorm.weight");
        CHECK(d.groups[ffn_norm1].dtype == stell::tiers::DType::F32);  // dtype источника
        const int head = find_group(d, "head");
        CHECK((head >= 0) && d.group_source_patterns[head] == "lm_head.weight");
        const int kv0 = find_group(d, "l000.kv");
        CHECK((kv0 >= 0) &&
              d.group_source_patterns[kv0] == "layers.0.kv_cache");

        // Roundtrip: логиты по layout == прямому пути плана (побитно).
        const std::string single =
            write_single("/tmp/stell_hf_dense_single.safetensors", ts);
        roundtrip_logits(d, dense_dir, single, "/tmp/stell_hf_dense_out",
                         se::PlainWeightsSafetensors);
        std::printf("dense (LlamaForCausalLM, F16): roundtrip логитов OK (побитно)\n");
    }

    // =========================================================================
    // 2. Смешанный план (канон ADR-09 block32): NVFP4-профиль — линейные веса
    //    NVFP4, нормы/роутер — F32 источника, embed/head — F16.
    // =========================================================================
    {
        const auto ts = dense_hf_tensors();
        const std::string json = hf_to_description(dense_dir, "NVFP4");
        const ModelDescription d = parse_model_description(json, "hf-nvfp4");
        CHECK(d.dtype_weights == "NVFP4");
        const int wq0 = find_group(d, "l000.attn.wq");
        CHECK(d.groups[wq0].dtype == stell::tiers::DType::NVFP4);
        const int nf = find_group(d, "norm_final");
        CHECK(d.groups[nf].dtype == stell::tiers::DType::F32);
        const int embed = find_group(d, "embed");
        CHECK(d.groups[embed].dtype == stell::tiers::DType::F16);

        // Roundtrip (квантование — отдельный проход конвертера/лоадера):
        // логиты по layout == прямому NVFP4W4A16-пути того же плана (побитно).
        const std::string single =
            write_single("/tmp/stell_hf_dense_single.safetensors", ts);
        roundtrip_logits(d, dense_dir, single, "/tmp/stell_hf_nvfp4_out",
                         se::PlainWeightsNVFP4W4A16);
        std::printf("dense (NVFP4 смешанный план): roundtrip логитов OK (побитно)\n");
    }

    // =========================================================================
    // 3. MoE, поэертельные тензоры (Qwen2MoeForCausalLM), NVFP4-профиль.
    // =========================================================================
    const std::string moe_dir = out_dir("/tmp/stell_hf_moe");
    {
        const auto ts = moe_hf_tensors();
        write_config(moe_dir, "Qwen2MoeForCausalLM", true);
        write_shards(moe_dir, ts);
        const std::string json = hf_to_description(moe_dir, "NVFP4");
        const JsonValue root = json_parse(json);
        CHECK(root.get_string("class") == "moe");
        CHECK(root.get_string("name") == "stell_hf_moe");
        const JsonValue& moe = root.get_object("moe");
        CHECK(moe.get_uint("n_experts") == 4);
        CHECK(moe.get_uint("n_experts_active") == 2);
        CHECK(moe.get_uint("expert_d_ff") == 8);

        const ModelDescription d = parse_model_description(json, "hf-generated-moe");
        const int router0 = find_group(d, "l000.mlp.router");
        CHECK((router0 >= 0) &&
              d.group_source_patterns[router0] == "model.layers.0.mlp.router.weight");
        CHECK(d.groups[router0].dtype == stell::tiers::DType::F32);  // роутер — F32 источника
        const int gate00 = find_group(d, "l000.e000.gate");
        CHECK((gate00 >= 0) &&
              d.group_source_patterns[gate00] ==
                  "model.layers.0.mlp.experts.0.gate_proj.weight");
        CHECK(d.groups[gate00].dtype == stell::tiers::DType::NVFP4);
        const int down11 = find_group(d, "l001.e003.down");
        CHECK((down11 >= 0) &&
              d.group_source_patterns[down11] ==
                  "model.layers.1.mlp.experts.3.down_proj.weight");
        CHECK((d.groups[down11].shape == std::vector<std::int64_t>{8, 8}));

        const std::string single =
            write_single("/tmp/stell_hf_moe_single.safetensors", ts);
        roundtrip_logits(d, moe_dir, single, "/tmp/stell_hf_moe_out",
                         se::PlainWeightsNVFP4W4A16);
        std::printf("moe (Qwen2MoeForCausalLM, NVFP4): roundtrip логитов OK (побитно)\n");
    }

    // =========================================================================
    // 4. MoE, стековные тензоры (MixtralForCausalLM), source_slice "{expert}".
    // =========================================================================
    const std::string mix_dir = out_dir("/tmp/stell_hf_mixtral");
    {
        const auto ts = mixtral_hf_tensors();
        write_config(mix_dir, "MixtralForCausalLM", true);
        write_shards(mix_dir, ts);
        const std::string json = hf_to_description(mix_dir, "F16");
        const JsonValue root = json_parse(json);
        CHECK(root.get_string("class") == "moe");
        CHECK(root.get_object("moe").get_uint("n_experts") == 4);

        const ModelDescription d = parse_model_description(json, "hf-mixtral");
        const int gate0 = find_group(d, "l000.e000.gate");
        CHECK((gate0 >= 0) &&
              d.group_source_patterns[gate0] ==
                  "model.layers.0.block_sparse_moe.w1.weight");
        CHECK(d.group_source_slices[gate0] == 0);  // срез стековного тензора
        const int up1 = find_group(d, "l001.e003.up");
        CHECK((up1 >= 0) &&
              d.group_source_patterns[up1] == "model.layers.1.block_sparse_moe.w3.weight");
        CHECK(d.group_source_slices[up1] == 3);
        const int dn = find_group(d, "l001.e001.down");
        CHECK((dn >= 0) &&
              d.group_source_patterns[dn] == "model.layers.1.block_sparse_moe.w2.weight");
        CHECK((d.groups[dn].shape == std::vector<std::int64_t>{8, 8}));

        const std::string single =
            write_single("/tmp/stell_hf_mixtral_single.safetensors", ts);
        roundtrip_logits(d, mix_dir, single, "/tmp/stell_hf_mixtral_out",
                         se::PlainWeightsSafetensors);
        std::printf("moe (MixtralForCausalLM, стековные w1/w3/w2): roundtrip логитов OK (побитно)\n");
    }

    // =========================================================================
    // 5. BF16-источник (R3-подготовка): конверсия BF16→F16 — побайтово ==
    //    референсу bf16_bits_to_f32 → f32_to_f16_bits.
    // =========================================================================
    {
        const std::string bf_dir = out_dir("/tmp/stell_hf_bf16");
        const auto ts = dense_hf_tensors(/*bf16_norms=*/true);
        write_config(bf_dir, "LlamaForCausalLM", false);
        write_shards(bf_dir, ts);
        const std::string json = hf_to_description(bf_dir, "F16");
        const ModelDescription d = parse_model_description(json, "hf-bf16");
        const int an0 = find_group(d, "l000.attn.norm");
        CHECK(d.groups[an0].dtype == stell::tiers::DType::F16);  // BF16 → F16-конверсия

        auto loader = make_loader(WeightsFormat::Safetensors);
        const std::string single =
            write_single("/tmp/stell_hf_bf16_single.safetensors", ts);
        auto src = loader->open(single, d);
        std::vector<std::uint8_t> buf(d.groups[an0].slot_bytes, 0);
        src->read({d.groups[an0].id, d.groups[an0].role}, buf.data(), buf.size());
        src->close();
        // Референс: f32-значения тензора → f16 биты (младший байт первый).
        std::size_t n = 1;
        for (auto x : d.groups[an0].shape) n *= static_cast<std::size_t>(x);
        const std::vector<std::uint8_t> ref_f32 = f32_det(n, 0.5f);
        for (std::size_t k = 0; k < n; ++k) {
            float v;
            std::memcpy(&v, ref_f32.data() + 4 * k, 4);
            const std::uint16_t h = f32_to_f16_bits(v);
            CHECK(buf[2 * k] == static_cast<std::uint8_t>(h & 0xFF));
            CHECK(buf[2 * k + 1] == static_cast<std::uint8_t>((h >> 8) & 0xFF));
        }
        // Хвост слота — нули.
        for (std::size_t k = n * 2; k < buf.size(); ++k) CHECK(buf[k] == 0);
        std::printf("BF16 → F16 конверсия: OK (побайтово == референсу)\n");
    }

    // =========================================================================
    // 6. Error-кейсы.
    // =========================================================================
    auto expect_error = [&](const std::string& dir, const std::string& profile,
                            LoaderError::Code code, const char* what) {
        try {
            hf_to_description(dir, profile);
            CHECK_MSG(false && "expected LoaderError", what);
        } catch (const LoaderError& e) {
            CHECK_MSG(e.code() == code,
                      std::string(what) + ": код ошибки != ожидаемому");
        }
        (void)what;
    };
    // (а) нет config.json → SchemaMismatch ([R4]).
    {
        const std::string no_cfg = out_dir("/tmp/stell_hf_nocfg");
        std::filesystem::create_directories(no_cfg);
        // Шарды есть, config.json нет.
        const auto ts = dense_hf_tensors();
        stell_test::write_safetensors(no_cfg + "/model-00001-of-00002.safetensors", ts);
        expect_error(no_cfg, "F16", LoaderError::Code::SchemaMismatch, "нет config.json");
    }
    // (б) шард не соответствует плану (тензора нет у группы) → TensorNotFound.
    {
        const std::string miss = out_dir("/tmp/stell_hf_missing");
        write_config(miss, "LlamaForCausalLM", false);
        auto ts = dense_hf_tensors();
        // Убираем тензор одного слоя (k_proj слоя 0) — план его требует.
        ts.erase(std::remove_if(ts.begin(), ts.end(),
                                [](const SynthTensor& t) {
                                    return t.name == "model.layers.0.self_attn.k_proj.weight";
                                }),
                 ts.end());
        write_shards(miss, ts);
        expect_error(miss, "F16", LoaderError::Code::TensorNotFound,
                     "шард не соответствует плану");
    }
    // (в) форма тензора не согласована с config → SchemaMismatch.
    {
        const std::string bad = out_dir("/tmp/stell_hf_shape");
        write_config(bad, "LlamaForCausalLM", false);  // hidden_size 8
        auto ts = dense_hf_tensors();
        // q_proj [16, 8] вместо [8, 8] (hidden_size в конфиге — 8).
        for (auto& t : ts)
            if (t.name == "model.layers.0.self_attn.q_proj.weight") t.shape = {16, 8};
        write_shards(bad, ts);
        expect_error(bad, "F16", LoaderError::Code::SchemaMismatch, "shape mismatch");
    }
    // (г) неизвестная архитектура → SchemaMismatch.
    {
        const std::string arch = out_dir("/tmp/stell_hf_arch");
        write_config(arch, "GPT2ForCausalLM", false);
        write_shards(arch, dense_hf_tensors());
        expect_error(arch, "F16", LoaderError::Code::SchemaMismatch, "unknown arch");
    }
    // (д) неизвестный профиль → SchemaMismatch.
    {
        expect_error(dense_dir, "E4M3", LoaderError::Code::SchemaMismatch, "bad profile");
    }
    // (е) каталог без .safetensors-шардов → IoFailure (честный отказ).
    {
        const std::string empty = out_dir("/tmp/stell_hf_noshards");
        write_config(empty, "LlamaForCausalLM", false);
        expect_error(empty, "F16", LoaderError::Code::IoFailure, "нет шардов");
    }
}

TEST_MAIN("test_models_hf_description")

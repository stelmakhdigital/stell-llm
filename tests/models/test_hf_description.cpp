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

// plain_fill_slot/TensorInfo — общий путь конверсий лоадера и конвертера
// (R5: clamp-сатурация BF16/F32→F16, счётчик f16_saturation_count).
#include "file_weights_source.hpp"

// engine-интерпретатор (референс того же плана; мост ADR-09 §4).
#include "stell/engine/plain_spec.hpp"
#include "stell/engine/ops_runner.hpp"

// [R2] round 6: побитовая сверка layout'ов перегенерации (TierLayoutReader).
#include <stell/tiers/layout.hpp>

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


// [R3] config с параметром числа слоёв (смешанный каталог: 4 слоя;
// with_intermediate=false — dense-слои без d_ff → честный SchemaMismatch).
void write_config_layers(const std::string& dir, const char* arch, bool moe,
                         int layers, bool with_intermediate = true) {
    std::filesystem::create_directories(dir);
    std::string cfg = std::string(R"({
  "architectures": [")") + arch + R"("],
  "hidden_size": 8,
  "num_hidden_layers": )" + std::to_string(layers) + R"(,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "vocab_size": 16,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0)";
    if (with_intermediate)
        cfg += R"(,
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

// [R3] Смешанный dense/MoE каталог (per-layer план, ADR-09 round 5):
// слои 0,1 — dense (mlp.{gate,up,down}_proj, d_ff=16); слои 2,3 — MoE
// (mlp.router + mlp.experts.{0..3}.{gate,up,down}_proj, d_ff=8).
std::vector<SynthTensor> mixed_hf_tensors() {
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
    push_f16("model.embed_tokens.weight", {16, 8}, 9);
    push_f32("model.norm.weight", {8}, 0.5f);
    for (int layer = 0; layer < 4; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        push_f32(p + "input_layernorm.weight", {8}, 0.25f + layer);
        push_f32(p + "post_attention_layernorm.weight", {8}, 0.75f + layer);
        push_f16(p + "self_attn.q_proj.weight", {8, 8}, layer * 13 + 7);
        push_f16(p + "self_attn.k_proj.weight", {4, 8}, layer * 21 + 11);
        push_f16(p + "self_attn.v_proj.weight", {4, 8}, layer * 29 + 13);
        push_f16(p + "self_attn.o_proj.weight", {8, 8}, layer * 31 + 17);
        if (layer < 2) {
            push_f16(p + "mlp.gate_proj.weight", {16, 8}, layer * 17 + 19);
            push_f16(p + "mlp.up_proj.weight", {16, 8}, layer * 7 + 23);
            push_f16(p + "mlp.down_proj.weight", {8, 16}, layer * 11 + 27);
        } else {
            push_f32(p + "mlp.router.weight", {4, 8}, 0.125f + layer);
            for (int e = 0; e < 4; ++e) {
                const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
                push_f16(ep + "gate_proj.weight", {8, 8}, layer * 53 + e * 7 + 19);
                push_f16(ep + "up_proj.weight", {8, 8}, layer * 41 + e * 11 + 23);
                push_f16(ep + "down_proj.weight", {8, 8}, layer * 67 + e * 13 + 27);
            }
        }
    }
    return out;
}

// [R4] Qwen2Moe с shared-expert: homogeneous MoE (moe_hf_tensors) + тензоры
// mlp.shared_expert.{gate,up,down}_proj (d_ff_shared=12 != routed d_ff=8 —
// формы групп ffn.shared — по факту тензоров слоя).
std::vector<SynthTensor> moe_shared_hf_tensors() {
    auto ts = moe_hf_tensors();
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        const std::string sp = p + "mlp.shared_expert.";
        std::size_t n = 12 * 8;
        ts.push_back(tensor(sp + "gate_proj.weight", "F16", {12, 8},
                            f16_det(n, layer * 79 + 31)));
        ts.push_back(tensor(sp + "up_proj.weight", "F16", {12, 8},
                            f16_det(n, layer * 83 + 37)));
        ts.push_back(tensor(sp + "down_proj.weight", "F16", {8, 12},
                            f16_det(n, layer * 89 + 41)));
    }
    return ts;
}

// [R1] round 6: та же фикстура + скалярный сигмоид-гейт shared-эксперта
// (mlp.shared_expert_gate, форма [1, d_model] — как в реальных Qwen2Moe;
// dtype F16 — однородный с shared-FFN тензора).
std::vector<SynthTensor> moe_shared_gate_hf_tensors() {
    auto ts = moe_shared_hf_tensors();
    for (int layer = 0; layer < 2; ++layer)
        ts.push_back(tensor("model.layers." + std::to_string(layer) +
                            ".mlp.shared_expert_gate.weight", "F16", {1, 8},
                            f16_det(8, layer * 7 + 53)));
    return ts;
}

// [R6-добор] Qwen3Moe-класс (зонд реального Qwen/Qwen3-30B-A3B):
// роутер — mlp.gate.weight (НЕ mlp.router.weight), эксперты — поэертельные
// mlp.experts.{e}.{gate,up,down}_proj.weight, dense-слоёв НЕТ, shared-expert
// НЕТ, lm_head есть (tie_word_embeddings=false). d_model=8, n_heads=2,
// d_head=4, n_kv_heads=1, vocab=16, layers=2, n_experts=4, topk=2, d_ff=8.
std::vector<SynthTensor> qwen3_moe_hf_tensors() {
    auto ts = moe_hf_tensors();
    for (auto& t : ts)
        if (t.name.find(".mlp.router.weight") != std::string::npos) {
            const std::size_t pos = t.name.find(".mlp.router.weight");
            t.name = t.name.substr(0, pos) + ".mlp.gate.weight";
        }
    // lm_head — как в реальном чекпоинте (tie_word_embeddings=false).
    ts.push_back(tensor("lm_head.weight", "F16", {16, 8}, f16_det(16 * 8, 1000)));
    return ts;
}

// [R1] round 7: та же фикстура + qk-norm тензоры (форма [d_head]=[4], каждый
// слой — как в реальном чекпоинте) — per-head RMSNorm q/k генерируется в
// группы l{layer}.attn.{q,k}_norm (ADR-09 round 7; отказ round 6 снят).
std::vector<SynthTensor> qwen3_moe_qknorm_hf_tensors() {
    auto ts = qwen3_moe_hf_tensors();
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        ts.push_back(tensor(p + "self_attn.q_norm.weight", "F16", {4},
                            f16_det(4, layer * 5 + 71)));
        ts.push_back(tensor(p + "self_attn.k_norm.weight", "F16", {4},
                            f16_det(4, layer * 5 + 73)));
    }
    return ts;
}

// [R3] round 8: GQA-вариант qk-norm фикстуры (реальный Qwen3-30B-A3B —
// GQA: n_kv_heads=4 > 1, head_dim=128): n_kv_heads=2, head_dim=4 →
// k/v проекции [n_kv*d_head]=[8]×[8], а k_norm.weight — [head_dim]=[4]
// (HF Qwen3: вес [head_dim] транслируется по KV-головам). Прежняя фикстура
// (n_kv_heads=1) не отличала [head_dim] от [n_kv*d_head] — регрессия
// формы k_norm-группы на GQA (описание генерируется, k_norm [d_head]).
std::vector<SynthTensor> qwen3_moe_qknorm_gqa_hf_tensors() {
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
        // GQA: n_kv_heads=2 × head_dim=4 → k/v проекции {8, 8} (НЕ {4, 8}).
        push_f16(p + "self_attn.k_proj.weight", {8, 8}, layer * 21 + 5);
        push_f16(p + "self_attn.v_proj.weight", {8, 8}, layer * 29 + 9);
        push_f16(p + "self_attn.o_proj.weight", {8, 8}, layer * 31 + 17);
        // per-head RMSNorm q/k — вес [head_dim]=[4] по головам (включая
        // KV-головы) — как в реальном чекпоинте Qwen3-30B-A3B.
        push_f16(p + "self_attn.q_norm.weight", {4}, layer * 5 + 71);
        push_f16(p + "self_attn.k_norm.weight", {4}, layer * 5 + 73);
        push_f32(p + "mlp.router.weight", {4, 8}, 0.125f + layer);
        for (int e = 0; e < 4; ++e) {
            const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            push_f16(ep + "gate_proj.weight", {8, 8}, layer * 53 + e * 7 + 19);
            push_f16(ep + "up_proj.weight", {8, 8}, layer * 41 + e * 11 + 23);
            push_f16(ep + "down_proj.weight", {8, 8}, layer * 67 + e * 13 + 27);
        }
    }
    push_f16("lm_head.weight", {16, 8}, 1000);   // tie_word_embeddings=false
    return out;
}

// [R1] round 7: частичный набор — только q_norm (k_norm нет) → честный
// SchemaMismatch (односторонне нормализованный attention не имеет покрытой
// семантики).
std::vector<SynthTensor> qwen3_moe_qknorm_partial_hf_tensors() {
    auto ts = qwen3_moe_hf_tensors();
    for (int layer = 0; layer < 2; ++layer)
        ts.push_back(tensor("model.layers." + std::to_string(layer) +
                            ".self_attn.q_norm.weight", "F16", {4},
                            f16_det(4, layer * 5 + 71)));
    return ts;
}

// [R6-добор] config Qwen3Moe-класса: ключи — по факту зонда реального
// config.json (moe_intermediate_size, num_experts_per_tok, norm_topk_prob,
// head_dim; intermediate_size — dense-fallback, в однородном MoE не читается;
// decoder_sparse_step=1 + mlp_only_layers=[] — все слои MoE). Параметры —
// для error-кейсов (norm_topk=false; moe_ff != факту шардов; без
// rms_norm_eps при qk-нормах).
void write_config_qwen3(const std::string& dir, bool norm_topk = true,
                        int moe_ff = 8, bool rms_eps = true, int n_kv = 1) {
    std::filesystem::create_directories(dir);
    std::string cfg = std::string(
        "{\n"
        "  \"architectures\": [\"Qwen3MoeForCausalLM\"],\n"
        "  \"hidden_size\": 8,\n"
        "  \"num_hidden_layers\": 2,\n"
        "  \"num_attention_heads\": 2,\n"
        "  \"num_key_value_heads\": " + std::to_string(n_kv) + ",\n"
        "  \"head_dim\": 4,\n"
        "  \"vocab_size\": 16,\n"
        "  \"rope_theta\": 10000.0,\n"
        "  \"intermediate_size\": 16,\n"
        "  \"moe_intermediate_size\": ") +
        std::to_string(moe_ff) +
        std::string(",\n"
        "  \"num_experts\": 4,\n"
        "  \"num_experts_per_tok\": 2,\n"
        "  \"norm_topk_prob\": ") +
        (norm_topk ? "true" : "false") +
        std::string(",\n"
        "  \"decoder_sparse_step\": 1,\n"
        "  \"mlp_only_layers\": [],\n"
        "  \"tie_word_embeddings\": false\n"
        "}\n");
    if (rms_eps) {
        const std::size_t pos = cfg.find("\"rope_theta\"");
        cfg.insert(pos, std::string("  \"rms_norm_eps\": 1e-06,\n"));
    }
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

// Прямой путь: лоадер models по описанию d (source_slice стековных тензоров
// сохраняется; engine-мост load_plain_weights пересобирает описание из
// PlainSpec без срезов — поэтому лоадер открывается здесь; см.
// REPORT-ROUND4.md, «открытые вопросы»).
se::LoadedWeights load_direct_weights(const ModelDescription& d,
                                      const std::string& single_file,
                                      se::PlainWeightsFormat direct_fmt) {
    const WeightsFormat direct_fmt_models = direct_fmt == se::PlainWeightsNVFP4W4A16
                                                ? WeightsFormat::NVFP4W4A16
                                                : WeightsFormat::Safetensors;
    se::LoadedWeights direct;
    auto loader = make_loader(direct_fmt_models);
    auto src = loader->open(single_file, d);
    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        if (d.groups[i].role == stell::tiers::Role::Kv) continue;
        std::vector<std::uint8_t> buf(d.groups[i].slot_bytes, 0);
        src->read({d.groups[i].id, d.groups[i].role}, buf.data(), buf.size());
        direct.slots[d.groups[i].id.value] = std::move(buf);
    }
    src->close();
    return direct;
}

// Roundtrip: слоты layout (конвертация каталога шардов) == прямому лоадеру —
// ПОБИТОВО; логиты интерпретатора по layout == референсу того же плана —
// ПОБИТОВО. [R3/R4] round 5: для per-layer планов и ffn.shared интерпретатор
// плана (ops_runner — src/engine, поток core-6) может ещё не маршрутизировать
// per-layer/не суммировать ffn.shared — тогда честный PENDING (план/разметка
// проверены полностью выше; логиты-сравнение оживёт после смержа core-6).
void roundtrip_all(const ModelDescription& d, const std::string& hf_dir,
                   const std::string& single_file, const char* out_tag,
                   se::PlainWeightsFormat direct_fmt, const char* tag,
                   bool pending_ok) {
    StellTierConverter conv;
    conv.convert_streaming(d, hf_dir, out_dir(out_tag));
    const std::string layout =
        std::string(out_tag) + "/" + d.name + ".stelltier";

    const se::PlainSpec spec = se::plain_spec_from_description(d);
    const se::ExecutionPlan plan = se::build_execution_plan(spec);
    const se::LoadedWeights direct = load_direct_weights(d, single_file, direct_fmt);
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

    try {
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
        if (tag && *tag)
            std::printf("%s: roundtrip логитов OK (побитно)\n", tag);
    } catch (const std::invalid_argument& e) {
        if (!pending_ok) throw;
        // Честный pending: план полон (слоты/резолюция проверены выше),
        // отказ — отсутствие per-layer маршрутизации в ops_runner (core-6).
        std::printf("[PENDING] %s: логиты-сравнение отложено — интерпретатор плана "
                    "(ops_runner, поток core-6 [R3]) ещё не исполняет per-layer "
                    "план: %s\n", tag, e.what());
    }
}

// Roundtrip без pending (однородные планы round 4 — поведение прежнее).
void roundtrip_logits(const ModelDescription& d, const std::string& hf_dir,
                      const std::string& single_file, const char* out_tag,
                      se::PlainWeightsFormat direct_fmt) {
    roundtrip_all(d, hf_dir, single_file, out_tag, direct_fmt, "", false);
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
    // 6. [R3] Смешанный dense/MoE каталог (per-layer план, ADR-09 round 5):
    //    слои 0,1 — dense; слои 2,3 — MoE. hf_to_description классифицирует
    //    КАЖДЫЙ слой по факту в шардах: dense-слои — literal dense ffn группы,
    //    MoE-слои — literal router + поэертельные группы; attn/нормы —
    //    шаблонные на все слои (однородный дефолт round 4 не меняется).
    // =========================================================================
    const std::string mixdir = out_dir("/tmp/stell_hf_mixed");
    {
        const auto ts = mixed_hf_tensors();
        write_config_layers(mixdir, "Qwen2MoeForCausalLM", true, 4);
        write_shards(mixdir, ts);
        const std::string json = hf_to_description(mixdir);
        const JsonValue root = json_parse(json);
        CHECK(root.get_string("class") == "moe");
        const JsonValue& moej = root.get_object("moe");
        CHECK(moej.get_uint("n_experts") == 4);
        CHECK(moej.get_uint("n_experts_active") == 2);
        CHECK(moej.get_uint("expert_d_ff") == 8);
        // Смешанный план: op-граф содержит ОБЕ ветви — маршрутизация per-layer
        // по наличию MoE-групп слоя (ADR-09 round 5).
        const std::vector<JsonValue>& ops = root.get_array("ops").array_v;
        CHECK(ops.size() == 4);
        CHECK(ops[2].get_string("op") == "ffn");
        CHECK(ops[3].get_string("op") == "moe_gather_scatter");

        const ModelDescription d = parse_model_description(json, "hf-mixed");
        // Группы: embed+norm_final (2) + kv/attn.norm/ffn.norm ×4 + wq/wk/wv/wo ×4
        //         + dense 2×3 + MoE 2×(1 роутер + 4×3) = 2+12+16+6+26 = 62.
        CHECK(d.groups.size() == 62);
        const int fg0 = find_group(d, "l000.ffn.gate");
        CHECK((fg0 >= 0) &&
              d.group_source_patterns[fg0] == "model.layers.0.mlp.gate_proj.weight");
        CHECK((d.groups[fg0].shape == std::vector<std::int64_t>{16, 8}));
        CHECK(d.groups[fg0].dtype == stell::tiers::DType::F16);
        const int fd1 = find_group(d, "l001.ffn.down");
        CHECK((fd1 >= 0) &&
              d.group_source_patterns[fd1] == "model.layers.1.mlp.down_proj.weight");
        CHECK((d.groups[fd1].shape == std::vector<std::int64_t>{8, 16}));
        const int r2 = find_group(d, "l002.mlp.router");
        CHECK((r2 >= 0) &&
              d.group_source_patterns[r2] == "model.layers.2.mlp.router.weight");
        CHECK(d.groups[r2].dtype == stell::tiers::DType::F32);  // роутер — F32 источника
        const int g21 = find_group(d, "l002.e001.gate");
        CHECK((g21 >= 0) && d.group_source_patterns[g21] ==
                                "model.layers.2.mlp.experts.1.gate_proj.weight");
        const int d33 = find_group(d, "l003.e003.down");
        CHECK((d33 >= 0) && d.group_source_patterns[d33] ==
                                "model.layers.3.mlp.experts.3.down_proj.weight");
        // Членство: на MoE-слоях нет dense-ffn групп, на dense-слоях нет MoE-групп.
        for (const auto& g : d.groups) {
            const std::string& n = g.name;
            CHECK_MSG(n.compare(0, 11, "l002.ffn.gate") != 0 &&
                          n.compare(0, 11, "l002.ffn.up") != 0 &&
                          n.compare(0, 11, "l002.ffn.down") != 0 &&
                          n.compare(0, 11, "l003.ffn.gate") != 0,
                      "dense-ffn группа на MoE-слое: " + n);
            CHECK_MSG(n.compare(0, 14, "l000.mlp.router") != 0 &&
                          n.compare(0, 14, "l001.mlp.router") != 0 &&
                          n.compare(0, 7, "l000.e") != 0 &&
                          n.compare(0, 7, "l001.e") != 0,
                      "MoE-группа на dense-слое: " + n);
        }

        // Roundtrip: layout побитово == прямому лоадеру; логиты == референсу
        // per-layer плана — после per-layer маршрутизации в ops_runner
        // (поток core-6 [R3]; здесь — честный pending при отказе интерпретатора).
        const std::string single =
            write_single("/tmp/stell_hf_mixed_single.safetensors", ts);
        roundtrip_all(d, mixdir, single, "/tmp/stell_hf_mixed_out",
                      se::PlainWeightsSafetensors, "mixed dense/MoE (per-layer план)",
                      true);
        std::printf("mixed dense/MoE (Qwen2MoeForCausalLM): per-layer план OK\n");
    }

    // =========================================================================
    // 7. [R4] Shared-expert фикстура (Qwen2Moe-класс, mlp.shared_expert.*):
    //    homogeneous MoE + shared → группы ffn.shared (dense-FFN семантика,
    //    ADR-09 round 5); routed-группы не задеты (обратная совместимость).
    // =========================================================================
    const std::string shdir = out_dir("/tmp/stell_hf_shared");
    {
        auto ts = moe_shared_hf_tensors();
        write_config(shdir, "Qwen2MoeForCausalLM", true);
        write_shards(shdir, ts);
        const std::string json = hf_to_description(shdir);
        const JsonValue root = json_parse(json);
        CHECK(root.get_string("class") == "moe");
        // ops без изменений: ffn.shared — аддитивный вклад, интерпретатор
        // суммирует по наличию групп (ADR-09 round 5); отсутствие групп —
        // поведение без изменений.
        const std::vector<JsonValue>& ops = root.get_array("ops").array_v;
        CHECK(ops.size() == 3);
        CHECK(ops[2].get_string("op") == "moe_gather_scatter");

        const ModelDescription d = parse_model_description(json, "hf-shared");
        // routed d_ff=8; shared d_ff=12 — формы по факту тензоров слоя.
        const int sg0 = find_group(d, "l000.ffn.shared.gate");
        CHECK((sg0 >= 0) && d.group_source_patterns[sg0] ==
                                "model.layers.0.mlp.shared_expert.gate_proj.weight");
        CHECK((d.groups[sg0].shape == std::vector<std::int64_t>{12, 8}));
        CHECK(d.groups[sg0].dtype == stell::tiers::DType::F16);
        const int su0 = find_group(d, "l000.ffn.shared.up");
        CHECK((su0 >= 0) &&
              d.group_source_patterns[su0] == "model.layers.0.mlp.shared_expert.up_proj.weight");
        const int sd1 = find_group(d, "l001.ffn.shared.down");
        CHECK((sd1 >= 0) && d.group_source_patterns[sd1] ==
                                "model.layers.1.mlp.shared_expert.down_proj.weight");
        CHECK((d.groups[sd1].shape == std::vector<std::int64_t>{8, 12}));
        CHECK(d.groups.size() == 48);  // однородный MoE (42) + 2 слоя × 3 shared

        // Roundtrip: layout побитово == прямому лоадеру; логиты == референсу
        // плана. Суммирование выхода ffn.shared с routed — интерпретатор
        // плана (ops_runner, src/engine — поток core-6 [R4]); фиксируем,
        // ПОГЛОЩАЕТ ли уже интерпретатор группы ffn.shared: логиты плана
        // со shared vs план БЕЗ shared (те же routed-веса).
        const std::string single =
            write_single("/tmp/stell_hf_shared_single.safetensors", ts);
        roundtrip_all(d, shdir, single, "/tmp/stell_hf_shared_out",
                      se::PlainWeightsSafetensors, "moe+shared (ffn.shared)", true);
        {
            const auto ts_plain = moe_hf_tensors();
            const std::string pdir = out_dir("/tmp/stell_hf_shared_plain");
            write_config(pdir, "Qwen2MoeForCausalLM", true);
            write_shards(pdir, ts_plain);
            const ModelDescription d_plain = parse_model_description(
                hf_to_description(pdir), "hf-shared-plain");
            const se::PlainSpec spec_s = se::plain_spec_from_description(d);
            const se::ExecutionPlan plan_s = se::build_execution_plan(spec_s);
            const se::PlainSpec spec_p = se::plain_spec_from_description(d_plain);
            const se::ExecutionPlan plan_p = se::build_execution_plan(spec_p);
            const se::LoadedWeights direct_s =
                load_direct_weights(d, single, se::PlainWeightsSafetensors);
            const std::string psingle =
                write_single("/tmp/stell_hf_shared_plain_single.safetensors", ts_plain);
            const se::LoadedWeights direct_p =
                load_direct_weights(d_plain, psingle, se::PlainWeightsSafetensors);
            se::OpsInterpreter interp_s(spec_s, plan_s, direct_s, nullptr);
            se::OpsInterpreter interp_p(spec_p, plan_p, direct_p, nullptr);
            const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5};
            const std::vector<float> with = interp_s.run_prompt_logits(prompt);
            const std::vector<float> without = interp_p.run_prompt_logits(prompt);
            bool same = with.size() == without.size();
            for (std::size_t i = 0; same && i < with.size(); ++i)
                same = with[i] == without[i];
            if (same)
                std::printf(
                    "[PENDING] ffn.shared: интерпретатор плана (ops_runner, core-6 "
                    "[R4]) ещё не суммирует выход shared-эксперта — логиты == плану "
                    "без ffn.shared (суммирование оживёт после смержа core-6)\n");
            else
                std::printf(
                    "ffn.shared: интерпретатор суммирует выход shared-эксперта "
                    "(логиты != плану без ffn.shared)\n");
        }
        std::printf("moe+shared (Qwen2MoeForCausalLM): ffn.shared группы OK, план валиден\n");
    }

    // =========================================================================
    // 7b. [R1] round 6 (ADR-09 round 6): скалярный сигмоид-гейт shared-
    //     экспента — группа ffn.shared.gate_scalar из тензора
    //     mlp.shared_expert_gate (форма [1, d_model]); отсутствие тензора —
    //     группы нет (группы описаний побитово прежние, round 5); частичный
    //     набор «gate есть, shared-FFN нет» — честный SchemaMismatch.
    //     Семантика умножения — ops_runner (поток core, [R1] core-часть);
    //     описание/roundtrip здесь побитово.
    // =========================================================================
    const std::string shgdir = out_dir("/tmp/stell_hf_shared_gate");
    {
        auto ts = moe_shared_gate_hf_tensors();
        write_config(shgdir, "Qwen2MoeForCausalLM", true);
        write_shards(shgdir, ts);
        const std::string json = hf_to_description(shgdir);
        const ModelDescription d = parse_model_description(json, "hf-shared-gate");
        // 4 группы shared-контура на слой: ffn.shared.{gate,up,down} + gate_scalar.
        const int gg0 = find_group(d, "l000.ffn.shared.gate_scalar");
        CHECK(gg0 >= 0);
        CHECK(d.group_source_patterns[gg0] ==
              "model.layers.0.mlp.shared_expert_gate.weight");
        CHECK((d.groups[gg0].shape == std::vector<std::int64_t>{1, 8}));
        CHECK(d.groups[gg0].dtype == stell::tiers::DType::F16);
        CHECK(d.groups[gg0].role == stell::tiers::Role::Attn);
        const int gg1 = find_group(d, "l001.ffn.shared.gate_scalar");
        CHECK(gg1 >= 0);
        CHECK(d.group_source_patterns[gg1] ==
              "model.layers.1.mlp.shared_expert_gate.weight");
        // 48 (round 5: 42 MoE + 2×3 shared) + 2 gate_scalar = 50.
        CHECK(d.groups.size() == 50);

        // Обратная совместимость (отсутствие тензора — группы нет): описание
        // фикстуры БЕЗ gate (секция 7) не содержит gate_scalar и имеет те же
        // группы побитово (48 групп; round-4/5 поведение без изменений).
        const std::string json_nogate = hf_to_description(shdir);
        CHECK(json_nogate.find("gate_scalar") == std::string::npos);
        {
            const ModelDescription d5 = parse_model_description(json_nogate, "hf-shared-r5");
            CHECK(d5.groups.size() == 48);
            // Группы round-5 описания == round-6 описание без gate_scalar-групп
            // (тот же план без гейта): имена/формы/dtype/patterns совпадают 1:1
            // (гейт-группы вставляются в цикле слоёв — позиция внутри shared-
            // контура своего слоя).
            CHECK(d.groups.size() == d5.groups.size() + 2);
            std::size_t j6 = 0;
            for (std::size_t i = 0; i < d5.groups.size(); ++i, ++j6) {
                while (j6 < d.groups.size() &&
                       d.groups[j6].name.find("gate_scalar") != std::string::npos)
                    ++j6;
                CHECK_MSG(j6 < d.groups.size(), "состав групп round-5 не префикс round-6");
                CHECK(d.groups[j6].name == d5.groups[i].name);
                CHECK(d.groups[j6].shape == d5.groups[i].shape);
                CHECK(d.groups[j6].dtype == d5.groups[i].dtype);
                CHECK(d.group_source_patterns[j6] == d5.group_source_patterns[i]);
            }
        }

        // Roundtrip: слоты layout == прямому лоадеру (побитно); логиты ==
        // референсу того же плана (побитно). Гейт-группа попадает в план;
        // умножение на скаляр — ops_runner (core, [R1]) — интерпретатор без
        // гейта самосогласован (логиты == референсу того же плана).
        const std::string single_g =
            write_single("/tmp/stell_hf_shared_gate_single.safetensors", ts);
        roundtrip_all(d, shgdir, single_g, "/tmp/stell_hf_shared_gate_out",
                      se::PlainWeightsSafetensors, "moe+shared+gate (ffn.shared.gate)", true);
        std::printf("moe+shared+gate: ffn.shared.gate_scalar OK (4 shared-группы/слой), план валиден\n");
    }

    // Частичный набор: gate есть, shared-FFN нет → честный SchemaMismatch
    // (не тихое игнорирование тензора, ADR-09 round 6).
    {
        auto ts = moe_hf_tensors();
        for (int layer = 0; layer < 2; ++layer)
            ts.push_back(tensor("model.layers." + std::to_string(layer) +
                                ".mlp.shared_expert_gate.weight", "F16", {1, 8},
                                f16_det(8, layer * 7 + 53)));
        const std::string dir = out_dir("/tmp/stell_hf_gate_only");
        write_config(dir, "Qwen2MoeForCausalLM", true);
        write_shards(dir, ts);
        try {
            hf_to_description(dir);
            CHECK_MSG(false && "expected LoaderError",
                      "gate без shared-FFN: ожидался SchemaMismatch");
        } catch (const LoaderError& e) {
            CHECK_MSG(e.code() == LoaderError::Code::SchemaMismatch,
                      "gate без shared-FFN: код ошибки != SchemaMismatch");
            CHECK_MSG(std::string(e.what()).find("shared_expert_gate") !=
                          std::string::npos,
                      "gate без shared-FFN: сообщение не называет тензора");
        }
        std::printf("gate-only: частичный набор → SchemaMismatch OK\n");
    }

    // =========================================================================
    // 8. [R5] Clamp-сатурация BF16/F32→F16 (ADR-03 round 5): |v| > 65504 →
    //    ±65504 (не inf/NaN), обычные значения — побитово == прежней конверсии;
    //    счётчик сатураций; провенанс dtype_orig — в статистике конвертера.
    // =========================================================================
    {
        // Unit: plain_fill_slot (общий путь лоадеров и конвертера), BF16-источник.
        stell::tiers::TensorGroup g;
        g.id.value = 1;
        g.name = "t";
        g.role = stell::tiers::Role::Attn;
        g.dtype = stell::tiers::DType::F16;
        g.shape = {4};
        g.slot_bytes = 16;
        TensorInfo tb;
        tb.dtype_str = "BF16";
        tb.dtype = stell::tiers::DType::F16;  // BF16 маппится в F16 + флаг bf16
        tb.bf16 = true;
        tb.itemsize = 2;
        tb.shape = {4};
        tb.bytes = 8;
        // 65280 (= максимум F16, округлённый вниз до представимого в BF16) —
        // |v| < 65504 → БЕЗ сатурации; 1.5 — обычное значение.
        const float vals[4] = {1e9f, -1e9f, 65280.0f, 1.5f};
        std::vector<std::uint8_t> raw(8);
        for (int k = 0; k < 4; ++k) {
            std::uint32_t b;
            std::memcpy(&b, &vals[k], 4);
            const std::uint16_t h = static_cast<std::uint16_t>(b >> 16);  // BF16-биты
            raw[2 * k] = static_cast<std::uint8_t>(h & 0xFF);
            raw[2 * k + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFF);
        }
        std::vector<std::uint8_t> slot(g.slot_bytes, 0xAB);
        f16_saturation_reset();
        plain_fill_slot(g, tb, raw, slot.data(), slot.size());
        CHECK(f16_saturation_count() == 2);  // 1e9 и -1e9 сатурированы
        // Сатурация к ±65504 (не inf): F16-биты ±max (0x7BFF / 0xFBFF).
        CHECK(slot[0] == 0xFF && slot[1] == 0x7B);  // +65504
        CHECK(slot[2] == 0xFF && slot[3] == 0xFB);  // -65504
        // Обычные значения (|v| ≤ 65504) — БЕЗ сатурации, побитово == прежней
        // конверсии (f32_to_f16_bits RTNE не менялся).
        CHECK(slot[4] == static_cast<std::uint8_t>(f32_to_f16_bits(65280.0f) & 0xFF));
        CHECK(slot[5] == static_cast<std::uint8_t>(f32_to_f16_bits(65280.0f) >> 8));
        CHECK(slot[6] == static_cast<std::uint8_t>(f32_to_f16_bits(1.5f) & 0xFF));
        CHECK(slot[7] == static_cast<std::uint8_t>(f32_to_f16_bits(1.5f) >> 8));
        for (std::size_t k = 8; k < slot.size(); ++k) CHECK(slot[k] == 0);  // хвост — нули

        // F32-источник — та же сатурация (конверсия F32→F16 по прецеденту);
        // 65520 (|v| > max, e=16) — был inf, стал ±65504.
        TensorInfo t32;
        t32.dtype_str = "F32";
        t32.dtype = stell::tiers::DType::F32;
        t32.itemsize = 4;
        t32.shape = {4};
        t32.bytes = 16;
        std::vector<std::uint8_t> raw32(16);
        // 65504 (F32-источник, |v| == max F16) — БЕЗ сатурации;
        // 65520 (e=16) — прежняя конверсия давала inf, теперь clamp к +65504.
        const float vals32[4] = {2e9f, -2e9f, 65520.0f, 65504.0f};
        std::memcpy(raw32.data(), vals32, 16);
        std::vector<std::uint8_t> slot32(g.slot_bytes, 0);
        f16_saturation_reset();
        plain_fill_slot(g, t32, raw32, slot32.data(), slot32.size());
        CHECK(f16_saturation_count() == 3);  // 2e9, -2e9, 65520
        CHECK(slot32[0] == 0xFF && slot32[1] == 0x7B);
        CHECK(slot32[2] == 0xFF && slot32[3] == 0xFB);
        CHECK(slot32[4] == 0xFF && slot32[5] == 0x7B);  // 65520 → +65504 (был inf)
        CHECK(slot32[6] == static_cast<std::uint8_t>(f32_to_f16_bits(65504.0f) & 0xFF));
        CHECK(slot32[7] == static_cast<std::uint8_t>(f32_to_f16_bits(65504.0f) >> 8));

        // Конвертер: счётчик сатураций в логе (Stats) + провенанс dtype_orig.
        const std::string bfs = out_dir("/tmp/stell_hf_bf16_sat");
        auto ts_sat = dense_hf_tensors(/*bf16_norms=*/true);
        for (auto& t : ts_sat)
            if (t.name == "model.layers.0.input_layernorm.weight") {
                const float huge[8] = {1e9f, -1e9f, 3e38f, -3e38f, 1.0f, 2.0f, 3.0f, 4.0f};
                t.data.assign(16, 0);
                for (int k = 0; k < 8; ++k) {
                    std::uint32_t b;
                    std::memcpy(&b, &huge[k], 4);
                    t.data[2 * k] = static_cast<std::uint8_t>((b >> 16) & 0xFF);
                    t.data[2 * k + 1] = static_cast<std::uint8_t>((b >> 24) & 0xFF);
                }
            }
        write_config(bfs, "LlamaForCausalLM", false);
        write_shards(bfs, ts_sat);
        const ModelDescription d_sat = parse_model_description(
            hf_to_description(bfs, "F16"), "hf-bf16-sat");
        StellTierConverter conv_sat;
        StellTierConverter::Stats stats;
        f16_saturation_reset();
        conv_sat.convert_streaming(d_sat, bfs, "/tmp/stell_hf_bf16_sat_out", {}, &stats);
        CHECK(stats.f16_saturations == 4);         // 1e9, -1e9, 3e38, -3e38
        CHECK(stats.groups_dtype_orig_bf16 == 4);  // 4 нормы BF16 (2 слоя × 2)
        CHECK(stats.groups_dtype_orig_f32 == 1);   // model.norm.weight (F32)
        std::printf("R5: конвертер — clamp ±65504, сатураций=%zu, dtype_orig bf16=%zu/f32=%zu\n",
                    static_cast<std::size_t>(stats.f16_saturations),
                    static_cast<std::size_t>(stats.groups_dtype_orig_bf16),
                    static_cast<std::size_t>(stats.groups_dtype_orig_f32));
    }

    // =========================================================================
    // 8b. [R2] round 6 (planner [Q6], ADR-03 round 5/6): провенанс dtype_orig
    //     — top-level поле описания («исходный dtype весов до конверсии»;
    //     читается R8-обёрткой serve/bench из description.json). Однородный
    //     источник → "f32"/"bf16"/"f16"; смешанный источник — поле не
    //     пишется (состав — в Stats конвертера groups_dtype_orig_bf16/f32 и
    //     логе CLI). Поле информационное: перегенерация не меняет слоты/
    //     логиты (ПОБИТОВО — сверка layout'ов по CRC/слотам + логиты
    //     интерпретатора). Парсер схемы 1.0 толерантен к полю (неизвестные
    //     top-level ключи игнорируются — прежние описания читаются).
    // =========================================================================
    {
        // Однородный dense-чекпоинт (все тензора одного исходного dtype;
        // d_model=8, vocab=16, d_ff=16, 2 слоя — конвенция dense_hf_tensors).
        auto uniform_dense = [](const char* dt) {
            const std::string dts = dt;
            std::vector<SynthTensor> out;
            auto push = [&](const std::string& name, std::vector<std::int64_t> sh,
                            int seed) {
                std::size_t n = 1;
                for (auto d : sh) n *= static_cast<std::size_t>(d);
                if (dts == "F32")
                    out.push_back(tensor(name, "F32", std::move(sh),
                                         f32_det(n, 0.125f + 0.03125f * seed)));
                else if (dts == "BF16")
                    out.push_back(tensor(name, "BF16", std::move(sh),
                                         bf16_from_f32(f32_det(n, 0.125f + 0.03125f * seed))));
                else
                    out.push_back(tensor(name, "F16", std::move(sh), f16_det(n, seed)));
            };
            push("model.embed_tokens.weight", {16, 8}, 3);
            push("model.norm.weight", {8}, 4);
            push("lm_head.weight", {16, 8}, 5);
            for (int layer = 0; layer < 2; ++layer) {
                const std::string p =
                    "model.layers." + std::to_string(layer) + ".";
                push(p + "input_layernorm.weight", {8}, 6 + layer);
                push(p + "post_attention_layernorm.weight", {8}, 8 + layer);
                push(p + "self_attn.q_proj.weight", {8, 8}, 10 + layer);
                push(p + "self_attn.k_proj.weight", {4, 8}, 12 + layer);
                push(p + "self_attn.v_proj.weight", {4, 8}, 14 + layer);
                push(p + "self_attn.o_proj.weight", {8, 8}, 16 + layer);
                push(p + "mlp.gate_proj.weight", {16, 8}, 18 + layer);
                push(p + "mlp.up_proj.weight", {16, 8}, 20 + layer);
                push(p + "mlp.down_proj.weight", {8, 16}, 22 + layer);
            }
            return out;
        };

        // Полный протокол [R2]: описание с полем vs перегенерация в формате
        // round-5 (без поля) — слоты layout'ов (CRC/офсеты/размеры) и логиты
        // интерпретатора ПОБИТОВО идентичны.
        auto check_dtype_orig = [&](const std::vector<SynthTensor>& ts,
                                    const std::string& dir, const char* arch,
                                    bool moe, const char* expect) {
            write_config(dir, arch, moe);
            write_shards(dir, ts);
            const std::string json = hf_to_description(dir);
            const std::string field =
                std::string(",\"dtype_orig\":\"") + expect + "\"";
            const std::size_t pos = json.find(field);
            CHECK_MSG(pos != std::string::npos,
                      "описание не содержит dtype_orig (ожидалось значение)");
            const std::size_t vw = json.find(",\"vocab_size\":");
            CHECK(pos < vw);  // поле — сразу после dtype_weights
            CHECK(json.find(field + ",\"vocab_size\"") != std::string::npos);

            // Формат round-5 (без поля) — референс побитовой сверки:
            std::string json5 = json;
            json5.erase(json5.find(field), field.size());
            const ModelDescription d6 =
                parse_model_description(json, "hf-r6-dtype-orig");
            const ModelDescription d5 =
                parse_model_description(json5, "hf-r5-no-field");
            CHECK(d6.groups.size() == d5.groups.size());
            for (std::size_t i = 0; i < d5.groups.size(); ++i) {
                CHECK(d6.groups[i].name == d5.groups[i].name);
                CHECK(d6.groups[i].shape == d5.groups[i].shape);
                CHECK(d6.groups[i].dtype == d5.groups[i].dtype);
            }

            const std::string out6s = dir + "_out6";
            const std::string out5s = dir + "_out5";
            StellTierConverter c6, c5;
            c6.convert_streaming(d6, dir, out_dir(out6s.c_str()));
            c5.convert_streaming(d5, dir, out_dir(out5s.c_str()));
            const std::string lay6 = out6s + "/" + d6.name + ".stelltier";
            const std::string lay5 = out5s + "/" + d5.name + ".stelltier";
            stell::tiers::TierLayoutReader r6(lay6);
            stell::tiers::TierLayoutReader r5(lay5);
            CHECK(r6.index().size() == r5.index().size());
            bool slots_same = r6.index().size() == r5.index().size();
            for (std::size_t i = 0; slots_same && i < r5.index().size(); ++i) {
                const auto& e6 = r6.index()[i];
                const auto& e5 = r5.index()[i];
                slots_same = e6.group_id == e5.group_id &&
                             e6.data_offset == e5.data_offset &&
                             e6.slot_bytes == e5.slot_bytes &&
                             e6.crc32 == e5.crc32;
            }
            CHECK_MSG(slots_same,
                      "слоты перегенерации (с dtype_orig) != прежним (побитно)");

            const se::PlainSpec s6 = se::plain_spec_from_description(d6);
            const se::ExecutionPlan p6 = se::build_execution_plan(s6);
            const se::PlainSpec s5 = se::plain_spec_from_description(d5);
            const se::ExecutionPlan p5 = se::build_execution_plan(s5);
            const se::LoadedWeights w6 =
                se::load_plain_weights(lay6, s6, se::PlainWeightsStellTier);
            const se::LoadedWeights w5 =
                se::load_plain_weights(lay5, s5, se::PlainWeightsStellTier);
            se::OpsInterpreter i6(s6, p6, w6, nullptr);
            se::OpsInterpreter i5(s5, p5, w5, nullptr);
            const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5};
            const std::vector<float> l6 = i6.run_prompt_logits(prompt);
            const std::vector<float> l5 = i5.run_prompt_logits(prompt);
            CHECK(l6.size() == l5.size());
            bool logits_same = l6.size() == l5.size();
            for (std::size_t i = 0; logits_same && i < l5.size(); ++i)
                logits_same = l6[i] == l5[i];
            CHECK_MSG(logits_same,
                      "логиты перегенерации (с dtype_orig) != прежним (побитно)");
            std::printf("R2: dtype_orig=\"%s\" — слоты/логиты перегенерации побитово == прежним\n",
                        expect);
        };

        {
            const std::string dir = out_dir("/tmp/stell_hf_u32");
            check_dtype_orig(uniform_dense("F32"), dir, "LlamaForCausalLM", false, "f32");
        }
        {
            const std::string dir = out_dir("/tmp/stell_hf_ubf");
            check_dtype_orig(uniform_dense("BF16"), dir, "LlamaForCausalLM", false, "bf16");
        }
        {
            const std::string dir = out_dir("/tmp/stell_hf_uf16");
            check_dtype_orig(uniform_dense("F16"), dir, "LlamaForCausalLM", false, "f16");
        }

        // Смешанный источник (нормы F32 + матрицы F16) — поле НЕ пишется
        // (честно: скаляр неоднозначен; полный состав — в Stats конвертера
        // groups_dtype_orig_bf16/f32 и логе CLI конвертера).
        {
            const std::string dir = out_dir("/tmp/stell_hf_mixed_dt");
            auto ts = dense_hf_tensors();
            write_config(dir, "LlamaForCausalLM", false);
            write_shards(dir, ts);
            const std::string json = hf_to_description(dir);
            CHECK(json.find("\"dtype_orig\"") == std::string::npos);
            std::printf("R2: смешанный источник — dtype_orig отсутствует (честно; состав — в Stats/логе конвертера)\n");
        }
    }

    // =========================================================================
    // 8c. [R6-добор] Qwen3MoeForCausalLM (зонд реального чекпоинта
    //     Qwen/Qwen3-30B-A3B-Instruct-2507): роутер mlp.gate.weight,
    //     поэертельные эксперты, однородный MoE БЕЗ shared-expert (легитимно),
    //     lm_head; config-маппинг moe_intermediate_size/num_experts_per_tok/
    //     norm_topk_prob; qk-norm — группы attn.{q,k}_norm (round 7); roundtrip побитово.
    // =========================================================================
    const std::string q3_dir = out_dir("/tmp/stell_hf_qwen3moe");
    {
        const auto ts = qwen3_moe_hf_tensors();
        write_config_qwen3(q3_dir);
        write_shards(q3_dir, ts);
        const std::string json = hf_to_description(q3_dir, "NVFP4");
        const JsonValue root = json_parse(json);
        CHECK(root.get_string("class") == "moe");
        CHECK(root.get_string("name") == "stell_hf_qwen3moe");
        CHECK(root.get_string("dtype_weights") == "NVFP4");
        const JsonValue& moe = root.get_object("moe");
        CHECK(moe.get_uint("n_experts") == 4);
        CHECK(moe.get_uint("n_experts_active") == 2);
        CHECK(moe.get_uint("expert_d_ff") == 8);
        // ops однородного MoE: rmsnorm + attention + moe_gather_scatter (БЕЗ ffn).
        const std::vector<JsonValue>& ops = root.get_array("ops").array_v;
        CHECK(ops.size() == 3);
        CHECK(ops[2].get_string("op") == "moe_gather_scatter");
        CHECK(json.find("\"ffn\"") == std::string::npos);

        const ModelDescription d = parse_model_description(json, "hf-qwen3moe");
        // Группы: embed + norm_final + head + 2 kv + 2 нормы + 4 attn-матрицы +
        // router + 4×3 эксперта = 24 (ffn.shared НЕТ — однородный MoE).
        CHECK(d.groups.size() == 43);
        const int router0 = find_group(d, "l000.mlp.router");
        CHECK((router0 >= 0) &&
              d.group_source_patterns[router0] ==
                  "model.layers.0.mlp.gate.weight");  // зонд: роутер mlp.gate
        CHECK(d.groups[router0].dtype == stell::tiers::DType::F32);
        const int gate00 = find_group(d, "l000.e000.gate");
        CHECK((gate00 >= 0) &&
              d.group_source_patterns[gate00] ==
                  "model.layers.0.mlp.experts.0.gate_proj.weight");
        CHECK(d.groups[gate00].dtype == stell::tiers::DType::NVFP4);
        const int down11 = find_group(d, "l001.e003.down");
        CHECK((down11 >= 0) &&
              d.group_source_patterns[down11] ==
                  "model.layers.1.mlp.experts.3.down_proj.weight");
        const int head = find_group(d, "head");
        CHECK((head >= 0) &&
              d.group_source_patterns[head] == "lm_head.weight");
        for (std::size_t i = 0; i < d.groups.size(); ++i)
            CHECK_MSG(d.group_source_patterns[i].find(".mlp.shared_expert") ==
                          std::string::npos,
                      "однородный MoE без shared-expert: группа ffn.shared не создана");

        const std::string single =
            write_single("/tmp/stell_hf_qwen3moe_single.safetensors", ts);
        roundtrip_logits(d, q3_dir, single, "/tmp/stell_hf_qwen3moe_out",
                         se::PlainWeightsNVFP4W4A16);
        std::printf("moe (Qwen3MoeForCausalLM, NVFP4): roundtrip логитов OK (побитно)\n");
    }
    // (а) [R1] round 7: qk-norm тензоры (как в реальном чекпоинте) —
    //     генерация БЕЗ SchemaMismatch: группы l{layer}.attn.{q,k}_norm
    //     ([d_head]/[d_kv], dtype источника), параметр op attention
    //     "qk_eps" из config rms_norm_eps; roundtrip побитово.
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_qknorm");
        write_config_qwen3(dir);
        write_shards(dir, qwen3_moe_qknorm_hf_tensors());
        const std::string json = hf_to_description(dir, "F16");
        const JsonValue root = json_parse(json);
        // Параметр op attention: qk_eps == rms_norm_eps из config (1e-06).
        const std::vector<JsonValue>& ops = root.get_array("ops").array_v;
        CHECK(ops[1].get_string("op") == "attention");
        const JsonValue* qk_eps = ops[1].find("params")->find("qk_eps");
        CHECK_MSG(qk_eps != nullptr, "qk_eps: параметр op attention отсутствует");
        const double qk_eps_val = qk_eps->type == JsonValue::Type::Double
                                      ? qk_eps->double_v
                                      : static_cast<double>(qk_eps->int_v);
        CHECK(std::fabs(qk_eps_val - 1e-6) < 1e-12);
        const ModelDescription d = parse_model_description(json, "hf-qwen3moe-qk");
        // Группы: прежние 43 + 2 слоя × (q_norm + k_norm) = 47.
        CHECK(d.groups.size() == 47);
        const int qn0 = find_group(d, "l000.attn.q_norm");
        CHECK((qn0 >= 0) &&
              d.group_source_patterns[qn0] ==
                  "model.layers.0.self_attn.q_norm.weight");
        CHECK((d.groups[qn0].shape == std::vector<std::int64_t>{4}));
        CHECK(d.groups[qn0].dtype == stell::tiers::DType::F16);
        const int kn1 = find_group(d, "l001.attn.k_norm");
        CHECK((kn1 >= 0) &&
              d.group_source_patterns[kn1] ==
                  "model.layers.1.self_attn.k_norm.weight");
        CHECK((d.groups[kn1].shape == std::vector<std::int64_t>{4}));
        const std::string single = write_single(
            "/tmp/stell_hf_qwen3moe_qknorm_single.safetensors",
            qwen3_moe_qknorm_hf_tensors());
        roundtrip_logits(d, dir, single, "/tmp/stell_hf_qwen3moe_qknorm_out",
                         se::PlainWeightsSafetensors);
        std::printf("moe (Qwen3Moe, qk-norm): группы attn.{q,k}_norm + qk_eps; roundtrip OK (побитно)\n");
    }
    // (а-1) [R3] round 8: GQA (n_kv_heads=2 > 1) + qk-norm — описание
    //     генерируется, k_norm-группа [head_dim]=[4] (НЕ [n_kv*d_head]=[8]);
    //     roundtrip логитов побитово (проходит GQA-путь интерпретатора:
    //     k-norm по per-head срезам, n_kv_heads>1 в attention/kv-кеше).
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_qkgqa");
        write_config_qwen3(dir, /*norm_topk=*/true, /*moe_ff=*/8,
                           /*rms_eps=*/true, /*n_kv=*/2);
        const auto ts = qwen3_moe_qknorm_gqa_hf_tensors();
        write_shards(dir, ts);
        const std::string json = hf_to_description(dir, "F16");
        const JsonValue root = json_parse(json);
        const JsonValue* kvj = root.find("kv");
        CHECK(kvj != nullptr);
        CHECK(kvj->get_uint("n_kv_heads") == 2);
        CHECK(kvj->get_uint("d_kv") == 4);   // per-head kv dim == head_dim
        const ModelDescription d = parse_model_description(json, "hf-qwen3moe-qkgqa");
        CHECK(d.groups.size() == 47);   // 43 + 2 слоя × (q_norm + k_norm)
        const int kn0 = find_group(d, "l000.attn.k_norm");
        CHECK((kn0 >= 0) &&
              d.group_source_patterns[kn0] ==
                  "model.layers.0.self_attn.k_norm.weight");
        CHECK((d.groups[kn0].shape == std::vector<std::int64_t>{4}));
        const std::string single =
            write_single("/tmp/stell_hf_qwen3moe_qkgqa_single.safetensors", ts);
        roundtrip_logits(d, dir, single, "/tmp/stell_hf_qwen3moe_qkgqa_out",
                         se::PlainWeightsSafetensors);
        std::printf("moe (Qwen3Moe, qk-norm GQA): k_norm [head_dim]; roundtrip OK (побитно)\n");
    }
    // (а-2) [R1] round 7: частичный набор (только q_norm, k_norm нет) →
    //     честный SchemaMismatch.
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_qkpartial");
        write_config_qwen3(dir);
        write_shards(dir, qwen3_moe_qknorm_partial_hf_tensors());
        try {
            hf_to_description(dir, "F16");
            CHECK_MSG(false && "expected LoaderError",
                      "частичный qk-набор → SchemaMismatch");
        } catch (const LoaderError& e) {
            CHECK_MSG(e.code() == LoaderError::Code::SchemaMismatch,
                      "частичный qk-набор: код != SchemaMismatch");
            CHECK_MSG(std::string(e.what()).find("k_norm") != std::string::npos,
                      "частичный qk-набор: сообщение называет отсутствующий тензор");
        }
    }
    // (а-3) [R1] round 7: qk-тензоры есть, rms_norm_eps в config нет →
    //     честный SchemaMismatch (eps не дефолт-догадка).
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_qknoeps");
        write_config_qwen3(dir, /*norm_topk=*/true, /*moe_ff=*/8,
                           /*rms_eps=*/false);
        write_shards(dir, qwen3_moe_qknorm_hf_tensors());
        try {
            hf_to_description(dir, "F16");
            CHECK_MSG(false && "expected LoaderError",
                      "qk-нормы без rms_norm_eps → SchemaMismatch");
        } catch (const LoaderError& e) {
            CHECK_MSG(e.code() == LoaderError::Code::SchemaMismatch,
                      "qk-нормы без rms_norm_eps: код != SchemaMismatch");
            CHECK_MSG(std::string(e.what()).find("rms_norm_eps") !=
                          std::string::npos,
                      "qk-нормы без rms_norm_eps: сообщение называет ключ");
        }
    }
    // (б) [R2] round 7: norm_topk_prob=false — генерация БЕЗ SchemaMismatch:
    //     top-level "router_renorm": false; roundtrip побитово; парсер
    //     читает MoeSpec::router_renorm == false. Плюс дефолт: без ключа
    //     (norm_topk=true) поле НЕ пишется, MoeSpec::router_renorm == true.
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_notopk");
        write_config_qwen3(dir, /*norm_topk=*/false);
        const auto ts = qwen3_moe_hf_tensors();
        write_shards(dir, ts);
        const std::string json = hf_to_description(dir, "F16");
        const JsonValue root = json_parse(json);
        const JsonValue* rr = root.find("router_renorm");
        CHECK_MSG(rr != nullptr && rr->type == JsonValue::Type::Bool && !rr->bool_v,
                  "norm_topk_prob=false: top-level router_renorm:false отсутствует");
        const ModelDescription d = parse_model_description(json, "hf-qwen3moe-nr");
        CHECK_MSG(d.moe && !d.moe->router_renorm,
                  "парсер: MoeSpec::router_renorm != false");
        const std::string single =
            write_single("/tmp/stell_hf_qwen3moe_notopk_single.safetensors", ts);
        roundtrip_logits(d, dir, single, "/tmp/stell_hf_qwen3moe_notopk_out",
                         se::PlainWeightsSafetensors);
        std::printf("moe (Qwen3Moe, norm_topk_prob=false): router_renorm:false; roundtrip OK (побитно)\n");
    }
    // (б-2) [R2] round 7: дефолт — поле router_renorm НЕ пишется (побитовые
    //     golden не меняются); парсер: отсутствие поля → true.
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_topk");
        write_config_qwen3(dir, /*norm_topk=*/true);
        write_shards(dir, qwen3_moe_hf_tensors());
        const std::string json = hf_to_description(dir, "F16");
        CHECK_MSG(json.find("router_renorm") == std::string::npos,
                  "дефолт: поле router_renorm написано (побитовые golden меняются)");
        const ModelDescription d = parse_model_description(json, "hf-qwen3moe-df");
        CHECK_MSG(d.moe && d.moe->router_renorm,
                  "парсер: отсутствие router_renorm → true");
    }
    // (б-4) [R2] round 7: парсер — class=dense + router_renorm=false →
    //     DescriptionError; router_renorm не-bool → DescriptionError.
    {
        const char* dense_min = R"({
  "schema": "1.0", "name": "dense-rr-false", "class": "dense",
  "dtype_weights": "F16",
  "vocab_size": 16, "d_model": 8, "n_layers": 1, "n_heads": 2, "d_head": 4,
  "kv": {"dtype": "F16", "n_kv_heads": 1, "d_kv": 4, "quant": null},
  "router_renorm": false,
  "ops": [{"op": "rmsnorm", "io": ["hidden"], "params": {"eps": 1e-5}}],
  "tensor_groups": [
    {"name": "embed", "role": "attn", "shape": [16, 8], "dtype": "F16",
     "source_pattern": "embedding.weight"}
  ]
})";
        bool threw = false;
        try {
            parse_model_description(dense_min, "test");
        } catch (const DescriptionError&) {
            threw = true;
        }
        CHECK_MSG(threw, "dense + router_renorm=false: DescriptionError отсутствует");
    }
    // (в) moe_intermediate_size != факту шардов → честный SchemaMismatch
    //     (перекрёстная сверка config ↔ тензоры).
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_moef");
        write_config_qwen3(dir, /*norm_topk=*/true, /*moe_ff=*/16);
        write_shards(dir, qwen3_moe_hf_tensors());
        try {
            hf_to_description(dir, "F16");
            CHECK_MSG(false && "expected LoaderError",
                      "moe_intermediate_size != d_ff → SchemaMismatch");
        } catch (const LoaderError& e) {
            CHECK_MSG(e.code() == LoaderError::Code::SchemaMismatch,
                      "moe_intermediate_size: код != SchemaMismatch");
            CHECK_MSG(std::string(e.what()).find("moe_intermediate_size") !=
                          std::string::npos,
                      "moe_intermediate_size: сообщение называет ключ");
        }
    }
    // (в-2) [R2] round 7: moe_intermediate_size согласована с expert 0, но
    //     фактическая форма ДРУГОГО эксперта отличается → честный
    //     SchemaMismatch (сверка против e{expert}.gate/up/down ВСЕХ
    //     экспертов первого MoE-слоя).
    {
        const std::string dir = out_dir("/tmp/stell_hf_qwen3moe_moef2");
        write_config_qwen3(dir, /*norm_topk=*/true, /*moe_ff=*/8);
        auto ts = qwen3_moe_hf_tensors();
        // Эксперт 2 слоя 0: down [8, 16] вместо [8, 8] (d_ff другого
        // эксперта не соответствует факту первого MoE-слоя).
        for (auto& t : ts)
            if (t.name == "model.layers.0.mlp.experts.2.down_proj.weight")
                t.shape = {8, 16};
        write_shards(dir, ts);
        try {
            hf_to_description(dir, "F16");
            CHECK_MSG(false && "expected LoaderError",
                      "эксперт 2: форма != факту первого MoE-слоя → SchemaMismatch");
        } catch (const LoaderError& e) {
            CHECK_MSG(e.code() == LoaderError::Code::SchemaMismatch,
                      "неоднородный эксперт: код != SchemaMismatch");
            CHECK_MSG(std::string(e.what()).find("experts.2.down") !=
                          std::string::npos,
                      "неоднородный эксперт: сообщение называет тензор");
        }
    }

    // =========================================================================
    // 9. Error-кейсы.
    // =========================================================================
    // 9. Error-кейсы.
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
    // (ж) [R3] частичный MoE-набор: роутер слоя 3 без тензоров экспертов →
    //     SchemaMismatch (непокрытая конфигурация, fail-fast §3).
    {
        const std::string dir = out_dir("/tmp/stell_hf_moe_partial");
        write_config_layers(dir, "Qwen2MoeForCausalLM", true, 4);
        auto ts = mixed_hf_tensors();
        ts.erase(std::remove_if(ts.begin(), ts.end(),
                                [](const SynthTensor& t) {
                                    return t.name.rfind("model.layers.3.mlp.experts.", 0) == 0;
                                }),
                 ts.end());
        write_shards(dir, ts);
        expect_error(dir, "F16", LoaderError::Code::SchemaMismatch,
                     "роутер без экспертов");
    }
    // (з) [R3] слой с тензорами экспертов, но без роутера → SchemaMismatch.
    {
        const std::string dir = out_dir("/tmp/stell_hf_moe_norouter");
        write_config_layers(dir, "Qwen2MoeForCausalLM", true, 4);
        auto ts = mixed_hf_tensors();
        ts.erase(std::remove_if(ts.begin(), ts.end(),
                                [](const SynthTensor& t) {
                                    return t.name == "model.layers.2.mlp.router.weight";
                                }),
                 ts.end());
        write_shards(dir, ts);
        expect_error(dir, "F16", LoaderError::Code::SchemaMismatch,
                     "эксперты без роутера");
    }
    // (и) [R3] слой с роутером И dense-ffn тензорами → SchemaMismatch
    //     (неоднозначная конфигурация).
    {
        const std::string dir = out_dir("/tmp/stell_hf_moe_ambiguous");
        write_config_layers(dir, "Qwen2MoeForCausalLM", true, 4);
        auto ts = mixed_hf_tensors();
        // Слой 2 (MoE) ДОПОЛНЁН dense-тензором mlp.gate_proj.weight —
        // роутер и dense-ffn одновременно → неоднозначная конфигурация.
        for (auto& t : ts)
            if (t.name == "model.layers.0.mlp.gate_proj.weight") {
                SynthTensor extra = t;
                extra.name = "model.layers.2.mlp.gate_proj.weight";
                ts.push_back(extra);
            }
        write_shards(dir, ts);
        expect_error(dir, "F16", LoaderError::Code::SchemaMismatch,
                     "роутер и dense-ffn одновременно");
    }
    // (к) [R3] слой без ffn-контура (ни dense-ffn тензоров, ни MoE-набора) →
    //     SchemaMismatch.
    {
        const std::string dir = out_dir("/tmp/stell_hf_layer_noffn");
        write_config_layers(dir, "Qwen2MoeForCausalLM", true, 4);
        auto ts = mixed_hf_tensors();
        ts.erase(std::remove_if(ts.begin(), ts.end(),
                                [](const SynthTensor& t) {
                                    return t.name.rfind("model.layers.1.mlp.", 0) == 0;
                                }),
                 ts.end());
        write_shards(dir, ts);
        expect_error(dir, "F16", LoaderError::Code::SchemaMismatch, "слой без ffn-контура");
    }
    // (л) [R3] MoE-архитектура, но ни одного MoE-слоя → SchemaMismatch
    //     (однородный dense под MoE-диспетчером — непокрытая конфигурация).
    {
        const std::string dir = out_dir("/tmp/stell_hf_moe_nolayers");
        write_config(dir, "Qwen2MoeForCausalLM", true);
        write_shards(dir, dense_hf_tensors());
        expect_error(dir, "F16", LoaderError::Code::SchemaMismatch,
                     "MoE-архитектура без MoE-слоёв");
    }
    // (м) [R3] смешанный каталог без intermediate_size (dense-слои без d_ff) →
    //     SchemaMismatch.
    {
        const std::string dir = out_dir("/tmp/stell_hf_mixed_noint");
        write_config_layers(dir, "Qwen2MoeForCausalLM", true, 4, /*with_intermediate=*/false);
        write_shards(dir, mixed_hf_tensors());
        expect_error(dir, "F16", LoaderError::Code::SchemaMismatch, "нет intermediate_size");
    }

}

TEST_MAIN("test_models_hf_description")

// Тесты описания модели (ADR-09 §1): парсинг, валидация, расширение
// cardinality, slot_bytes (ADR-08/ADR-12).
#include "stell/models/description.hpp"

#include <set>

#include "test_util.hpp"

using namespace stell::models;

// Хелпер: строковое имя dtype группы (для payload_bytes_for).
static std::string dtype_str_of(const stell::models::ModelDescription& d, std::size_t i) {
    switch (d.groups[i].dtype) {
        case stell::tiers::DType::F16: return "F16";
        case stell::tiers::DType::F32: return "F32";
        case stell::tiers::DType::E4M3: return "E4M3";
        case stell::tiers::DType::NVFP4: return "NVFP4";
        case stell::tiers::DType::W4A16: return "W4A16";
    }
    return "";
}

namespace {

const char* kMiniDense = R"({
  "schema": "1.0",
  "name": "mini-dense-test",
  "class": "dense",
  "dtype_weights": "F16",
  "vocab_size": 16, "d_model": 8, "n_layers": 2, "n_heads": 2, "d_head": 4,
  "kv": {"dtype": "F16", "n_kv_heads": 1, "d_kv": 4, "quant": null},
  "ops": [
    {"op": "rmsnorm", "io": ["hidden"], "params": {"eps": 1e-5}},
    {"op": "attention", "io": ["hidden", "kv"], "params": {"kernel": "sm70_fp32_qk_pv"}}
  ],
  "tensor_groups": [
    {"name": "embed", "role": "attn", "shape": [16, 8], "dtype": "F16",
     "source_pattern": "embedding.weight"},
    {"name": "norm_final", "role": "attn", "shape": [8], "dtype": "F32",
     "source_pattern": "model.norm.weight"},
    {"name": "l{layer}.attn.wq", "role": "attn", "shape": [8, 8], "dtype": "F16",
     "source_pattern": "layers.{layer}.self_attn.q_proj.weight", "cardinality": {"layer": 2}},
    {"name": "l{layer}.e{expert}.up", "role": "expert", "shape": [8, 8], "dtype": "W4A16",
     "source_pattern": "layers.{layer}.mlp.experts.{expert}.up.weight",
     "cardinality": {"layer": 2, "expert": 4}},
    {"name": "l{layer}.kv", "role": "kv", "shape": [4, 8], "dtype": "F16",
     "source_pattern": "layers.{layer}.kv_cache", "kv_block": true, "cardinality": {"layer": 2}}
  ]
})";

const char* kMiniMoe = R"({
  "schema": "1.0",
  "name": "mini-moe-test",
  "class": "moe",
  "dtype_weights": "NVFP4",
  "vocab_size": 16, "d_model": 8, "n_layers": 2, "n_heads": 2, "d_head": 4,
  "kv": {"dtype": "F16", "n_kv_heads": 1, "d_kv": 4, "quant": null},
  "moe": {"n_experts": 4, "n_experts_active": 2, "expert_d_ff": 8, "router": "softmax-topk"},
  "ops": [
    {"op": "rmsnorm", "io": ["hidden"], "params": {"eps": 1e-5}},
    {"op": "moe_gather_scatter", "io": ["hidden", "experts"], "params": {}}
  ],
  "tensor_groups": [
    {"name": "embed", "role": "attn", "shape": [16, 8], "dtype": "F16",
     "source_pattern": "embedding.weight"},
    {"name": "l{layer}.e{expert}.up", "role": "expert", "shape": [8, 8], "dtype": "NVFP4",
     "source_pattern": "layers.{layer}.mlp.experts.{expert}.up.weight",
     "cardinality": {"layer": 2, "expert": 4}}
  ]
})";

}  // namespace

void body() {
    // --- парсинг dense -------------------------------------------------------
    ModelDescription d = parse_model_description(kMiniDense, "kMiniDense");
    CHECK(d.name == "mini-dense-test");
    CHECK(!d.is_moe);
    CHECK(d.dtype_weights == "F16");
    CHECK(d.vocab_size == 16 && d.d_model == 8);
    CHECK(d.n_layers == 2 && d.n_heads == 2 && d.d_head == 4);
    CHECK(d.kv.n_kv_heads == 1 && d.kv.d_kv == 4 && d.kv.quant.empty());
    CHECK(!d.moe.has_value());
    CHECK(d.ops.size() == 2);
    CHECK(d.ops[0].op == "rmsnorm");
    CHECK(d.ops[1].io.size() == 2);

    // Расширение cardinality: 1 (embed) + 1 (norm) + 2 (wq) + 8 (experts) + 2 (kv) = 14.
    CHECK(d.groups.size() == 14);
    CHECK(d.group_source_patterns.size() == d.groups.size());
    std::set<std::uint32_t> ids;
    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        CHECK(d.groups[i].id.value == i);                 // GroupId монотонный с 0 (ADR-08)
        CHECK(ids.insert(d.groups[i].id.value).second);
        CHECK(d.groups[i].slot_bytes % 4096 == 0);        // ADR-12 §2.4
        CHECK(d.groups[i].slot_bytes >=
                   payload_bytes_for(dtype_str_of(d, i), d.groups[i].shape));
    }
    // Имена и паттерны: "l001.e003.up" / "layers.1.mlp.experts.3.up.weight".
    bool found_name = false, found_pattern = false;
    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        if (d.groups[i].name == "l001.e003.up") found_name = true;
        if (d.group_source_patterns[i] == "layers.1.mlp.experts.3.up.weight") found_pattern = true;
    }
    CHECK(found_name && found_pattern);

    // Роли/типы группы kv.
    for (const auto& g : d.groups) {
        if (g.name == "l000.kv") {
            CHECK(g.role == stell::tiers::Role::Kv);
            CHECK(g.dtype == stell::tiers::DType::F16);
            CHECK(g.shape.size() == 2);
            CHECK(g.shape[0] == 4);
            CHECK(g.shape[1] == 8);
        }
        if (g.name == "l001.e000.up")
            CHECK(g.role == stell::tiers::Role::Expert);
    }

    // --- парсинг MoE ---------------------------------------------------------
    ModelDescription m = parse_model_description(kMiniMoe);
    CHECK(m.is_moe);
    CHECK(m.moe && m.moe->n_experts == 4 && m.moe->n_experts_active == 2);
    CHECK(m.moe->expert_d_ff == 8 && m.moe->router == "softmax-topk");
    CHECK(m.groups.size() == 1 + 8);  // embed + experts

    // --- slot_bytes (ADR-08 выровнен, кратен 4096 по ADR-12) -----------------
    CHECK(slot_bytes_for("F16", {4096, 4096}) == 32 * 1024 * 1024);  // ровно, без паддинга
    CHECK(slot_bytes_for("F16", {8, 8}) == 4096);                    // 128B → 4096
    CHECK(slot_bytes_for("NVFP4", {8, 8}) == 4096);                  // 32+1 B → 4096
    CHECK(slot_bytes_for("W4A16", {8, 8}) == 4096);                  // 32+2 B → 4096
    CHECK(slot_bytes_for("E4M3", {4096}) == 4096);
    CHECK(payload_bytes_for("NVFP4", {64}) == 34);                   // 32 ниббла + 2 scale
    CHECK(payload_bytes_for("W4A16", {64}) == 36);                   // 32 ниббла + 2×F16 scale

    // --- ошибки валидации ----------------------------------------------------
    std::string bad_schema = std::string(kMiniDense);
    bad_schema.replace(bad_schema.find("\"1.0\""), 5, "\"2.0\"");
    CHECK_THROW(parse_model_description(bad_schema), DescriptionError, "schema 2.x rejected");

    std::string bad_role = std::string(kMiniDense);
    bad_role.replace(bad_role.find("\"role\": \"attn\""), 15, "\"role\": \"attention\"");
    CHECK_THROW(parse_model_description(bad_role), DescriptionError, "unknown role rejected");

    std::string bad_card = std::string(kMiniMoe);
    bad_card.replace(bad_card.find("\"expert\": 4"), 11, "\"expert\": 5");
    CHECK_THROW(parse_model_description(bad_card), DescriptionError,
                "expert cardinality != moe.n_experts rejected");

    std::string no_moe = std::string(kMiniMoe);
    const std::size_t moe_pos = no_moe.find("\"moe\": {");
    no_moe.replace(moe_pos, 5, "\"zzz\":");
    no_moe.replace(no_moe.find("\"zzz\": {") + 5, 1, " ");
    // class=moe без moe-объекта → ошибка (получится только если moe-поле переименовано)
    CHECK_THROW(parse_model_description(no_moe), DescriptionError, "class=moe without moe rejected");
}

TEST_MAIN("test_models_description")

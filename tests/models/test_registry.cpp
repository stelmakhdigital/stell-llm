// Тесты реестра архитектур (ADR-09 §2): 3 встроенные архитектуры
// (1 dense + 2 MoE), инварианты разметки, ошибки.
#include "stell/models/registry.hpp"

#include <set>
#include <vector>

#include "test_util.hpp"

using namespace stell::models;

namespace {
// Ожидаемые числа групп (embed/norm_final + per-layer × cardinality):
std::size_t expected_groups(const std::string& name) {
    if (name == "llama-class") return 3 + 10 * 32;                 // 323
    if (name == "glm-class") return 2 + 8 * 32 + 2 * 32 * 128;     // 8450
    if (name == "qwen-moe-class") return 2 + 8 * 48 + 2 * 48 * 128; // 12674
    return 0;
}
}  // namespace

void body() {
    Registry r;
    const std::vector<std::string> avail = r.available();
    CHECK(avail.size() == 3);
    std::set<std::string> names(avail.begin(), avail.end());
    CHECK(names.count("llama-class") == 1);
    CHECK(names.count("glm-class") == 1);
    CHECK(names.count("qwen-moe-class") == 1);

    for (const auto& name : avail) {
        const ModelDescription d = r.load_builtin(name);
        CHECK(d.name == name);
        CHECK(d.group_source_patterns.size() == d.groups.size());
        CHECK(d.groups.size() == expected_groups(name));
        std::set<std::uint32_t> ids;
        for (const auto& g : d.groups) {
            CHECK(ids.insert(g.id.value).second);          // GroupId уникален и монотонен
            CHECK(g.slot_bytes % 4096 == 0);               // ADR-12 §2.4
            CHECK(g.role == stell::tiers::Role::Attn ||
                   g.role == stell::tiers::Role::Expert ||
                   g.role == stell::tiers::Role::Kv);
        }
        // Роли из tensor_groups — единственный источник (ADR-09 §1): dense
        // не содержит expert-групп, MoE — содержит.
        std::size_t experts = 0, kv = 0;
        for (const auto& g : d.groups) {
            if (g.role == stell::tiers::Role::Expert) ++experts;
            if (g.role == stell::tiers::Role::Kv) ++kv;
        }
        if (d.is_moe) CHECK(experts > 0);
        else CHECK(experts == 0);
        CHECK(kv == d.n_layers);
    }

    // Различия профилей (ADR-03): плот — F16, glm — NVFP4, qwen — W4A16.
    const ModelDescription llama = r.load_builtin("llama-class");
    const ModelDescription glm = r.load_builtin("glm-class");
    const ModelDescription qwen = r.load_builtin("qwen-moe-class");
    CHECK(!llama.is_moe && llama.dtype_weights == "F16");
    CHECK(glm.is_moe && glm.dtype_weights == "NVFP4" && glm.moe->n_experts == 128);
    CHECK(qwen.is_moe && qwen.dtype_weights == "W4A16" && qwen.moe->n_experts_active == 8);

    // Паттерны экспертов: glm — fused gate_up, qwen — split up/down (ADR-09 §1).
    bool glm_fused = false, qwen_split_up = false, qwen_split_down = false;
    for (std::size_t i = 0; i < glm.groups.size(); ++i)
        if (glm.group_source_patterns[i] == "layers.0.mlp.experts.0.gate_up.weight")
            glm_fused = true;
    for (std::size_t i = 0; i < qwen.groups.size(); ++i) {
        if (qwen.group_source_patterns[i] == "layers.0.mlp.experts.0.up.weight")
            qwen_split_up = true;
        if (qwen.group_source_patterns[i] == "layers.0.mlp.experts.0.down.weight")
            qwen_split_down = true;
    }
    CHECK(glm_fused);
    CHECK(qwen_split_up && qwen_split_down);

    // Неизвестная встроенная архитектура → RegistryError.
    CHECK_THROW(r.load_builtin("gpt-999"), RegistryError, "unknown builtin rejected");

    // Битый JSON-файл → RegistryError/DescriptionError (через load()).
    const std::string bad_path = "/tmp/stell_models_registry_bad.json";
    {
        std::FILE* f = std::fopen(bad_path.c_str(), "wb");
        std::fwrite("{\"schema\": \"1.0\", \"name\":", 1, 21, f);
        std::fclose(f);
    }
    CHECK_THROW(r.load(bad_path), DescriptionError, "broken JSON rejected");
}

TEST_MAIN("test_models_registry")

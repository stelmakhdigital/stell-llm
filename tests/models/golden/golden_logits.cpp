// ============================================================================
// Golden-тест логитов (ADR-09 §4) — round 2.
//
// Фикстуры golden сгенерированы НЕЗАВИСИМОЙ реализацией (Python-интерпретатор
// ops-графа, tests/models/golden/generate_golden.py) и сверены против
// второго референса — HF LlamaForCausalLM / MixtralForCausalLM (CPU float32,
// см. hf_crosscheck.py и meta-блоки в файлах): max_rel_err ~1.5e-7.
//
// Фикстуры: tests/models/golden/logits_<fixture>.json (схема
// golden/logits_ref.schema.json; режим greedy — argmax, T-сэмплинга нет;
// «T=1» bench-контракта PLAN.md §5 — ДРУГОЙ режим, см. ADR-09 «Уточнение»).
// Прогон engine (core-2) дописывает logits_tested; без поля — SKIP (не
// молчаливый успех). Три пути ADR-09 §4 (loader/stelltier/tiering_nvme_miss)
// обязаны давать те же логиты, что референс того же профиля.
//
// Правила сравнения: rel-ошибка по каждому элементу
// |ref − tested| / max(|ref|, |tested|, 1) ≤ tolerance_rel.
// ============================================================================
#include "stell/models/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "test_util.hpp"

using namespace stell::models;

namespace {

const char* kGoldenDir = STELL_GOLDEN_FIXTURE;

// Проверка фикстуры по схеме (обязательные поля ADR-09 §4 + самосогласованность).
void validate_fixture(const JsonValue& fx, const std::string& name) {
    CHECK_MSG(fx.get_string("schema") == "1.0", name + ": schema must be \"1.0\"");
    CHECK_MSG(!fx.get_string("model").empty(), name + ": model must be non-empty");
    const std::string& profile = fx.get_string("profile");  // F16 | NVFP4 | W4A16
    CHECK_MSG(profile == "F16" || profile == "NVFP4" || profile == "W4A16",
              name + ": profile enum");
    const std::string& path_kind = fx.get_string("path");
    CHECK_MSG(path_kind == "loader" || path_kind == "stelltier" ||
                  path_kind == "tiering_nvme_miss",
              name + ": path enum (ADR-09 §4)");
    CHECK_MSG(fx.get_uint("seed") == 0, name + ": seed must be 0");
    CHECK_MSG(fx.find("greedy")->bool_v == true, name + ": greedy must be true");
    CHECK_MSG(fx.get_double("tolerance_rel") > 0.0, name + ": tolerance_rel > 0");
    const std::vector<JsonValue>& toks = fx.get_array("prompt_tokens").array_v;
    CHECK_MSG(!toks.empty(), name + ": prompt_tokens non-empty");
    for (const auto& tok : toks)
        CHECK_MSG(tok.type == JsonValue::Type::Int && tok.int_v >= 0,
                  name + ": prompt_tokens must be non-negative ints");
    const std::vector<JsonValue>& ref = fx.get_array("logits_ref").array_v;
    CHECK_MSG(!ref.empty(), name + ": logits_ref non-empty");
    // Самосогласованность greedy: greedy_token (если задан) = argmax(logits_ref).
    const JsonValue* gt = fx.find("greedy_token");
    if (gt && gt->type == JsonValue::Type::Int) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < ref.size(); ++i) {
            const double ri = ref[i].type == JsonValue::Type::Double
                                  ? ref[i].double_v
                                  : static_cast<double>(ref[i].int_v);
            const double rb = ref[best].type == JsonValue::Type::Double
                                  ? ref[best].double_v
                                  : static_cast<double>(ref[best].int_v);
            if (ri > rb) best = i;  // ничья → меньший индекс (строгий >)
        }
        CHECK_MSG(static_cast<std::size_t>(gt->int_v) == best,
                  name + ": greedy_token must equal argmax(logits_ref)");
    }
}

void compare_logits(const JsonValue& fx, const std::string& name) {
    const double tol = fx.get_double("tolerance_rel");
    const std::vector<JsonValue>& ref = fx.get_array("logits_ref").array_v;
    const std::vector<JsonValue>& tested = fx.get_array("logits_tested").array_v;
    CHECK_MSG(ref.size() == tested.size(),
              name + ": logits_tested size mismatch");
    double max_rel = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double rv = ref[i].type == JsonValue::Type::Double
                              ? ref[i].double_v
                              : static_cast<double>(ref[i].int_v);
        const double tv = tested[i].type == JsonValue::Type::Double
                              ? tested[i].double_v
                              : static_cast<double>(tested[i].int_v);
        const double err = std::fabs(rv - tv);
        const double mag = std::max({std::fabs(rv), std::fabs(tv), 1.0});
        const double rel = err / mag;
        max_rel = std::max(max_rel, rel);
        CHECK_MSG(rel <= tol,
                  name + ": rel err " + std::to_string(rel) + " > tolerance " +
                      std::to_string(tol) + " at element " + std::to_string(i));
    }
    std::printf("golden %s: OK (max_rel=%.3e, tol=%g)\n", name.c_str(), max_rel, tol);
}

}  // namespace

void body() {
    // Enumerate logits_*.json в каталоге golden (файлы коммитятся;
    // прогоны engine/core-2 дописывают logits_tested).
    std::vector<std::string> files;
    {
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(kGoldenDir, ec)) {
            const std::string name = de.path().filename().string();
            // logits_*.json — golden-фикстуры; logits_ref.schema.json —
            // ДОКУМЕНТ схемы (не фикстура): исключаем.
            if (name.rfind("logits_", 0) == 0 && name.size() > 5 &&
                name.substr(name.size() - 5) == ".json" &&
                name.find(".schema.") == std::string::npos)
                files.push_back(name);
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        std::printf(
            "SKIP: golden-фикстуры logits_*.json отсутствуют в %s — pending "
            "(см. README в tests/models/golden/)\n",
            kGoldenDir);
    std::size_t compared = 0, skipped = 0;
    for (const auto& name : files) {
        const std::string path = std::string(kGoldenDir) + "/" + name;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open golden fixture: " + path);
        std::string json;
        int c;
        while ((c = std::fgetc(f)) != EOF) json.push_back(static_cast<char>(c));
        std::fclose(f);
        const JsonValue fx = json_parse(json);
        validate_fixture(fx, name);
        if (fx.find("logits_tested") == nullptr) {
            ++skipped;
            std::printf(
                "SKIP: %s — референс готов (независимый Python-интерпретатор + "
                "HF-кроссчек, см. meta), прогона engine ещё нет: сравнение "
                "включится, когда core-2 допишет logits_tested (схема: "
                "logits_ref.schema.json)\n",
                name.c_str());
            continue;
        }
        ++compared;
        compare_logits(fx, name);
    }
    if (compared)
        std::printf("golden: сравнено %zu фикстур, skip %zu\n", compared, skipped);
}

TEST_MAIN("test_models_golden_logits")

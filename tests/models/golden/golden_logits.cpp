// ============================================================================
// Golden-тест логитов (ADR-09 §4) — каркас раунда 1.
//
// Контракт фикстуры (tests/models/golden/logits_ref.json, схема —
// golden/logits_ref.schema.json): фиксированный вход (промпт-токены, seed=0,
// greedy T=1), эталон логитов последнего слоя против HF-референса.
//
// РЕАЛЬНОГО ПРОГОНА ПРОТИВ HF-РЕФЕРЕНСОВ НЕТ (весов локально нет, сервер
// занят): фикстуры отсутствуют → тест честно печатает SKIP и завершается
// успехом, не выдавая себя за проверку. Когда появятся фикстуры
// (logits_ref.json с полем logits_tested от прогона engine) — тест
// автоматически включит сравнение: rel-ошибка ≤ tolerance_rel для каждого
// токена. Три пути (ADR-09 §4): обычный лоадер, stell-tier, tiering с
// промахами NVMe — фиксируются отдельными полями `path`.
// ============================================================================
#include "stell/models/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "test_util.hpp"

using namespace stell::models;

namespace {

const char* kFixturePath = STELL_GOLDEN_FIXTURE "/logits_ref.json";
const char* kSchemaPath = STELL_GOLDEN_FIXTURE "/logits_ref.schema.json";

bool file_exists(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// Проверка фикстуры по схеме (поля ADR-09 §4 + допуски).
void validate_fixture(const JsonValue& fx) {
    CHECK(fx.get_string("schema") == "1.0");
    CHECK(!fx.get_string("model").empty());
    const std::string& profile = fx.get_string("profile");  // F16 | NVFP4 | W4A16
    CHECK(profile == "F16" || profile == "NVFP4" || profile == "W4A16");
    const std::string& path_kind = fx.get_string("path");
    CHECK(path_kind == "loader" || path_kind == "stelltier" || path_kind == "tiering_nvme_miss");
    CHECK(fx.get_uint("seed") == 0);
    CHECK(fx.find("greedy")->bool_v == true);
    CHECK(!fx.get_array("prompt_tokens").array_v.empty());
    CHECK(!fx.get_array("logits_ref").array_v.empty());
    for (const auto& tok : fx.get_array("prompt_tokens").array_v)
        CHECK(tok.type == JsonValue::Type::Int && tok.int_v >= 0);
}

void compare_logits(const JsonValue& fx) {
    const double tol = fx.get_double("tolerance_rel");
    const JsonValue& ref = fx.get_array("logits_ref");
    const JsonValue& tested = fx.get_array("logits_tested");
    CHECK(ref.array_v.size() == tested.array_v.size());
    for (std::size_t i = 0; i < ref.array_v.size(); ++i) {
        const double rv = ref.array_v[i].type == JsonValue::Type::Double
                              ? ref.array_v[i].double_v
                              : static_cast<double>(ref.array_v[i].int_v);
        const double tv = tested.array_v[i].type == JsonValue::Type::Double
                              ? tested.array_v[i].double_v
                              : static_cast<double>(tested.array_v[i].int_v);
        const double err = std::fabs(rv - tv);
        const double mag = std::max({std::fabs(rv), std::fabs(tv), 1.0});
        CHECK(err / mag <= tol);
    }
}

}  // namespace

void body() {
    if (!file_exists(kFixturePath)) {
        std::printf(
            "SKIP: golden-фикстура %s отсутствует — прогон против HF-референсов "
            "pending (план: генерация референсов на сервере, ssh -p 2214 192.168.1.114; "
            "см. README в tests/models/golden/). Каркас сравнения готов и включится "
            "автоматически при появлении logits_ref.json с полем logits_tested.\n",
            kFixturePath);
        return;
    }
    // Фикстура есть: валидируем схему; если logits_tested не заполнен — SKIP.
    std::FILE* f = std::fopen(kFixturePath, "rb");
    std::string json;
    int c;
    while ((c = std::fgetc(f)) != EOF) json.push_back(static_cast<char>(c));
    std::fclose(f);
    const JsonValue fx = json_parse(json);
    validate_fixture(fx);
    if (fx.find("logits_tested") == nullptr) {
        std::printf(
            "SKIP: в фикстуре нет logits_tested — референс готов, прогона engine ещё нет "
            "(golden сравнение включится при появлении прогонов; схема: %s)\n",
            kSchemaPath);
        return;
    }
    compare_logits(fx);
}

TEST_MAIN("test_models_golden_logits")

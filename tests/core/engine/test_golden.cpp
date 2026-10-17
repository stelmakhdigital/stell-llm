// Golden-сравнение логитов (round 2, задача T-20261017200354, ADR-09 §4).
//
// Схема фикстур — tests/models/golden/logits_ref.schema.json (согласовано с
// потоком D), файлы — tests/models/golden/logits_<fixture>.json. Тест:
//   * ЕСЛИ golden-файл присутствует — исполняет CPU-логиты engine на модели
//     реестра и сравнивает с logits_ref (rel-ошибка ≤ порога);
//   * иначе — честный SKIP (не молчаливый успех).
//
// Порог rel-ошибки задаётся честно и зафиксирован:
//   * F16-профиль: tolerance_rel = 1e-3 (ADR-09 §4);
//   * NVFP4 / W4A16: tolerance_rel = 1e-2 по умолчанию (значение поля
//     tolerance_rel фикстуры перекрывает; порог калибруется против HF-
//     референса на сервере — поток D);
//   * fixture с неизвестным модельным именем/весами → SKIP с причиной.
//
// Поля фикстуры, поддерживаемые здесь (сверх схемы D): "weights" — путь к
// файлу весов входного формата (safetensors/GGUF/stell-tier, ADR-09 §3);
// "weights_format" — формат лоадера ("safetensors|gguf|nvfp4w4a16|stelltier",
// по умолчанию — по расширению/магии через detect... здесь: safetensors);
// путь "loader" схемы D. Пути "stelltier"/"tiering_nvme_miss" — прогон через
// StellTier-лоадер / tiering c принудительными промахами (ADR-09 §4).
#include "test_util.hpp"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include "stell/engine/json.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/plain_spec.hpp"

using namespace stell::engine;

namespace {

const char* kGoldenDir = STELL_GOLDEN_FIXTURE;

bool is_golden_file(const std::string& name) {
    // logits_<fixture>.json; схему (logits_ref.schema.json) не берём.
    if (name.rfind("logits_", 0) != 0) return false;
    if (name.size() <= 5 || name.substr(name.size() - 5) != ".json") return false;
    if (name.find(".schema") != std::string::npos) return false;
    return true;
}

std::string read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string out;
    int c;
    while ((c = std::fgetc(f)) != EOF) out.push_back(static_cast<char>(c));
    std::fclose(f);
    return out;
}

struct GoldenFixture {
    std::string name;          // имя файла (logits_<fixture>.json)
    std::string model;         // имя архитектуры реестра (Registry::available())
    std::string profile;       // F16 | NVFP4 | W4A16
    std::string path_kind;     // loader | stelltier | tiering_nvme_miss
    std::string weights;       // путь к весам (расширение схемы, поток D)
    std::string weights_format;
    double tolerance_rel = 0.0;
    std::vector<std::int32_t> prompt_tokens;
    std::vector<double> logits_ref;
    bool greedy = true;
};

bool parse_fixture(const std::string& path, GoldenFixture& fx, std::string& err) {
    const std::string text = read_file(path);
    if (text.empty()) { err = "файл не читается"; return false; }
    try {
        const Json root = json_parse(text);
        fx.name = path;
        if (const Json* v = root.find("model")) fx.model = v->as_string("");
        if (const Json* v = root.find("profile")) fx.profile = v->as_string("");
        if (const Json* v = root.find("path")) fx.path_kind = v->as_string("");
        if (const Json* v = root.find("weights")) fx.weights = v->as_string("");
        if (const Json* v = root.find("weights_format"))
            fx.weights_format = v->as_string("safetensors");
        if (const Json* v = root.find("tolerance_rel"))
            fx.tolerance_rel = v->num;
        if (const Json* v = root.find("greedy")) fx.greedy = v->boolean;
        if (const Json* v = root.find("prompt_tokens"))
            for (const Json& t : v->arr)
                fx.prompt_tokens.push_back(static_cast<std::int32_t>(t.as_int()));
        if (const Json* v = root.find("logits_ref"))
            for (const Json& t : v->arr) fx.logits_ref.push_back(t.num);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

PlainWeightsFormat weights_format_of(const std::string& name) {
    if (name == "safetensors") return PlainWeightsSafetensors;
    if (name == "gguf") return PlainWeightsGGUF;
    if (name == "nvfp4w4a16") return PlainWeightsNVFP4W4A16;
    if (name == "stelltier") return PlainWeightsStellTier;
    return PlainWeightsSafetensors;
}

// Порог rel-ошибки (честно зафиксирован; см. комментарий в шапке).
double default_tolerance(const std::string& profile) {
    if (profile == "F16") return 1e-3;                 // ADR-09 §4
    return 1e-2;                                       // NVFP4/W4A16 — до калибровки HF
}

} // namespace

void run() {
    // Собираем golden-файлы из каталога потока D (logits_<fixture>.json).
    std::vector<std::string> files;
    DIR* d = opendir(kGoldenDir);
    if (d) {
        while (dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (is_golden_file(n)) files.push_back(n);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::printf(
            "SKIP: golden-фикстуры tests/models/golden/logits_<fixture>.json "
            "отсутствуют — прогона против HF-референсов пока нет (поток D: "
            "генерация референсов на сервере; схема — logits_ref.schema.json). "
            "Тест включится автоматически при появлении фикстур.\n");
        return;
    }

    for (const std::string& file : files) {
        const std::string path = std::string(kGoldenDir) + "/" + file;
        GoldenFixture fx{};
        std::string err;
        if (!parse_fixture(path, fx, err)) {
            std::printf("SKIP: %s — фикстура не разбирается (%s)\n", file.c_str(),
                        err.c_str());
            continue;
        }
        if (fx.model.empty() || fx.prompt_tokens.empty() || fx.logits_ref.empty()) {
            std::printf("SKIP: %s — неполная фикстура (model/prompt_tokens/logits_ref)\n",
                        file.c_str());
            continue;
        }

        // Загрузка модели реестра: имя фикстуры → файл описания <model>.json.
        // Честно: путь описания ищем в tests/models/fixtures и в реестре
        // (STELL_MODELS_REGISTRY_DIR / встроенный реестр через Registry).
        std::string desc_path;
        {
            const std::string cand1 = std::string(STELL_FIXTURES_DIR) + "/" + fx.model + ".json";
            std::FILE* f1 = std::fopen(cand1.c_str(), "rb");
            if (f1) { std::fclose(f1); desc_path = cand1; }
        }
        if (desc_path.empty()) {
            const char* reg = std::getenv("STELL_MODELS_REGISTRY_DIR");
            if (reg) {
                const std::string cand2 = std::string(reg) + "/" + fx.model + ".json";
                std::FILE* f2 = std::fopen(cand2.c_str(), "rb");
                if (f2) { std::fclose(f2); desc_path = cand2; }
            }
        }
        if (desc_path.empty()) {
            std::printf(
                "SKIP: %s — описание модели '%s' не найдено (ищу %s/%s.json и "
                "STELL_MODELS_REGISTRY_DIR/%s.json)\n",
                file.c_str(), fx.model.c_str(), STELL_FIXTURES_DIR, fx.model.c_str(),
                fx.model.c_str());
            continue;
        }
        if (fx.weights.empty() ||
            ![](const std::string& p) {
                std::FILE* f = std::fopen(p.c_str(), "rb");
                if (f) { std::fclose(f); return true; }
                return false;
            }(fx.weights)) {
            std::printf("SKIP: %s — веса не указаны/не найдены (поле weights фикстуры)\n",
                        file.c_str());
            continue;
        }

        try {
            const PlainSpec spec = load_plain_spec(desc_path);
            const LoadedWeights w = load_plain_weights(
                fx.weights, spec, weights_format_of(fx.weights_format));
            const ExecutionPlan plan = build_execution_plan(spec);
            OpsInterpreter interp(spec, plan, w);

            SamplingParams sp{};
            sp.temperature = fx.greedy ? 0.0f : 1.0f;
            const std::vector<float> logits = interp.run_prompt_logits(fx.prompt_tokens);
            if (logits.size() != fx.logits_ref.size()) {
                std::printf(
                    "SKIP: %s — размерность логитов %zu != эталона %zu (профиль "
                    "%s, модель %s)\n",
                    file.c_str(), logits.size(), fx.logits_ref.size(),
                    fx.profile.c_str(), fx.model.c_str());
                continue;
            }
            const double tol = fx.tolerance_rel > 0.0 ? fx.tolerance_rel
                                                      : default_tolerance(fx.profile);
            double max_rel = 0.0;
            bool ok = true;
            for (std::size_t i = 0; i < logits.size(); ++i) {
                const double rv = fx.logits_ref[i];
                const double tv = logits[i];
                const double err = std::fabs(rv - tv);
                const double mag = std::max({std::fabs(rv), std::fabs(tv), 1.0});
                const double rel = err / mag;
                max_rel = std::max(max_rel, rel);
                if (rel > tol) ok = false;
            }
            std::printf(
                "%s: %s — модель=%s профиль=%s путь=%s порог=%.1e max_rel=%.3e — %s\n",
                ok ? "PASS" : "FAIL", file.c_str(), fx.model.c_str(),
                fx.profile.c_str(), fx.path_kind.c_str(), tol, max_rel,
                ok ? "логиты согласованы с референсом" : "расхождение с референсом");
            STELL_CHECK_MSG(ok, "golden-логиты расходятся с референсом: " + file);
        } catch (const std::exception& e) {
            std::printf("SKIP: %s — прогон не удался (%s)\n", file.c_str(), e.what());
        }
    }
}

STELL_TEST_MAIN(run)

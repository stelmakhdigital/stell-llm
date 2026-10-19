// ABI-символ stell_logits (round 2, задача T-20261017200354 [R1], ADR-11 §1):
// логиты финального слоя (до softmax) на готовых token id через C ABI.
//
// Тест:
//   * mini_dense / mini_moe на prompt из golden-фикстуры
//     (tests/models/golden/logits_mini_*.json) — логиты ("last") совпадают с
//     logits_ref с rel ≤ 1e-6;
//   * режим "all" покрывает ≥2 позиций: n_positions == |prompt| (≥ 2),
//     последняя строка == строке "last" (побитово: одна и та же float-строка
//     после одинаковой %.9g-сериализации), строки позиций различаются;
//   * отсутствие "positions" → дефолт "last";
//   * ошибки: runner:"demo" → -1 + {"error":...}; пустой промпт → -1;
//     неизвестное "positions" → -1; строки free() на вызывающем.
//
// Разрешение весов — как в test_golden.cpp (интеграционный фикс): путь из
// имени фикстуры → fixtures/<name>.safetensors; формат: F16 → safetensors,
// NVFP4/W4A16 → nvfp4w4a16 (квантование на read, ADR-03).
#include "test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "stell/engine/capi.h"
#include "stell/engine/json.hpp"

using stell::engine::Json;
using stell::engine::json_parse;

namespace {

const char* kGoldenDir = STELL_GOLDEN_FIXTURE;
const char* kFixtures = STELL_FIXTURES_DIR;

struct StrOut {
    char* p = nullptr;
    ~StrOut() { if (p) std::free(p); }
    const char* get() const { return p ? p : ""; }
};

struct GoldenFixture {
    std::string model;
    std::string profile;
    std::vector<std::int32_t> prompt;
    std::vector<double> logits_ref;
};

GoldenFixture load_golden(const char* name) {
    GoldenFixture fx{};
    StrOut raw;
    // читаем файл напрямую (фикстура — тестовый ресурс потока D)
    std::FILE* f = std::fopen((std::string(kGoldenDir) + "/" + name).c_str(), "rb");
    STELL_CHECK_MSG(f != nullptr, name);
    std::string text;
    int c;
    while ((c = std::fgetc(f)) != EOF) text.push_back(static_cast<char>(c));
    std::fclose(f);
    const Json root = json_parse(text);
    if (const Json* v = root.find("model")) fx.model = v->as_string("");
    if (const Json* v = root.find("profile")) fx.profile = v->as_string("");
    if (const Json* v = root.find("prompt_tokens"))
        for (const Json& t : v->arr)
            fx.prompt.push_back(static_cast<std::int32_t>(t.as_int()));
    if (const Json* v = root.find("logits_ref"))
        for (const Json& t : v->arr) fx.logits_ref.push_back(t.num);
    STELL_CHECK(!fx.model.empty());
    STELL_CHECK(fx.prompt.size() >= 2);          // "all" должен покрыть ≥2 позиций
    STELL_CHECK(!fx.logits_ref.empty());
    return fx;
}

// weights_format по профилю (тест golden: F16 → safetensors, иначе nvfp4w4a16)
std::string weights_format_of(const std::string& profile) {
    return profile == "F16" ? "safetensors" : "nvfp4w4a16";
}

// вызов stell_logits + разбор ответа; rc наружу
Json call_logits(stell_engine_t* e, const std::string& req, int* rc_out) {
    StrOut out;
    const int rc = stell_logits(e, req.c_str(), &out.p);
    if (rc_out) *rc_out = rc;
    return json_parse(std::string(out.get()));
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double err = std::fabs(a[i] - b[i]);
        const double mag = std::max({std::fabs(a[i]), std::fabs(b[i]), 1.0});
        m = std::max(m, err / mag);
    }
    return m;
}

void run_case(const char* golden_name) {
    const GoldenFixture fx = load_golden(golden_name);
    const std::string base = std::string(golden_name).substr(7);   // logits_<base>.json
    const std::string fixture = base.substr(0, base.size() - 5);   // mini_dense / mini_moe

    // --- engine (runner:"ops") над фикстурой той же модели -------------------
    const std::string weights =
        std::string(kFixtures) + "/" + fixture + ".safetensors";
    std::string cfg = std::string("{\"runner\":\"ops\",") +
        "\"engine\":{\"max_slots\":4},"
        "\"model\":{\"description\":\"" + std::string(kFixtures) + "/" +
        fixture + ".json\",\"weights\":\"" + weights + "\",\"weights_format\":\"" +
        weights_format_of(fx.profile) + "\",\"tokenizer\":\"byte\"}}";
    stell_engine_t* e = nullptr;
    char* msg = nullptr;
    const int rc = stell_engine_create(cfg.c_str(), &e);
    STELL_CHECK_MSG(rc == 0, golden_name);
    if (msg) std::free(msg);

    // --- "last": логиты последней позиции == golden (rel ≤ 1e-6) -------------
    std::string req_last = "{\"prompt_tokens\":[";
    for (std::size_t i = 0; i < fx.prompt.size(); ++i) {
        if (i) req_last += ',';
        req_last += std::to_string(fx.prompt[i]);
    }
    req_last += "],\"positions\":\"last\"}";
    int rc_last = -2;
    const Json last = call_logits(e, req_last, &rc_last);
    STELL_CHECK_MSG(rc_last == 0, golden_name);

    std::vector<double> got;
    {
        const Json* arr = last.find("logits");
        STELL_CHECK(arr != nullptr && arr->t == Json::T::Array);
        STELL_CHECK(arr->arr.size() == 1);       // одна позиция
        for (const Json& t : arr->arr[0].arr) got.push_back(t.num);
    }
    STELL_CHECK(got.size() == fx.logits_ref.size());
    const double rel = max_rel(fx.logits_ref, got);
    STELL_CHECK_MSG(rel <= 1e-6,
                    (std::string(golden_name) + ": rel=" + std::to_string(rel))
                        .c_str());
    std::printf("%s: stell_logits last — max_rel=%.3e (порог 1e-6)\n",
                golden_name, rel);

    // n_positions / n_vocab
    STELL_CHECK(last.find("n_positions")->as_int() == 1);
    const std::int64_t n_vocab = last.find("n_vocab")->as_int();
    STELL_CHECK(n_vocab == static_cast<std::int64_t>(got.size()));

    // --- дефолт: без "positions" → "last" ------------------------------------
    std::string req_default = "{\"prompt_tokens\":[";
    for (std::size_t i = 0; i < fx.prompt.size(); ++i) {
        if (i) req_default += ',';
        req_default += std::to_string(fx.prompt[i]);
    }
    req_default += "]}";
    int rc_def = -2;
    const Json def = call_logits(e, req_default, &rc_def);
    STELL_CHECK(rc_def == 0);
    // (сравниваем поле logits напрямую)
    {
        const Json* a = def.find("logits");
        const Json* b = last.find("logits");
        STELL_CHECK(a && b && a->arr.size() == 1 && b->arr.size() == 1);
        for (std::size_t i = 0; i < a->arr[0].arr.size(); ++i)
            STELL_CHECK(a->arr[0].arr[i].num == b->arr[0].arr[i].num);
    }

    // --- "all": все позиции промпта ------------------------------------------
    {
        std::string req_all = "{\"prompt_tokens\":[";
        for (std::size_t i = 0; i < fx.prompt.size(); ++i) {
            if (i) req_all += ',';
            req_all += std::to_string(fx.prompt[i]);
        }
        req_all += "],\"positions\":\"all\"}";
        int rc2 = -2;
        const Json all2 = call_logits(e, req_all, &rc2);
        STELL_CHECK(rc2 == 0);
        const Json* rows = all2.find("logits");
        STELL_CHECK(rows != nullptr && rows->t == Json::T::Array);
        // покрывает ≥2 позиций: P == |prompt| ≥ 2
        STELL_CHECK(rows->arr.size() == fx.prompt.size());
        STELL_CHECK(rows->arr.size() >= 2);
        STELL_CHECK(all2.find("n_positions")->as_int() ==
                    static_cast<std::int64_t>(fx.prompt.size()));
        STELL_CHECK(all2.find("n_vocab")->as_int() == n_vocab);
        // последняя строка "all" == строка "last" (побитово: тот же float
        // после одинаковой %.9g-сериализации)
        for (std::size_t i = 0; i < rows->arr.back().arr.size(); ++i)
            STELL_CHECK(rows->arr.back().arr[i].num == got[i]);
        // строки позиций различаются (не продублированная последняя строка)
        bool any_diff = false;
        for (std::size_t p = 1; p < rows->arr.size(); ++p)
            for (std::size_t i = 0; i < rows->arr[p].arr.size(); ++i)
                if (rows->arr[p].arr[i].num != rows->arr[0].arr[i].num) {
                    any_diff = true;
                    break;
                }
        STELL_CHECK_MSG(any_diff, "режим all покрывает ≥2 различных позиций");
    }

    // --- ошибки ---------------------------------------------------------------
    {   // неизвестное positions
        int rcb = -2;
        const Json err = call_logits(e, "{\"prompt_tokens\":[1],\"positions\":\"x\"}", &rcb);
        STELL_CHECK(rcb == -1);
        STELL_CHECK(err.find("error") != nullptr);
    }
    {   // пустой промпт
        int rcb = -2;
        const Json err = call_logits(e, "{\"prompt_tokens\":[]}", &rcb);
        STELL_CHECK(rcb == -1);
        STELL_CHECK(err.find("error") != nullptr);
    }
    {   // prompt_tokens не массив
        int rcb = -2;
        const Json err = call_logits(e, "{\"prompt_tokens\":1}", &rcb);
        STELL_CHECK(rcb == -1);
        STELL_CHECK(err.find("error") != nullptr);
    }

    stell_engine_destroy(e);
}

void run_demo_error_case() {
    // runner:"demo" не имеет модели: честный отказ (не заглушка)
    stell_engine_t* e = nullptr;
    STELL_CHECK(stell_engine_create(
        "{\"runner\":\"demo\",\"engine\":{\"max_slots\":4}}", &e) == 0);
    int rc = -2;
    const Json err = call_logits(e, "{\"prompt_tokens\":[1,2]}", &rc);
    STELL_CHECK(rc == -1);
    STELL_CHECK(err.find("error") != nullptr);
    stell_engine_destroy(e);

    // деградация аргументов
    STELL_CHECK(stell_logits(nullptr, "{}", nullptr) == -1);
    stell_engine_t* e2 = nullptr;
    STELL_CHECK(stell_engine_create(
        "{\"runner\":\"demo\",\"engine\":{\"max_slots\":4}}", &e2) == 0);
    STELL_CHECK(stell_logits(e2, nullptr, nullptr) == -1);
    stell_engine_destroy(e2);
}

} // namespace

void run() {
    run_demo_error_case();
    run_case("logits_mini_dense.json");
    run_case("logits_mini_moe.json");
}

STELL_TEST_MAIN(run)

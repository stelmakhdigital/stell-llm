// ABI-контракт round 8: stell_spec_step (спекуляция на реальном runner'е) и
// spec-поля снапшота stell_metrics (features.spec_draft, latency.tok_s.
// spec_decode, spec.accept_rate/n_draft) — снапшот, который bench-харнесс
// уже читает (src/bench/stell_bench/harness.py).
#include "test_util.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "stell/engine/capi.h"
#include "stell/engine/json.hpp"

using namespace stell::engine;

namespace {

const std::string kFixtures = STELL_FIXTURES_DIR;

std::string spec_engine_cfg() {
    return std::string("{\"runner\":\"ops\",") +
        "\"model\":{\"description\":\"" + kFixtures + "/mini_moe.json\"," +
        "\"weights\":\"" + kFixtures + "/mini_moe.safetensors\"}," +
        "\"spec\":{\"enabled\":true,\"n_draft\":4,\"accept_target\":0.5," +
        "\"draft_model\":{\"description\":\"" + kFixtures + "/mini_dense.json\"," +
        "\"weights\":\"" + kFixtures + "/mini_dense.safetensors\"}}}";
}

} // namespace

void run() {
    stell_engine_t* e = nullptr;
    STELL_CHECK_MSG(stell_engine_create(spec_engine_cfg().c_str(), &e) == 0 && e,
                    "engine с runner=ops + spec создаётся");

    // Снапшот ДО декода: spec-поля присутствуют (честные нули/false —
    // «контур вооружён, но не использовался»; bench-контракт читает ровно их).
    {
        char* js = nullptr;
        STELL_CHECK(stell_metrics(e, &js) == 0 && js);
        Json snap = json_parse(js);
        const Json* f = snap.find("features");
        STELL_CHECK_MSG(f && f->find("spec_draft"),
                        "features.spec_draft обязан быть в снапшоте");
        STELL_CHECK(f->find("spec_draft")->t == Json::T::Bool);
        const Json* lat = snap.find("latency");
        STELL_CHECK_MSG(lat && lat->find("tok_s") &&
                            lat->find("tok_s")->find("spec_decode"),
                        "latency.tok_s.spec_decode обязан быть в снапшоте");
        const Json* sp = snap.find("spec");
        STELL_CHECK_MSG(sp && sp->find("accept_rate") && sp->find("n_draft"),
                        "spec.accept_rate/n_draft обязаны быть в снапшоте");
        std::free(js);
    }

    // Спекулятивный декод: токены побитово == plain greedy цели (сверка через
    // stell_logits: argmax логитов последней позиции совпадает с каждым
    // токеном спек-пути на соответствующем префиксе).
    {
        char* js = nullptr;
        const std::string req = "{\"prompt_tokens\":[1,2,3,4,5],\"n_steps\":12}";
        STELL_CHECK_MSG(stell_spec_step(e, req.c_str(), &js) == 0 && js,
                        "stell_spec_step: успешный вызов");
        Json out = json_parse(js);
        const Json* toks = out.find("tokens");
        STELL_CHECK_MSG(toks && toks->t == Json::T::Array && toks->arr.size() == 12,
                        "stell_spec_step: 12 токенов");
        // Побитовая сверка каждого токена с greedy-путём цели: argmax логитов
        // префикса [prompt + tokens[0..i-1]] == tokens[i].
        for (std::size_t i = 0; i < toks->arr.size(); ++i) {
            std::vector<std::int32_t> prefix = {1, 2, 3, 4, 5};
            for (std::size_t j = 0; j < i; ++j)
                prefix.push_back(static_cast<std::int32_t>(toks->arr[j].as_int()));
            char* lj = nullptr;
            std::string p = "{\"prompt_tokens\":[";
            for (std::size_t k = 0; k < prefix.size(); ++k)
                p += (k ? "," : "") + std::to_string(prefix[k]);
            p += "],\"positions\":\"last\"}";
            STELL_CHECK_MSG(stell_logits(e, p.c_str(), &lj) == 0 && lj,
                            "stell_logits на префиксе");
            Json logits = json_parse(lj);
            const Json* rows = logits.find("logits");
            STELL_CHECK(rows && rows->t == Json::T::Array && !rows->arr.empty());
            const Json& row = rows->arr[rows->arr.size() - 1];
            float best = row.arr[0].num;
            std::int64_t arg = 0;
            for (std::size_t k = 1; k < row.arr.size(); ++k)
                if (row.arr[k].num > best) { best = row.arr[k].num; arg = static_cast<std::int64_t>(k); }
            STELL_CHECK_MSG(static_cast<std::int64_t>(toks->arr[i].as_int()) == arg,
                            "токен " + std::to_string(i) + " спек-пути != greedy цели");
            std::free(lj);
        }
        const Json* en = out.find("enabled");
        STELL_CHECK(en && en->t == Json::T::Bool);
        std::free(js);
    }

    stell_engine_destroy(e);
    std::puts("spec capi: снапшот + stell_spec_step побитово == greedy — OK");
}

STELL_TEST_MAIN(run)

// [R3] round 5 (planner Q4, ADR-09 «Уточнение round 5» — per-layer план):
// описание со СМЕШАННЫМИ слоями (синтетика): слой 0 — dense (ffn.{gate,up,down}),
// слои 1–2 — MoE (mlp.router + e{000..003}); ops-граф общий
// [rmsnorm, attention, ffn, moe_gather_scatter], ops_runner маршрутизирует
// per-layer. Референс per-layer плана — эквивалентная композиция: dense-слой
// с ffn = w·x → silu(w·x)·(up·x) → down ПОБИТОВО равен MoE-слою с ОДНИМ
// экспертом и k_top=1 (softmax одного скорa ≡ 1, probs ≡ 1, scatter_add
// ≡ тождественность) — поэтому смешанный план M (dense L0) сверяется с
// однородным планом H (L0 — одно-экспертный MoE с ТЕМИ ЖЕ весами ffn):
// логиты M == H побитово на всех позициях. Любая неверная маршрутизация
// (ffn на MoE-слое, пропуск ffn на dense, двойное исполнение) даёт расхождение.
// Непокрытые конфигурации — честный SchemaMismatch (план не интерпретируется
// молча); однородные дефолты (mini_dense/mini_moe) не меняются (их тесты —
// engine/ops_dense, engine/ops_moe).
#include "test_util.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"

using namespace stell::engine;
using namespace stell_core_test;

namespace {

constexpr std::uint32_t kVocab = 16;
constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kExperts = 4;

std::vector<PlainGroup> common_groups(bool layer0_dense) {
    // Общие группы (embedding/norm/head/attn/kv); MoE-слои 1..2 — всегда;
    // слой 0 — dense ffn либо одно-экспертный MoE с теми же весами ffn.
    std::vector<PlainGroup> out;
    std::uint32_t id = 0;
    const auto add = [&](PlainGroup g) { g.id = id++; out.push_back(std::move(g)); };

    add(make_group(id, "embed", "F16", {kVocab, 8}, "embedding.weight"));
    add(make_group(id, "norm_final", "F32", {8}, "model.norm.weight"));
    add(make_group(id, "head", "F16", {kVocab, 8}, "head.weight"));
    for (std::size_t l = 0; l < kLayers; ++l) {
        const std::string p = "l" + std::to_string(1000 + l).substr(1, 3) + ".";
        const std::string ls = std::to_string(l);
        add(make_group(id, p + "attn.norm", "F32", {8},
                       "layers." + ls + ".input_layernorm.weight"));
        add(make_group(id, p + "attn.wq", "F16", {8, 8},
                       "layers." + ls + ".self_attn.q_proj.weight"));
        add(make_group(id, p + "attn.wk", "F16", {4, 8},
                       "layers." + ls + ".self_attn.k_proj.weight"));
        add(make_group(id, p + "attn.wv", "F16", {4, 8},
                       "layers." + ls + ".self_attn.v_proj.weight"));
        add(make_group(id, p + "attn.wo", "F16", {8, 8},
                       "layers." + ls + ".self_attn.o_proj.weight"));
        add(make_group(id, p + "kv", "F16", {4, 8},
                       "layers." + ls + ".kv_cache", /*kv=*/true));
        const bool dense = layer0_dense && l == 0;
        if (dense) {
            add(make_group(id, p + "ffn.gate", "F16", {16, 8},
                           "layers." + ls + ".mlp.gate_proj.weight"));
            add(make_group(id, p + "ffn.up", "F16", {16, 8},
                           "layers." + ls + ".mlp.up_proj.weight"));
            add(make_group(id, p + "ffn.down", "F16", {8, 16},
                           "layers." + ls + ".mlp.down_proj.weight"));
        } else {
            const std::size_t n_exp = (l == 0) ? 1 : kExperts;
            add(make_group(id, p + "mlp.router", "F32",
                           {std::int64_t(n_exp), 8},
                           "layers." + ls + ".mlp.router.weight"));
            for (std::size_t e = 0; e < n_exp; ++e) {
                char eb[32];
                std::snprintf(eb, sizeof eb, "e%03zu", e);
                // слой 0: веса ffn переиспользованы одно-экспертным MoE —
                // те же имена тензоров-источников (== побитовые значения)
                const std::string base = (l == 0)
                    ? "layers." + ls + ".mlp"
                    : "layers." + ls + ".mlp.experts." + std::to_string(e);
                add(make_group(id, p + eb + ".gate", "F16", {16, 8},
                               base + ".gate_proj.weight"));
                add(make_group(id, p + eb + ".up", "F16", {16, 8},
                               base + ".up_proj.weight"));
                add(make_group(id, p + eb + ".down", "F16", {8, 16},
                               base + ".down_proj.weight"));
            }
        }
    }
    return out;
}

PlainSpec make_spec(bool layer0_dense) {
    PlainSpec s{};
    s.name = layer0_dense ? "mini-mixed-test" : "mini-mixed-ref-moe";
    s.is_moe = true;                       // смешанный класс — class="moe"
    s.dtype_weights = "F16";
    s.vocab_size = kVocab;
    s.d_model = 8;
    s.n_layers = kLayers;
    s.n_heads = 2;
    s.d_head = 4;
    s.n_kv_heads = 1;
    s.d_kv = 4;
    s.n_experts = kExperts;
    s.n_experts_active = 2;
    s.expert_d_ff = 16;
    s.router = "softmax-topk";
    s.ops = base_ops(/*with_ffn=*/layer0_dense, /*with_moe=*/true);
    s.groups = common_groups(layer0_dense);
    return s;
}

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load(const PlainSpec& spec, const char* path) {
    Loaded l{};
    l.spec = spec;
    write_safetensors(path, fixture_tensors(spec));
    l.weights = load_plain_weights(path, spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(spec);
    return l;
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

// Ожидание честного SchemaMismatch при загрузке/исполнении плана.
bool throws_schema_mismatch(const PlainSpec& spec, const char* path,
                            const char* tag) {
    try {
        const Loaded l = load(spec, path);
        OpsInterpreter interp(l.spec, l.plan, l.weights);
        (void)interp.run_prompt_logits({1, 2, 3});
    } catch (const std::invalid_argument& e) {
        if (!contains(e.what(), "SchemaMismatch")) {
            std::printf("[FAIL] %s: отказ без пометки SchemaMismatch: %s\n",
                        tag, e.what());
            return false;
        }
        std::printf("schema-mismatch [%s]: %s\n", tag, e.what());
        return true;
    } catch (const std::exception& e) {
        std::printf("[FAIL] %s: неожиданный тип отказа: %s\n", tag, e.what());
        return false;
    }
    std::printf("[FAIL] %s: отказ НЕ произошёл (тихий неверный план)\n", tag);
    return false;
}

} // namespace

void run() {
    const PlainSpec mixed = make_spec(/*layer0_dense=*/true);
    const PlainSpec refmoe = make_spec(/*layer0_dense=*/false);
    STELL_CHECK(mixed.is_moe && refmoe.is_moe);

    const Loaded lm = load(mixed, "/tmp/stell_core_test_mixed.safetensors");
    const Loaded lh = load(refmoe, "/tmp/stell_core_test_mixed_ref.safetensors");

    OpsInterpreter im(lm.spec, lm.plan, lm.weights);
    OpsInterpreter ih(lh.spec, lh.plan, lh.weights);

    const std::vector<std::int32_t> prompt = {1, 2, 3, 5, 7, 9, 11, 13};

    // --- логиты смешанного плана == референсу per-layer плана, ПОБИТОВО -----
    const std::vector<float> last_m = im.run_prompt_logits(prompt);
    const std::vector<float> last_h = ih.run_prompt_logits(prompt);
    STELL_CHECK(last_m.size() == kVocab && last_h.size() == kVocab);
    for (std::size_t i = 0; i < last_m.size(); ++i)
        STELL_CHECK_MSG(last_m[i] == last_h[i],
                        "смешанный план != референсу (позиция last, индекс " +
                            std::to_string(i) + ")");

    const auto all_m = im.run_prompt_logits_all(prompt);
    const auto all_h = ih.run_prompt_logits_all(prompt);
    STELL_CHECK(all_m.size() == prompt.size());
    for (std::size_t p = 0; p < all_m.size(); ++p)
        for (std::size_t i = 0; i < all_m[p].size(); ++i)
            STELL_CHECK_MSG(all_m[p][i] == all_h[p][i],
                            "смешанный план != референсу (позиция " +
                                std::to_string(p) + ", индекс " +
                                std::to_string(i) + ")");
    for (float v : last_m) STELL_CHECK(std::isfinite(v));

    // детерминизм смешанного плана
    OpsInterpreter im2(lm.spec, lm.plan, lm.weights);
    const std::vector<float> again = im2.run_prompt_logits(prompt);
    for (std::size_t i = 0; i < last_m.size(); ++i)
        STELL_CHECK(again[i] == last_m[i]);

    // --- честный SchemaMismatch для непокрытых конфигураций ------------------
    const char* kPath = "/tmp/stell_core_test_mixed_bad.safetensors";
    std::size_t ok = 0;
    {
        // router без expert-групп (слой 1)
        PlainSpec bad = mixed;
        std::vector<PlainGroup> keep;
        for (const PlainGroup& g : bad.groups)
            if (g.name.find("l001.e") != 0) keep.push_back(g);
        bad.groups = keep;
        ok += throws_schema_mismatch(bad, kPath, "router-без-expertов") ? 1 : 0;
    }
    {
        // expert-группы без router (слой 1)
        PlainSpec bad = mixed;
        std::vector<PlainGroup> keep;
        for (const PlainGroup& g : bad.groups)
            if (g.name != "l001.mlp.router") keep.push_back(g);
        bad.groups = keep;
        ok += throws_schema_mismatch(bad, kPath, "expertы-без-router") ? 1 : 0;
    }
    {
        // неполный expert-набор слоя: нет e002 (router [4,8])
        PlainSpec bad = mixed;
        std::vector<PlainGroup> keep;
        for (const PlainGroup& g : bad.groups)
            if (g.name.find("l001.e002.") != 0) keep.push_back(g);
        bad.groups = keep;
        ok += throws_schema_mismatch(bad, kPath, "частичный-expert-набор") ? 1 : 0;
    }
    {
        // есть dense-слой (0), но в ops нет ffn — FFN плотного слоя не исполняется
        PlainSpec bad = mixed;
        bad.ops = base_ops(/*with_ffn=*/false, /*with_moe=*/true);
        ok += throws_schema_mismatch(bad, kPath, "dense-слой-без-ffn-op") ? 1 : 0;
    }
    {
        // есть MoE-слои, но в ops нет moe_gather_scatter
        PlainSpec bad = mixed;
        bad.ops = base_ops(/*with_ffn=*/true, /*with_moe=*/false);
        ok += throws_schema_mismatch(bad, kPath, "moe-слои-без-moe-op") ? 1 : 0;
    }
    {
        // moe op в dense-описании: ни один слой не имеет mlp.router (мёртвый op)
        PlainSpec bad =
            load_plain_spec(STELL_FIXTURES_DIR "/mini_dense.json");
        PlainOp op{};
        op.op = "moe_gather_scatter";
        bad.ops.push_back(op);
        ok += throws_schema_mismatch(bad, kPath, "moe-op-без-moe-слоёв") ? 1 : 0;
    }
    {
        // class=dense при наличии MoE-слоёв (рассинхрон класса)
        PlainSpec bad = mixed;
        bad.is_moe = false;
        ok += throws_schema_mismatch(bad, kPath, "class-не-согласован") ? 1 : 0;
    }
    {
        // частичный набор ffn.shared (gate без up/down) на MoE-слое
        PlainSpec bad = mixed;
        bad.groups.push_back(make_group(999, "l001.ffn.shared.gate", "F32", {8},
                                        "layers.1.mlp.shared.gate.weight"));
        ok += throws_schema_mismatch(bad, kPath, "частичный-ffn.shared") ? 1 : 0;
    }
    STELL_CHECK(ok == 8);

    std::printf("ops-mixed: смешанный план (dense L0 + MoE L1/L2) == референсу "
                "per-layer плана побитово; непокрытые конфигурации — честный "
                "SchemaMismatch (8 кейсов); однородные дефолты не меняются\n");
}

STELL_TEST_MAIN(run)

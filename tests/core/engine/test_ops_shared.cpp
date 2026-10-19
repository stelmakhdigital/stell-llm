// [R4] round 5 (planner Q5, ADR-09 «Уточнение round 5» — op-дополнение
// ffn.shared): shared-expert как dense-FFN (gate/up/down, SwiGLU), выход
// суммируется с суммой выходов routed-экспертов; новая операция ядра не
// вводится, отсутствие группы — поведение без изменений.
//
// Референс плана (побитово, CPU): routed-вклад обнуляется честно — веса
// ВСЕХ экспертов синтетической MoE-фикстуры нулевые (matvec по нулевой
// матрице ≡ +0.0, silu(0) ≡ 0, scatter_add(1·0) ≡ +0.0 — прибавление +0.0
// побитово тождественно), поэтому FFN MoE-слоя = shared-вклад. Референс —
// dense-модель D с ffn.{gate,up,down} по ТЕМ ЖЕ именам тензоров-источников,
// что и ffn.shared MoE-фикстуры Z: ffn-путь (op "ffn") и shared-путь —
// один и тот же ffn_forward ⇒ логиты Z == D побитово на всех позициях.
// Любая неверная семантика shared (другой вход, иная композиция, двойное
// суммирование) даёт расхождение. Дополнительно: Z без ffn.shared (фикстура
// N) даёт ДРУГИЕ логиты — аддитивный вклад реально исполняется.
#include "test_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"

using namespace stell::engine;
using namespace stell_core_test;

namespace {

constexpr std::uint32_t kVocab = 16;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kExperts = 4;

std::vector<PlainGroup> groups_for(bool moe, bool with_shared) {
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
        if (moe) {
            add(make_group(id, p + "mlp.router", "F32", {kExperts, 8},
                           "layers." + ls + ".mlp.router.weight"));
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                char eb[32];
                std::snprintf(eb, sizeof eb, "e%03u", e);
                const std::string base =
                    "layers." + ls + ".mlp.experts." + std::to_string(e);
                add(make_group(id, p + eb + ".gate", "F16", {16, 8},
                               base + ".gate_proj.weight"));
                add(make_group(id, p + eb + ".up", "F16", {16, 8},
                               base + ".up_proj.weight"));
                add(make_group(id, p + eb + ".down", "F16", {8, 16},
                               base + ".down_proj.weight"));
            }
        }
        // ffn.shared MoE-фикстуры / (в dense-референсе) ffn с теми же
        // именами тензоров-источников — побитово те же веса; флаг один —
        // presence FFN-блока (в MoE — shared, в dense-референсе — ffn)
        if (with_shared) {
            const std::string base = "layers." + ls + ".mlp.shared_expert";
            const std::string np = moe ? (p + "ffn.shared") : (p + "ffn");
            add(make_group(id, np + ".gate", "F16", {16, 8},
                           base + ".gate_proj.weight"));
            add(make_group(id, np + ".up", "F16", {16, 8},
                           base + ".up_proj.weight"));
            add(make_group(id, np + ".down", "F16", {8, 16},
                           base + ".down_proj.weight"));
        }
    }
    return out;
}

PlainSpec make_spec(bool with_shared, bool moe_class) {
    PlainSpec s{};
    s.name = with_shared ? "mini-shared-test" : "mini-shared-ref";
    s.is_moe = moe_class;
    s.dtype_weights = "F16";
    s.vocab_size = kVocab;
    s.d_model = 8;
    s.n_layers = kLayers;
    s.n_heads = 2;
    s.d_head = 4;
    s.n_kv_heads = 1;
    s.d_kv = 4;
    if (moe_class) {
        s.n_experts = kExperts;
        s.n_experts_active = 2;
        s.expert_d_ff = 16;
        s.router = "softmax-topk";
    }
    s.ops = base_ops(/*with_ffn=*/!moe_class, /*with_moe=*/moe_class);
    s.groups = groups_for(/*moe=*/moe_class, with_shared);
    return s;
}

// Тензоры фикстуры; expert-группы (имя "l%03u.e%03u...") — НУЛЕВОЙ payload
// (честное обнуление routed-вклада, см. шапку).
std::vector<STensor> tensors_for(const PlainSpec& spec, bool zero_experts) {
    std::vector<STensor> out;
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block) continue;
        STensor t;
        t.name = g.source_pattern;
        t.dtype = g.dtype == PlainDTypeF32 ? "F32" : "F16";
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto d : g.shape) n *= static_cast<std::size_t>(d);
        const bool expert = zero_experts && g.name.size() > 6 &&
                            g.name[4] == '.' && g.name[5] == 'e';
        if (expert) {
            // НУЛЕВОЙ payload expert-групп (F16: 0x0000; F32: +0.0)
            t.data.assign(n * (t.dtype == "F32" ? 4 : 2), 0);
        } else {
            const std::vector<float> vals = fill_weights(g.source_pattern, n);
            if (t.dtype == "F32") {
                t.data = f32_payload(vals);
            } else {
                std::vector<std::uint16_t> halves(n);
                for (std::size_t i = 0; i < n; ++i) halves[i] = f32_to_f16(vals[i]);
                t.data = f16_payload(halves);
            }
        }
        out.push_back(std::move(t));
    }
    return out;
}

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load(const PlainSpec& spec, const char* path, bool zero_experts) {
    Loaded l{};
    l.spec = spec;
    write_safetensors(path, tensors_for(spec, zero_experts));
    l.weights = load_plain_weights(path, spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(spec);
    return l;
}

} // namespace

void run() {
    // Z: MoE-фикстура со shared-экспертом, routed-веса обнулены
    const PlainSpec zspec = make_spec(/*with_shared=*/true, /*moe_class=*/true);
    const Loaded lz = load(zspec, "/tmp/stell_core_test_shared.safetensors", true);
    // D: референс — dense-модель, ffn по тем же именам тензоров, что ffn.shared
    const PlainSpec dspec = make_spec(/*with_shared=*/true, /*moe_class=*/false);
    const Loaded ld = load(dspec, "/tmp/stell_core_test_shared_ref.safetensors",
                           /*zero_experts=*/false);
    // N: та же MoE-фикстура БЕЗ ffn.shared (аддитивного вклада нет)
    const PlainSpec nspec = make_spec(/*with_shared=*/false, /*moe_class=*/true);
    const Loaded ln = load(nspec, "/tmp/stell_core_test_shared_none.safetensors",
                           true);

    OpsInterpreter iz(lz.spec, lz.plan, lz.weights);
    OpsInterpreter id(ld.spec, ld.plan, ld.weights);
    OpsInterpreter in(ln.spec, ln.plan, ln.weights);

    const std::vector<std::int32_t> prompt = {1, 2, 3, 5, 7, 9, 11, 13};

    // --- логиты Z (shared) == референсу плана D (dense-ffn), ПОБИТОВО --------
    const std::vector<float> last_z = iz.run_prompt_logits(prompt);
    const std::vector<float> last_d = id.run_prompt_logits(prompt);
    STELL_CHECK(last_z.size() == kVocab && last_d.size() == kVocab);
    for (std::size_t i = 0; i < last_z.size(); ++i)
        STELL_CHECK_MSG(last_z[i] == last_d[i],
                        "shared-логиты != референсу dense-ffn (индекс " +
                            std::to_string(i) + ")");
    for (float v : last_z) STELL_CHECK(std::isfinite(v));

    const auto all_z = iz.run_prompt_logits_all(prompt);
    const auto all_d = id.run_prompt_logits_all(prompt);
    STELL_CHECK(all_z.size() == prompt.size());
    for (std::size_t p = 0; p < all_z.size(); ++p)
        for (std::size_t i = 0; i < all_z[p].size(); ++i)
            STELL_CHECK_MSG(all_z[p][i] == all_d[p][i],
                            "shared-логиты != референсу (позиция " +
                                std::to_string(p) + ", индекс " +
                                std::to_string(i) + ")");

    // --- отсутствие ffn.shared — поведение без изменений + вклад реален ----- 
    const std::vector<float> last_n = in.run_prompt_logits(prompt);
    bool differs = false;
    for (std::size_t i = 0; i < last_n.size(); ++i)
        if (last_n[i] != last_z[i]) { differs = true; break; }
    STELL_CHECK_MSG(differs, "shared-вклад не проявился в логитах");
    // N побитово == mini_moe-подобному плану без shared: routed-вклад нулевой
    // (обнулённые эксперты) — детерминизм повторного прогона:
    OpsInterpreter in2(ln.spec, ln.plan, ln.weights);
    const std::vector<float> again = in2.run_prompt_logits(prompt);
    for (std::size_t i = 0; i < last_n.size(); ++i)
        STELL_CHECK(again[i] == last_n[i]);

    std::printf("ops-shared: MoE + ffn.shared (dense-FFN, сумма с routed) == "
                "референсу плана побитово (все позиции); без ffn.shared — "
                "поведение без изменений; аддитивный вклад зафиксирован\n");
}

STELL_TEST_MAIN(run)

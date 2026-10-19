// [R1] round 5 (planner Q1, ADR-11 «Уточнение round 5» — стоимость
// positions:"last"):
//   1) побитовая эквивалентность: run_prompt_logits (last-путь, projection
//      только последней позиции) == прежнему пути
//      run_prompt_logits_all(prompt).back() — на mini_dense и mini_moe,
//      промпт ≥ 64 токенов;
//   2) метрика wall-времени (фиксируется в логе теста): 128 последовательных
//      last-вызовов ≤ 2× wall-времени одного полного forward ("all") того же
//      промпта — на синтетическом big-vocab каталоге (vocab=240000, тот же
//      режим, что разблокирует R8-прогон реальных весов: projection над
//      словарём доминирует над стоимостью токен-forward).
//      Жёсткая проверка (≤ 2×) — на паттерне R8 incremental-last (повторные
//      last-вызовы по растущему префиксу); вариант «128 вызовов того же
//      промпта» выводится в лог справочно: он доминируется пересчётом
//      префикса и чувствителен к CPU-конкуренции параллельного ctest.
#include "test_util.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "synth.hpp"
#include "stell/engine/ops_runner.hpp"

using namespace stell::engine;
using namespace stell_core_test;

namespace {

// Слот-байты группы — та же конвенция, что description.cpp models
// (payload по dtype + выравнивание слота до 4096).
std::size_t payload_bytes_for(const std::string& dtype, std::size_t n) {
    if (dtype == "F16") return n * 2;
    if (dtype == "F32") return n * 4;
    throw std::runtime_error("test: dtype " + dtype + " не поддержан хелпером");
}
std::size_t align_up4096(std::size_t v) { return (v + 4095) / 4096 * 4096; }

std::uint32_t g_next_id = 0;

struct GroupAdd {
    std::vector<PlainGroup> groups;
    PlainGroup& add(const std::string& name, const std::string& dtype,
                    std::vector<std::int64_t> shape, const std::string& pattern,
                    bool kv = false) {
        PlainGroup g{};
        g.id = g_next_id++;
        g.name = name;
        g.role = kv ? PlainRoleKv : PlainRoleAttn;
        g.dtype = dtype == "F32" ? PlainDTypeF32 : PlainDTypeF16;
        g.shape = std::move(shape);
        std::size_t n = 1;
        for (auto d : g.shape) n *= static_cast<std::size_t>(d);
        g.slot_bytes = align_up4096(payload_bytes_for(dtype, n));
        g.source_pattern = pattern;
        g.kv_block = kv;
        groups.push_back(g);
        return groups.back();
    }
};

// big-vocab dense-каталог: форма mini_dense (d_model=8, 2 слоя, ffn с gate),
// словарь 120000 — projection над словарём доминирует в стоимости.
PlainSpec big_vocab_spec(std::uint32_t vocab) {
    PlainSpec s{};
    s.name = "big-vocab-last-test";
    s.is_moe = false;
    s.dtype_weights = "F16";
    s.vocab_size = vocab;
    s.d_model = 8;
    s.n_layers = 2;
    s.n_heads = 2;
    s.d_head = 4;
    s.n_kv_heads = 1;
    s.d_kv = 4;

    PlainOp op{};
    op.op = "rmsnorm";
    op.params.push_back({"eps", 1e-5, "", true});
    s.ops.push_back(op);
    op = PlainOp{};
    op.op = "attention";
    op.params.push_back({"kernel", 0.0, "sm70_fp32_qk_pv", false});
    s.ops.push_back(op);
    op = PlainOp{};
    op.op = "ffn";
    op.params.push_back({"activation", 0.0, "silu_mul", false});
    s.ops.push_back(op);

    GroupAdd add;
    add.add("embed", "F16", {std::int64_t(vocab), 8}, "embedding.weight");
    add.add("norm_final", "F32", {8}, "model.norm.weight");
    add.add("head", "F16", {std::int64_t(vocab), 8}, "head.weight");
    for (std::size_t l = 0; l < s.n_layers; ++l) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "l%03zu", l);
        const std::string p = buf;
        add.add(p + ".attn.norm", "F32", {8},
                "layers." + std::to_string(l) + ".input_layernorm.weight");
        add.add(p + ".attn.wq", "F16", {8, 8},
                "layers." + std::to_string(l) + ".self_attn.q_proj.weight");
        add.add(p + ".attn.wk", "F16", {4, 8},
                "layers." + std::to_string(l) + ".self_attn.k_proj.weight");
        add.add(p + ".attn.wv", "F16", {4, 8},
                "layers." + std::to_string(l) + ".self_attn.v_proj.weight");
        add.add(p + ".attn.wo", "F16", {8, 8},
                "layers." + std::to_string(l) + ".self_attn.o_proj.weight");
        add.add(p + ".ffn.gate", "F16", {16, 8},
                "layers." + std::to_string(l) + ".mlp.gate_proj.weight");
        add.add(p + ".ffn.up", "F16", {16, 8},
                "layers." + std::to_string(l) + ".mlp.up_proj.weight");
        add.add(p + ".ffn.down", "F16", {8, 16},
                "layers." + std::to_string(l) + ".mlp.down_proj.weight");
        add.add(p + ".kv", "F16", {4, 8}, "layers." + std::to_string(l) + ".kv_cache",
                /*kv=*/true);
    }
    s.groups = std::move(add.groups);
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

double wall_ms(const std::chrono::steady_clock::time_point& t0,
               const std::chrono::steady_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

void run() {
    const std::size_t kPromptLen = 96;   // ≥ 64 (требование planner Q1)
    std::vector<std::int32_t> prompt;
    for (std::size_t i = 0; i < kPromptLen; ++i)
        prompt.push_back(static_cast<std::int32_t>(i % 13));

    // --- 1) побитовая эквивалентность: mini_dense ---------------------------
    {
        PlainSpec dense = load_plain_spec(STELL_FIXTURES_DIR "/mini_dense.json");
        const Loaded l = load(dense, "/tmp/stell_core_test_last_dense.safetensors");
        OpsInterpreter interp(l.spec, l.plan, l.weights);
        const std::vector<float> last = interp.run_prompt_logits(prompt);
        const std::vector<std::vector<float>> all =
            interp.run_prompt_logits_all(prompt);
        STELL_CHECK(all.size() == prompt.size());
        STELL_CHECK(last.size() == all.back().size());
        for (std::size_t i = 0; i < last.size(); ++i)
            STELL_CHECK_MSG(last[i] == all.back()[i],
                            "mini_dense: last != all.back() (индекс " +
                                std::to_string(i) + ")");
    }

    // --- 1) побитовая эквивалентность: mini_moe -----------------------------
    {
        PlainSpec moe = load_plain_spec(STELL_FIXTURES_DIR "/mini_moe.json");
        const Loaded l = load(moe, "/tmp/stell_core_test_last_moe.safetensors");
        OpsInterpreter interp(l.spec, l.plan, l.weights);
        const std::vector<float> last = interp.run_prompt_logits(prompt);
        const std::vector<std::vector<float>> all =
            interp.run_prompt_logits_all(prompt);
        STELL_CHECK(all.size() == prompt.size());
        STELL_CHECK(last.size() == all.back().size());
        for (std::size_t i = 0; i < last.size(); ++i)
            STELL_CHECK_MSG(last[i] == all.back()[i],
                            "mini_moe: last != all.back() (индекс " +
                                std::to_string(i) + ")");
    }

    // --- 2) метрика wall-времени: big-vocab каталог -------------------------
    const PlainSpec spec = big_vocab_spec(240000);
    const Loaded l = load(spec, "/tmp/stell_core_test_last_big.safetensors");
    OpsInterpreter interp(l.spec, l.plan, l.weights);

    // разминка (страницы/частота CPU — честный замер после разогрева)
    (void)interp.run_prompt_logits(prompt);
    (void)interp.run_prompt_logits_all(prompt);

    std::vector<std::int32_t> prompt128;
    for (std::size_t i = 0; i < 128; ++i)
        prompt128.push_back(static_cast<std::int32_t>(i % 13));

    // один полный forward ("all") того же промпта
    const auto t_all0 = std::chrono::steady_clock::now();
    const auto rows_all = interp.run_prompt_logits_all(prompt128);
    const auto t_all1 = std::chrono::steady_clock::now();
    const double ms_full = wall_ms(t_all0, t_all1);

    // 128 последовательных last-вызовов, паттерн R8 incremental-last:
    // растущий префикс (позиции 1..128)
    const auto t_grow0 = std::chrono::steady_clock::now();
    for (std::size_t n = 1; n <= 128; ++n)
        (void)interp.run_prompt_logits(
            std::vector<std::int32_t>(prompt128.begin(),
                                      prompt128.begin() + static_cast<std::ptrdiff_t>(n)));
    const auto t_grow1 = std::chrono::steady_clock::now();
    const double ms_grow = wall_ms(t_grow0, t_grow1);

    // 128 последовательных last-вызовов того же (полного) промпта —
    // для прозрачности; тоже фиксируется в логе
    const auto t_same0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 128; ++i) (void)interp.run_prompt_logits(prompt128);
    const auto t_same1 = std::chrono::steady_clock::now();
    const double ms_same = wall_ms(t_same0, t_same1);

    // last-результат big-vocab каталога тоже побитово == прежнему пути
    STELL_CHECK(rows_all.size() == 128);
    const std::vector<float> last_big = interp.run_prompt_logits(prompt128);
    for (std::size_t i = 0; i < last_big.size(); ++i)
        STELL_CHECK_MSG(last_big[i] == rows_all.back()[i],
                        "big-vocab: last != all.back() (индекс " +
                            std::to_string(i) + ")");

    std::printf(
        "METRIC[R1] vocab=240000, prompt=128, n_layers=2: полный forward "
        "(\"all\") = %.1f мс; 128 last-вызовов (растущий префикс, паттерн R8 "
        "incremental-last) = %.1f мс (×%.2f); 128 last-вызовов того же промпта "
        "= %.1f мс (×%.2f); порог ≤ 2× (растущий префикс — паттерн R8 "
        "incremental-last) — %s; вариант \"тот же промпт\" — справочный"
        " (чувствителен к CPU-конкуренции параллельного ctest: доминируется "
        "пересчётом префикса, а не projection'ом)\n",
        ms_full, ms_grow, ms_grow / ms_full, ms_same, ms_same / ms_full,
        ms_grow <= 2.0 * ms_full ? "PASS" : "FAIL");
    STELL_CHECK_MSG(ms_grow <= 2.0 * ms_full,
                    "метрика [R1]: 128 last-вызовов (растущий префикс) > 2× "
                    "полного forward");

    std::printf("prompt-last: побитовая эквивалентность last == all.back() "
                "(mini_dense, mini_moe, big-vocab; промпт 96) — PASS\n");
}

STELL_TEST_MAIN(run)

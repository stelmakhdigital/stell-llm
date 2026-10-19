// [R6] round 5 (planner Q7, ADR-09 «Уточнение round 5» — source_slice через
// мост): PlainGroup несёт source_slice (срез стековного тензора по старшей
// размерности); движок читает sliced-группы через ОБЩИЙ IWeightsSource —
// открытие источника ровно одно (метрика LoadedWeights::source_opens),
// собственного открытия лоадера движком по стороннему описанию нет.
//
// Тест: описание со sliced-группами — fused qkv-пример: q/k/v слоя — срезы
// 0/1/2 одного стековного тензора layers.{layer}.attn_qkv.weight [3,8,8]
// (равные формы q [8,8], k/v [8,8] при n_kv_heads=2 — канон стековных
// тензоров, R9/ADR-09). Срезы доезжают через JSON → description → PlainGroup
// → compat_description → лоадер. Референс — «прямой лоадер»: эквивалентное
// описание, где q/k/v — самостоятельные тензоры с ТЕМИ ЖЕ байтами; логиты
// побитово == референсу на всех позициях; payload каждой sliced-группы ==
// байтам соответствующего среза файла (и различаются между собой — срез не
// проигнорирован: проигнорированный срез = shape mismatch SchemaMismatch).
#include "test_util.hpp"

#include <cmath>
#include <map>
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

// Описание со fused qkv (slices 0/1/2) — валидный JSON схемы ADR-09 §1.
const char* kJsonFused = R"json({
  "schema": "1.0",
  "name": "mini-fused-qkv-test",
  "class": "dense",
  "dtype_weights": "F16",
  "vocab_size": 16,
  "d_model": 8,
  "n_layers": 2,
  "n_heads": 2,
  "d_head": 4,
  "kv": { "dtype": "F16", "n_kv_heads": 2, "d_kv": 4, "quant": null },
  "ops": [
    { "op": "rmsnorm", "io": ["hidden"], "params": { "eps": 1e-5 } },
    { "op": "attention", "io": ["hidden", "kv"], "params": { "kernel": "sm70_fp32_qk_pv" } },
    { "op": "ffn", "io": ["hidden"], "params": { "activation": "silu_mul" } }
  ],
  "tensor_groups": [
    { "name": "embed", "role": "attn", "shape": [16, 8], "dtype": "F16",
      "source_pattern": "embedding.weight" },
    { "name": "norm_final", "role": "attn", "shape": [8], "dtype": "F32",
      "source_pattern": "model.norm.weight" },
    { "name": "head", "role": "attn", "shape": [16, 8], "dtype": "F16",
      "source_pattern": "head.weight" },
    { "name": "l{layer}.attn.norm", "role": "attn", "shape": [8], "dtype": "F32",
      "source_pattern": "layers.{layer}.input_layernorm.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.attn.wq", "role": "attn", "shape": [8, 8], "dtype": "F16",
      "source_pattern": "layers.{layer}.attn_qkv.weight", "source_slice": 0,
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.attn.wk", "role": "attn", "shape": [8, 8], "dtype": "F16",
      "source_pattern": "layers.{layer}.attn_qkv.weight", "source_slice": 1,
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.attn.wv", "role": "attn", "shape": [8, 8], "dtype": "F16",
      "source_pattern": "layers.{layer}.attn_qkv.weight", "source_slice": 2,
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.attn.wo", "role": "attn", "shape": [8, 8], "dtype": "F16",
      "source_pattern": "layers.{layer}.self_attn.o_proj.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.ffn.gate", "role": "attn", "shape": [16, 8], "dtype": "F16",
      "source_pattern": "layers.{layer}.mlp.gate_proj.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.ffn.up", "role": "attn", "shape": [16, 8], "dtype": "F16",
      "source_pattern": "layers.{layer}.mlp.up_proj.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.ffn.down", "role": "attn", "shape": [8, 16], "dtype": "F16",
      "source_pattern": "layers.{layer}.mlp.down_proj.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.kv", "role": "kv", "shape": [4, 8], "dtype": "F16",
      "source_pattern": "layers.{layer}.kv_cache", "kv_block": true,
      "cardinality": { "layer": 2 } }
  ]
})json";

// Байты среза s стековного тензора слоя l (детерминированные, различные
// между срезами — срез обязан разрешаться в СВОИ данные).
std::vector<std::uint8_t> slice_payload(const std::string& fused, std::size_t rows,
                                        std::size_t cols, std::size_t s) {
    const std::vector<float> v = fill_weights(
        fused + ".s" + std::to_string(s), rows * cols);
    std::vector<std::uint16_t> halves(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) halves[i] = f32_to_f16(v[i]);
    return f16_payload(halves);
}

// Тензоры файла fused-описания: обычные группы — по fixture_tensors; срезные
// — стековный тензор [n_slices, rows, cols] = конкатенация байтов срезов.
std::vector<STensor> fused_tensors(const PlainSpec& spec) {
    std::vector<STensor> out;
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block || g.source_slice < 0) continue;
        // группа среза: собрать стек по всем группам того же паттерна
        std::map<std::int64_t, const PlainGroup*> by_slice;
        std::size_t count = 0;
        for (const PlainGroup& h : spec.groups) {
            if (h.kv_block || h.source_slice < 0) continue;
            if (h.source_pattern != g.source_pattern) continue;
            by_slice[h.source_slice] = &h;
            ++count;
        }
        STensor t;
        t.name = g.source_pattern;
        t.dtype = "F16";
        t.shape = {std::int64_t(count), g.shape[0], g.shape[1]};
        for (std::size_t s = 0; s < count; ++s) {
            const auto it = by_slice.find(static_cast<std::int64_t>(s));
            STELL_CHECK_MSG(it != by_slice.end(),
                            "срез " + std::to_string(s) + " отсутствует");
            std::vector<std::uint8_t> payload = slice_payload(
                g.source_pattern, static_cast<std::size_t>(g.shape[0]),
                static_cast<std::size_t>(g.shape[1]), s);
            t.data.insert(t.data.end(), payload.begin(), payload.end());
        }
        out.push_back(std::move(t));
    }
    // обычные (несрезные) группы — конвенция fixture_tensors
    for (const PlainGroup& g : spec.groups) {
        if (g.kv_block || g.source_slice >= 0) continue;
        STensor t;
        t.name = g.source_pattern;
        t.dtype = g.dtype == PlainDTypeF32 ? "F32" : "F16";
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto d : g.shape) n *= static_cast<std::size_t>(d);
        const std::vector<float> vals = fill_weights(g.source_pattern, n);
        if (t.dtype == "F32") {
            t.data = f32_payload(vals);
        } else {
            std::vector<std::uint16_t> halves(n);
            for (std::size_t i = 0; i < n; ++i) halves[i] = f32_to_f16(vals[i]);
            t.data = f16_payload(halves);
        }
        out.push_back(std::move(t));
    }
    return out;
}

// Референс: то же описание, но q/k/v — самостоятельные тензоры (без срезов)
// с байтами == байтам соответствующих срезов («прямой лоадер»).
PlainSpec unfused_reference(const PlainSpec& fused) {
    PlainSpec ref = fused;
    ref.name = "mini-fused-qkv-ref";
    for (PlainGroup& g : ref.groups) {
        if (g.kv_block || g.source_slice < 0) continue;
        const std::string suffix =
            g.name.find(".wq") != std::string::npos ? "s0" :
            g.name.find(".wk") != std::string::npos ? "s1" : "s2";
        g.source_pattern += "." + suffix;      // имя отдельного тензора
        g.source_slice = -1;
    }
    return ref;
}

struct Loaded {
    PlainSpec spec;
    ExecutionPlan plan;
    LoadedWeights weights;
};

Loaded load(const PlainSpec& spec, const char* path,
            const std::vector<STensor>& tensors) {
    Loaded l{};
    l.spec = spec;
    write_safetensors(path, tensors);
    l.weights = load_plain_weights(path, spec, PlainWeightsSafetensors);
    l.plan = build_execution_plan(spec);
    return l;
}

} // namespace

void run() {
    // JSON → description → PlainSpec: source_slice доезжает до PlainGroup
    const std::string json_path = "/tmp/stell_core_test_fused_qkv.json";
    {
        std::FILE* f = std::fopen(json_path.c_str(), "wb");
        STELL_CHECK(f != nullptr);
        std::fwrite(kJsonFused, 1, std::strlen(kJsonFused), f);
        std::fclose(f);
    }
    const PlainSpec fused = load_plain_spec(json_path);
    std::size_t sliced = 0;
    for (const PlainGroup& g : fused.groups) {
        if (g.name.find(".wq") != std::string::npos) STELL_CHECK(g.source_slice == 0);
        if (g.name.find(".wk") != std::string::npos) STELL_CHECK(g.source_slice == 1);
        if (g.name.find(".wv") != std::string::npos) STELL_CHECK(g.source_slice == 2);
        if (g.source_slice >= 0) ++sliced;
    }
    STELL_CHECK(sliced == 6);                  // 2 слоя × {wq, wk, wv}

    // Ровно одно открытие источника; sliced-группы читаются общим IWeightsSource
    const Loaded lf = load(fused, "/tmp/stell_core_test_fused_qkv.safetensors",
                           fused_tensors(fused));
    STELL_CHECK_MSG(lf.weights.source_opens == 1,
                    "источник открыт не ровно один раз");

    // Референс «прямой лоадер»: q/k/v — самостоятельные тензоры, те же байты
    const PlainSpec refspec = unfused_reference(fused);
    const Loaded lr = load(refspec, "/tmp/stell_core_test_fused_qkv_ref.safetensors",
                           fixture_tensors(refspec));

    // payload каждой sliced-группы == байтам среза файла (и различаются)
    for (const PlainGroup& g : fused.groups) {
        if (g.kv_block || g.source_slice < 0) continue;
        const auto it = lf.weights.slots.find(g.id);
        STELL_CHECK_MSG(it != lf.weights.slots.end(), "нет payload " + g.name);
        const std::vector<std::uint8_t> expect = slice_payload(
            g.source_pattern, static_cast<std::size_t>(g.shape[0]),
            static_cast<std::size_t>(g.shape[1]),
            static_cast<std::size_t>(g.source_slice));
        STELL_CHECK_MSG(expect.size() <= it->second.size(),
                        "размер payload " + g.name);
        STELL_CHECK_MSG(std::memcmp(expect.data(), it->second.data(),
                                    expect.size()) == 0,
                        "payload " + g.name + " != байтам среза файла");
    }
    {
        // срезы различаются между собой (срез не проигнорирован)
        const PlainGroup* wq = nullptr; const PlainGroup* wk = nullptr;
        for (const PlainGroup& g : fused.groups) {
            if (g.name == "l000.attn.wq") wq = &g;
            if (g.name == "l000.attn.wk") wk = &g;
        }
        STELL_CHECK(wq && wk);
        const auto a = lf.weights.slots.find(wq->id);
        const auto b = lf.weights.slots.find(wk->id);
        bool differs = false;
        for (std::size_t i = 0; i < 16; ++i)
            if (a->second[i] != b->second[i]) { differs = true; break; }
        STELL_CHECK_MSG(differs, "срезы q/k совпадают — срез проигнорирован");
    }

    // --- логиты побитово == прямому лоадеру (все позиции) --------------------
    OpsInterpreter ifused(lf.spec, lf.plan, lf.weights);
    OpsInterpreter iref(lr.spec, lr.plan, lr.weights);
    const std::vector<std::int32_t> prompt = {1, 2, 3, 5, 7, 9, 11, 13};

    const std::vector<float> last_f = ifused.run_prompt_logits(prompt);
    const std::vector<float> last_r = iref.run_prompt_logits(prompt);
    STELL_CHECK(last_f.size() == kVocab);
    for (std::size_t i = 0; i < last_f.size(); ++i)
        STELL_CHECK_MSG(last_f[i] == last_r[i],
                        "sliced-план != прямому лоадеру (индекс " +
                            std::to_string(i) + ")");
    for (float v : last_f) STELL_CHECK(std::isfinite(v));

    const auto all_f = ifused.run_prompt_logits_all(prompt);
    const auto all_r = iref.run_prompt_logits_all(prompt);
    for (std::size_t p = 0; p < all_f.size(); ++p)
        for (std::size_t i = 0; i < all_f[p].size(); ++i)
            STELL_CHECK_MSG(all_f[p][i] == all_r[p][i],
                            "sliced-план != прямому лоадеру (позиция " +
                                std::to_string(p) + ", индекс " +
                                std::to_string(i) + ")");

    std::printf("weights-slice: source_slice доезжает через мост (JSON → "
                "PlainGroup → compat_description → лоадер), открытие источника "
                "ровно одно (source_opens=1), payload срезов == байтам файла, "
                "логиты fused-qkv побитово == прямому лоадеру (все позиции)\n");
}

STELL_TEST_MAIN(run)

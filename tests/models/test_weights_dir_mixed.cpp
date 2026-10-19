// ============================================================================
// Mixed-шардовый тест (round 4, [R6]; planner [Q7]): IWeightsSource на каталоге
// шардов, где в одном каталоге соседствуют группы разного dtype (NVFP4
// block32 + F32/F16) — чтение ОДНИМ источником, побитово == по-групповому
// чтению (прямые лоадеры того же плана на объединённом файле).
//
// «Профильные» ветвления в источнике запрещены ([Q7]: смешанность — свойство
// плана, dtype — атрибут каждой tensor_group, ADR-09 round 2; источник — без
// состояния профиля): проверка — один и тот же источник выдаёт ПОБИТОВО
// одинаковые байты для двух описаний, различающихся ТОЛЬКО dtype_weights
// (групповые dtype идентичны). Чтение диспетчируется по dtype ГРУППЫ, не по
// профилю описания.
// ============================================================================
#include "stell/models/convert.hpp"
#include "stell/models/registry.hpp"
#include "stell/models/weights.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "quant.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"

using namespace stell::models;
using namespace stell_test;

namespace {

SynthTensor tensor(const std::string& name, const std::string& dtype,
                   std::vector<std::int64_t> shape, std::vector<std::uint8_t> data) {
    SynthTensor t;
    t.name = name;
    t.dtype = dtype;
    t.shape = std::move(shape);
    t.data = std::move(data);
    return t;
}

std::vector<std::uint8_t> f16_det(std::size_t n, int seed) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint16_t h =
            static_cast<std::uint16_t>(((k * 3 + seed + 5) % 31) + 1);
        out[2 * k] = static_cast<std::uint8_t>(h & 0xFF);
        out[2 * k + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFF);
    }
    return out;
}

std::vector<std::uint8_t> f32_det(std::size_t n, float seed) {
    std::vector<std::uint8_t> out(n * 4);
    for (std::size_t k = 0; k < n; ++k) {
        const float v = seed + 0.0625f * static_cast<float>((k * 7) % 13);
        std::memcpy(out.data() + 4 * k, &v, 4);
    }
    return out;
}

// Смешанный план (канон ADR-09 block32): матрицы — NVFP4, нормы/роутер — F32,
// embed/kv — F16. Соседствующие группы разных dtype живут в одном каталоге
// (2 шарда); разбиение по шардам сознательно НЕ по dtype (в каждом шарде есть
// и квантованные, и плоские группы).
// d_model=8, vocab=16, layers=2, heads=2, d_head=4, kv_heads=1, d_ff=16,
// n_experts=4, n_active=2, expert_d_ff=8.
const char* kMixedJson = R"({
  "schema": "1.0",
  "name": "mixed-shards",
  "class": "moe",
  "dtype_weights": "NVFP4",
  "vocab_size": 16,
  "d_model": 8,
  "n_layers": 2,
  "n_heads": 2,
  "d_head": 4,
  "kv": { "dtype": "F16", "n_kv_heads": 1, "d_kv": 4, "quant": null },
  "moe": { "n_experts": 4, "n_experts_active": 2, "expert_d_ff": 8,
           "router": "softmax-topk" },
  "ops": [
    { "op": "rmsnorm", "io": ["hidden"], "params": { "eps": 1e-5 } },
    { "op": "attention", "io": ["hidden", "kv"], "params": { "kernel": "sm70_fp32_qk_pv" } },
    { "op": "moe_gather_scatter", "io": ["hidden", "experts"], "params": { "kernel": "sm70_moe_gs" } }
  ],
  "tensor_groups": [
    { "name": "embed", "role": "attn", "shape": [16, 8], "dtype": "F16",
      "source_pattern": "model.embed_tokens.weight" },
    { "name": "norm_final", "role": "attn", "shape": [8], "dtype": "F32",
      "source_pattern": "model.norm.weight" },
    { "name": "l{layer}.attn.norm", "role": "attn", "shape": [8], "dtype": "F32",
      "source_pattern": "model.layers.{layer}.input_layernorm.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.attn.wq", "role": "attn", "shape": [8, 8], "dtype": "NVFP4",
      "source_pattern": "model.layers.{layer}.self_attn.q_proj.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.attn.wo", "role": "attn", "shape": [8, 8], "dtype": "W4A16",
      "source_pattern": "model.layers.{layer}.self_attn.o_proj.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.kv", "role": "kv", "shape": [8, 4], "dtype": "F16",
      "source_pattern": "layers.{layer}.kv_cache", "kv_block": true,
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.mlp.router", "role": "expert", "shape": [4, 8], "dtype": "F32",
      "source_pattern": "model.layers.{layer}.mlp.router.weight",
      "cardinality": { "layer": 2 } },
    { "name": "l{layer}.e{expert}.gate", "role": "expert", "shape": [8, 8], "dtype": "NVFP4",
      "source_pattern": "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
      "cardinality": { "layer": 2, "expert": 4 } },
    { "name": "l{layer}.e{expert}.down", "role": "expert", "shape": [8, 8], "dtype": "W4A16",
      "source_pattern": "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
      "cardinality": { "layer": 2, "expert": 4 } }
  ]
})";

std::vector<SynthTensor> mixed_source_tensors() {
    std::vector<SynthTensor> out;
    auto push_f16 = [&](const std::string& name, std::vector<std::int64_t> sh, int seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F16", std::move(sh), f16_det(n, seed)));
    };
    auto push_f32 = [&](const std::string& name, std::vector<std::int64_t> sh, float seed) {
        std::size_t n = 1;
        for (auto d : sh) n *= static_cast<std::size_t>(d);
        out.push_back(tensor(name, "F32", std::move(sh), f32_det(n, seed)));
    };
    push_f16("model.embed_tokens.weight", {16, 8}, 3);
    push_f32("model.norm.weight", {8}, 0.25f);
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        push_f32(p + "input_layernorm.weight", {8}, 0.5f + layer);
        push_f16(p + "self_attn.q_proj.weight", {8, 8}, layer * 13 + 7);
        push_f16(p + "self_attn.o_proj.weight", {8, 8}, layer * 31 + 17);
        push_f32(p + "mlp.router.weight", {4, 8}, 0.125f + layer);
        for (int e = 0; e < 4; ++e) {
            const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            push_f16(ep + "gate_proj.weight", {8, 8}, layer * 53 + e * 7 + 19);
            push_f16(ep + "down_proj.weight", {8, 8}, layer * 67 + e * 13 + 27);
        }
    }
    return out;
}

std::vector<std::size_t> nonkv_indices(const ModelDescription& d) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < d.groups.size(); ++i)
        if (d.groups[i].role != stell::tiers::Role::Kv) out.push_back(i);
    return out;
}

// Чтение ВСЕХ некешевых групп одним источником (каталог шардов).
std::vector<std::vector<std::uint8_t>> read_all(IWeightsSource* src,
                                                const ModelDescription& d,
                                                const std::vector<std::size_t>& nonkv) {
    std::vector<std::vector<std::uint8_t>> out;
    for (const std::size_t i : nonkv) {
        std::vector<std::uint8_t> buf(d.groups[i].slot_bytes, 0);
        src->read({d.groups[i].id, d.groups[i].role}, buf.data(), buf.size());
        out.push_back(std::move(buf));
    }
    return out;
}

}  // namespace

void body() {
    // =========================================================================
    // 1. Каталог шардов: соседствующие группы разного dtype (NVFP4 block32 +
    //    W4A16 + F32/F16), чтение ОДНИМ IWeightsSource (NVFP4W4A16 — профильный
    //    лоадер ADR-03), побитово == по-групповому чтению (те же группы через
    //    лоадеры на объединённом одиночном файле — Safetensors-путь для плоских
    //    групп недоступен (квант), поэтому референс — тот же NVFP4W4A16-лоадер
    //    на одиночном файле: единственная разница — раскладка источника).
    // =========================================================================
    const ModelDescription mixed = parse_model_description(kMixedJson, "mixed-shards");
    const std::vector<std::size_t> nonkv = nonkv_indices(mixed);
    CHECK(nonkv.size() == 26);  // 2 верхних + 2×(норма, wq, wo, router) + 2×4×(e.gate, e.down)

    // Каталог: 2 шарда; в каждом — и квантованные, и плоские группы (разбиение
    // НЕ по dtype — соседство разных dtype в одном каталоге и в одном шарде).
    const std::string shard_dir = "/tmp/stell_mixed_shards";
    std::filesystem::remove_all(shard_dir);
    std::filesystem::create_directories(shard_dir);
    const auto ts = mixed_source_tensors();
    const std::size_t half = ts.size() / 2 + (ts.size() % 2);
    write_safetensors(shard_dir + "/model-00001-of-00002.safetensors",
                      std::vector<SynthTensor>(ts.begin(), ts.begin() + half));
    write_safetensors(shard_dir + "/model-00002-of-00002.safetensors",
                      std::vector<SynthTensor>(ts.begin() + half, ts.end()));
    // Объединённый одиночный файл (референс по-группового чтения).
    const std::string single = "/tmp/stell_mixed_single.safetensors";
    write_safetensors(single, ts);

    // Один источник на каталог (смешанный план в одном IWeightsSource).
    std::vector<std::vector<std::uint8_t>> via_dir;
    {
        auto loader = make_loader(WeightsFormat::NVFP4W4A16);
        auto src = loader->open(shard_dir, mixed);
        via_dir = read_all(src.get(), mixed, nonkv);
        src->close();
    }
    // Референс: по-групповое чтение (тот же план, одиночный файл).
    std::vector<std::vector<std::uint8_t>> via_single;
    {
        auto loader = make_loader(WeightsFormat::NVFP4W4A16);
        auto src = loader->open(single, mixed);
        via_single = read_all(src.get(), mixed, nonkv);
        src->close();
    }
    CHECK(via_dir.size() == via_single.size());
    for (std::size_t k = 0; k < via_dir.size(); ++k) {
        const auto& g = mixed.groups[nonkv[k]];
        CHECK_MSG(via_dir[k].size() == via_single[k].size() &&
                      std::memcmp(via_dir[k].data(), via_single[k].data(),
                                  via_dir[k].size()) == 0,
                  "группа '" + g.name + "' (dtype " +
                      std::string(g.dtype == stell::tiers::DType::NVFP4 ? "NVFP4"
                                  : g.dtype == stell::tiers::DType::W4A16 ? "W4A16"
                                  : g.dtype == stell::tiers::DType::F32 ? "F32"
                                                                        : "F16") +
                      "): каталог шардов != одиночному файлу (побитно)");
    }
    // Смешанность фактически проверена: среди прочитанных групп есть и
    // квантованные (NVFP4/W4A16), и плоские (F32/F16).
    {
        bool has_quant = false, has_plain = false;
        for (const std::size_t i : nonkv) {
            const auto dt = mixed.groups[i].dtype;
            has_quant = has_quant ||
                        dt == stell::tiers::DType::NVFP4 || dt == stell::tiers::DType::W4A16;
            has_plain = has_plain ||
                        dt == stell::tiers::DType::F32 || dt == stell::tiers::DType::F16;
        }
        CHECK(has_quant && has_plain);
    }
    std::printf("каталог шардов (NVFP4+W4A16+F32/F16 в одном источнике): "
                "побитно == по-групповому чтению: OK\n");

    // =========================================================================
    // 2. Отсутствие «профильных» ветвлений: два описания, различающиеся ТОЛЬКО
    //    dtype_weights (NVFP4 → W4A16; групповые dtype идентичны), — байты
    //    чтения ПОБИТОВО идентичны (источник диспетчирует по dtype группы).
    // =========================================================================
    {
        std::string alt_json = kMixedJson;
        const std::string from = "\"dtype_weights\": \"NVFP4\"";
        const std::string to = "\"dtype_weights\": \"W4A16\"";
        const std::size_t pos = alt_json.find(from);
        CHECK(pos != std::string::npos);
        alt_json.replace(pos, from.size(), to);
        const ModelDescription alt = parse_model_description(alt_json, "mixed-shards-w4a16");
        CHECK(alt.dtype_weights == "W4A16");
        CHECK(alt.groups.size() == mixed.groups.size());

        std::vector<std::vector<std::uint8_t>> via_alt;
        auto loader = make_loader(WeightsFormat::NVFP4W4A16);
        auto src = loader->open(shard_dir, alt);
        via_alt = read_all(src.get(), alt, nonkv_indices(alt));
        src->close();

        CHECK(via_alt.size() == via_dir.size());
        for (std::size_t k = 0; k < via_dir.size(); ++k)
            CHECK_MSG(via_alt[k].size() == via_dir[k].size() &&
                          std::memcmp(via_alt[k].data(), via_dir[k].data(),
                                      via_dir[k].size()) == 0,
                      "dtype_weights описания не влияет на байты чтения "
                      "(источник без «профильных» ветвлений)");
        std::printf("два профиля (NVFP4 vs W4A16, групповые dtype одинаковы): "
                    "байты чтения побитово идентичны: OK\n");
    }

    // =========================================================================
    // 3. Каталог шардов и у plain-лоадера (Safetensors): те же плоские группы —
    //    побитово == одиночному файлу (единый контракт каталога для обоих
    //    лоадеров; смешанность — свойство плана, а не источника).
    // =========================================================================
    {
        // План только с плоскими группами (F32/F16): тот же каталог, те же
        // тензора; квантованных групп в плане нет — plain-лоадер обязан
        // обслужить каталог идентично одиночному файлу.
        const char* plain_json = R"({
  "schema": "1.0",
  "name": "mixed-shards-plain",
  "class": "dense",
  "dtype_weights": "F16",
  "vocab_size": 16,
  "d_model": 8,
  "n_layers": 1,
  "n_heads": 2,
  "d_head": 4,
  "kv": { "dtype": "F16", "n_kv_heads": 1, "d_kv": 4, "quant": null },
  "ops": [
    { "op": "rmsnorm", "io": ["hidden"], "params": { "eps": 1e-5 } },
    { "op": "attention", "io": ["hidden", "kv"], "params": { "kernel": "sm70_fp32_qk_pv" } },
    { "op": "ffn", "io": ["hidden"], "params": { "activation": "silu_mul" } }
  ],
  "tensor_groups": [
    { "name": "embed", "role": "attn", "shape": [16, 8], "dtype": "F16",
      "source_pattern": "model.embed_tokens.weight" },
    { "name": "norm_final", "role": "attn", "shape": [8], "dtype": "F32",
      "source_pattern": "model.norm.weight" },
    { "name": "l{layer}.attn.norm", "role": "attn", "shape": [8], "dtype": "F32",
      "source_pattern": "model.layers.{layer}.input_layernorm.weight",
      "cardinality": { "layer": 1 } },
    { "name": "l{layer}.attn.wq", "role": "attn", "shape": [8, 8], "dtype": "F16",
      "source_pattern": "model.layers.{layer}.self_attn.q_proj.weight",
      "cardinality": { "layer": 1 } },
    { "name": "l{layer}.kv", "role": "kv", "shape": [8, 4], "dtype": "F16",
      "source_pattern": "layers.{layer}.kv_cache", "kv_block": true,
      "cardinality": { "layer": 1 } }
  ]
})";
        const ModelDescription plain = parse_model_description(plain_json, "plain-plan");
        const std::vector<std::size_t> pnonkv = nonkv_indices(plain);

        std::vector<std::vector<std::uint8_t>> via_dir_plain, via_single_plain;
        {
            auto loader = make_loader(WeightsFormat::Safetensors);
            auto src = loader->open(shard_dir, plain);
            via_dir_plain = read_all(src.get(), plain, pnonkv);
            src->close();
            auto loader2 = make_loader(WeightsFormat::Safetensors);
            auto src2 = loader2->open(single, plain);
            via_single_plain = read_all(src2.get(), plain, pnonkv);
            src2->close();
        }
        for (std::size_t k = 0; k < via_dir_plain.size(); ++k)
            CHECK_MSG(via_dir_plain[k].size() == via_single_plain[k].size() &&
                          std::memcmp(via_dir_plain[k].data(),
                                      via_single_plain[k].data(),
                                      via_dir_plain[k].size()) == 0,
                      "plain-лоадер на каталоге шардов: побитно == одиночному файлу");
        std::printf("plain-лоадер (Safetensors) на каталоге шардов: OK (побитно)\n");
    }

    // =========================================================================
    // 4. Ошибки каталога (честные отказы): каталог без .safetensors-шардов;
    //    тензор в двух шардах → SchemaMismatch.
    // =========================================================================
    {
        const std::string empty = "/tmp/stell_mixed_empty_dir";
        std::filesystem::remove_all(empty);
        std::filesystem::create_directories(empty);
        auto loader = make_loader(WeightsFormat::NVFP4W4A16);
        bool thrown = false;
        try {
            auto src = loader->open(empty, mixed);
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::IoFailure);
            thrown = true;
        }
        CHECK(thrown);
    }
    {
        const std::string dup = "/tmp/stell_mixed_dup_dir";
        std::filesystem::remove_all(dup);
        std::filesystem::create_directories(dup);
        // Оба шарда содержат один и тот же тензор.
        write_safetensors(dup + "/model-00001-of-00002.safetensors", {ts[0]});
        write_safetensors(dup + "/model-00002-of-00002.safetensors", {ts[0]});
        auto loader = make_loader(WeightsFormat::NVFP4W4A16);
        bool thrown = false;
        try {
            auto src = loader->open(dup, mixed);
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::SchemaMismatch);
            thrown = true;
        }
        CHECK(thrown);
    }
    std::printf("ошибки каталога (нет шардов / дубль тензора): OK\n");
}

TEST_MAIN("test_models_weights_dir_mixed")

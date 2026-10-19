// ============================================================================
// Потоковая конвертация в stell-tier layout (round 3, R7 — подготовка среза 3).
//
// Проверяемые срезы:
//   1. mini_moe (committed fixture): потоковая конвертация → layout → логиты
//      интерпретатора (engine OpsInterpreter, stell-tier путь) == committed
//      golden-референсу того же плана (независимый Python-интерпретатор,
//      tests/models/golden/logits_mini_moe.json).
//   2. Синтетическая MoE-модель заметного размера (64 эксперта × 4 слоя,
//      d_model=256, d_ff=512, NVFP4 — ~52 МиБ слотов layout, ~192 МиБ
//      F16-источник): потоковая конвертация →
//        2a. файл читается reader'ом models (stelltier_read_header_and_index)
//            И reader'ом tiers (TierLayoutReader) — раскладка/CRC/индекс;
//        2b. слоты loader'а StellTier == прямому NVFP4W4A16-лоадеру (побитно);
//        2c. логиты интерпретатора по stell-tier пути == референсу того же
//            плана (прямой путь того же плана) — побитово;
//        2d. ПИКОВАЯ ПАМЯТЬ writer'а: замер в изолированном потомке (fork +
//            wait4, ru_maxrss потомка) — потоковый writer против полного
//            (convert(), держит все слоты в RAM). Метрика печатается и
//            фиксируется в отчёте: пик потокового << полного и << размера
//            модели.
//   3. Конфигурация: writer принимает параметры тиров извне (согласованы с
//      TierConfig): vram_slot_size валидируется против slot_bytes плана;
//      выравнивания layout v1 фиксированы (отвергаются); flush по группам
//      не меняет байты файла (сравнение с convert() по байтам, кроме
//      created_ts).
//
// Пиковая память меряется в fork-потомке: ru_maxrss потомка (wait4) = пик
// процесса, выполняющего ТОЛЬКО конвертацию — без генерации источника и без
// остальной части теста (честный замек writer'а).
// ============================================================================
#include "stell/models/convert.hpp"

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "quant.hpp"
#include "stelltier_format.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"
#include "tensor_file_formats.hpp"

#include "stell/models/json.hpp"
#include "stell/models/registry.hpp"

// engine-интерпретатор (референс того же плана; линкуется stell_engine —
// мост ADR-09 §4, TU-изолирован в engine).
#include "stell/engine/execution_plan.hpp"
#include "stell/engine/ops_runner.hpp"
#include "stell/engine/plain_spec.hpp"

// tiers-сторона контракта ADR-12: файл обязан читаться и tiers'овским reader'ом.
#include <stell/tiers/layout.hpp>

using namespace stell::models;
using namespace stell_test;
namespace se = stell::engine;  // engine-типы квалифицированно (мост ADR-09 §4)

namespace {

std::string fixtures_dir() { return STELL_FIXTURES_DIR; }
std::string golden_dir() { return STELL_GOLDEN_FIXTURE; }

std::string read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string out;
    int c;
    while ((c = std::fgetc(f)) != EOF) out.push_back(static_cast<char>(c));
    std::fclose(f);
    return out;
}

// --- синтетическая MoE-модель заметного размера (64 эксперта × 4 слоя) ------
// Шаблоны групп — конвенция mini_moe (интерпретатор engine поддерживает ровно
// эти имена); размеры увеличены: d_model=256, d_ff=512, слоёв 4, экспертов 64.
const char* kSynthMoeJson = R"({
  "schema": "1.0",
  "name": "stream-moe-synth",
  "class": "moe",
  "dtype_weights": "NVFP4",
  "vocab_size": 1024,
  "d_model": 256,
  "n_layers": 4,
  "n_heads": 4,
  "d_head": 64,
  "kv": { "dtype": "F16", "n_kv_heads": 1, "d_kv": 64, "quant": null },
  "moe": { "n_experts": 64, "n_experts_active": 2, "expert_d_ff": 512,
           "router": "softmax-topk" },
  "ops": [
    { "op": "rmsnorm", "io": ["hidden"], "params": { "eps": 1e-5 } },
    { "op": "attention", "io": ["hidden", "kv"], "params": { "kernel": "sm70_fp32_qk_pv" } },
    { "op": "moe_gather_scatter", "io": ["hidden", "experts"], "params": { "kernel": "sm70_moe_gs" } }
  ],
  "tensor_groups": [
    { "name": "embed", "role": "attn", "shape": [1024, 256], "dtype": "F16",
      "source_pattern": "embedding.weight" },
    { "name": "norm_final", "role": "attn", "shape": [256], "dtype": "F32",
      "source_pattern": "model.norm.weight" },
    { "name": "l{layer}.attn.norm", "role": "attn", "shape": [256], "dtype": "F32",
      "source_pattern": "layers.{layer}.input_layernorm.weight", "cardinality": { "layer": 4 } },
    { "name": "l{layer}.attn.wq", "role": "attn", "shape": [256, 256], "dtype": "NVFP4",
      "source_pattern": "layers.{layer}.self_attn.q_proj.weight", "cardinality": { "layer": 4 } },
    { "name": "l{layer}.attn.wk", "role": "attn", "shape": [64, 256], "dtype": "NVFP4",
      "source_pattern": "layers.{layer}.self_attn.k_proj.weight", "cardinality": { "layer": 4 } },
    { "name": "l{layer}.attn.wv", "role": "attn", "shape": [64, 256], "dtype": "NVFP4",
      "source_pattern": "layers.{layer}.self_attn.v_proj.weight", "cardinality": { "layer": 4 } },
    { "name": "l{layer}.attn.wo", "role": "attn", "shape": [256, 256], "dtype": "NVFP4",
      "source_pattern": "layers.{layer}.self_attn.o_proj.weight", "cardinality": { "layer": 4 } },
    { "name": "l{layer}.kv", "role": "kv", "shape": [64, 256], "dtype": "F16",
      "source_pattern": "layers.{layer}.kv_cache", "kv_block": true,
      "cardinality": { "layer": 4 } },
    { "name": "l{layer}.mlp.router", "role": "expert", "shape": [64, 256], "dtype": "F32",
      "source_pattern": "layers.{layer}.mlp.router.weight", "cardinality": { "layer": 4 } },
    { "name": "l{layer}.e{expert}.gate", "role": "expert", "shape": [512, 256], "dtype": "NVFP4",
      "source_pattern": "layers.{layer}.mlp.experts.{expert}.gate.weight",
      "cardinality": { "layer": 4, "expert": 64 } },
    { "name": "l{layer}.e{expert}.up", "role": "expert", "shape": [512, 256], "dtype": "NVFP4",
      "source_pattern": "layers.{layer}.mlp.experts.{expert}.up.weight",
      "cardinality": { "layer": 4, "expert": 64 } },
    { "name": "l{layer}.e{expert}.down", "role": "expert", "shape": [256, 512], "dtype": "NVFP4",
      "source_pattern": "layers.{layer}.mlp.experts.{expert}.down.weight",
      "cardinality": { "layer": 4, "expert": 64 } }
  ]
})";

// Плоский список некешевых групп плана (порядок = порядок layout).
std::vector<std::size_t> nonkv_indices(const ModelDescription& d) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < d.groups.size(); ++i)
        if (d.groups[i].role != stell::tiers::Role::Kv) out.push_back(i);
    return out;
}

std::size_t numel_of(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

// Детерминированный F16-payload (конвенция make_tensors test_convert.cpp).
std::vector<std::uint8_t> f16_payload_det(std::size_t n) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint16_t h = static_cast<std::uint16_t>(((k * 3 + 5) % 31) + 1);
        out[2 * k] = static_cast<std::uint8_t>(h & 0xFF);
        out[2 * k + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFF);
    }
    return out;
}

std::vector<std::uint8_t> f32_payload_const(std::size_t n, float v) {
    std::vector<std::uint8_t> out(n * 4);
    for (std::size_t k = 0; k < n; ++k) std::memcpy(out.data() + 4 * k, &v, 4);
    return out;
}

// Потоковая генерация синтетического safetensors-источника: payload тензора
// строится и пишется по одному (память генератора O(тензора), не O(модели) —
// сам тест соблюдает ту же дисциплину, которую проверяет).
struct SourceTensorDesc {
    std::string name;
    std::string dtype;  // "F16" | "F32"
    std::vector<std::int64_t> shape;
};

void write_safetensors_stream(const std::string& path,
                              const std::vector<SourceTensorDesc>& descs) {
    // Заголовок: смещения считаются заранее (payload подряд в порядке descs).
    std::string header = "{";
    std::size_t off = 0;
    for (std::size_t i = 0; i < descs.size(); ++i) {
        const auto& t = descs[i];
        const std::size_t bytes = numel_of(t.shape) * (t.dtype == "F32" ? 4 : 2);
        std::string shape = "[";
        for (std::size_t k = 0; k < t.shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(t.shape[k]);
        shape += "]";
        header += (i ? "," : "") + std::string("\"") + t.name +
                  "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":" + shape +
                  ",\"data_offsets\":[" + std::to_string(off) + "," +
                  std::to_string(off + bytes) + "]}";
        off += bytes;
    }
    header += ",\"__metadata__\":{\"stell_test\":\"synthetic\"}}";
    std::size_t padded = header.size();
    while ((8 + padded) % 8 != 0) ++padded;
    header.resize(padded, ' ');

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot create " + path);
    stell_test::put_u64_file(f, header.size());
    std::fwrite(header.data(), 1, header.size(), f);
    // Payload по одному тензору (O(тензора) в RAM).
    std::size_t k_idx = 0;
    for (const auto& t : descs) {
        const std::size_t n = numel_of(t.shape);
        if (t.dtype == "F16") {
            const std::vector<std::uint8_t> payload = f16_payload_det(n);
            std::fwrite(payload.data(), 1, payload.size(), f);
        } else {
            const std::vector<std::uint8_t> payload =
                f32_payload_const(n, 0.25f + 0.001f * static_cast<float>(k_idx++ % 7));
            std::fwrite(payload.data(), 1, payload.size(), f);
        }
    }
    if (std::ferror(f)) {
        std::fclose(f);
        throw std::runtime_error("write failed: " + path);
    }
    std::fclose(f);
}

std::vector<SourceTensorDesc> synth_source_descs(const ModelDescription& d) {
    std::vector<SourceTensorDesc> out;
    for (std::size_t i = 0; i < d.group_source_patterns.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;
        SourceTensorDesc t;
        t.name = d.group_source_patterns[i];
        t.dtype = g.dtype == stell::tiers::DType::F32 ? "F32" : "F16";
        t.shape = g.shape;
        out.push_back(std::move(t));
    }
    return out;
}

// --- изолированный замек пиковой памяти writer'а (fork + wait4) --------------
struct ChildResult {
    int exit_code = -1;
    long peak_rss_kb = -1;  // ru_maxrss потомка (Linux: КиБ)
};

template <class Fn>
ChildResult run_in_child(Fn&& fn) {
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    CHECK_MSG(pid >= 0, "fork failed");
    if (pid == 0) {
        int rc = 0;
        try {
            fn();
        } catch (const std::exception& e) {
            std::printf("child error: %s\n", e.what());
            rc = 2;
        } catch (...) {
            rc = 3;
        }
        _exit(rc);
    }
    int status = 0;
    rusage ru{};
    const pid_t got = ::wait4(pid, &status, 0, &ru);
    CHECK_MSG(got == pid, "wait4: wrong pid");
    CHECK_MSG(WIFEXITED(status), "child: abnormal termination");
    return ChildResult{WEXITSTATUS(status), ru.ru_maxrss};
}

// Файлы идентичны кроме created_ts (0x10..0x18 — метка конвертера, ADR-12 §2).
bool layout_bytes_equal_except_ts(const std::string& a, const std::string& b) {
    const std::string ba = read_file(a);
    const std::string bb = read_file(b);
    if (ba.size() != bb.size()) return false;
    for (std::size_t i = 0; i < ba.size(); ++i) {
        if (i >= 0x10 && i < 0x18) continue;
        if (static_cast<unsigned char>(ba[i]) != static_cast<unsigned char>(bb[i])) return false;
    }
    return true;
}

double max_rel_err(const std::vector<float>& got, const std::vector<double>& ref) {
    CHECK(got.size() == ref.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
        const double mag = std::max({std::fabs(ref[i]), std::fabs(static_cast<double>(got[i])), 1.0});
        worst = std::max(worst, err / mag);
    }
    return worst;
}

}  // namespace

void body() {
    // =========================================================================
    // 1. mini_moe (committed fixture): поток → логиты == golden-референс плана.
    // =========================================================================
    const ModelDescription mini = Registry().load(fixtures_dir() + "/mini_moe.json");
    const std::string mini_src = fixtures_dir() + "/mini_moe.safetensors";
    const std::string mini_out = "/tmp/stell_stream_mini_out";
    {
        StellTierConverter conv;
        conv.convert_streaming(mini, mini_src, mini_out);
        const std::string layout = mini_out + "/" + mini.name + ".stelltier";

        // Файл читается reader'ом models...
        std::uint64_t fsz = 0;
        const StellTierIndex idx = stelltier_read_header_and_index(layout, &fsz);
        const std::size_t n_nonkv = nonkv_indices(mini).size();
        CHECK(idx.n_groups == n_nonkv);
        CHECK(idx.entries.size() == n_nonkv);
        CHECK(idx.dtype_profile == 2);  // NVFP4 (профиль mini_moe)
        CHECK(idx.model_name == mini.name);

        // ...и reader'ом tiers (ADR-12: единый контракт, поток B — читатель).
        stell::tiers::TierLayoutReader trd(layout);
        CHECK(trd.header().dtype_profile == 2);
        CHECK(trd.index().size() == n_nonkv);
        CHECK(trd.header().n_groups == n_nonkv);

        // Логиты интерпретатора (engine, stell-tier путь) == committed golden
        // референс того же плана (независимый Python-интерпретатор, ADR-09 §4).
        const JsonValue fx = json_parse(read_file(golden_dir() + "/logits_mini_moe.json"));
        const se::PlainSpec spec = se::plain_spec_from_description(mini);
        const se::ExecutionPlan plan = se::build_execution_plan(spec);
        const se::LoadedWeights w =
            load_plain_weights(layout, spec, se::PlainWeightsStellTier);
        se::OpsInterpreter interp(spec, plan, w, /*tiers=*/nullptr);
        std::vector<std::int32_t> prompt;
        for (const auto& t : fx.get_array("prompt_tokens").array_v)
            prompt.push_back(static_cast<std::int32_t>(t.int_v));
        std::vector<double> ref;
        for (const auto& v : fx.get_array("logits_ref").array_v)
            ref.push_back(v.type == JsonValue::Type::Double ? v.double_v
                                                            : static_cast<double>(v.int_v));
        const std::vector<float> got = interp.run_prompt_logits(prompt);
        const double rel = max_rel_err(got, ref);
        const double tol = fx.get_double("tolerance_rel");
        CHECK_MSG(rel <= tol, "mini_moe: логиты потокового layout != golden-референсу "
                              "(rel=" + std::to_string(rel) + " > " + std::to_string(tol) + ")");
        std::printf("mini_moe golden (streaming path): OK (max_rel=%.3e, tol=%g)\n", rel, tol);
    }

    // =========================================================================
    // 2. Синтетическая MoE заметного размера: 64 эксперта × 4 слоя.
    // =========================================================================
    const ModelDescription big = parse_model_description(kSynthMoeJson, "synth-stream-moe");
    const std::vector<std::size_t> nonkv = nonkv_indices(big);
    std::uint64_t sum_slot_bytes = 0;
    for (const auto& g : big.groups)
        if (g.role != stell::tiers::Role::Kv) sum_slot_bytes += g.slot_bytes;

    const std::string big_src = "/tmp/stell_stream_big_in.safetensors";
    {
        const std::vector<SourceTensorDesc> descs = synth_source_descs(big);
        write_safetensors_stream(big_src, descs);
        std::FILE* f = std::fopen(big_src.c_str(), "rb");
        CHECK(f);
        std::fseek(f, 0, SEEK_END);
        const long src_sz = std::ftell(f);
        std::fclose(f);
        std::printf("synthetic MoE: групп плана %zu (некешевых %zu), слоты layout % .1f МиБ, "
                    "F16-источник %.1f МиБ\n",
                    big.groups.size(), nonkv.size(),
                    static_cast<double>(sum_slot_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(src_sz) / (1024.0 * 1024.0));
    }

    const std::string big_out_stream = "/tmp/stell_stream_big_out";
    const std::string big_out_full = "/tmp/stell_stream_big_out_full";

    // --- 2d. Пиковая память writer'а (изолированные потомки) ------------------
    // Порядок честный: сначала потоковый (его пик фиксируется первым), затем
    // полный convert() (держит ВСЕ слоты в RAM — верхняя граница по конструкции).
    const ChildResult peak_stream = run_in_child([&] {
        StellTierConverter conv;
        conv.convert_streaming(big, big_src, big_out_stream);
    });
    CHECK_MSG(peak_stream.exit_code == 0, "потоковая конвертация потомка упала");
    const ChildResult peak_full = run_in_child([&] {
        StellTierConverter conv;
        conv.convert(big, big_src, big_out_full);
    });
    CHECK_MSG(peak_full.exit_code == 0, "полная конвертация потомка упала");

    std::printf("пиковая память writer'а: потоковый %ld КиБ, полный (convert) %ld КиБ, "
                "слоты layout %.1f МиБ → потоковый/полный = 1/%.1f, потоковый/размер модели = 1/%.1f\n",
                peak_stream.peak_rss_kb, peak_full.peak_rss_kb,
                static_cast<double>(sum_slot_bytes) / (1024.0 * 1024.0),
                static_cast<double>(peak_full.peak_rss_kb) / static_cast<double>(peak_stream.peak_rss_kb),
                static_cast<double>(sum_slot_bytes / 1024) / static_cast<double>(peak_stream.peak_rss_kb));
    // Метрика R7: пик потокового << полного (≥4×) и << размера модели.
    CHECK_MSG(peak_full.peak_rss_kb > peak_stream.peak_rss_kb,
              "пик потокового writer'а не меньше полного — потоковый дизайн не подтверждён");
    CHECK_MSG(peak_stream.peak_rss_kb * 4 < peak_full.peak_rss_kb,
              "пик потокового writer'а ≥ 1/4 полного — выигрыша O(1)-по-payload нет");
    CHECK_MSG(static_cast<double>(peak_stream.peak_rss_kb) <
                  static_cast<double>(sum_slot_bytes / 1024) * 0.5,
              "пик потокового writer'а ≥ половины размера модели");

    const std::string big_layout = big_out_stream + "/" + big.name + ".stelltier";

    // --- 2a. Раскладка: свой reader + tiers'овский reader --------------------
    std::uint64_t fsz = 0;
    const StellTierIndex idx = stelltier_read_header_and_index(big_layout, &fsz);
    CHECK(idx.entries.size() == nonkv.size());
    CHECK(idx.dtype_profile == 2);  // NVFP4
    CHECK(idx.model_name == big.name);
    CHECK(idx.data_offset % kStellTierSlotAlign == 0);
    {
        std::uint64_t cursor = idx.data_offset;
        for (std::size_t i = 0; i < idx.entries.size(); ++i) {
            const StellTierEntry& e = idx.entries[i];
            const auto& g = big.groups[nonkv[i]];
            CHECK(e.data_offset == cursor);
            CHECK(e.slot_bytes == g.slot_bytes);
            CHECK(e.slot_bytes % kStellTierSlotAlign == 0);
            CHECK(e.group_id == g.id.value);
            CHECK(e.role == stelltier_role_code(g.role));
            CHECK(e.dtype == stelltier_dtype_code(g.dtype));
            cursor += e.slot_bytes;
        }
        CHECK(cursor == fsz);  // без дыр, до конца файла (монотонно, ADR-12 §2.4)
    }
    // CRC заголовка (канон ADR-12: [0x00..0x2C)).
    {
        std::FILE* f = std::fopen(big_layout.c_str(), "rb");
        std::array<std::uint8_t, 512> h{};
        CHECK(std::fread(h.data(), 1, h.size(), f) == h.size());
        std::fclose(f);
        std::uint32_t stored = 0;
        for (int b = 0; b < 4; ++b)
            stored |= static_cast<std::uint32_t>(h[0x2C + b]) << (8 * b);
        CHECK(stelltier_header_crc32(h) == stored);
    }
    // tiers'овский reader: тот же файл, та же раскладка (ADR-12 §2, поток B).
    {
        stell::tiers::TierLayoutReader trd(big_layout);
        CHECK(trd.header().n_groups == nonkv.size());
        CHECK(trd.header().dtype_profile == 2);
        // Слот через tiers-reader == слоту через models-reader (CRC групп проверен).
        const std::size_t probe = 5;  // некешевая группа 5 (wv слоя 0)
        std::vector<std::uint8_t> via_tiers(idx.entries[probe].slot_bytes, 0);
        trd.read_group(idx.entries[probe].group_id, via_tiers.data());
        std::vector<std::uint8_t> via_models(idx.entries[probe].slot_bytes, 0);
        {
            auto loader = make_loader(WeightsFormat::StellTier);
            auto src = loader->open(big_layout, big);
            src->read({stell::tiers::GroupId{idx.entries[probe].group_id},
                       big.groups[nonkv[probe]].role},
                      via_models.data(), via_models.size());
            src->close();
        }
        CHECK(std::memcmp(via_tiers.data(), via_models.data(), via_tiers.size()) == 0);
    }

    // --- 2b/2c. Слоты побитно == прямому лоадеру; логиты == референсу плана ---
    const se::PlainSpec spec = se::plain_spec_from_description(big);
    const se::ExecutionPlan plan = se::build_execution_plan(spec);
    const se::LoadedWeights direct =
        load_plain_weights(big_src, spec, se::PlainWeightsNVFP4W4A16);
    const se::LoadedWeights via_layout = se::load_plain_weights(big_layout, spec, se::PlainWeightsStellTier);

    // 2b: побитовая идентичность всех слотов (payload конвертера корректен).
    CHECK(direct.slots.size() == via_layout.slots.size());
    for (const auto& [gid, buf] : via_layout.slots) {
        const auto it = direct.slots.find(gid);
        CHECK_MSG(it != direct.slots.end(), "группы нет в прямом пути");
        CHECK_MSG(it->second.size() == buf.size() &&
                      std::memcmp(it->second.data(), buf.data(), buf.size()) == 0,
                  "слот группы (id " + std::to_string(gid) +
                      ") потокового layout != прямому лоадеру");
    }
    std::printf("слоты layout (%zu групп) == прямому NVFP4W4A16-лоадеру: OK (побитно)\n",
                via_layout.slots.size());

    // 2c: логиты интерпретатора по stell-tier пути == референсу того же плана
    // (прямой путь того же плана) — побитово (слоты идентичны, ADR-09 §4:
    // физический путь весов не меняет вычисление).
    {
        se::OpsInterpreter ref_interp(spec, plan, direct, /*tiers=*/nullptr);
        se::OpsInterpreter tier_interp(spec, plan, via_layout, /*tiers=*/nullptr);
        const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5};
        const std::vector<float> ref = ref_interp.run_prompt_logits(prompt);
        const std::vector<float> got = tier_interp.run_prompt_logits(prompt);
        CHECK(ref.size() == static_cast<std::size_t>(big.vocab_size));
        CHECK(got.size() == ref.size());
        for (std::size_t i = 0; i < ref.size(); ++i)
            CHECK_MSG(ref[i] == got[i],
                      "логиты интерпретатора по потоковому layout != референсу того же плана");
        std::printf("логиты интерпретатора (64 экспертов × 4 слоя, vocab %zu): OK "
                    "(побитно == прямому пути, %zu значений)\n",
                    static_cast<std::size_t>(big.vocab_size), ref.size());
    }

    // =========================================================================
    // 3. Конфигурация: параметры тиров извне + flush по группам.
    // =========================================================================
    // Эталон для секций (в) и 4: полный convert() (та же раскладка по
    // конструкции ADR-12) на mini_moe.
    const std::string ref_out = "/tmp/stell_stream_mini_fullref";
    {
        StellTierConverter full;
        full.convert(mini, mini_src, ref_out);
    }
    // (а) vram_slot_size: группа эксперта (69632 Б) не помещается в 4096 Б —
    //     профиль «свободно 2–11 ГБ на V100» неисполним для этого плана.
    {
        StellTierWriterConfig bad;
        bad.vram_slot_size = 4096;
        StellTierConverter conv;
        CHECK_THROW(conv.convert_streaming(big, big_src, "/tmp/stell_stream_cfg_bad", bad),
                    LoaderError, "vram_slot_size: план не валидирован против профиля");
    }
    // (б) выравнивания фиксированы layout v1 (ADR-12 §2/§2.4) — отвергаются.
    {
        StellTierWriterConfig bad;
        bad.slot_align = 8192;
        StellTierConverter conv;
        CHECK_THROW(conv.convert_streaming(mini, mini_src, "/tmp/stell_stream_cfg_align", bad),
                    LoaderError, "slot_align != 4096 отвергнут (layout v1)");
    }
    // (в) flush по группам (порог 1 Б → fflush на каждую группу) не меняет
    //     байты файла; ram_capacity принимается извне без влияния на layout.
    {
        const std::string fl_out = "/tmp/stell_stream_mini_flush";
        StellTierWriterConfig cfg;
        cfg.flush_group_bytes = 1;
        cfg.ram_capacity = 100500;
        cfg.vram_slot_size = 4096;  // mini_moe: максимальный slot_bytes = 4096 — совместимо
        StellTierConverter conv;
        conv.convert_streaming(mini, mini_src, fl_out, cfg);
        CHECK(layout_bytes_equal_except_ts(fl_out + "/" + mini.name + ".stelltier",
                                           ref_out + "/" + mini.name + ".stelltier"));
        std::printf("конфигурация: flush по группам + tiers-параметры извне — "
                    "файл побитово == convert() (кроме created_ts): OK\n");
    }

    // =========================================================================
    // 4. Каталог HF-шардов (инжест реального чекпоинта): source — каталог с
    //    model-00001-of-00002.safetensors / model-00002-of-00002.safetensors;
    //    результат побитово == конвертации объединённого файла (кроме created_ts).
    // =========================================================================
    {
        const std::string shard_dir = "/tmp/stell_stream_mini_shards";
        std::filesystem::create_directories(shard_dir);
        const std::vector<SourceTensorDesc> descs = synth_source_descs(mini);
        const std::size_t half = descs.size() / 2;
        write_safetensors_stream(shard_dir + "/model-00001-of-00002.safetensors",
                                 std::vector<SourceTensorDesc>(descs.begin(), descs.begin() + half));
        write_safetensors_stream(shard_dir + "/model-00002-of-00002.safetensors",
                                 std::vector<SourceTensorDesc>(descs.begin() + half, descs.end()));
        const std::string sh_out = "/tmp/stell_stream_mini_shards_out";
        // Эталон: те же тензоры одним файлом (F32-значения синтетического
        // генератора отличаются от committed-фикстуры — сравнение честное
        // только против того же набора payload'ов).
        const std::string single = "/tmp/stell_stream_mini_synth_single.safetensors";
        const std::string single_out = "/tmp/stell_stream_mini_synth_out";
        write_safetensors_stream(single, descs);
        StellTierConverter conv;
        conv.convert_streaming(mini, single, single_out);
        conv.convert_streaming(mini, shard_dir, sh_out);
        CHECK(layout_bytes_equal_except_ts(sh_out + "/" + mini.name + ".stelltier",
                                           single_out + "/" + mini.name + ".stelltier"));
        std::printf("каталог HF-шардов (2 шарда): layout побитово == одному файлу: OK\n");
    }

    // =========================================================================
    // 5. Реальные веса (Mixtral-8x7B и т.п.): честный pending по постановке
    //    planner'а ([R7]: «если реальные веса недоступны в раунде — пайплайн и
    //    конфигурация закоммичены, прогон честно pending с причиной»).
    //    Прогон включается env: STELL_R7_REAL_WEIGHTS_DIR (каталог safetensors-
    //    шардов HF) + STELL_R7_REAL_DESC_JSON (описание ADR-09). Без них —
    //    pending с причиной «веса не предоставлены» (НЕ молчаливый успех).
    // =========================================================================
    {
        const char* real_dir = std::getenv("STELL_R7_REAL_WEIGHTS_DIR");
        const char* real_desc = std::getenv("STELL_R7_REAL_DESC_JSON");
        const bool have_real = real_dir && real_desc &&
                               std::filesystem::exists(real_dir) &&
                               std::filesystem::exists(real_desc);
        if (have_real) {
            const ModelDescription real = Registry().load(std::string(real_desc));
            const std::string real_out = "/tmp/stell_stream_real_out";
            StellTierWriterConfig rcfg;  // профиль co-tenancy: границы задаёт tiers-конфиг
            // Round 4 (R3-подготовка): нижняя граница профиля co-tenancy
            // «свободно 2–11 ГБ на V100»: VRAM-слот 2 ГиБ — план реального
            // чекпоинта (embed F32/F16 [vocab,d_model] ~0.3–0.6 ГиБ, экспертные
            // группы-срезы ~десятки МиБ) помещается; ram_capacity — верхняя
            // граница профиля (в layout семантики не имеет, ADR-12).
            rcfg.vram_slot_size = 2u << 30;    // 2 ГиБ VRAM-слот (co-tenancy, низ профиля)
            rcfg.ram_capacity = 11u << 30;     // 11 ГиБ RAM-тира (co-tenancy, верх профиля)
            StellTierConverter conv;
            conv.convert_streaming(real, std::string(real_dir), real_out, rcfg);
            const std::string layout = real_out + "/" + real.name + ".stelltier";
            std::uint64_t rsz = 0;
            const StellTierIndex ridx = stelltier_read_header_and_index(layout, &rsz);
            stell::tiers::TierLayoutReader rr(layout);
            CHECK(rr.index().size() == ridx.entries.size());
            std::printf("реальные веса (%s): layout готов, групп %zu, %.1f МиБ — OK\n",
                        real.name.c_str(), ridx.entries.size(),
                        static_cast<double>(rsz) / (1024.0 * 1024.0));
        } else {
            std::printf(
                "PENDING: прогон на реальных MoE-весах (напр. Mixtral-8x7B) — "
                "веса не предоставлены (STELL_R7_REAL_WEIGHTS_DIR / STELL_R7_REAL_DESC_JSON "
                "не заданы); пайплайн и конфигурация закоммичены и покрыты синтетикой "
                "(срезы 1–4 этого теста)\n");
        }
    }
}

TEST_MAIN("test_models_convert_streaming")

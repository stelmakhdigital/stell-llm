// ============================================================================
// Тесты конвертера GGUF → описание модели (round 2, задача 5; ADR-09
// «Уточнение»): GGUF не несёт JSON-описания — models генерирует его из
// метаданных GGUF с НАТИВНЫМИ именами тензоров формата GGUF (см. ADR-09):
// (token_embd.weight,
// blk.{i}.attn_q.weight, …). Проверки:
//   - happy path: описание собирается (schema 1.0, метаданные, группы),
//     парсится parse_model_description (roundtrip), GGUF-лоадер резолвит
//     НАТИВНЫЕ source_pattern и побайтово сверяет payload;
//   - опциональные метаданные: general.name, head_count_kv (умолчание
//     = head_count), key_length (умолчание d_model/n_heads);
//   - ошибки: unsupported architecture, missing llama.block_count,
//     unsupported tensor dtype (код 2), shape mismatch, missing tensor;
//   - квантованные GGUF-блоки Q4_0 (GGML 20) и Q8_0 (GGML 36): dequant →
//     f32 → F16-группа (значения побайтово сверяются с референсом блоков).
// ============================================================================
#include "stell/models/convert.hpp"
#include "stell/models/registry.hpp"
#include "stell/models/weights.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "quant.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"

using namespace stell::models;
using namespace stell_test;

namespace {

// Синтетический GGUF llama с НАТИВНЫМИ именами; dims — GGUF-native
// {in, out} (разворот делает parse_gguf_header; см. loader_gguf.cpp).
// d_model=8, n_heads=2, d_head=4, n_kv_heads=1, vocab=16, d_ff=16, eps=1e-5.
std::vector<SynthTensor> make_gguf_tensors(int kv_heads) {
    std::vector<SynthTensor> out;
    auto push_f32 = [&](const std::string& name, std::vector<std::int64_t> dims,
                        float seed) {
        SynthTensor t;
        t.name = name;
        t.dtype = "F32";
        t.shape = dims;  // GGUF-native (writer пишет dims как есть)
        std::size_t n = 1;
        for (auto d : dims) n *= static_cast<std::size_t>(d);
        std::vector<float> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) * 0.125f + seed;
        t.data = f32_payload(v);
        out.push_back(std::move(t));
    };
    auto push_f16 = [&](const std::string& name, std::vector<std::int64_t> dims,
                        int seed) {
        SynthTensor t;
        t.name = name;
        t.dtype = "F16";
        t.shape = dims;
        std::size_t n = 1;
        for (auto d : dims) n *= static_cast<std::size_t>(d);
        std::vector<std::uint16_t> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint16_t>((i + seed) & 0x3FFF);
        t.data = f16_payload(v);
        out.push_back(std::move(t));
    };
    push_f32("token_embd.weight", {8, 16}, -0.5f);   // ne {n_embd, vocab}
    push_f32("output_norm.weight", {8}, 0.25f);
    push_f16("output.weight", {8, 16}, 1000);
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        push_f32(p + "attn_norm.weight", {8}, 0.5f);
        push_f16(p + "attn_q.weight", {8, 8}, layer * 13 + 3);
        // ne {in, out}: out = n_kv_heads * d_head
        push_f16(p + "attn_k.weight", {8, static_cast<std::int64_t>(kv_heads * 4)}, layer * 21 + 5);
        push_f16(p + "attn_v.weight", {8, static_cast<std::int64_t>(kv_heads * 4)}, layer * 29 + 9);
        push_f16(p + "attn_output.weight", {8, 8}, layer * 31 + 17);
        push_f32(p + "ffn_norm.weight", {8}, 0.75f);
        push_f16(p + "ffn_gate.weight", {8, 16}, layer * 17 + 19);  // ne {d_model, d_ff}
        push_f16(p + "ffn_up.weight", {8, 16}, layer * 7 + 23);
        push_f16(p + "ffn_down.weight", {16, 8}, layer * 11 + 27);  // ne {d_ff, d_model}
    }
    return out;
}

std::vector<GgufMetaKV> make_meta(int kv_heads, bool with_kv_meta, bool with_name) {
    std::vector<GgufMetaKV> meta;
    GgufMetaKV arch;
    arch.key = "general.architecture";
    arch.type = kGGMLMetaString;
    arch.str = "llama";
    meta.push_back(arch);
    if (with_name) {
        GgufMetaKV n;
        n.key = "general.name";
        n.type = kGGMLMetaString;
        n.str = "mini-llama-gguf";
        meta.push_back(n);
    }
    GgufMetaKV kv;
    kv.key = "llama.block_count";
    kv.u64 = 2;
    meta.push_back(kv);
    GgufMetaKV hc;
    hc.key = "llama.attention.head_count";
    hc.u64 = 2;
    meta.push_back(hc);
    if (with_kv_meta) {
        GgufMetaKV hk;
        hk.key = "llama.attention.head_count_kv";
        hk.u64 = static_cast<std::uint64_t>(kv_heads);
        meta.push_back(hk);
    }
    GgufMetaKV el;
    el.key = "llama.embedding_length";
    el.u64 = 8;
    meta.push_back(el);
    GgufMetaKV eps;
    eps.key = "llama.attention.layer_norm_rms_epsilon";
    eps.type = kGGMLMetaFloat32;
    eps.f64 = 1e-5;
    meta.push_back(eps);
    return meta;
}

int find_group(const ModelDescription& d, const std::string& name) {
    for (std::size_t i = 0; i < d.groups.size(); ++i)
        if (d.groups[i].name == name) return static_cast<int>(i);
    return -1;
}

}  // namespace

void body() {
    // --- happy path ----------------------------------------------------------
    const std::string path = "/tmp/stell_test_gguf_desc.gguf";
    write_gguf_meta(path, make_gguf_tensors(1), make_meta(1, true, false));
    const std::string json = gguf_to_description(path);

    const JsonValue root = json_parse(json);
    CHECK(root.get_string("schema") == "1.0");
    CHECK(root.get_string("name") == "llama-gguf");   // general.name отсутствует
    CHECK(root.get_string("class") == "dense");
    CHECK(root.get_string("dtype_weights") == "F16");
    CHECK(root.get_uint("vocab_size") == 16);
    CHECK(root.get_uint("d_model") == 8);
    CHECK(root.get_uint("n_layers") == 2);
    CHECK(root.get_uint("n_heads") == 2);
    CHECK(root.get_uint("d_head") == 4);              // key_length отсутствует → 8/2
    const JsonValue* kvj = root.find("kv");
    CHECK(kvj != nullptr);
    CHECK(kvj->get_uint("n_kv_heads") == 1);
    CHECK(kvj->get_uint("d_kv") == 4);
    CHECK(kvj->find("quant")->type == JsonValue::Type::Null);
    const std::vector<JsonValue>& ops = root.get_array("ops").array_v;
    CHECK(ops.size() == 3);
    CHECK(ops[0].get_string("op") == "rmsnorm");
    CHECK(ops[1].get_string("op") == "attention");
    CHECK(ops[1].find("params")->find("kernel")->string_v == "sm70_fp32_qk_pv");
    CHECK(ops[2].get_string("op") == "ffn");
    CHECK(ops[2].find("params")->find("activation")->string_v == "silu_mul");
    CHECK(std::fabs(ops[0].find("params")->find("eps")->double_v - 1e-5) < 1e-9);

    // Roundtrip: сгенерированное описание парсится валидатором.
    const ModelDescription d = parse_model_description(json, "gguf-generated");
    CHECK(d.name == "llama-gguf");
    CHECK(d.n_layers == 2);
    // Групп: 3 верхних + 2 слоя × (2 нормы + 7 матриц) + 2 kv = 23.
    CHECK(d.groups.size() == 23);
    // НАТИВНЫЕ имена: embed, l000.attn.wq, l001.ffn.gate, kv.
    const int embed = find_group(d, "embed");
    CHECK((embed >= 0) && (d.group_source_patterns[embed] == "token_embd.weight"));
    CHECK((d.groups[embed].shape == std::vector<std::int64_t>{16, 8}));
    const int wq0 = find_group(d, "l000.attn.wq");
    CHECK((wq0 >= 0) && (d.group_source_patterns[wq0] == "blk.0.attn_q.weight"));
    CHECK((d.groups[wq0].shape == std::vector<std::int64_t>{8, 8}));
    const int gate1 = find_group(d, "l001.ffn.gate");
    CHECK((gate1 >= 0) && (d.group_source_patterns[gate1] == "blk.1.ffn_gate.weight"));
    CHECK((d.groups[gate1].shape == std::vector<std::int64_t>{16, 8}));
    const int down1 = find_group(d, "l001.ffn.down");
    CHECK((down1 >= 0) && (d.group_source_patterns[down1] == "blk.1.ffn_down.weight"));
    CHECK((d.groups[down1].shape == std::vector<std::int64_t>{8, 16}));
    const int kv0 = find_group(d, "l000.kv");
    CHECK((kv0 >= 0) && (d.group_source_patterns[kv0] == "blk.0.kv_cache"));
    CHECK((d.groups[kv0].shape == std::vector<std::int64_t>{8, 4}));

    // GGUF-лоадер резолвит НАТИВНЫЕ source_pattern + сверяет payload.
    auto loader = make_loader(WeightsFormat::GGUF);
    auto src = loader->open(path, d);
    // wq l000: F16-тензор → побайтовое совпадение.
    {
        std::vector<std::uint8_t> buf(d.groups[wq0].slot_bytes, 0);
        src->read({d.groups[wq0].id, d.groups[wq0].role}, buf.data(), buf.size());
        for (const auto& t : make_gguf_tensors(1))
            if (t.name == "blk.0.attn_q.weight")
                CHECK(std::memcmp(buf.data(), t.data.data(), t.data.size()) == 0);
    }
    // norm_final: F32-тензор, F32-группа → побайтово.
    {
        const int nf = find_group(d, "norm_final");
        CHECK((nf >= 0) && (d.group_source_patterns[nf] == "output_norm.weight"));
        std::vector<std::uint8_t> buf(d.groups[nf].slot_bytes, 0);
        src->read({d.groups[nf].id, d.groups[nf].role}, buf.data(), buf.size());
        for (const auto& t : make_gguf_tensors(1))
            if (t.name == "output_norm.weight")
                CHECK(std::memcmp(buf.data(), t.data.data(), t.data.size()) == 0);
    }
    // embed: F32-тензор → группа F32 (dtype тензора, конверсии нет):
    // payload побайтово равен тензору (token_embd.weight).
    {
        std::vector<std::uint8_t> buf(d.groups[embed].slot_bytes, 0);
        src->read({d.groups[embed].id, d.groups[embed].role}, buf.data(), buf.size());
        for (const auto& t : make_gguf_tensors(1))
            if (t.name == "token_embd.weight")
                CHECK(std::memcmp(buf.data(), t.data.data(), t.data.size()) == 0);
    }
    src->close();

    // --- опциональные метаданные --------------------------------------------
    // general.name → имя из метаданных; head_count_kv отсутствует → = head_count
    // (kv-тензоры {8,8} → out = 2*4).
    const std::string p2 = "/tmp/stell_test_gguf_desc2.gguf";
    write_gguf_meta(p2, make_gguf_tensors(2), make_meta(2, false, true));
    const ModelDescription d2 = parse_model_description(gguf_to_description(p2));
    CHECK(d2.name == "mini-llama-gguf");
    CHECK(d2.kv.n_kv_heads == 2 && d2.kv.d_kv == 4);
    const int wk0 = find_group(d2, "l000.attn.wk");
    CHECK((wk0 >= 0) && (d2.group_source_patterns[wk0] == "blk.0.attn_k.weight"));
    CHECK((d2.groups[wk0].shape == std::vector<std::int64_t>{8, 8}));

    // --- ошибки ---------------------------------------------------------------
    auto expect_error = [&](const std::string& json_text,
                            LoaderError::Code code, const char* what) {
        try {
            gguf_to_description(json_text);
            CHECK(false && "expected LoaderError");
        } catch (const LoaderError& e) {
            CHECK(e.code() == code);
        }
        (void)what;
    };
    // Unsupported architecture (mixtral поддержан с round 2, R9; glm — нет).
    {
        auto meta3 = make_meta(1, true, false);
        for (auto& kv : meta3)
            if (kv.key == "general.architecture") kv.str = "glm";
        const std::string p3 = "/tmp/stell_test_gguf_arch.gguf";
        write_gguf_meta(p3, make_gguf_tensors(1), meta3);
        expect_error(p3, LoaderError::Code::SchemaMismatch, "unsupported arch rejected");
    }
    // Missing llama.block_count.
    {
        std::vector<GgufMetaKV> meta4;
        for (const auto& kv : make_meta(1, true, false))
            if (kv.key != "llama.block_count") meta4.push_back(kv);
        const std::string p4 = "/tmp/stell_test_gguf_nolayers.gguf";
        write_gguf_meta(p4, make_gguf_tensors(1), meta4);
        expect_error(p4, LoaderError::Code::SchemaMismatch, "missing block_count rejected");
    }
    // Unsupported tensor dtype (код 2 → не F32/F16/Q4_0/Q8_0).
    {
        std::vector<SynthTensor> ts = make_gguf_tensors(1);
        ts[0].gguf_code = 2;
        const std::string p5 = "/tmp/stell_test_gguf_dtype.gguf";
        write_gguf_meta(p5, ts, make_meta(1, true, false));
        expect_error(p5, LoaderError::Code::UnsupportedQuant, "unsupported dtype rejected");
    }
    // Shape mismatch (wk out ≠ n_kv_heads*d_head).
    {
        std::vector<SynthTensor> ts = make_gguf_tensors(1);
        for (auto& t : ts)
            if (t.name == "blk.0.attn_k.weight") t.shape = {8, 8};
        const std::string p6 = "/tmp/stell_test_gguf_shape.gguf";
        write_gguf_meta(p6, ts, make_meta(1, true, false));
        expect_error(p6, LoaderError::Code::SchemaMismatch, "shape mismatch rejected");
    }
    // Missing tensor (нет output.weight → head).
    {
        std::vector<SynthTensor> ts;
        for (auto& t : make_gguf_tensors(1))
            if (t.name != "output.weight") ts.push_back(t);
        const std::string p7 = "/tmp/stell_test_gguf_nohead.gguf";
        write_gguf_meta(p7, ts, make_meta(1, true, false));
        expect_error(p7, LoaderError::Code::TensorNotFound, "missing tensor rejected");
    }

    // --- квантованные GGUF-блоки Q4_0/Q8_0 ------------------------------------
    // GGML 20 (Q4_0): блок 32 эл. = F16 delta (НАЧАЛО) + 16 байт нибблов;
    // значение = (nibble − 8) × delta; GGML 36 (Q8_0): 32 байта int8 +
    // F16 delta (КОНЕЦ); значение = q × delta.
    {
        const std::string p8 = "/tmp/stell_test_gguf_quant.gguf";
        std::vector<SynthTensor> ts;
        auto push_raw = [&](const std::string& name, std::vector<std::int64_t> dims,
                            std::int64_t code, const std::vector<std::uint8_t>& data) {
            SynthTensor t;
            t.name = name;
            t.dtype = "F16";
            t.gguf_code = code;
            t.shape = dims;
            t.data = data;
            ts.push_back(std::move(t));
        };
        auto push_f32_2 = [&](const std::string& name, std::vector<std::int64_t> dims,
                              float seed) {
            SynthTensor t;
            t.name = name;
            t.dtype = "F32";
            t.shape = dims;
            std::size_t n = 1;
            for (auto d : dims) n *= static_cast<std::size_t>(d);
            std::vector<float> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) * 0.125f + seed;
            t.data = f32_payload(v);
            ts.push_back(std::move(t));
        };
        auto push_f16_2 = [&](const std::string& name, std::vector<std::int64_t> dims,
                              int seed) {
            SynthTensor t;
            t.name = name;
            t.dtype = "F16";
            t.shape = dims;
            std::size_t n = 1;
            for (auto d : dims) n *= static_cast<std::size_t>(d);
            std::vector<std::uint16_t> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint16_t>((i + seed) & 0x3FFF);
            t.data = f16_payload(v);
            ts.push_back(std::move(t));
        };

        push_f32_2("token_embd.weight", {8, 16}, -0.5f);
        push_f32_2("output_norm.weight", {8}, 0.25f);
        push_f16_2("output.weight", {8, 16}, 1000);
        // blk.0: attn_q = Q4_0 (64 эл. = 2 блока), attn_v = Q8_0 (32 эл. = 1 блок).
        // Q4_0 блок 0: delta 1.0 (F16 0x3C00 → LE 00 3C), значения эл. i:
        // (nibble − 8) × 1.0, nibble = i % 16 → значения −8…7, −8…7 (эл. 0..31).
        std::vector<std::uint8_t> q40(64);
        q40[0] = 0x00;
        q40[1] = 0x3C;                       // delta 1.0 (НАЧАЛО блока)
        for (int i = 0; i < 32; ++i) {
            const int nib0 = (2 * i + 0) % 16;
            const int nib1 = (2 * i + 1) % 16;
            q40[2 + i] = static_cast<std::uint8_t>((nib1 << 4) | nib0);
        }
        // Блок 1: delta 0.5 (F16 0x3800 → LE 00 38), nibbles 0x08 → значение 0.
        q40[18] = 0x00;
        q40[19] = 0x38;
        for (int i = 0; i < 16; ++i) q40[20 + i] = 0x88;
        push_raw("blk.0.attn_q.weight", {8, 8}, 20, q40);
        // Q8_0: 32 int8 (значения q = i − 16 → значения q×0.5 = −8…7.5),
        // delta 0.5 в КОНЦЕ блока (байты 32..33).
        std::vector<std::uint8_t> q80(34);
        for (int i = 0; i < 32; ++i) q80[i] = static_cast<std::uint8_t>(i - 16);
        q80[32] = 0x00;
        q80[33] = 0x38;                      // delta 0.5 (КОНЕЦ блока)
        push_raw("blk.0.attn_v.weight", {8, 8}, 36, q80);
        // Остальные тензора слоя 0 — F16/F32 (n_kv_heads по умолчанию = 2).
        push_f32_2("blk.0.attn_norm.weight", {8}, 0.5f);
        push_f16_2("blk.0.attn_k.weight", {8, 8}, 5);
        push_f16_2("blk.0.attn_output.weight", {8, 8}, 17);
        push_f32_2("blk.0.ffn_norm.weight", {8}, 0.75f);
        push_f16_2("blk.0.ffn_gate.weight", {8, 16}, 19);
        push_f16_2("blk.0.ffn_up.weight", {8, 16}, 23);
        push_f16_2("blk.0.ffn_down.weight", {16, 8}, 27);

        auto meta8 = make_meta(1, true, false);
        for (auto& kv : meta8)
            if (kv.key == "llama.block_count") kv.u64 = 1;
            else if (kv.key == "llama.attention.head_count_kv") kv.u64 = 2;
        write_gguf_meta(p8, ts, meta8);

        const ModelDescription d8 = parse_model_description(gguf_to_description(p8));
        auto loader8 = make_loader(WeightsFormat::GGUF);
        auto src8 = loader8->open(p8, d8);
        // wq: Q4_0 → dequant → f32 → группа F16: значения побайтово.
        const int wq = find_group(d8, "l000.attn.wq");
        CHECK((wq >= 0) && (d8.group_source_patterns[wq] == "blk.0.attn_q.weight"));
        std::vector<std::uint8_t> buf(d8.groups[wq].slot_bytes, 0);
        src8->read({d8.groups[wq].id, d8.groups[wq].role}, buf.data(), buf.size());
        for (int i = 0; i < 64; ++i) {
            const float v = (i < 32) ? static_cast<float>((i % 16) - 8) * 1.0f : 0.0f;
            const std::uint16_t h = f32_to_f16_bits(v);
            CHECK(buf[2 * i] == static_cast<std::uint8_t>(h & 0xFF));
            CHECK(buf[2 * i + 1] == static_cast<std::uint8_t>((h >> 8) & 0xFF));
        }
        // wv: Q8_0 → значения (i − 16) × 0.5.
        const int wv = find_group(d8, "l000.attn.wv");
        CHECK((wv >= 0) && (d8.group_source_patterns[wv] == "blk.0.attn_v.weight"));
        std::vector<std::uint8_t> bufv(d8.groups[wv].slot_bytes, 0);
        src8->read({d8.groups[wv].id, d8.groups[wv].role}, bufv.data(), bufv.size());
        for (int i = 0; i < 32; ++i) {
            const float v = static_cast<float>(i - 16) * 0.5f;
            const std::uint16_t h = f32_to_f16_bits(v);
            CHECK(bufv[2 * i] == static_cast<std::uint8_t>(h & 0xFF));
            CHECK(bufv[2 * i + 1] == static_cast<std::uint8_t>((h >> 8) & 0xFF));
        }
        src8->close();
    }
}

TEST_MAIN("test_models_gguf_description")

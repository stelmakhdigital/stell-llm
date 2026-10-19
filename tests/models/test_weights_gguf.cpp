// Тесты лоадера GGUF v3 (subset) — контракт ADR-09 §3.
#include "stell/models/weights.hpp"
#include "stell/models/registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "quant.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"

using namespace stell::models;
using namespace stell_test;

namespace {

ModelDescription load_mini_dense() {
    Registry r;
    return r.load(STELL_FIXTURES_DIR "/mini_dense.json");
}

std::vector<SynthTensor> make_tensors() {
    // ВАЖНО (round 2, конвенция shape GGUF): для матриц GGUF ne хранит
    // {in, out} (байты C-порядка совпадают с row-major [out, in]); лоадер
    // возвращает развернутый dims (логическую форму по конвенции раскладки
// safetensors/JSON-конфигов). Писатeль
    // write_gguf пишет dims КАК ЕСТЬ, поэтому тест передаёт GGUF-нативные
    // dims (развернутые относительно логических форм).
    std::vector<SynthTensor> out;
    auto push_f32 = [&](const std::string& name, std::vector<std::int64_t> shape, float seed) {
        std::reverse(shape.begin(), shape.end());  // GGUF-native dims (ne[0] = input dim)
        SynthTensor t;
        t.name = name;
        t.dtype = "F32";
        t.shape = shape;
        std::size_t n = 1;
        for (auto d : shape) n *= static_cast<std::size_t>(d);
        std::vector<float> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) * 0.125f + seed;
        t.data = f32_payload(v);
        out.push_back(std::move(t));
    };
    auto push_f16 = [&](const std::string& name, std::vector<std::int64_t> shape, int seed) {
        std::reverse(shape.begin(), shape.end());  // GGUF-native dims (ne[0] = input dim)
        SynthTensor t;
        t.name = name;
        t.dtype = "F16";
        t.shape = shape;
        std::size_t n = 1;
        for (auto d : shape) n *= static_cast<std::size_t>(d);
        std::vector<std::uint16_t> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint16_t>((i + seed) & 0x3FFF);
        t.data = f16_payload(v);
        out.push_back(std::move(t));
    };

    push_f32("embedding.weight", {16, 8}, -0.5f);
    push_f32("model.norm.weight", {8}, 0.25f);
    push_f32("head.weight", {16, 8}, 0.0f);
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        push_f32(p + "input_layernorm.weight", {8}, 0.5f);
        push_f16(p + "self_attn.q_proj.weight", {8, 8}, layer * 13 + 3);
        push_f16(p + "self_attn.k_proj.weight", {4, 8}, layer * 21 + 5);
        push_f16(p + "self_attn.v_proj.weight", {4, 8}, layer * 29 + 9);
        push_f16(p + "self_attn.o_proj.weight", {8, 8}, layer * 31 + 17);
        push_f16(p + "mlp.gate_proj.weight", {16, 8}, layer * 17 + 19);
        push_f16(p + "mlp.up_proj.weight", {16, 8}, layer * 7 + 23);
        push_f16(p + "mlp.down_proj.weight", {8, 16}, layer * 11 + 27);
    }
    return out;
}

}  // namespace

void body() {
    const ModelDescription d = load_mini_dense();
    const std::string path = "/tmp/stell_test_mini_dense.gguf";
    const std::vector<SynthTensor> ts = make_tensors();
    write_gguf(path, ts);
    CHECK(detect_format(path) == WeightsFormat::GGUF);

    auto loader = make_loader(WeightsFormat::GGUF);
    auto src = loader->open(path, d);

    // embedding.weight: F32 [16,8] → группа F16: лоадер конвертирует F32→F16.
    // Первый элемент 16*8-tensora = -0.5 → half 0xFB00 → LE-байты 00 FB.
    {
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            if (d.groups[i].name != "embed") continue;
            const auto& g = d.groups[i];
            std::vector<std::uint8_t> buf(g.slot_bytes, 0xAB);
            src->read({g.id, g.role}, buf.data(), buf.size());
            CHECK(buf.size() == g.slot_bytes);
            CHECK(buf[0] == 0x00 && buf[1] == 0xB8);  // -0.5 half = 0xB800
            break;
        }
    }

    // wq layers.0: F16-тензор → побайтовое совпадение payload (первый элемент
    // = 3 → half 0x4300? нет: 3.0 half = 0x4000+0x0300... проверяем по битам).
    {
        bool wq_checked = false;
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            if (d.group_source_patterns[i] != "layers.0.self_attn.q_proj.weight") continue;
            const auto& g = d.groups[i];
            std::vector<std::uint8_t> buf(g.slot_bytes, 0);
            src->read({g.id, g.role}, buf.data(), buf.size());
            const SynthTensor* tq = nullptr;
            for (const auto& x : ts)
                if (x.name == "layers.0.self_attn.q_proj.weight") tq = &x;
            CHECK(tq != nullptr);
            CHECK(std::memcmp(buf.data(), tq->data.data(), tq->data.size()) == 0);
            wq_checked = true;
            break;
        }
        CHECK(wq_checked);
    }

    // down layers.1: F16-тензор → побайтовое совпадение payload (ts[16]).
    {
        bool down_checked = false;
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            if (d.group_source_patterns[i] != "layers.1.mlp.down_proj.weight") continue;
            const auto& g = d.groups[i];
            std::vector<std::uint8_t> buf(g.slot_bytes, 0);
            src->read({g.id, g.role}, buf.data(), buf.size());
            const SynthTensor* td = nullptr;
            for (const auto& x : ts)
                if (x.name == "layers.1.mlp.down_proj.weight") td = &x;
            CHECK(td != nullptr);
            CHECK(std::memcmp(buf.data(), td->data.data(), td->data.size()) == 0);
            down_checked = true;
            break;
        }
        CHECK(down_checked);
    }
    src->close();

    // --- ошибки --------------------------------------------------------------
    // Bad magic → SchemaMismatch.
    {
        const std::string p2 = "/tmp/stell_test_bad.gguf";
        std::FILE* f = std::fopen(p2.c_str(), "wb");
        std::fwrite("XXXX", 1, 4, f);
        std::fwrite("GGUF", 1, 4, f);
        std::fclose(f);
        CHECK_THROW(loader->open(p2, d), LoaderError, "bad magic rejected");
        try {
            loader->open(p2, d);
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::SchemaMismatch);
        }
    }
    // Версия 2 → SchemaMismatch.
    {
        const std::string p3 = "/tmp/stell_test_v2.gguf";
        std::FILE* f = std::fopen(p3.c_str(), "wb");
        std::fwrite("GGUF", 1, 4, f);
        stell_test::put_u32_file(f, 2);
        std::fclose(f);
        CHECK_THROW(loader->open(p3, d), LoaderError, "unsupported version rejected");
    }
    // Квантованный dtype GGUF (Q4_0 = 2) → UnsupportedQuant.
    {
        const std::string p4 = "/tmp/stell_test_q40.gguf";
        std::vector<SynthTensor> ts4 = make_tensors();
        // Подмена dtype-кода тензора: перепишем файл руками (magic+version+counts).
        std::FILE* f = std::fopen(p4.c_str(), "wb");
        std::fwrite("GGUF", 1, 4, f);
        stell_test::put_u32_file(f, 3);
        // n_tensors=1, n_kv=0, тензор: name, n_dims=1, dim=1, dtype=2, offset=0.
        stell_test::put_u64_file(f, 1);
        stell_test::put_u64_file(f, 0);
        stell_test::put_string_file(f, "embedding.weight");
        stell_test::put_u32_file(f, 1);
        stell_test::put_u64_file(f, 1);
        stell_test::put_u32_file(f, 2);   // неподдержанный dtype-код → UnsupportedQuant
        stell_test::put_u64_file(f, 0);
        std::fclose(f);
        CHECK_THROW(loader->open(p4, d), LoaderError, "quantized gguf dtype rejected");
        try {
            loader->open(p4, d);
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::UnsupportedQuant);
        }
    }
    // Отсутствующий тензор → TensorNotFound.
    {
        ModelDescription d2 = load_mini_dense();
        d2.group_source_patterns[0] = "nope.weight";
        CHECK_THROW(loader->open(path, d2), LoaderError, "missing gguf tensor rejected");
    }
}

TEST_MAIN("test_models_weights_gguf")

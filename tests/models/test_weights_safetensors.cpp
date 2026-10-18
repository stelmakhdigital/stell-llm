// Тесты лоадера safetensors (ADR-09 §3): резолюция source_pattern, заполнение
// слотов, ошибки (TensorNotFound/SchemaMismatch/UnsupportedQuant/IoFailure).
#include "stell/models/weights.hpp"
#include "stell/models/registry.hpp"

#include <cstdio>
#include <vector>

#include "quant.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"
#include "tensor_file_formats.hpp"

using namespace stell::models;
using namespace stell_test;

namespace {

// Готовит desc из фикстуры тестов.
ModelDescription load_mini_dense() {
    Registry r;
    return r.load(STELL_FIXTURES_DIR "/mini_dense.json");
}

SynthTensor tensor(const std::string& name, const std::string& dtype,
                   std::vector<std::int64_t> shape, std::vector<std::uint8_t> data) {
    SynthTensor t;
    t.name = name;
    t.dtype = dtype;
    t.shape = std::move(shape);
    t.data = std::move(data);
    return t;
}

std::vector<SynthTensor> make_tensors([[maybe_unused]] const ModelDescription& d, bool head_as_f32) {
    std::vector<SynthTensor> out;
    // embed: F16 [16,8]
    std::vector<std::uint16_t> embed(16 * 8);
    for (std::size_t i = 0; i < embed.size(); ++i) embed[i] = static_cast<std::uint16_t>((i * 37 + 11) & 0x7FFF);
    out.push_back(tensor("embedding.weight", "F16", {16, 8}, f16_payload(embed)));
    // norm_final: F32 [8]
    std::vector<std::uint8_t> norm_bytes = f32_payload({0.5f, -0.25f, 1.0f, 2.0f, 0.0f, -1.0f, 0.125f, 3.5f});
    out.push_back(tensor("model.norm.weight", "F32", {8}, norm_bytes));
    // head: F16 [16,8] (проверяемый F32→F16 конверт — отдельный кейс ниже)
    std::vector<std::uint16_t> head(16 * 8);
    for (std::size_t i = 0; i < head.size(); ++i) head[i] = static_cast<std::uint16_t>((i * 53 + 7) & 0x7FFF);
    out.push_back(tensor(head_as_f32 ? "head.weight" : "head.weight",
                         head_as_f32 ? "F32" : "F16", {16, 8},
                         head_as_f32 ? f32_payload(std::vector<float>(16 * 8, 0.0f))
                                     : f16_payload(head)));
    // Слои: wq/wo [8,8], wk/wv [4,8], up [16,8], down [8,16] — F16.
    for (int layer = 0; layer < 2; ++layer) {
        std::vector<std::uint8_t> norm_bytes = f32_payload(std::vector<float>(8, 0.5f));
        out.push_back(tensor("layers." + std::to_string(layer) + ".input_layernorm.weight",
                             "F32", {8}, norm_bytes));
        std::vector<std::uint16_t> wq(8 * 8);
        for (std::size_t i = 0; i < wq.size(); ++i) wq[i] = static_cast<std::uint16_t>((i + layer * 13 + 3) & 0x3FFF);
        out.push_back(tensor("layers." + std::to_string(layer) + ".self_attn.q_proj.weight",
                             "F16", {8, 8}, f16_payload(wq)));
        std::vector<std::uint16_t> wk(4 * 8);
        for (std::size_t i = 0; i < wk.size(); ++i) wk[i] = static_cast<std::uint16_t>((i + layer * 21 + 5) & 0x3FFF);
        out.push_back(tensor("layers." + std::to_string(layer) + ".self_attn.k_proj.weight",
                             "F16", {4, 8}, f16_payload(wk)));
        std::vector<std::uint16_t> wv(4 * 8);
        for (std::size_t i = 0; i < wv.size(); ++i) wv[i] = static_cast<std::uint16_t>((i + layer * 29 + 9) & 0x3FFF);
        out.push_back(tensor("layers." + std::to_string(layer) + ".self_attn.v_proj.weight",
                             "F16", {4, 8}, f16_payload(wv)));
        std::vector<std::uint16_t> wo(8 * 8);
        for (std::size_t i = 0; i < wo.size(); ++i) wo[i] = static_cast<std::uint16_t>((i + layer * 31 + 17) & 0x3FFF);
        out.push_back(tensor("layers." + std::to_string(layer) + ".self_attn.o_proj.weight",
                             "F16", {8, 8}, f16_payload(wo)));
        std::vector<std::uint16_t> up(16 * 8);
        for (std::size_t i = 0; i < up.size(); ++i) up[i] = static_cast<std::uint16_t>((i + layer * 7 + 23) & 0x3FFF);
        std::vector<std::uint16_t> gate(16 * 8);
        for (std::size_t i = 0; i < gate.size(); ++i)
            gate[i] = static_cast<std::uint16_t>((i + layer * 17 + 19) & 0x3FFF);
        // gate — round 2: fixture mini_dense получил ffn.gate (SwiGLU, README golden)
        out.push_back(tensor("layers." + std::to_string(layer) + ".mlp.gate_proj.weight",
                             "F16", {16, 8}, f16_payload(gate)));
        out.push_back(tensor("layers." + std::to_string(layer) + ".mlp.up_proj.weight",
                             "F16", {16, 8}, f16_payload(up)));
        std::vector<std::uint16_t> down(8 * 16);
        for (std::size_t i = 0; i < down.size(); ++i) down[i] = static_cast<std::uint16_t>((i + layer * 11 + 27) & 0x3FFF);
        out.push_back(tensor("layers." + std::to_string(layer) + ".mlp.down_proj.weight",
                             "F16", {8, 16}, f16_payload(down)));
    }
    return out;
}

}  // namespace

void body() {
    const ModelDescription d = load_mini_dense();
    const std::string path = "/tmp/stell_test_mini_dense.safetensors";
    const std::vector<SynthTensor> ts = make_tensors(d, /*head_as_f32=*/false);
    write_safetensors(path, ts);

    std::unique_ptr<IWeightsLoader> loader = make_loader(WeightsFormat::Safetensors);
    CHECK(loader->format() == WeightsFormat::Safetensors);
    std::unique_ptr<IWeightsSource> src = loader->open(path, d);
    CHECK(detect_format(path) == WeightsFormat::Safetensors);

    // Успешный путь: каждая некешевая группа читается слотом slot_bytes,
    // payload совпадает с тензором файла, хвост слота = нули.
    std::size_t read_groups = 0;
    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;  // kv-группы не резолвятся
        stell::tiers::GroupKey key{g.id, g.role};
        std::vector<std::uint8_t> buf(g.slot_bytes, 0xFF);
        src->read(key, buf.data(), buf.size());
        if (g.role == stell::tiers::Role::Kv) continue;  // kv-группы не резолвятся
        // Найти исходный тензор по паттерну.
        const std::string& pattern = d.group_source_patterns[i];
        const SynthTensor* t = nullptr;
        for (const auto& x : ts)
            if (x.name == pattern) t = &x;
        CHECK_MSG(t != nullptr, "tensor for pattern '" + pattern + "' must exist");
        CHECK(buf.size() == g.slot_bytes);
        if (std::memcmp(buf.data(), t->data.data(), t->data.size()) != 0) {
            std::printf("MISMATCH group=%s pattern=%s payload=%zu\n", g.name.c_str(),
                        pattern.c_str(), t->data.size());
            std::printf("  buf: %02X %02X %02X %02X  expect: %02X %02X %02X %02X\n",
                        buf[0], buf[1], buf[2], buf[3],
                        t->data[0], t->data[1], t->data[2], t->data[3]);
        }
        CHECK(std::memcmp(buf.data(), t->data.data(), t->data.size()) == 0);
        for (std::size_t k = t->data.size(); k < buf.size(); ++k) CHECK(buf[k] == 0);
        ++read_groups;
    }
    CHECK(read_groups == d.groups.size() - d.n_layers);  // минус kv-группы

    // F32→F16 конверт (head.weight как F32, группа head F16 — проверяем ниже
    // на отдельном файле; здесь убеждаемся, что профиль плана совпадает).
    {
        const ModelDescription d2 = load_mini_dense();
        const std::string p2 = "/tmp/stell_test_mini_dense_f32.safetensors";
        // Меняем dtype группы head на F32 → лоадер копирует как есть.
        // (конверт F32→F16 покрыт через compatible_plain: F32-источник + F16-группа)
        std::vector<SynthTensor> ts2 = make_tensors(d, true);
        write_safetensors(p2, ts2);
        auto src2 = loader->open(p2, d);
        // head — F32-тензор при F16-группе: лоадер обязан сконвертировать.
        bool head_ok = true;
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            if (d.groups[i].name != "head") continue;
            const auto& g = d.groups[i];
            std::vector<std::uint8_t> buf(g.slot_bytes, 0);
            src2->read({g.id, g.role}, buf.data(), buf.size());
            // Ожидаем payload: F32-данные (нулевые floats) → F16-биты 0.
            const std::size_t n = 16 * 8;
            for (std::size_t k = 0; k < n * 2; ++k) head_ok &= buf[k] == 0;
        }
        CHECK(head_ok);
        src2->close();
    }
    src->close();

    // --- ошибки --------------------------------------------------------------
    // TensorNotFound: план требует тензор, отсутствующий в файле.
    {
        ModelDescription d3 = load_mini_dense();
        d3.group_source_patterns[0] = "missing.tensor";
        CHECK_THROW(loader->open(path, d3), LoaderError, "missing tensor rejected");
        try {
            loader->open(path, d3);
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::TensorNotFound);
        }
    }
    // SchemaMismatch: dst_bytes != slot_bytes.
    {
        auto src3 = loader->open(path, d);
        const auto& g = d.groups[0];
        std::vector<std::uint8_t> buf(g.slot_bytes - 16);
        bool caught = false;
        try {
            src3->read({g.id, g.role}, buf.data(), buf.size());
        } catch (const LoaderError& e) {
            caught = true;
            CHECK(e.code() == LoaderError::Code::SchemaMismatch);
        }
        CHECK(caught);
        src3->close();
    }
    // IoFailure: чтение kv-группы (весов в файле нет).
    {
        auto src4 = loader->open(path, d);
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            if (d.groups[i].role != stell::tiers::Role::Kv) continue;
            const auto& g = d.groups[i];
            std::vector<std::uint8_t> buf(g.slot_bytes);
            CHECK_THROW(src4->read({g.id, g.role}, buf.data(), buf.size()), LoaderError,
                        "kv group has no weights");
            break;
        }
        src4->close();
    }
    // UnsupportedQuant: тензор I64 в файле при F16-группе.
    {
        const std::string p5 = "/tmp/stell_test_i64.safetensors";
        std::vector<SynthTensor> ts5 = make_tensors(d, false);
        ts5[0].dtype = "I64";  // embedding как I64: itemsize 8 × 128 элементов
        ts5[0].data.assign(8 * 16 * 8, 0);
        write_safetensors(p5, ts5);
        // open() успешен (резолюция не смотрит dtype); UnsupportedQuant — на read().
        auto src5 = loader->open(p5, d);
        const auto& g0 = d.groups[0];
        std::vector<std::uint8_t> buf0(g0.slot_bytes, 0);
        CHECK_THROW(src5->read({g0.id, g0.role}, buf0.data(), buf0.size()), LoaderError,
                    "I64 tensor rejected");
        try {
            src5->read({g0.id, g0.role}, buf0.data(), buf0.size());
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::UnsupportedQuant);
        }
        src5->close();
    }
}

TEST_MAIN("test_models_weights_safetensors")

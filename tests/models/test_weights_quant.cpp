// Тесты профильного лоадера NVFP4-W4A16 (ADR-03: вариант лоадера, не входной
// формат): квантование на read(), размеры payload, ошибки.
#include "stell/models/weights.hpp"
#include "stell/models/registry.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include "quant.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"

using namespace stell::models;
using namespace stell_test;

namespace {

ModelDescription load_fixture(const std::string& file) {
    Registry r;
    return r.load(STELL_FIXTURES_DIR "/" + file);
}

// Все весовые паттерны mini_moe как F16-тензоры (источник квантования).
std::vector<SynthTensor> make_tensors(const ModelDescription& d) {
    std::vector<SynthTensor> out;
    for (std::size_t i = 0; i < d.group_source_patterns.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;
        SynthTensor t;
        t.name = d.group_source_patterns[i];
        // Группы F16/F32 отдаём как есть; NVFP4/W4A16 — источник F16.
        const bool quant = g.dtype == stell::tiers::DType::NVFP4 ||
                           g.dtype == stell::tiers::DType::W4A16;
        const std::string dtype = quant ? "F16" : (g.dtype == stell::tiers::DType::F32 ? "F32" : "F16");
        t.dtype = dtype;
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto dim : g.shape) n *= static_cast<std::size_t>(dim);
        if (dtype == "F32") {
            std::vector<float> v(n);
            for (std::size_t k = 0; k < n; ++k) v[k] = 0.5f;  // нормы
            t.data = f32_payload(v);
        } else {
            std::vector<std::uint16_t> v(n);
            for (std::size_t k = 0; k < n; ++k)
                v[k] = static_cast<std::uint16_t>(((k * 5 + 7) % 15) + 1);  // малые F16-биты
            t.data = f16_payload(v);
        }
        out.push_back(std::move(t));
    }
    return out;
}

float rel_err_nvfp4(float orig, float deq) {
    const float scale = std::max(std::max(std::fabs(orig), std::fabs(deq)), 1.0f);
    return std::fabs(orig - deq) / scale;
}

}  // namespace

void body() {
    const ModelDescription d = load_fixture("mini_moe.json");
    CHECK(d.dtype_weights == "NVFP4");
    const std::string path = "/tmp/stell_test_mini_moe.safetensors";
    const std::vector<SynthTensor> ts = make_tensors(d);
    write_safetensors(path, ts);

    // Готовим пары (индекс группы, тензор) для сравнения без переиндексации.
    std::vector<std::pair<std::size_t, const SynthTensor*>> tensor_of_group;
    {
        std::size_t k = 0;
        for (std::size_t i = 0; i < d.group_source_patterns.size(); ++i) {
            const auto& g = d.groups[i];
            if (g.role == stell::tiers::Role::Kv) continue;
            tensor_of_group.push_back({i, &ts[k++]});
        }
    }

    auto loader = make_loader(WeightsFormat::NVFP4W4A16);
    CHECK(loader->format() == WeightsFormat::NVFP4W4A16);
    auto src = loader->open(path, d);

    // NVFP4-группа wq layers.0 [8,8]: payload = 32 ниббла + 1 scale = 33 байта.
    {
        for (const auto& [gi, t] : tensor_of_group) {
            if (d.group_source_patterns[gi] != "layers.0.self_attn.q_proj.weight") continue;
            const auto& g = d.groups[gi];
            std::vector<std::uint8_t> buf(g.slot_bytes, 0xEE);
            src->read({g.id, g.role}, buf.data(), buf.size());
            CHECK(buf.size() == g.slot_bytes);
            const std::size_t n = 64;
            const std::size_t payload = 34;
            for (std::size_t k = payload; k < buf.size(); ++k) CHECK(buf[k] == 0);  // хвост слота
            // Референсный dequant: ошибка ≤ ~1/8 от maxabs (E2M1, блок 32).
            std::vector<float> deq(n);
            dequant_nvfp4(buf.data(), n, deq.data());
            for (std::size_t k = 0; k < n; ++k) {
                std::uint16_t h = static_cast<std::uint16_t>(t->data[2 * k]) |
                                  (static_cast<std::uint16_t>(t->data[2 * k + 1]) << 8);
                float orig = f16_bits_to_f32(h);
                if (orig == 0.0f) continue;  // нуль квантуется точно
                CHECK(rel_err_nvfp4(orig, deq[k]) < 0.25f);
            }
            break;
        }
    }
    src->close();

    // W4A16-профиль: описание собираем на лету (64-элементный тензор).
    {
        const std::string w4_json = R"({
          "schema": "1.0", "name": "mini-w4a16", "class": "dense",
          "dtype_weights": "W4A16",
          "vocab_size": 16, "d_model": 8, "n_layers": 1, "n_heads": 2, "d_head": 4,
          "kv": {"dtype": "F16", "n_kv_heads": 1, "d_kv": 4, "quant": null},
          "ops": [{"op": "attention", "io": ["hidden", "kv"], "params": {}}],
          "tensor_groups": [
            {"name": "wq", "role": "attn", "shape": [64], "dtype": "W4A16",
             "source_pattern": "wq.weight"}
          ]
        })";
        ModelDescription wd = parse_model_description(w4_json);
        const std::string p2 = "/tmp/stell_test_w4a16.safetensors";
        std::vector<SynthTensor> ts2;
        SynthTensor t;
        t.name = "wq.weight";
        t.dtype = "F32";
        t.shape = {64};
        std::vector<float> v(64);
        for (std::size_t k = 0; k < 64; ++k) v[k] = static_cast<float>(k % 9) - 4.0f;
        t.data = f32_payload(v);
        ts2.push_back(std::move(t));
        write_safetensors(p2, ts2);

        auto loader2 = make_loader(WeightsFormat::NVFP4W4A16);
        auto src2 = loader2->open(p2, wd);
        const auto& g = wd.groups[0];
        std::vector<std::uint8_t> buf(g.slot_bytes, 0);
        src2->read({g.id, g.role}, buf.data(), buf.size());
        CHECK(payload_bytes_for("W4A16", {64}) == 36);
        std::vector<float> deq(64);
        dequant_w4a16(buf.data(), 64, deq.data());
        for (std::size_t k = 0; k < 64; ++k) {
            const float orig = v[k];
            if (orig == 0.0f) { CHECK(deq[k] == 0.0f); continue; }
            // int4 [-8..7] со scale = maxabs/7: ошибка ≤ scale/2 ≤ ~0.35 для maxabs 4.
            CHECK(std::fabs(orig - deq[k]) <= 0.4f);
        }
        // Хвост слота = 0.
        for (std::size_t k = 36; k < buf.size(); ++k) CHECK(buf[k] == 0);
        src2->close();
    }

    // --- ошибки --------------------------------------------------------------
    // F16-профиль описания → UnsupportedQuant в конструкторе источника.
    {
        auto loader3 = make_loader(WeightsFormat::NVFP4W4A16);
        CHECK_THROW(loader3->open(path, load_fixture("mini_dense.json")), LoaderError,
                    "non-quantized profile rejected");
        try {
            loader3->open(path, load_fixture("mini_dense.json"));
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::UnsupportedQuant);
        }
    }
    // I64-источник при NVFP4-группе → UnsupportedQuant на read.
    {
        const std::string p4 = "/tmp/stell_test_nvfp4_bad.safetensors";
        std::vector<SynthTensor> ts4 = make_tensors(d);
        // wq — первый NVFP4-тензор: подменим dtype на I64.
        for (auto& t : ts4) {
            if (t.name == "layers.0.self_attn.q_proj.weight") {
                t.dtype = "I64";
                const std::size_t n = 64;
                t.data.assign(n * 8, 0);  // I64 itemsize 8: размер согласован с header
            }
        }
        write_safetensors(p4, ts4);
        auto loader4 = make_loader(WeightsFormat::NVFP4W4A16);
        auto src4 = loader4->open(p4, d);
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            if (d.group_source_patterns[i] != "layers.0.self_attn.q_proj.weight") continue;
            const auto& g = d.groups[i];
            std::vector<std::uint8_t> buf(g.slot_bytes, 0);
            CHECK_THROW(src4->read({g.id, g.role}, buf.data(), buf.size()), LoaderError,
                        "I64 source for NVFP4 group rejected");
            break;
        }
        src4->close();
    }
}

TEST_MAIN("test_models_weights_quant")

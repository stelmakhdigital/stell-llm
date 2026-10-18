// Golden-фикстуры весов (round 2): проверка, что committed
// tests/models/fixtures/<fixture>.safetensors согласованы с описанием
// (<fixture>.json) и с референсным квантованием quant.hpp:
//   - лоадер резолвит ВСЕ некешевые source_pattern (fail-fast, ADR-09 §3);
//   - слоты групп: payload по dtype, хвост выравнивания = нули;
//   - NVFP4-группы: payload == quantize_nvfp4(F16-тензора файла) ПОБИТНО
//     (референс quant.cpp == Python-порт generate_golden.py, на котором
//      посчитаны golden-логиты) + фиксированные hex-эталоны;
//   - golden-логиты (logits_<fixture>.json) согласованы с fixture:
//     model/profile, vocab_size, prompt_tokens в словаре.
// Это каркас интеграции с engine (core-2 в tests/core использует те же файлы).
// ============================================================================
#include "stell/models/convert.hpp"
#include "stell/models/registry.hpp"
#include "stell/models/weights.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "quant.hpp"
#include "synth_writers.hpp"
#include "tensor_file_formats.hpp"
#include "test_util.hpp"

using namespace stell::models;
using namespace stell_test;

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

std::string fixture_json(const std::string& name) {
    return fixtures_dir() + "/" + name + ".json";
}
std::string fixture_weights(const std::string& name) {
    return fixtures_dir() + "/" + name + ".safetensors";
}
std::string golden_logits(const std::string& name) {
    return golden_dir() + "/logits_" + name + ".json";
}

std::string bytes_hex(const std::uint8_t* p, std::size_t n) {
    static const char* kHex = "0123456789ABCDEF";
    std::string out;
    for (std::size_t i = 0; i < n; ++i) {
        out += kHex[(p[i] >> 4) & 0xF];
        out += kHex[p[i] & 0xF];
    }
    return out;
}

void check_hex(const std::string& name, const std::uint8_t* payload,
               std::size_t payload_bytes, const char* expected_hex) {
    const std::string got = bytes_hex(payload, payload_bytes);
    CHECK_MSG(got == expected_hex,
              name + ": NVFP4 payload != референс quant.cpp (Python-порт): got " +
                  got.substr(0, 16) + "... expected " +
                  std::string(expected_hex).substr(0, 16) + "...");
}

// Ожидаемые NVFP4 payload (биты) для выбранных тензоров mini_moe — сгенерированы
// tests/models/golden/generate_golden.py (Python-порт quant.cpp; сверка
// побитно против C++ quant.cpp выполнена при генерации раунда).
const char* kExpectedQ0 = "5E391585C2AE54C344D1E23429E3C06A619C5D6D6C8D52299E68D25C91CD69581D1C";
const char* kExpectedE3DownL1 = "AE05FFE1D9C6F74FF7FD7DFADEDE2359EA928DBAEF252FDD7E566605DEDFFC571818";
const char* kExpectedE0UpL0 = "55E155D234D1D62C19496224DEC56DDB24CE55D2C11DCB65D5D5BDDB35BDDC691C1C";

void check_fixture(const std::string& fixture) {
    Registry r;
    const ModelDescription d = r.load(fixture_json(fixture));
    const std::string weights_path = fixture_weights(fixture);
    CHECK(std::filesystem::exists(weights_path));
    CHECK(detect_format(weights_path) == WeightsFormat::Safetensors);

    // Лоадер по профилю: F16 → Safetensors; NVFP4/W4A16 → NVFP4W4A16.
    // (Round 2: NVFP4W4A16-лоадер обслуживает плоские группы тоже.)
    const WeightsFormat fmt = d.dtype_weights == "F16" ? WeightsFormat::Safetensors
                                                       : WeightsFormat::NVFP4W4A16;
    std::unique_ptr<IWeightsLoader> loader = make_loader(fmt);
    std::unique_ptr<IWeightsSource> src = loader->open(weights_path, d);

    // Карта тензоров файла (для независимой проверки NVFP4-групп).
    std::FILE* fw = std::fopen(weights_path.c_str(), "rb");
    if (!fw) throw std::runtime_error("cannot open " + weights_path);
    TensorFileMap tfm = parse_safetensors_header(fw);

    for (std::size_t i = 0; i < d.groups.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;  // kv-группы не резолвятся
        const std::string& pattern = d.group_source_patterns[i];
        auto it = tfm.by_name.find(pattern);
        CHECK_MSG(it != tfm.by_name.end(),
                  fixture + ": tensor '" + pattern + "' must exist");
        const TensorInfo& t = it->second;
        CHECK_MSG(t.shape == g.shape, fixture + ": shape mismatch for " + pattern);

        // Размер payload по dtype плана (ADR-03), хвост слота = нули.
        std::size_t n = 1;
        for (auto dim : g.shape) n *= static_cast<std::size_t>(dim);
        std::vector<std::uint8_t> buf(g.slot_bytes, 0x5A);
        src->read({g.id, g.role}, buf.data(), buf.size());
        CHECK(buf.size() == g.slot_bytes);
        const std::size_t payload = [&] {
            switch (g.dtype) {
                case stell::tiers::DType::F16: return n * 2;
                case stell::tiers::DType::F32: return n * 4;
                case stell::tiers::DType::E4M3: return n;
                case stell::tiers::DType::NVFP4: return (n + 1) / 2 + (n + 31) / 32;
                case stell::tiers::DType::W4A16:
                    return (n + 1) / 2 + 2 * ((n + 31) / 32);
            }
            return std::size_t{0};
        }();
        CHECK(payload <= g.slot_bytes);
        for (std::size_t k = payload; k < buf.size(); ++k)
            CHECK_MSG(buf[k] == 0, fixture + ": slot tail must be zero");

        // NVFP4-группы: независимая сверка payload == quantize_nvfp4(F16→f32).
        if (g.dtype == stell::tiers::DType::NVFP4) {
            std::vector<std::uint8_t> raw(t.bytes);
            read_exact(fw, t.offset, raw.data(), raw.size());
            std::vector<float> f32(n);
            for (std::size_t k = 0; k < n; ++k) {
                const std::uint16_t h = static_cast<std::uint16_t>(raw[2 * k]) |
                                        (static_cast<std::uint16_t>(raw[2 * k + 1]) << 8);
                f32[k] = f16_bits_to_f32(h);
            }
            std::vector<std::uint8_t> q(payload);
            quantize_nvfp4(f32.data(), n, q.data());
            CHECK_MSG(std::memcmp(q.data(), buf.data(), payload) == 0,
                      fixture + ": NVFP4 group '" + g.name +
                          "' payload != quantize(file tensor) (quant.cpp)");
            // Фиксированные hex-эталоны (Python↔C++ побитно, раунд 2).
            if (pattern == "layers.0.self_attn.q_proj.weight")
                check_hex(g.name, buf.data(), payload, kExpectedQ0);
            if (pattern == "layers.1.mlp.experts.3.down.weight")
                check_hex(g.name, buf.data(), payload, kExpectedE3DownL1);
            if (pattern == "layers.0.mlp.experts.0.up.weight")
                check_hex(g.name, buf.data(), payload, kExpectedE0UpL0);
        }
    }
    src->close();
    std::fclose(fw);
    std::printf("golden fixtures %s: OK\n", fixture.c_str());
}

void check_golden_logits_files() {
    for (const char* name : {"mini_dense", "mini_moe"}) {
        const std::string path = golden_logits(name);
        CHECK(std::filesystem::exists(path));
        const JsonValue fx = json_parse(read_file(path));
        CHECK(fx.get_string("schema") == "1.0");
        CHECK(fx.get_string("model") == std::string(name == "mini_dense"
                                                        ? "mini-dense-test"
                                                        : "mini-moe-test"));
        CHECK(fx.find("greedy")->bool_v == true);
        CHECK(fx.get_uint("seed") == 0);
        CHECK(fx.get_double("tolerance_rel") > 0.0);
        CHECK(!fx.get_array("logits_ref").array_v.empty());
        // prompt_tokens в словаре fixture
        const ModelDescription d = Registry().load(fixture_json(name));
        for (const auto& tok : fx.get_array("prompt_tokens").array_v)
            CHECK(tok.int_v < static_cast<std::int64_t>(d.vocab_size));
    }
    std::printf("golden logits files: OK\n");
}

}  // namespace

void body() {
    check_fixture("mini_dense");
    check_fixture("mini_moe");
    check_golden_logits_files();
}

TEST_MAIN("test_models_golden_fixtures")

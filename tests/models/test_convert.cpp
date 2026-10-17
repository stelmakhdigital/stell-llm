// Тесты конвертера stell-tier (ADR-12 §1/§2) + лоадера StellTier roundtrip:
// раскладка header/index/data по спецификации, CRC, согласованность с планом.
#include "stell/models/convert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "quant.hpp"
#include "stelltier_format.hpp"
#include "synth_writers.hpp"
#include "test_util.hpp"
#include "tensor_file_formats.hpp"
#include "stell/models/weights.hpp"
#include "stell/models/registry.hpp"

using namespace stell::models;
using namespace stell_test;

namespace {

ModelDescription load_fixture(const std::string& file) {
    Registry r;
    return r.load(STELL_FIXTURES_DIR "/" + file);
}

std::vector<SynthTensor> make_tensors(const ModelDescription& d) {
    std::vector<SynthTensor> out;
    for (std::size_t i = 0; i < d.group_source_patterns.size(); ++i) {
        const auto& g = d.groups[i];
        if (g.role == stell::tiers::Role::Kv) continue;
        SynthTensor t;
        t.name = d.group_source_patterns[i];
        const bool quant = g.dtype == stell::tiers::DType::NVFP4 ||
                           g.dtype == stell::tiers::DType::W4A16;
        t.dtype = quant ? "F16" : (g.dtype == stell::tiers::DType::F32 ? "F32" : "F16");
        t.shape = g.shape;
        std::size_t n = 1;
        for (auto dim : g.shape) n *= static_cast<std::size_t>(dim);
        if (t.dtype == "F32") {
            std::vector<float> v(n);
            for (std::size_t k = 0; k < n; ++k) v[k] = 0.25f;
            t.data = f32_payload(v);
        } else {
            std::vector<std::uint16_t> v(n);
            for (std::size_t k = 0; k < n; ++k)
                v[k] = static_cast<std::uint16_t>(((k * 3 + 5) % 31) + 1);
            t.data = f16_payload(v);
        }
        out.push_back(std::move(t));
    }
    return out;
}

}  // namespace

void body() {
    // --- safetensors → stell-tier (F16-профиль) ------------------------------
    const ModelDescription d = load_fixture("mini_dense.json");
    const std::string in_path = "/tmp/stell_conv_in.safetensors";
    const std::vector<SynthTensor> ts = make_tensors(d);
    write_safetensors(in_path, ts);

    StellTierConverter conv;
    const std::string out_dir = "/tmp/stell_conv_out";
    conv.convert(d, in_path, out_dir);
    const std::string layout_path = out_dir + "/" + d.name + ".stelltier";

    std::uint64_t file_size = 0;
    const StellTierIndex idx = stelltier_read_header_and_index(layout_path, &file_size);

    // Ожидаемое число групп = не-KV группы плана.
    std::size_t n_expected = 0;
    for (const auto& g : d.groups)
        if (g.role != stell::tiers::Role::Kv) ++n_expected;
    CHECK(idx.n_groups == n_expected);
    CHECK(idx.entries.size() == n_expected);
    CHECK(idx.dtype_profile == 1);            // F16 (ADR-12 §2)
    CHECK(idx.model_name == d.name);

    // Раскладка: header 512, индекс на 0x200, данные монотонно с 4096, без дыр.
    // Индекс записи соответствуют некешевым группам плана в их порядке.
    std::vector<std::size_t> nonkv;
    for (std::size_t i = 0; i < d.groups.size(); ++i)
        if (d.groups[i].role != stell::tiers::Role::Kv) nonkv.push_back(i);
    CHECK(idx.data_offset == 4096);           // 512 + 17*64 = 1600 → align 4096
    std::uint64_t cursor = idx.data_offset;
    for (std::size_t i = 0; i < idx.entries.size(); ++i) {
        const StellTierEntry& e = idx.entries[i];
        CHECK(e.data_offset == cursor);
        CHECK(e.slot_bytes % kStellTierSlotAlign == 0);
        const auto& g = d.groups[nonkv[i]];
        CHECK(e.group_id == g.id.value);
        CHECK(e.role == stelltier_role_code(g.role));
        CHECK(e.dtype == stelltier_dtype_code(g.dtype));
        CHECK(e.slot_bytes == g.slot_bytes);
        CHECK(e.shape_rank == g.shape.size());
        if (e.name != g.name.substr(0, 20))
            std::printf("NAME MISMATCH idx=%zu entry(len=%zu)='%s' group(len=%zu)='%s'\n",
                        i, e.name.size(), e.name.c_str(), g.name.size(),
                        g.name.substr(0, 16).c_str());
        CHECK(e.name == g.name.substr(0, 16));
        cursor += e.slot_bytes;
    }
    CHECK(cursor <= file_size);

    // CRC заголовка: пересчёт по [0..0x2C) == поле 0x2C (ADR-12 §2.3).
    {
        std::FILE* f = std::fopen(layout_path.c_str(), "rb");
        std::array<std::uint8_t, 512> h{};
        CHECK(std::fread(h.data(), 1, h.size(), f) == h.size());
        std::fclose(f);
        std::uint32_t stored = 0;
        for (int b = 0; b < 4; ++b)
            stored |= static_cast<std::uint32_t>(h[0x2C + b]) << (8 * b);
        CHECK(stelltier_header_crc32(h) == stored);
    }

    // Roundtrip: лоадер StellTier читает группы → payload == исходный тензор.
    {
        auto loader = make_loader(WeightsFormat::StellTier);
        auto src = loader->open(layout_path, d);
        std::size_t k = 0;
        for (std::size_t i = 0; i < d.groups.size(); ++i) {
            const auto& g = d.groups[i];
            if (g.role == stell::tiers::Role::Kv) continue;
            std::vector<std::uint8_t> buf(g.slot_bytes, 0xCD);
            src->read({g.id, g.role}, buf.data(), buf.size());
            const SynthTensor& t = ts[k++];
            CHECK(std::memcmp(buf.data(), t.data.data(), t.data.size()) == 0);
            for (std::size_t z = t.data.size(); z < buf.size(); ++z) CHECK(buf[z] == 0);
        }
        CHECK(k == n_expected);
        src->close();
    }

    // --- повреждение CRC группы → ChecksumMismatch --------------------------
    {
        const std::string corrupt_path = "/tmp/stell_conv_corrupt.stelltier";
        std::FILE* srcf = std::fopen(layout_path.c_str(), "rb");
        std::vector<std::uint8_t> blob;
        int c;
        while ((c = std::fgetc(srcf)) != EOF) blob.push_back(static_cast<std::uint8_t>(c));
        std::fclose(srcf);
        blob[idx.entries[1].data_offset + 3] ^= 0xFF;  // третий байт payload группы 1
        std::FILE* out = std::fopen(corrupt_path.c_str(), "wb");
        std::fwrite(blob.data(), 1, blob.size(), out);
        std::fclose(out);

        auto loader = make_loader(WeightsFormat::StellTier);
        auto src = loader->open(corrupt_path, d);  // open OK (CRC ленивый, ADR-12 §2.3)
        const auto& g = d.groups[nonkv[1]];
        std::vector<std::uint8_t> buf(g.slot_bytes, 0);
        CHECK_THROW(src->read({g.id, g.role}, buf.data(), buf.size()), LoaderError,
                    "corrupted group CRC rejected");
        try {
            src->read({g.id, g.role}, buf.data(), buf.size());
        } catch (const LoaderError& e) {
            CHECK(e.code() == LoaderError::Code::ChecksumMismatch);
        }
        src->close();
    }

    // --- повреждение заголовка → ChecksumMismatch ---------------------------
    {
        const std::string corrupt_path = "/tmp/stell_conv_corrupt_header.stelltier";
        std::FILE* srcf = std::fopen(layout_path.c_str(), "rb");
        std::vector<std::uint8_t> blob;
        int c;
        while ((c = std::fgetc(srcf)) != EOF) blob.push_back(static_cast<std::uint8_t>(c));
        std::fclose(srcf);
        blob[0x10] ^= 0xFF;  // created_ts
        std::FILE* out = std::fopen(corrupt_path.c_str(), "wb");
        std::fwrite(blob.data(), 1, blob.size(), out);
        std::fclose(out);
        auto loader = make_loader(WeightsFormat::StellTier);
        CHECK_THROW(loader->open(corrupt_path, d), LoaderError, "corrupted header rejected");
    }

    // --- MoE NVFP4-профиль: conversion квантует, dtype_profile = 2 ----------
    {
        const ModelDescription m = load_fixture("mini_moe.json");
        const std::string in2 = "/tmp/stell_conv_moe.safetensors";
        const std::vector<SynthTensor> ts2 = make_tensors(m);
        write_safetensors(in2, ts2);
        conv.convert(m, in2, out_dir);
        const std::string lp = out_dir + "/" + m.name + ".stelltier";
        const StellTierIndex idx2 = stelltier_read_header_and_index(lp, nullptr);
        CHECK(idx2.dtype_profile == 2);  // NVFP4
        // wq layers.0 — пятая некешевая группа плана (gid 4): dtype код 3 (NVFP4).
        CHECK(idx2.entries[4].dtype == 3);
        auto loader = make_loader(WeightsFormat::StellTier);
        auto src = loader->open(lp, m);
        std::size_t k = 0;
        for (std::size_t i = 0; i < m.groups.size(); ++i) {
            const auto& g = m.groups[i];
            if (g.role == stell::tiers::Role::Kv) continue;
            std::vector<std::uint8_t> buf(g.slot_bytes, 0);
            src->read({g.id, g.role}, buf.data(), buf.size());
            const SynthTensor& t = ts2[k++];
            if (g.dtype == stell::tiers::DType::NVFP4 || g.dtype == stell::tiers::DType::W4A16) {
                const std::size_t n = 1;
                (void)n;
                std::size_t numel = 1;
                for (auto dim : g.shape) numel *= static_cast<std::size_t>(dim);
                std::vector<float> deq(numel);
                if (g.dtype == stell::tiers::DType::NVFP4)
                    dequant_nvfp4(buf.data(), numel, deq.data());
                else
                    dequant_w4a16(buf.data(), numel, deq.data());
                // Источник F16: значения ((k*3+5)%31)+1 → half bits, большие числа;
                // квантование с block-32 scale: ошибка ≤ ~12% от maxabs блока.
                for (std::size_t z = 0; z < numel; ++z) {
                    std::uint16_t h = static_cast<std::uint16_t>(t.data[2 * z]) |
                                      (static_cast<std::uint16_t>(t.data[2 * z + 1]) << 8);
                    float orig = f16_bits_to_f32(h);
                    float err = std::fabs(orig - deq[z]);
                    float mag = std::max(std::max(orig, deq[z]), 1.0f);
                    CHECK(err / mag < 0.25f);
                }
            } else {
                CHECK(std::memcmp(buf.data(), t.data.data(), t.data.size()) == 0);
            }
        }
        src->close();
    }
}

TEST_MAIN("test_models_convert_stelltier")

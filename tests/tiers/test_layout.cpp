// Тесты stell-tier layout v1 (ADR-12): roundtrip, выравнивания, CRC-повреждение.
#include <stell/tiers/layout.hpp>

#include <fstream>
#include <stdexcept>
#include <vector>

#include "harness.hpp"

using namespace stell::tiers;

static std::vector<TensorGroup> sample_plan(std::size_t slot_bytes) {
    return {
        make_group(1, "dense.l000.attn.qkv", Role::Attn, DType::F16, slot_bytes, {3, 512, 128}),
        make_group(2, "dense.l000.kv.cache", Role::Kv, DType::F32, slot_bytes, {512, 64}),
        make_group(3, "moe.l007.e003.wq", Role::Expert, DType::NVFP4, slot_bytes, {128, 256}),
        make_group(4, "moe.l007.e004.wk", Role::Expert, DType::W4A16, slot_bytes, {128, 256}),
    };
}

static void write_layout(const std::string& path, const std::vector<TensorGroup>& groups) {
    TierLayoutHeader h;
    h.dtype_profile = 1; // F16 (профиль — только метка, ADR-12 §2)
    h.n_groups = groups.size();
    h.model_name = "test-model";
    TierLayoutWriter w(path, h);
    std::uint8_t payload_byte = 0;
    for (const auto& g : groups) {
        // payload меньше слота: проверяем, что остаток (padding) нулевой.
        std::vector<std::uint8_t> payload(g.slot_bytes / 2, payload_byte++);
        w.add_group(g, /*tensor_count=*/1, payload.data(), payload.size());
    }
    w.finalize();
}

T_CASE(layout_roundtrip) {
    const std::size_t slot = 4096; // ADR-12 §2.4: кратен 4096
    const std::string path = "/tmp/stell_tier_test_layout.stelltier";
    const auto groups = sample_plan(slot);
    write_layout(path, groups);

    TierLayoutReader r(path);
    const auto& h = r.header();
    T_REQUIRE(h.version == 1, "version");
    T_REQUIRE(h.n_groups == groups.size(), "n_groups");
    T_REQUIRE(h.n_tensors == groups.size(), "n_tensors");
    T_REQUIRE(h.index_offset == 512, "index_offset == 512 (ADR-12 §2.2)");
    T_REQUIRE(h.data_offset == ((512 + 64 * groups.size() + 4095) / 4096) * 4096,
              "data_offset выровнен на 4096");
    T_REQUIRE(r.index().size() == groups.size(), "index size");

    // Монотонность data_offset без дыр и выравнивание.
    std::uint64_t prev_end = h.data_offset;
    for (std::size_t i = 0; i < r.index().size(); ++i) {
        const auto& e = r.index()[i];
        T_REQUIRE(e.data_offset == prev_end, "монотонный data_offset, без дыр");
        T_REQUIRE(e.data_offset % 4096 == 0, "выравнивание на slot_align");
        T_REQUIRE(e.slot_bytes == slot, "slot_bytes");
        T_REQUIRE(e.slot_bytes % 4096 == 0, "slot_bytes кратен 4096");
        prev_end = e.data_offset + e.slot_bytes;
    }

    // Соответствие плану: роль/dtype/имя.
    const auto& e0 = r.index()[0];
    T_REQUIRE(e0.role == role_code(Role::Attn), "role attn");
    T_REQUIRE(e0.dtype == dtype_code(DType::F16), "dtype f16");
    T_REQUIRE(r.index()[2].role == role_code(Role::Expert), "role expert");
    T_REQUIRE(r.index()[3].dtype == dtype_code(DType::W4A16), "dtype w4a16");
    // Имя группы обрезается под 64-байтную запись индекса (16 байт, см. layout.hpp).
    T_REQUIRE(e0.name == groups[0].name.substr(0, 16), "имя группы");

    // Payload roundtrip: слот целиком (padding нулями), CRC на первом чтении.
    for (std::size_t i = 0; i < groups.size(); ++i) {
        std::vector<std::uint8_t> buf(slot);
        r.read_group(groups[i].id.value, buf.data());
        T_REQUIRE(buf[0] == static_cast<std::uint8_t>(i), "payload byte");
        T_REQUIRE(buf.back() == 0, "padding нулями");
        // Повторное чтение — без повторной CRC (ADR-12 §2.3).
        r.read_group(groups[i].id.value, buf.data());
        T_REQUIRE(buf[0] == static_cast<std::uint8_t>(i), "repeat read");
    }
    T_REQUIRE(r.io_bytes() == 2 * groups.size() * slot, "io_bytes counter");
}

T_CASE(layout_header_crc_corruption) {
    const std::string path = "/tmp/stell_tier_test_layout_hdr.stelltier";
    write_layout(path, sample_plan(4096));
    // Порча любого поля CRC-диапазона заголовка (0x00..0x28) → файл невалиден.
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(0x12); // created_ts — поле внутри диапазона CRC (ADR-12 §2)
    f.write("\xff", 1);
    f.flush();
    f.close();
    bool thrown = false;
    try {
        TierLayoutReader r(path);
        (void)r;
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
    }
    T_REQUIRE(thrown, "порча заголовка → IoFailure при открытии");
}

T_CASE(layout_group_crc_corruption) {
    const std::string path = "/tmp/stell_tier_test_layout_grp.stelltier";
    const auto groups = sample_plan(4096);
    write_layout(path, groups);

    TierLayoutReader r1(path);
    const auto& e2 = r1.index()[2];
    T_REQUIRE(e2.data_offset != 0, "offset известен");

    // Порча середины слота группы 3 (индекс 2).
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(static_cast<std::streamoff>(e2.data_offset + e2.slot_bytes / 2));
    f.write("\xba", 1);
    f.flush();
    f.close();

    TierLayoutReader r2(path);
    std::vector<std::uint8_t> buf(e2.slot_bytes);
    bool thrown = false;
    try {
        r2.read_group(groups[2].id.value, buf.data());
    } catch (const TierError& e) {
        thrown = e.code() == TierError::Code::IoFailure;
    }
    T_REQUIRE(thrown, "CRC группы → IoFailure при первом чтении (ADR-12 §2.3)");
}

T_CASE(layout_writer_misuse) {
    const std::string path = "/tmp/stell_tier_test_layout_misuse.stelltier";
    // slot_bytes не кратен 4096 → invalid_argument (ADR-12 §2.4).
    TierLayoutHeader h;
    h.n_groups = 1;
    TierLayoutWriter w(path, h);
    bool thrown = false;
    try {
        auto g = make_group(1, "g", Role::Kv, DType::F16, 512);
        w.add_group(g, 1, nullptr, 0);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    T_REQUIRE(thrown, "slot_bytes не кратен 4096 отклонён");
    // finalize() без валидных групп обязан бросить; деструктор удаляет файл.
    thrown = false;
    try {
        w.finalize();
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    T_REQUIRE(thrown, "finalize с меньшим числом групп → invalid_argument");
}

T_CASE(layout_name_truncation_and_shape) {
    const std::string path = "/tmp/stell_tier_test_layout_name.stelltier";
    TierLayoutHeader h;
    h.n_groups = 1;
    TierLayoutWriter w(path, h);
    auto g = make_group(7, "very-long-name-0123456789abcdefghij", Role::Expert,
                        DType::E4M3, 4096, {2, 3});
    w.add_group(g, 3, nullptr, 0);
    w.finalize();

    TierLayoutReader r(path);
    const auto& e = r.index()[0];
    T_REQUIRE(e.tensor_count == 3, "tensor_count");
    T_REQUIRE(e.shape_rank == 2, "shape_rank");
    T_REQUIRE(e.shape[0] == 2 && e.shape[1] == 3, "shape");
    // Имя обрезано под запись 64 байта (см. примечание в layout.hpp).
    T_REQUIRE(e.name.size() <= 16, "имя обрезано до 16 байт");
    T_REQUIRE(e.name == g.name.substr(0, 16), "обрезанное имя");
}

T_MAIN

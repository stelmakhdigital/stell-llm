// [R3] round 8 — побитовая сверка VRAM-слотов при eviction-чередовании
// на РЕАЛЬНОМ stell-tier layout (qwen3-30b-a3b, 16 ГиБ, 18867 групп, NVFP4).
//
// Метод (контрольное подмножество групп — полный проход 57 ГиБ-модели слишком
// долог для окна; честная пометка в README артефакта):
//   1. База: слот каждой контрольной группы читается ИНДЕПЕНДЕННЫМ путём
//      (TierLayoutReader::read_group — raw-читатель layout, payload-CRC
//      проверяется при первом чтении, ADR-12 §2.3) — это эталон «что лежит
//      на NVMe».
//   2. TierManager с МАЛЫМ VRAM-тиром (6 слотов — реальный eviction-чурн,
//      в т.ч. внутри подмножества) и теским RAM-тиром (1 ГиБ — RAM-eviction
//      тоже чередуется): группы пинаются в ЧЕРЕДУЮЩЕМСЯ порядке (ротация на
//      каждой проходке), 4 проходки; после каждого wait-resident слот группы
//      читается через read_vram_slot и сравнивается с базой ПОБИТОВО.
//   3. Инвариант ([R5] round 4, класс aliasing-багов ba42267): «одна группа
//      = один слот» — данные в VRAM после любого числа eviction-перезагрузок
//      совпадают с NVMe-слотом побитово.
//
// Сборка (сервер, из корня репо; бинарь в артефакт не пишется — только лог):
//   g++ -std=c++20 -O2 -I src/tiers/include \
//       bench/results/<dir>/harness_slot_verify.cpp \
//       build/cpu/src/tiers/libstell_tiers.a -lpthread -o /tmp/slot_verify
// Запуск:
//   STELL_TIER_ODIRECT=1 /tmp/slot_verify <layout.stelltier> > slot_verify.log
//
// Только CPU-контур; co-tenancy — NVMe общий (см. README артефакта).

#include <stell/tiers/layout.hpp>
#include <stell/tiers/tier_manager.hpp>
#include <stell/tiers/types.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace stell::tiers;

namespace {

bool name_selected(const std::string& n) {
    // Глобальные группы + контрольные тензоры 4 слоев (0, 1, 23, 47):
    // attention (wq/wo/q_norm) и эксперты (первый/последний, gate/up/down).
    const bool global = (n == "embed" || n == "head" || n == "norm_final");
    if (global) return true;
    const char* layers[] = {"l000.", "l001.", "l023.", "l047."};
    bool layer_ok = false;
    for (const char* l : layers)
        if (n.rfind(l, 0) == 0) layer_ok = true;
    if (!layer_ok) return false;
    static const char* parts[] = {
        ".attn.wq", ".attn.wo", ".attn.q_norm", ".attn.k_norm",
        ".e000.gate", ".e000.up", ".e000.down",
        ".e127.gate", ".e127.up", ".e127.down"};
    for (const char* p : parts)
        if (n.find(p) != std::string::npos) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: slot_verify <layout.stelltier>\n");
        return 2;
    }
    const std::string path = argv[1];

    // --- план и контрольное подмножество ------------------------------------
    TierLayoutReader rdr(path);   // payload-CRC при первом чтении (§2.3)
    std::vector<TensorGroup> plan;
    std::vector<const TierGroupIndexEntry*> subset;
    std::uint64_t subset_bytes = 0;
    for (const auto& e : rdr.index()) {
        TensorGroup g;
        g.id = GroupId{e.group_id};
        g.name = e.name;
        g.role = role_from_code(e.role);
        g.dtype = dtype_from_code(e.dtype);
        g.slot_bytes = e.slot_bytes;
        for (int d = 0; d < static_cast<int>(e.shape_rank) && d < 4; ++d)
            g.shape.push_back(static_cast<std::int64_t>(e.shape[(size_t)d]));
        plan.push_back(g);
        if (name_selected(e.name)) {
            subset.push_back(&e);
            subset_bytes += e.slot_bytes;
        }
    }
    std::printf("layout: %u групп, план %llu байт; контрольных групп %zu "
                "(%llu байт)\n",
                (unsigned)rdr.header().n_groups,
                (unsigned long long)rdr.header().data_offset, subset.size(),
                (unsigned long long)subset_bytes);

    // --- база: независимое чтение слотов (raw-читатель, payload-CRC) --------
    std::vector<std::vector<std::uint8_t>> base(subset.size());
    for (std::size_t i = 0; i < subset.size(); ++i) {
        base[i].resize((*subset[i]).slot_bytes);
        rdr.read_group((*subset[i]).group_id, base[i].data());
    }
    std::printf("база: %zu слотов прочитано независимым читателем "
                "(payload-CRC OK при первом чтении)\n", subset.size());

    // --- менеджер: 6 VRAM-слотов (чурн) + теский RAM (1 ГиБ) ----------------
    TierConfig cfg{};
    cfg.vram_slots = 6;
    cfg.ram_capacity = 1ull << 30;
    cfg.nvme_layout = path;
    TierManager mgr(cfg);
    mgr.set_plan(plan);

    // 4 проходки, порядок ротируется → eviction-чередование групп внутри
    // подмножества (6 слотов < 23 групп; RAM 1 ГиБ < ~1.6 ГиБ подмножества).
    std::size_t checks = 0, mismatches = 0;
    for (int round = 0; round < 4; ++round) {
        const std::size_t rot = (std::size_t)round * 7 % subset.size();
        for (std::size_t i = 0; i < subset.size(); ++i) {
            const TierGroupIndexEntry* e = subset[(i + rot) % subset.size()];
            const GroupId gid{e->group_id};
            const GroupKey key{gid, role_from_code(e->role)};
            auto a = mgr.pin(key);
            a->wait();
            std::vector<std::uint8_t> got(e->slot_bytes);
            mgr.read_vram_slot(gid, got.data(), got.size());
            mgr.unpin(key);
            ++checks;
            if (std::memcmp(got.data(), base[(i + rot) % subset.size()].data(),
                            e->slot_bytes) != 0) {
                ++mismatches;
                std::printf("[FAIL] группа %u (%s): слот VRAM != базе NVMe "
                            "(побитово)\n", e->group_id, e->name.c_str());
            }
        }
        std::printf("проходка %d: %zu сверок, несовпадений %zu, "
                    "vram_resident=%llu байт\n",
                    round + 1, checks, mismatches,
                    (unsigned long long)mgr.stats().tier_bytes[0]);
    }

    std::printf("ИТОГ: сверок=%u несовпадений=%u → %s\n", checks, mismatches,
                mismatches == 0 ? "ПОБИТОВАЯ СВЕРКА OK"
                                : "ПОБИТОВАЯ СВЕРКА ПРОВАЛЕНА");
    return mismatches == 0 ? 0 : 1;
}

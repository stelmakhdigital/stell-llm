// ============================================================================
// stell_convert_streaming — публичный CLI потоковой конвертации в stell-tier
// layout (round 3, R7 — подготовка среза 3; ADR-12 §1/§2, ADR-03).
//
// Writer БЕЗ полной загрузки модели в RAM: источник (safetensors-файл /
// каталог safetensors-шардов HF / GGUF) читается последовательно, группы
// материализуются и пишутся по одной, память — O(максимальной группы), не
// O(размера модели). Параметры тиров принимаются извне (согласованы с
// TierConfig, src/tiers): границы тиров задаёт tiers-конфиг, writer принимает
// их и валидирует совместимость плана (см. StellTierWriterConfig).
//
// Использование:
//   stell_convert_streaming <desc.json> <source: файл|каталог шардов> <out_dir>
//        [--vram-slot-size BYTES] [--ram-capacity BYTES] [--flush-mb N]
//
// Профиль «свободно 2–11 ГБ на V100 при co-tenancy»: границы тиров задаёт
// tiers-конфиг; сюда передаются те же величины (пример: при 2 ГБ свободной
// VRAM и slot_bytes групп ≤ 64 МиБ — --vram-slot-size 67108864; конкретные
// числа — решение деплоя, writer только валидирует план).
//
// Печать: сводка (группы, профиль, размер файла) + пиковая память процесса
// (getrusage RUSAGE_SELF maxrss). Коды: 0 — успех, 1 — ошибка.
// ============================================================================
#include <getopt.h>
#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "stell/models/convert.hpp"
#include "stell/models/weights.hpp"
#include "stell/models/registry.hpp"

using namespace stell::models;

namespace {

void print_usage() {
    std::printf(
        "stell_convert_streaming — потоковая конвертация в stell-tier layout (ADR-12)\n"
        "использование:\n"
        "  stell_convert_streaming <desc.json> <source> <out_dir>\n"
        "      [--vram-slot-size BYTES] [--ram-capacity BYTES] [--flush-mb N]\n"
        "  desc.json — JSON-описание модели (ADR-09, схема 1.0);\n"
        "  source    — safetensors-файл, каталог .safetensors-шардов HF или GGUF;\n"
        "  out_dir   — каталог вывода (<model_name>.stelltier);\n"
        "  --vram-slot-size — параметр tiers-профиля: max slot_bytes группы, байт\n"
        "                     (валидация плана; 0 = не задан);\n"
        "  --ram-capacity   — параметр tiers-профиля (RAM-тир), байт\n"
        "                     (принимается извне; в layout семантики нет);\n"
        "  --flush-mb       — порог flush'а по группам, МиБ (дефолт 8).\n");
}

std::string read_peak_rss_kb() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return "?";
    return std::to_string(ru.ru_maxrss) + " КиБ";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 1;
    }
    const std::string desc_path = argv[1];
    const std::string source_uri = argv[2];
    const std::string out_dir = argv[3];

    StellTierWriterConfig cfg;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                std::printf("ERROR: у флага %s нет значения\n", name);
                std::exit(1);
            }
            return std::string(argv[++i]);
        };
        try {
            if (arg == "--vram-slot-size") {
                cfg.vram_slot_size = std::stoull(need("--vram-slot-size"));
            } else if (arg == "--ram-capacity") {
                cfg.ram_capacity = std::stoull(need("--ram-capacity"));
            } else if (arg == "--flush-mb") {
                cfg.flush_group_bytes = std::stoull(need("--flush-mb")) << 20;
            } else {
                std::printf("ERROR: неизвестный флаг '%s'\n", arg.c_str());
                print_usage();
                return 1;
            }
        } catch (const std::exception&) {
            std::printf("ERROR: некорректное число у флага '%s'\n", arg.c_str());
            return 1;
        }
    }

    try {
        Registry registry;
        const ModelDescription desc = registry.load(desc_path);

        const auto t0 = std::chrono::steady_clock::now();
        StellTierConverter conv;
        StellTierConverter::Stats stats;
        conv.convert_streaming(desc, source_uri, out_dir, cfg, &stats);
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        const std::string out_path = out_dir + "/" + desc.name + ".stelltier";
        std::FILE* f = std::fopen(out_path.c_str(), "rb");
        if (!f) throw LoaderError(LoaderError::Code::IoFailure, "cannot open " + out_path);
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fclose(f);

        std::size_t n_nonkv = 0;
        for (const auto& g : desc.groups)
            if (g.role != stell::tiers::Role::Kv) ++n_nonkv;

        std::printf("модель: '%s' (%s, профиль %s), MoE=%s\n", desc.name.c_str(),
                    desc.is_moe ? "moe" : "dense", desc.dtype_weights.c_str(),
                    desc.is_moe ? "да" : "нет");
        std::printf("групп в layout: %zu (план: %zu, kv-группы в файл не пишутся)\n", n_nonkv,
                    desc.groups.size());
        std::printf("layout: %s (%ld байт)\n", out_path.c_str(), sz);
        std::printf("потоковая конвертация: %.2f c, пиковая память процесса: %s\n", sec,
                    read_peak_rss_kb().c_str());
        // R5 ([Q6], ADR-03 round 5): счётчик сатураций + провенанс dtype_orig
        // — в логе конвертера (clamp-сатурация F32/BF16→F16 к ±65504; inf в
        // весах не порождается; обычные значения — без изменений).
        std::printf("сатураций F32/BF16→F16: %zu (clamp к ±65504); dtype_orig: bf16=%zu, f32=%zu групп\n",
                    stats.f16_saturations, stats.groups_dtype_orig_bf16,
                    stats.groups_dtype_orig_f32);
        if (cfg.vram_slot_size)
            std::printf("tiers-профиль: vram_slot_size=%zu, ram_capacity=%zu "
                        "(валидация плана пройдена; границы тиров — стороны tiers, ADR-12)\n",
                        cfg.vram_slot_size, cfg.ram_capacity);
        return 0;
    } catch (const std::exception& e) {
        std::printf("ERROR: %s\n", e.what());
        return 1;
    }
}

// [R4] round 8 — декод-контур на реальном чекпоинте Qwen3-30B-A3B:
// контрольные логиты == прямому лоадеру (побитово) на КОНТРОЛЬНОМ размещении.
//
// Честные пометки (см. README артефакта):
//   * Полный 48-слойный декод через OpsInterpreter НЕВОЗМОЖЕН на CPU-контуре
//     по RAM: конструктор интерпретатора материализует ВСЕ группы в F32
//     (~120 ГиБ на 48 слоях, 128 экспертов) — не помещается в 62 ГиБ машины.
//     Контрольное размещение — 2-слойный срез реальной архитектуры (GQA
//     n_kv_heads=4/head_dim=128, qk-norm, 128 экспертов MoE, router_renorm
//     дефолт, head_dim*32 != d_model): те же кодовые пути плана/конвертера/
//     лоадеров/интерпретатора; контроль — «одна и та же арифметика плана
//     на обоих лоадерах» (прецедент roundtrip_logits, tests/models).
//   * wall/token измеряется на 2 слоях; экстраполяция на полный PPL-прогон
//     (128 токенов, инкрементально) — в README артефакта (не измерение).
//
// Сборка (сервер, из корня репо):
//   g++ -std=c++20 -O2 -I src/tiers/include -I src/models/include \
//       -I src/engine/include -I src/kernels/include \
//       bench/results/20261020-decode-real-qwen3-30b-a3b/harness_decode_logits.cpp \
//       build/cpu/src/engine/libstell_engine.a build/cpu/src/models/libstell_models.a \
//       build/cpu/src/tiers/libstell_tiers.a build/cpu/src/kernels/libstell_kernels.a \
//       build/cpu/src/core/libstell_core.a -lpthread -o /tmp/decode_logits
// Запуск:
//   /tmp/decode_logits <hf_dir контроля> <layout.stelltier>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "stell/engine/ops_runner.hpp"
#include "stell/engine/execution_plan.hpp"
#include "stell/models/convert.hpp"
#include "stell/models/description.hpp"
#include "stell/models/weights.hpp"

namespace se = stell::engine;
using namespace stell::models;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: decode_logits <hf_dir> <layout.stelltier>\n");
        return 2;
    }
    const std::string hf_dir = argv[1];
    const std::string layout_path = argv[2];
    const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};

    const auto t00 = std::chrono::steady_clock::now();
    const std::string json = hf_to_description(hf_dir, "NVFP4");
    const ModelDescription d = parse_model_description(json, "qwen3-ctrl2l");
    const se::PlainSpec spec = se::plain_spec_from_description(d);
    const se::ExecutionPlan plan = se::build_execution_plan(spec);
    std::size_t n_groups = 0;
    for (const auto& g : d.groups)
        if (g.role != stell::tiers::Role::Kv) ++n_groups;
    // dtype_orig — top-level поле JSON (провенанс [R2] round 6); в структуре
    // ModelDescription его нет — честно ищем в тексте описания.
    const std::size_t do_pos = json.find("\"dtype_orig\"");
    std::printf("описание: %s класс=%s dtype=%s dtype_orig=%s групп(не-kv)=%zu "
                "wall=%.1fs\n",
                d.name.c_str(), d.is_moe ? "moe" : "dense",
                d.dtype_weights.c_str(),
                do_pos == std::string::npos ? "<нет>" : "см. desc json",
                n_groups, seconds_since(t00));

    // --- прямое размещение: лоадер NVFP4W4A16 из источника -------------------
    auto t0 = std::chrono::steady_clock::now();
    se::LoadedWeights direct;
    {
        auto loader = make_loader(WeightsFormat::NVFP4W4A16);
        auto src = loader->open(hf_dir, d);
        for (const auto& g : d.groups) {
            if (g.role == stell::tiers::Role::Kv) continue;
            std::vector<std::uint8_t> buf(g.slot_bytes, 0);
            src->read({g.id, g.role}, buf.data(), buf.size());
            direct.slots[g.id.value] = std::move(buf);
        }
        src->close();
    }
    const double direct_load_s = seconds_since(t0);
    std::printf("прямой лоадер: %zu групп, wall=%.1fs\n", direct.slots.size(),
                direct_load_s);

    // один forward-прогон короткого промпта (8 токенов, инкрементально) —
    // контрольные логиты последней позиции. Интерпретатор материализует
    // группы в F32 при конструировании — далее слоты лоадера не нужны.
    std::vector<float> ref_logits;
    double ref_build_s = 0, ref_run_s = 0;
    {
        auto t0 = std::chrono::steady_clock::now();
        se::OpsInterpreter ref(spec, plan, direct, /*tiers=*/nullptr);
        ref_build_s = seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        ref_logits = ref.run_prompt_logits(prompt);
        ref_run_s = seconds_since(t0);
    }   // интерпретатор уничтожен — прямое размещение освобождаем честно
    direct.slots.clear();
    direct.slots = {};
    std::printf("прямая сторона: материализация=%.1fs forward(8 токенов)=%.3fs "
                "(wall/token=%.4fs) логитов=%zu; размещение освобождено\n",
                ref_build_s, ref_run_s, ref_run_s / prompt.size(),
                ref_logits.size());

    // --- stell-tier layout: тот же план, лоадер StellTier --------------------
    const se::LoadedWeights via =
        se::load_plain_weights(layout_path, spec, se::PlainWeightsStellTier);
    const double layout_load_s = seconds_since(t0);
    std::vector<float> tier_logits;
    double tier_build_s = 0, tier_run_s = 0;
    {
        auto t1 = std::chrono::steady_clock::now();
        se::OpsInterpreter tier(spec, plan, via, /*tiers=*/nullptr);
        tier_build_s = seconds_since(t1);
        t1 = std::chrono::steady_clock::now();
        tier_logits = tier.run_prompt_logits(prompt);
        tier_run_s = seconds_since(t1);
    }

    std::printf("layout: слотов=%zu загрузка=%.1fs материализация=%.1fs "
                "forward=%.3fs (wall/token=%.4fs)\n",
                via.slots.size(), layout_load_s, tier_build_s, tier_run_s,
                tier_run_s / prompt.size());

    // --- побитовая сверка логитов --------------------------------------------
    bool ok = ref_logits.size() == tier_logits.size();
    std::size_t diffs = 0;
    for (std::size_t i = 0; ok && i < ref_logits.size(); ++i) {
        if (ref_logits[i] != tier_logits[i]) {
            ++diffs;
            ok = false;
        }
    }
    std::printf("СВЕРКА ЛОГИТОВ: %zu позиций, побитовых несовпадений=%zu → %s\n",
                ref_logits.size(), diffs, ok ? "OK (побитно)" : "FAIL");
    std::printf("extrapolation_note: wall/token 2 слоев; полный PPL-128 "
                "(48 слоёв) ≈ wall/token × 24 × 128 (см. README)\n");
    return ok ? 0 : 1;
}

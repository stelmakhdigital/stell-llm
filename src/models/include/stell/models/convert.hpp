// ============================================================================
// Конвертер в stell-tier layout — контракт ADR-12 §1.
//
// Разбиение весов на тензорные группы — ИСКЛЮЧИТЕЛЬНО на стороне models
// (ADR-12 §3); tiers — только читатель layout. Writer формата живёт здесь,
// а не в tiers: ADR-12 §4 прямо назначает конвертер (запись layout) стороне
// models, поэтому дублирования кода с tiers нет — у tiers свой reader
// (Tier Manager), общий для обеих сторон только spec (ADR-12 §2) и CRC32.
// ============================================================================
#pragma once

#include <cstddef>
#include <string>

#include "stell/models/description.hpp"

namespace stell::models {

// ----------------------------------------------------------------------------
// Параметры writer'а stell-tier, согласованные с tiers-конфигом (TierConfig):
// границы тиров задаёт конфиг tiers, writer принимает их извне и валидирует
// совместимость плана (см. convert_streaming). Выравнивания фиксированы
// layout v1 (ADR-12 §2: slot_align=4096, index_align=512) — кастомные значения
// отвергаются с SchemaMismatch (новая версия формата — через ADR, ADR-12).
// ----------------------------------------------------------------------------
struct StellTierWriterConfig {
    std::size_t slot_align = 4096;   // ADR-12 §2.4 (kStellTierSlotAlign, layout v1)
    std::size_t index_align = 512;   // ADR-12 §2.2 (kStellTierIndexAlign, layout v1)
    // Профиль тиров (из tiers-конфига, «свободно 2–11 ГБ на V100»):
    std::size_t vram_slot_size = 0;  // 0 = проверка не задана (TierConfig: 0 → max slot_bytes)
    std::size_t ram_capacity = 0;    // принимается извне; в layout семантики нет
    // Flush по группам: fflush вывода, когда не сброшенных ≥ порога (байт).
    std::size_t flush_group_bytes = 8u << 20;
};

class IStellTierConverter {
public:
    virtual ~IStellTierConverter() = default;

    // Конвертер «входной формат → stell-tier layout» (ADR-03, ADR-12 §1).
    // Профиль dtype берётся из desc.dtype_weights (F16|NVFP4|W4A16); при
    // квантованном профиле payload квантуется на этапе конвертации (см. weights.hpp).
    // KV-группы (role=kv) в layout не попадают — весов/KV-кеша в файле нет (ADR-12 §2).
    // Пишет файл <out_dir>/<model_name>.stelltier.
    virtual void convert(const ModelDescription& desc,
                         const std::string& source_uri,  // safetensors/GGUF
                         const std::string& out_dir) = 0;
};

class StellTierConverter : public IStellTierConverter {
public:
    void convert(const ModelDescription& desc, const std::string& source_uri,
                 const std::string& out_dir) override;

    // ------------------------------------------------------------
    // R5 ([Q6], ADR-03 round 5): статистика/провенанс конвертации — заполняется
    // convert_streaming, печатается в лог CLI конвертера (счётчик сатураций
    // clamp-сатурации F32/BF16→F16; исходные dtype групп — dtype_orig: bf16/f32,
    // честность профиля: точка регрессии/замер знает о сжатии из BF16).
    struct Stats {
        std::uint64_t f16_saturations = 0;      // значений, сатурированных к ±65504
        std::uint64_t groups_dtype_orig_bf16 = 0;  // группы с исходным dtype bf16
        std::uint64_t groups_dtype_orig_f32 = 0;   // группы с исходным dtype f32
    };

    // Потоковая конвертация (round 3, R7 — подготовка среза 3): writer БЕЗ
    // полной загрузки модели в RAM. Источник (safetensors-файл / каталог
    // safetensors-шардов HF / GGUF) читается ПОСЛЕДОВАТЕЛЬНО по группам плана;
    // каждая группа материализуется по одной (слот + source-буфер), пишется в
    // layout с flush'ем по группам. Память writer'а — O(максимальной группы)
    // по payload-байтам и O(числа групп) по метаданным (записи индекса 64 Б/гр,
    // ADR-12 §2.2) — НЕ O(размера модели).
    //
    // Раскладка файла идентична convert() (ADR-12 §2/§2.2/§2.4): заголовок 512,
    // индекс на 0x200, данные монотонно с выравниванием 4096, без дыр; CRC32
    // группы проставляется в индекс сразу после записи слота (fixup записи
    // индекса). Файл пишется во временный <name>.stelltier.tmp и атомарно
    // переименовывается по успехе (частичный файл не остаётся под финальным
    // именем). Пишет <out_dir>/<model_name>.stelltier.
    //
    // Параметры тиров — ИЗВНЕ (постановка R7: границы тиров задаёт tiers-конфиг;
    // интерфейс согласован с TierConfig, src/tiers/include/stell/tiers/
    // tier_manager.hpp): writer НЕ размещает тиры сам и не пишет границы в файл
    // (ADR-12: layout хранит ровно то, что нужно tiers, без семантики тиров),
    // но принимает параметры профиля и валидирует совместимость плана:
    //   - vram_slot_size (>0): каждая группа обязана помещаться в VRAM-слот
    //     (slot_bytes ≤ vram_slot_size), иначе профиль «свободно 2–11 ГБ на
    //     V100 при co-tenancy» неисполним — SchemaMismatch на этапе конвертации;
    //   - ram_capacity: принимается для выравнивания интерфейса с tiers-конфигом;
    //   семантики в файле не имеет (проверяется честно в тесте).
    //
    // source_uri: файл safetensors/GGUF ИЛИ каталог safetensors-шардов (HF:
    // model-00001-of-00002.safetensors, …) — шард-карты разбираются по метаданным
    // сразу (мало), payload читается из шарда последовательно, шард за шардом;
    // для стековных тензоров (GGUF-mixtral, source_slice) срезы соседних
    // экспертов смежны — чтение последовательное и в этом случае.
    void convert_streaming(const ModelDescription& desc, const std::string& source_uri,
                           const std::string& out_dir,
                           const StellTierWriterConfig& cfg = {},
                           Stats* stats = nullptr);
};

// ----------------------------------------------------------------------------
// Конвертер GGUF → описание модели (ADR-09 «Уточнение», round 2): GGUF не
// поставляется с JSON-описанием, поэтому models генерирует его из метаданных
// и тензорных имён файла. Имена тензоров — НАТИВНЫЕ имена тензоров формата
// GGUF (см. ADR-09 «Уточнение»; token_embd.weight, blk.{i}.attn_q.weight, …);
// source_pattern в описании — те же нативные имена, лоадер сравнивает их
// буквально. Round 2 (R9): архитектуры llama (dense) И mixtral (MoE) —
// стековные экспертные тензоры blk.{i}.ffn_*_exps.weight, смещение роутера
// blk.{i}.exp_probs_b.weight, n_experts/n_experts_active из GGUF-метаданных
// (llama.expert_count / llama.expert_used_count), срезы экспертов —
// source_slice.
// Возвращает JSON-текст описания (схема "1.0", как у llama-class.json).
// Ошибки — LoaderError (SchemaMismatch/UnsupportedQuant/TensorNotFound).
// ----------------------------------------------------------------------------
std::string gguf_to_description(const std::string& gguf_uri);

// ----------------------------------------------------------------------------
// Конвертер HF-чекпоинта → описание модели (round 4, [R4]; ADR-09; по прецеденту
// gguf_to_description): вход — каталог модели HF (config.json + safetensors-
// шарды; канонический порядок шардов — отсортированный) или одиночный
// .safetensors; выход — JSON-текст описания (схема "1.0") с НАТИВНЫМИ именами
// тензоров HF (model.embed_tokens.weight, model.layers.{layer}.self_attn.
// q_proj.weight, …), source_pattern — те же нативные имена, лоадер/конвертер
// сравнивают их буквально (ADR-09 §3).
// Профиль dtype_weights — аргумент profile ("F16" по умолчанию; "NVFP4"/
// "W4A16" — квантование выполняет отдельный проход конвертера/лоадера;
// канон ADR-09 block32: линейные веса — dtype профиля, нормы/роутер —
// dtype источника F32/F16, embed/head/kv — F16).
// Архитектуры (subset): LlamaForCausalLM / Qwen2ForCausalLM (dense);
// Qwen2MoeForCausalLM / Qwen1.5MoeForCausalLM / OlmoeForCausalLM (MoE,
// поэертельные тензоры экспертов; роутер mlp.router|mlp.gate — по факту в
// шардах); MixtralForCausalLM (MoE, стековные block_sparse_moe.w1/w2/w3,
// source_slice "{expert}", канон R9).
// rope-параметры config.json (rope_theta) → параметр ops attention.
// Ошибки — LoaderError: нет config.json → SchemaMismatch; шард не соответствует
// плану (тензора нет у группы) → TensorNotFound; форма/dtype — SchemaMismatch/
// UnsupportedQuant; валидация энергичная — все некешевые группы расширенного
// плана резолвятся по метаданным шардов (fail-fast, ADR-09 §3).
// ----------------------------------------------------------------------------
std::string hf_to_description(const std::string& hf_dir,
                              const std::string& profile = "F16");

}  // namespace stell::models

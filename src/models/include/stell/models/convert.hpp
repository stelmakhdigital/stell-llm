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

#include <string>

#include "stell/models/description.hpp"

namespace stell::models {

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

}  // namespace stell::models

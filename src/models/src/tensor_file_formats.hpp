// Общие функции разбора файловых форматов (safetensors/GGUF) — используются
// файловыми лоадерами и NVFP4W4A16-вариантом (TensorFileMap из каркаса).
#pragma once

#include "file_weights_source.hpp"

namespace stell::models {

// safetensors (HF): u64 header_len (LE) → JSON-заголовок → данные.
// Поддержанные dtype: F16, F32, F8_E4M3 (остальные — UnsupportedQuant при
// обращении; itemsize известен для пропуска/валидации размеров).
TensorFileMap parse_safetensors_header(std::FILE* f);

// GGUF v3 (subset): magic "GGUF", version 3, метаданные (u32/i32/u64/i64/f32/
// f64/bool/string/array), тензорные описания; dtype тензора только F32(0)/F16(1).
TensorFileMap parse_gguf_header(std::FILE* f);

}  // namespace stell::models

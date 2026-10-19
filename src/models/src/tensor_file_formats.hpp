// Общие функции разбора файловых форматов (safetensors/GGUF) — используются
// файловыми лоадерами и NVFP4W4A16-вариантом (TensorFileMap из каркаса).
#pragma once

#include "file_weights_source.hpp"

namespace stell::models {

// safetensors: u64 header_len (LE) → JSON-заголовок → данные.
// Поддержанные dtype: F16, F32, F8_E4M3 (остальные — UnsupportedQuant при
// обращении; itemsize известен для пропуска/валидации размеров).
TensorFileMap parse_safetensors_header(std::FILE* f);

// GGUF v3 (subset): magic "GGUF", version 3, метаданные (u32/i32/u64/i64/f32/
// f64/bool/string/array), тензорные описания; dtype тензора только F32(0)/F16(1).
//
// КОНВЕНЦИЯ SHAPE (round 2, зафиксировано): GGUF ne хранит dims в порядке
// {in, out} для матриц ({n_embd, vocab} для token_embd и т.п.), при этом
// байты в GGUF C-порядке совпадают с row-major [out, in] — то есть
// РАЗВЕРНУТЫЙ порядок ne == логической форме (конвенция раскладки
// safetensors/JSON-конфигов). Поэтому
// parse_gguf_header возвращает TensorInfo.shape = РАЗВЕРНУТЫЙ dims
// (1-D тензоры не меняются); байты payload не меняются.
TensorFileMap parse_gguf_header(std::FILE* f);

// --- GGUF v3: примитивные читатели (LE) — общие для лоадера и конвертера
// описаний (gguf_description.cpp; дублирование запрещено). --------------------

std::uint32_t gguf_read_u32(std::FILE* f);
std::uint64_t gguf_read_u64(std::FILE* f);
// u64 length + bytes (GGUF string).
std::string gguf_read_string(std::FILE* f);

// Коды типов метаданных GGUF (subset, см. loader_gguf.cpp).
constexpr std::uint32_t kGgufMetaUint8 = 0, kGgufMetaInt8 = 1;
constexpr std::uint32_t kGgufMetaUint16 = 2, kGgufMetaInt16 = 3;
constexpr std::uint32_t kGgufMetaUint32 = 4, kGgufMetaInt32 = 5;
constexpr std::uint32_t kGgufMetaUint64 = 6, kGgufMetaInt64 = 7;
constexpr std::uint32_t kGgufMetaFloat32 = 8, kGgufMetaFloat64 = 9;
constexpr std::uint32_t kGgufMetaBool = 10, kGgufMetaString = 11,
                         kGgufMetaArray = 12;

// Значение метаданных заданного типа (файл уже позиционирован на значении).
// Массивы (kGgufMetaArray) round 1 пропускаются: в value возвращается только
// type и число элементов (u64), содержимое не материализуется.
struct GgufMetaValue {
    std::uint32_t type = 0;
    std::string str;
    std::uint64_t u64 = 0;
    std::int64_t i64 = 0;
    double f64 = 0.0;
};

GgufMetaValue gguf_read_meta_value(std::FILE* f, std::uint32_t type);

// Код квантованных GGML-блоков (subset round 2): Q4_0 (GGML 20) и Q8_0 (GGML 36).
constexpr std::uint32_t kGgmlQ40 = 20, kGgmlQ80 = 36;
// Имена/размеры блоков — в quant.hpp (is_quantized_ggml/ggml_block_bytes):
// общие для лоадера и срезов стековных тензоров (file_weights_source.hpp).

// Dequant квантованного GGUF-тензора → f32 (референс ggml-блоков):
//   Q4_0: блок 32 эл. = 2 байта F16 delta (в НАЧАЛЕ блока) + 16 байт нибблов;
//         значение = (nibble − 8) * delta; младший ниббл = чётный элемент.
//   Q8_0: блок 32 эл. = 32 байта int8 + 2 байта F16 delta (в КОНЦЕ блока);
//         значение = q * delta.
// numel = numel(TensorInfo.shape) (развернутая логическая форма).
void gguf_dequant_tensor(std::FILE* f, const TensorInfo& t,
                         std::vector<float>& out);

}  // namespace stell::models

// ============================================================================
// Byte-level BPE токенизатор (round 2, задача R10) — реализация ITokenizer
// (src/engine, инжект в Scheduler::set_tokenizer; ADR-11 §3, план Q7).
//
// Формат загрузки — tokenizer.json каталога модели (формат эталонных
// токенизаторов; fixture — tests/models/fixtures/tokenizer/):
//   {"model": {"vocab": {token: id, ...}, "merges": [["a","b"], ...],
//              "unk_token": null, ...},
//    "pre_tokenizer": {"type": "ByteLevel", ...},
//    "decoder": {"type": "ByteLevel", ...}}
//
// Реализованный subset (честно зафиксирован в отчёте раунда):
//   - byte-level Pretokenizer БЕЗ регэкспа (use_regex=false): вся строка —
//     один чанк; байты UTF-8 маппятся в алфавитные символы по таблице
//     byte→unicode (конвенция GPT-2-семейства: печатные ASCII — как есть,
//     остальные — кодовые точки 256+n; таблица выводится формулой, см. ниже);
//   - merges: жадное слияние по рангу — на каждом шаге выбирается пара
//     с МИНИМАЛЬНЫМ рангом среди смежных, сливаются ВСЕ её вхождения
//     слева направо (семантика классического BPE-алгоритма; сверено с
//     эталонной реализацией — тест test_models_tokenizer, расхождение 0
//     на committed fixture);
//   - decode: id → token-строка → обратная маппинга символов в байты →
//     UTF-8 строка; неизвестный id → пустая строка (конвенция byte-fallback
//     ByteTokenizer в engine).
// Не поддержано (TokenizerError при загрузке, тишина запрещена): нормализаторы,
// post-processor'ы, added/special tokens, unk/byte_fallback, иной pretokenizer.
// ============================================================================
#pragma once

#include <map>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "stell/engine/tokenizer.hpp"

namespace stell::models {

class TokenizerError : public std::runtime_error {
public:
    explicit TokenizerError(const std::string& what)
        : std::runtime_error(what) {}
};

class ByteLevelBpeTokenizer final : public stell::engine::ITokenizer {
public:
    // Загрузка tokenizer.json: vocab + merges + byte-level
    // pretokenizer/decoder. Бросает TokenizerError при несоответствии subset.
    static ByteLevelBpeTokenizer load(const std::string& tokenizer_json_path);

    ByteLevelBpeTokenizer() = default;

    // ITokenizer (stell::engine): текст → id (byte-level BPE).
    std::vector<std::int32_t> encode(const std::string& text) const override;
    // Текст ОДНОГО токена; неизвестный id → пустая строка.
    std::string decode(std::int32_t id) const override;
    // Конкатенация текстов токенов (детокенизация сессии, ADR-11 §3).
    std::string decode(const std::vector<std::int32_t>& ids) const override;

    std::size_t vocab_size() const { return id2token_.size(); }
    std::size_t n_merges() const { return ranks_.size(); }

private:
    // Символы алфавита: byte (0..255) → строка из одного юникод-символа.
    static std::string byte_to_char(std::uint8_t b);
    // Обратная маппинга: юникод-символ (кодовая точка) → байт (−1 = нет).
    static int char_to_byte(std::int64_t codepoint);

    // Жадное слияние по рангу (см. шапку): сливает символы на месте.
    void merge_symbols(std::vector<std::string>& symbols) const;

    std::map<std::string, std::int32_t> vocab_;  // token-строка → id
    std::vector<std::string> id2token_;          // id → token-строка
    std::map<std::pair<std::string, std::string>, std::size_t> ranks_;  // merge → ранг
};

}  // namespace stell::models

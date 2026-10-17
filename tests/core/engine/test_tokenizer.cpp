// Токенизация (round 2): ITokenizer + ByteTokenizer (byte-level fallback).
// ЧЕСТНАЯ ПОМЕТКА: прод-токенизатор (BPE/SentencePiece из реестра моделей) —
// отдельная задача; byte-fallback — только для интеграционных тестов срезов
// (text в событиях ADR-11 §3 заполняется честно тем, что известно engine).
#include "test_util.hpp"

#include <string>
#include <vector>

#include "stell/engine/tokenizer.hpp"

using namespace stell::engine;

void run() {
    ByteTokenizer tok;

    // encode: строка → байты (byte-level)
    const std::vector<std::int32_t> ids = tok.encode("hi!");
    STELL_CHECK(ids.size() == 3);
    STELL_CHECK(ids[0] == 'h' && ids[1] == 'i' && ids[2] == '!');

    // decode(id): один токен → один байт; вне [0,255] — пустая строка
    STELL_CHECK(tok.decode('a') == "a");
    STELL_CHECK(tok.decode(255) == std::string(1, static_cast<char>(255)));
    STELL_CHECK(tok.decode(-1).empty());
    STELL_CHECK(tok.decode(256).empty());        // byte-level знает только байты

    // decode(последовательность)
    STELL_CHECK(tok.decode({104, 105}) == "hi");
    STELL_CHECK(tok.decode(std::vector<std::int32_t>{}).empty());

    // roundtrip: encode → decode
    const std::string text = "round trip 42";
    STELL_CHECK(tok.decode(tok.encode(text)) == text);
}

STELL_TEST_MAIN(run)

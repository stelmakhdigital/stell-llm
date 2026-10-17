// Токенизация (round 2, интеграция): интерфейс ITokenizer + минимальная
// CPU-реализация. ЧЕСТНАЯ ПОМЕТКА: прод-токенизатор (BPE/SentencePiece из
// реестра моделей) — ОТДЕЛЬНАЯ ЗАДАЧА, в этом раунде не реализован.
// ByteTokenizer — byte-level fallback: id = байты строки; decode(id) — один
// байт (для ids вне [0,255] — пустая строка). Годится для интеграционных
// тестов срезов (text в events заполняется честно тем, что известно engine);
// НЕ годится как прод-детокенизатор — помечено в отчёте потока core.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace stell::engine {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    // Текст → id токенов (для byte-level — байты UTF-8).
    virtual std::vector<std::int32_t> encode(const std::string& text) const = 0;

    // Текст ОДНОГО токена (инкрементальный decode для событий ADR-11 §3).
    virtual std::string decode(std::int32_t id) const = 0;

    // Текст последовательности (для детокенизации контекста сессии).
    virtual std::string decode(const std::vector<std::int32_t>& ids) const = 0;
};

// Byte-level fallback: id = байт [0,255]; вне диапазона — пустая строка.
class ByteTokenizer final : public ITokenizer {
public:
    std::vector<std::int32_t> encode(const std::string& text) const override;
    std::string decode(std::int32_t id) const override;
    std::string decode(const std::vector<std::int32_t>& ids) const override;
};

} // namespace stell::engine

// ============================================================================
// Минимальный собственный JSON-парсер для декларативных описаний моделей
// (ADR-09 §1, схема 1.0).
//
// Выбор собственного парсера вместо стороннего (nlohmann/json упомянут в
// ADR-09 как вариант) зафиксирован в отчёте раунда: добавление зависимостей
// в ядро запрещено без анализа; для схемы 1.0 достаточно ~300 строк
// стандартного JSON (RFC 8259): объекты, массивы, строки с escapes
// (\uXXXX + суррогатные пары), int/double, bool, null. Полная спецификация
// чисел IEEE (hex-экспоненты и т.п.) не нужна — описания машинно-генерируются.
//
// Парсер строгий: неизвестные конструкции = исключение JsonError с позицией.
// ============================================================================
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stell::models {

class JsonError : public std::runtime_error {
public:
    JsonError(std::size_t pos, const std::string& what)
        : std::runtime_error(what), pos_(pos) {}
    std::size_t pos() const noexcept { return pos_; }

private:
    std::size_t pos_;
};

struct JsonValue {
    enum class Type : std::uint8_t { Null, Bool, Int, Double, String, Array, Object };

    Type type = Type::Null;
    bool bool_v = false;
    std::int64_t int_v = 0;
    double double_v = 0.0;
    std::string string_v;
    std::vector<JsonValue> array_v;
    // Порядок ключей сохраняется (описания генерируются конвертерами — порядок
    // детерминирован входным файлом); поиск — линейный, ключей мало.
    std::vector<std::pair<std::string, JsonValue>> object_v;

    bool is(Type t) const noexcept { return type == t; }

    // Возвращает nullptr, если ключа нет.
    const JsonValue* find(const std::string& key) const;
    // Бросает JsonError, если ключа нет.
    const JsonValue& at(const std::string& key) const;

    // Типизированные доступы: бросают JsonError при несовпадении типа.
    std::int64_t get_int(const std::string& key) const;
    std::uint64_t get_uint(const std::string& key) const;
    double get_double(const std::string& key) const;
    const std::string& get_string(const std::string& key) const;
    const JsonValue& get_array(const std::string& key) const;
    const JsonValue& get_object(const std::string& key) const;
};

// Разбор JSON-текста. Бросает JsonError при любой ошибке синтаксиса.
JsonValue json_parse(const std::string& text);

}  // namespace stell::models

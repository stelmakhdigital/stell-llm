// Юнит-тесты собственного JSON-парсера (include/stell/models/json.hpp).
#include "stell/models/json.hpp"

#include "test_util.hpp"

using namespace stell::models;

void body() {
    // Базовый объект + массивы + числа.
    JsonValue v = json_parse(
        "{\"schema\":\"1.0\",\"n_layers\":32,\"eps\":1.5e-5,\"flag\":true,\"none\":null,"
        "\"arr\":[1,-2,3.5,\"str\"],\"obj\":{\"k\":{\"deep\":[true,false]}}}");
    CHECK(v.is(JsonValue::Type::Object));
    CHECK(v.get_string("schema") == "1.0");
    CHECK(v.get_uint("n_layers") == 32);
    CHECK(v.get_int("n_layers") == 32);
    const double eps = v.get_double("eps");
    CHECK(eps > 1.4e-5 && eps < 1.6e-5);
    CHECK(v.at("flag").type == JsonValue::Type::Bool && v.at("flag").bool_v);
    CHECK(v.at("none").type == JsonValue::Type::Null);
    const JsonValue& arr = v.get_array("arr");
    CHECK(arr.array_v.size() == 4);
    CHECK(arr.array_v[1].type == JsonValue::Type::Int && arr.array_v[1].int_v == -2);
    CHECK(arr.array_v[3].type == JsonValue::Type::String && arr.array_v[3].string_v == "str");
    CHECK(v.get_object("obj").get_object("k").get_array("deep").array_v.size() == 2);

    // Строки: escapes + \uXXXX + суррогатная пара → UTF-8.
    JsonValue esc = json_parse(
        "{\"s\":\"a\\\"b\\\\c\\/d\\b\\f\\n\\r\\t\\u0041\\u0440\\ud83d\\ude00\"}");
    const std::string& s = esc.get_string("s");
    const std::string expected =
        std::string("a\"b\\c/d\b\f\n\r\t") + "\x41" + "\xd1\x80" + "\xF0\x9F\x98\x80";
    CHECK(s == expected);

    // Ошибки парсинга → JsonError.
    CHECK_THROW(json_parse("{\"a\":1} trailing"), JsonError, "trailing garbage rejected");
    CHECK_THROW(json_parse("{\"a\":}"), JsonError, "bad value rejected");
    CHECK_THROW(json_parse("{\"a\":\"\\q\"}"), JsonError, "unknown escape rejected");
    CHECK_THROW(json_parse("{\"a\":\"\\uD833\"}"), JsonError, "lone high surrogate rejected");
    CHECK_THROW(json_parse("{\"a\":\"\\uDC00\"}"), JsonError, "lone low surrogate rejected");
    CHECK_THROW(json_parse("{\"a\":01}"), JsonError, "leading-zero number rejected");
    CHECK_THROW(json_parse("{\"a\":1.}"), JsonError, "empty fraction rejected");
    CHECK_THROW(json_parse("[1,2"), JsonError, "unterminated array rejected");
    CHECK_THROW(json_parse("{\"a\":1"), JsonError, "unterminated object rejected");
    CHECK_THROW(json_parse("{\"a\":1e}"), JsonError, "empty exponent rejected");

    // Типизированные доступы бросают JsonError на несовпадении типа.
    JsonValue wrong = json_parse("{\"n_layers\":\"32\",\"eps\":true}");
    CHECK_THROW((void)wrong.get_int("n_layers"), JsonError, "string instead of int rejected");
    CHECK_THROW((void)wrong.get_string("eps"), JsonError, "bool instead of string rejected");
    CHECK(wrong.find("missing") == nullptr);
}

TEST_MAIN("test_models_json")

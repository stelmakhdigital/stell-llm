#include "stell/engine/json.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace stell::engine {

namespace {

struct Parser {
    const std::string& src;
    std::size_t pos = 0;

    explicit Parser(const std::string& s) : src(s) {}

    void skip_ws() {
        while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos]))) ++pos;
    }

    [[noreturn]] void fail(const char* what) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "json: %s at offset %zu", what, pos);
        throw std::runtime_error(buf);
    }

    char peek() { return pos < src.size() ? src[pos] : '\0'; }

    Json parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    Json parse_object() {
        Json v; v.t = Json::T::Object;
        ++pos;                       // '{'
        skip_ws();
        if (peek() == '}') { ++pos; return v; }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("expect object key");
            Json key = parse_string();
            skip_ws();
            if (peek() != ':') fail("expect ':'");
            ++pos;
            v.obj.emplace_back(key.str, parse_value());
            skip_ws();
            const char c = peek();
            if (c == ',') { ++pos; continue; }
            if (c == '}') { ++pos; return v; }
            fail("expect ',' or '}'");
        }
    }

    Json parse_array() {
        Json v; v.t = Json::T::Array;
        ++pos;                       // '['
        skip_ws();
        if (peek() == ']') { ++pos; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            const char c = peek();
            if (c == ',') { ++pos; continue; }
            if (c == ']') { ++pos; return v; }
            fail("expect ',' or ']'");
        }
    }

    Json parse_string() {
        Json v; v.t = Json::T::String;
        ++pos;                       // '"'
        std::string out;
        while (pos < src.size() && src[pos] != '"') {
            // escapes не поддержаны (см. json.hpp): '"' внутри строки — ошибка
            if (src[pos] == '\\') fail("string escapes not supported (round 1)");
            out.push_back(src[pos]);
            ++pos;
        }
        if (pos >= src.size()) fail("unterminated string");
        ++pos;                       // '"'
        v.str = out;
        return v;
    }

    Json parse_bool() {
        Json v; v.t = Json::T::Bool;
        if (src.compare(pos, 4, "true") == 0) { pos += 4; v.boolean = true; return v; }
        if (src.compare(pos, 5, "false") == 0) { pos += 5; v.boolean = false; return v; }
        fail("bad literal");
    }

    Json parse_null() {
        Json v; v.t = Json::T::Null;
        if (src.compare(pos, 4, "null") == 0) { pos += 4; return v; }
        fail("bad literal");
    }

    Json parse_number() {
        std::size_t begin = pos;
        if (peek() == '-') ++pos;
        bool dot = false, exp = false;
        while (pos < src.size()) {
            const char c = src[pos];
            if (std::isdigit(static_cast<unsigned char>(c))) { ++pos; continue; }
            if (c == '.' && !dot) { dot = true; ++pos; continue; }
            if ((c == 'e' || c == 'E') && !exp) { exp = true; ++pos;
                if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) ++pos;
                continue; }
            break;
        }
        if (pos == begin) fail("expect number");
        Json v; v.t = Json::T::Number;
        std::string tok = src.substr(begin, pos - begin);
        v.num = std::strtod(tok.c_str(), nullptr);
        return v;
    }
};

} // namespace

Json json_parse(const std::string& src) {
    Parser p(src);
    Json v = p.parse_value();
    p.skip_ws();
    if (p.pos != src.size()) p.fail("trailing characters");
    return v;
}

std::string json_dump(const Json& v) {
    std::string out;
    auto numstr = [](double x) {
        char buf[64];
        if (std::fabs(x - (double)(std::int64_t)x) < 1e-9)
            std::snprintf(buf, sizeof buf, "%lld", (long long)(std::int64_t)x);
        else
            std::snprintf(buf, sizeof buf, "%g", x);
        return std::string(buf);
    };
    switch (v.t) {
    case Json::T::Null:   out += "null"; break;
    case Json::T::Bool:   out += v.boolean ? "true" : "false"; break;
    case Json::T::Number: out += numstr(v.num); break;
    case Json::T::String: out += '"' + v.str + '"'; break;
    case Json::T::Array:
        out += '[';
        for (std::size_t i = 0; i < v.arr.size(); ++i) {
            if (i) out += ',';
            out += json_dump(v.arr[i]);
        }
        out += ']';
        break;
    case Json::T::Object:
        out += '{';
        for (std::size_t i = 0; i < v.obj.size(); ++i) {
            if (i) out += ',';
            out += '"' + v.obj[i].first + "\":" + json_dump(v.obj[i].second);
        }
        out += '}';
        break;
    }
    return out;
}

} // namespace stell::engine

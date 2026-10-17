#include "stell/models/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace stell::models {
namespace {

constexpr int kMaxDepth = 128;

const JsonValue kNullValue{};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    JsonValue run() {
        JsonValue v = parse_value(0);
        skip_ws();
        if (i_ != s_.size()) fail("trailing characters after JSON value");
        return v;
    }

private:
    const std::string& s_;
    std::size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "JSON error at byte %zu: %s", i_, msg.c_str());
        throw JsonError(i_, buf);
    }

    void skip_ws() {
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i_;
            else break;
        }
    }

    char peek() {
        if (i_ >= s_.size()) fail("unexpected end of input");
        return s_[i_];
    }

    char next() {
        char c = peek();
        ++i_;
        return c;
    }

    void expect(char c) {
        skip_ws();
        if (i_ >= s_.size() || s_[i_] != c) fail(std::string("expected '") + c + "'");
        ++i_;
    }

    bool match(char c) {
        skip_ws();
        if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; }
        return false;
    }

    JsonValue parse_value(int depth) {
        if (depth > kMaxDepth) fail("nesting too deep");
        skip_ws();
        char c = peek();
        switch (c) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': { JsonValue v; v.type = JsonValue::Type::String; parse_string(&v.string_v); return v; }
            case 't': case 'f': return parse_bool();
            case 'n': return parse_null();
            default: return parse_number();
        }
    }

    JsonValue parse_object(int depth) {
        JsonValue v;
        v.type = JsonValue::Type::Object;
        ++i_;  // '{'
        skip_ws();
        if (peek() == '}') { ++i_; return v; }
        for (;;) {
            skip_ws();
            if (peek() != '"') fail("expected string key in object");
            std::string key;
            parse_string(&key);
            expect(':');
            v.object_v.emplace_back(std::move(key), parse_value(depth + 1));
            if (!match(',')) break;
        }
        expect('}');
        return v;
    }

    JsonValue parse_array(int depth) {
        JsonValue v;
        v.type = JsonValue::Type::Array;
        ++i_;  // '['
        skip_ws();
        if (peek() == ']') { ++i_; return v; }
        for (;;) {
            v.array_v.push_back(parse_value(depth + 1));
            if (!match(',')) break;
        }
        expect(']');
        return v;
    }

    void parse_string(std::string* out) {
        ++i_;  // '"'
        for (;;) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') return;
            if (c != '\\') { out->push_back(c); continue; }
            if (i_ >= s_.size()) fail("unterminated escape");
            char e = s_[i_++];
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': parse_utf8_escape(out); break;
                default: fail(std::string("invalid escape '\\") + e + "'");
            }
        }
    }

    // \uXXXX (+ суррогатные пары) → UTF-8.
    void parse_utf8_escape(std::string* out) {
        std::uint32_t cp = read_hex4();
        if (cp >= 0xD800 && cp <= 0xDBFF) {  // старшая половина суррогатной пары
            std::size_t save = i_;
            if (i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                i_ += 2;
                std::uint32_t lo = read_hex4();
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                } else {
                    i_ = save;  // не пара — вернём как есть (одиночный суррогат запрещён)
                    fail("lone high surrogate in \\u escape");
                }
            } else {
                fail("lone high surrogate in \\u escape");
            }
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("lone low surrogate in \\u escape");
        }
        // UTF-8 encode
        if (cp < 0x80) {
            out->push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    std::uint32_t read_hex4() {
        if (i_ + 4 > s_.size()) fail("truncated \\u escape");
        std::uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            if (c >= '0' && c <= '9') v = v * 16 + (c - '0');
            else if (c >= 'a' && c <= 'f') v = v * 16 + (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v = v * 16 + (c - 'A' + 10);
            else fail("invalid hex digit in \\u escape");
        }
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonValue::Type::Bool;
        if (s_.compare(i_, 4, "true") == 0) { i_ += 4; v.bool_v = true; return v; }
        if (s_.compare(i_, 5, "false") == 0) { i_ += 5; v.bool_v = false; return v; }
        fail("invalid literal");
    }

    JsonValue parse_null() {
        JsonValue v;
        v.type = JsonValue::Type::Null;
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return v; }
        fail("invalid literal");
    }

    JsonValue parse_number() {
        std::size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') ++i_;
        bool has_digits = false;
        while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') { ++i_; has_digits = true; }
        if (!has_digits) fail("invalid number");
        if (s_[start] == '-' ? (i_ - start > 2 && s_[start + 1] == '0')
                             : (i_ - start > 1 && s_[start] == '0'))
            fail("leading zeros in number are not allowed");
        bool is_double = false;
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            is_double = true;
            bool frac = false;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') { ++i_; frac = true; }
            if (!frac) fail("invalid number: empty fraction");
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            is_double = true;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            bool exp_digits = false;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') { ++i_; exp_digits = true; }
            if (!exp_digits) fail("invalid number: empty exponent");
        }
        std::string tok = s_.substr(start, i_ - start);
        JsonValue v;
        if (is_double) {
            v.type = JsonValue::Type::Double;
            v.double_v = std::strtod(tok.c_str(), nullptr);
        } else {
            v.type = JsonValue::Type::Int;
            errno = 0;
            v.int_v = std::strtoll(tok.c_str(), nullptr, 10);
            if (errno == ERANGE) fail("integer out of range");
        }
        return v;
    }
};

}  // namespace

JsonValue json_parse(const std::string& text) {
    Parser p(text);
    return p.run();
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& kv : object_v)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

const JsonValue& JsonValue::at(const std::string& key) const {
    const JsonValue* v = find(key);
    if (!v) throw JsonError(0, "missing key '" + key + "'");
    return *v;
}

static std::string type_name(JsonValue::Type t) {
    switch (t) {
        case JsonValue::Type::Null: return "null";
        case JsonValue::Type::Bool: return "bool";
        case JsonValue::Type::Int: return "int";
        case JsonValue::Type::Double: return "double";
        case JsonValue::Type::String: return "string";
        case JsonValue::Type::Array: return "array";
        case JsonValue::Type::Object: return "object";
    }
    return "?";
}

static void require(const JsonValue& v, JsonValue::Type t, const std::string& key) {
    if (v.type != t) {
        std::string msg = "key '" + key + "': expected " + type_name(t) + ", got " + type_name(v.type);
        throw JsonError(0, msg);
    }
}

std::int64_t JsonValue::get_int(const std::string& key) const {
    const JsonValue& v = at(key);
    require(v, Type::Int, key);
    return v.int_v;
}

std::uint64_t JsonValue::get_uint(const std::string& key) const {
    const JsonValue& v = at(key);
    require(v, Type::Int, key);
    if (v.int_v < 0) throw JsonError(0, "key '" + key + "': negative value for unsigned field");
    return static_cast<std::uint64_t>(v.int_v);
}

double JsonValue::get_double(const std::string& key) const {
    const JsonValue& v = at(key);
    if (v.type == Type::Int) return static_cast<double>(v.int_v);
    require(v, Type::Double, key);
    return v.double_v;
}

const std::string& JsonValue::get_string(const std::string& key) const {
    const JsonValue& v = at(key);
    require(v, Type::String, key);
    return v.string_v;
}

const JsonValue& JsonValue::get_array(const std::string& key) const {
    const JsonValue& v = at(key);
    require(v, Type::Array, key);
    return v;
}

const JsonValue& JsonValue::get_object(const std::string& key) const {
    const JsonValue& v = at(key);
    require(v, Type::Object, key);
    return v;
}

}  // namespace stell::models

// Общий JSON-писатель описания модели (round 4): используется gguf_to_description
// (round 2) и hf_to_description (round 4, [R4]) — единственный источник
// конвенции сериализации tensor_groups (дублирование запрещено).
// Парсер — stell/models/json.hpp (сериализатора в json.cpp нет — пишем руками).
#pragma once

#include <cinttypes>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace stell::models {

inline void json_escape(std::string& out, const std::string& s) {
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
}

inline std::string fmt_int(std::int64_t v) { return std::to_string(v); }

// Числа с плавающей точкой: "%.8g" → валидный JSON-литерал (1e-05 → "1e-05").
inline std::string fmt_f64(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.8g", v);
    return buf;
}

inline std::string fmt_shape(const std::vector<std::int64_t>& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) out += ",";
        out += fmt_int(s[i]);
    }
    return out + "]";
}

inline std::string fmt_card(const std::vector<std::pair<std::string, std::int64_t>>& card) {
    if (card.empty()) return "";
    std::string out = ",\"cardinality\":{";
    for (std::size_t i = 0; i < card.size(); ++i) {
        if (i) out += ",";
        out += "\"" + card[i].first + "\":" + fmt_int(card[i].second);
    }
    return out + "}";
}

inline std::string make_group(const std::string& name, const std::string& role,
                              const std::vector<std::int64_t>& shape,
                              const std::string& dtype, const std::string& pattern,
                              const std::vector<std::pair<std::string, std::int64_t>>& card =
                                  {},
                              const std::string& slice = "", bool kv_block = false) {
    std::string out = "{\"name\":\"";
    json_escape(out, name);
    out += "\",\"role\":\"" + role + "\",\"shape\":" + fmt_shape(shape);
    out += ",\"dtype\":\"" + dtype + "\",\"source_pattern\":\"";
    json_escape(out, pattern);
    out += "\"";
    if (!slice.empty()) {
        out += ",\"source_slice\":\"";
        json_escape(out, slice);
        out += "\"";
    }
    if (kv_block) out += ",\"kv_block\":true";
    out += fmt_card(card);
    return out + "}";
}

}  // namespace stell::models

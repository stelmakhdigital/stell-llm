// Реализация ByteLevelBpeTokenizer — см. tokenizer.hpp (шапка с контрактом).
#include "stell/models/tokenizer.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "stell/models/json.hpp"

namespace stell::models {
namespace {

std::string read_file_or_throw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw TokenizerError("tokenizer: cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// UTF-8-кодирование кодовой точки (алфавит byte-level маппинга ≤ U+01FF).
std::string utf8_encode(std::int64_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        throw TokenizerError("tokenizer: codepoint out of byte-level range");
    }
    return out;
}

// Декодирование ОДНОЙ кодовой точки из UTF-8 строки (pos — позиция, после
// чтения сдвигается). Ошибка → −1.
std::int64_t utf8_next(const std::string& s, std::size_t& pos) {
    if (pos >= s.size()) return -1;
    const unsigned char c = static_cast<unsigned char>(s[pos]);
    std::int64_t cp = -1;
    std::size_t len = 0;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
    else return -1;
    if (pos + len > s.size()) return -1;
    for (std::size_t i = 1; i < len; ++i) {
        const unsigned char cc = static_cast<unsigned char>(s[pos + i]);
        if ((cc & 0xC0) != 0x80) return -1;
        cp = (cp << 6) | (cc & 0x3F);
    }
    pos += len;
    return cp;
}

bool is_printable_range(std::int64_t b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
           (b >= 174 && b <= 255);
}

}  // namespace

// --- таблица byte ↔ unicode (GPT-2-конвенция, см. tokenizer.hpp) -------------

std::string ByteLevelBpeTokenizer::byte_to_char(std::uint8_t b) {
    // Ленивая таблица: печатные диапазоны ASCII ('!'..'~', '¡'..'¬',
    // '®'..'ÿ') — сам байт; прочие 68 байтов — кодовые точки 256, 257, …
    // по возрастанию байта (конвенция GPT-2-семейства byte-level).
    static const std::array<std::string, 256> table = [] {
        std::array<std::string, 256> t;
        for (int bb = 0; bb < 256; ++bb) {
            std::int64_t n = 0;
            for (int prev = 0; prev < bb; ++prev)
                if (!is_printable_range(prev)) ++n;
            const std::int64_t cp = is_printable_range(bb) ? bb : 256 + n;
            t[static_cast<std::size_t>(bb)] = utf8_encode(cp);
        }
        return t;
    }();
    return table[b];
}

int ByteLevelBpeTokenizer::char_to_byte(std::int64_t cp) {
    if (is_printable_range(cp) && cp <= 255) return static_cast<int>(cp);
    if (cp >= 256 && cp < 256 + 68) {
        std::int64_t n = cp - 256;
        for (int b = 0; b < 256; ++b) {
            if (!is_printable_range(b)) {
                if (n == 0) return b;
                --n;
            }
        }
    }
    return -1;
}

// --- загрузка tokenizer.json ---------------------------------------------------

ByteLevelBpeTokenizer ByteLevelBpeTokenizer::load(
    const std::string& tokenizer_json_path) {
    const std::string text = read_file_or_throw(tokenizer_json_path);
    JsonValue root;
    try {
        root = json_parse(text);
    } catch (const JsonError& e) {
        throw TokenizerError("tokenizer.json: parse error at " +
                             std::to_string(e.pos()) + ": " + e.what());
    }

    // Subset-проверки: только чистый byte-level BPE (тишина запрещена).
    if (const JsonValue* pt = root.find("pre_tokenizer")) {
        if (pt->type != JsonValue::Type::Object ||
            pt->get_string("type") != "ByteLevel")
            throw TokenizerError(
                "tokenizer.json: поддержан только ByteLevel pretokenizer "
                "(use_regex=false — конвенция fixture, R10)");
        if (pt->find("use_regex") && pt->find("use_regex")->bool_v)
            throw TokenizerError(
                "tokenizer.json: ByteLevel use_regex=true не поддержан "
                "(regex-сплиты — вне subset R10)");
    } else {
        throw TokenizerError("tokenizer.json: отсутствует pre_tokenizer");
    }
    if (const JsonValue* dec = root.find("decoder")) {
        if (dec->type != JsonValue::Type::Object ||
            dec->get_string("type") != "ByteLevel")
            throw TokenizerError("tokenizer.json: поддержан только ByteLevel "
                                 "decoder");
    } else {
        throw TokenizerError("tokenizer.json: отсутствует decoder");
    }
    if (root.find("normalizer") &&
        root.find("normalizer")->type != JsonValue::Type::Null)
        throw TokenizerError("tokenizer.json: normalizer не поддержан");
    if (const JsonValue* added = root.find("added_tokens")) {
        if (!added->array_v.empty())
            throw TokenizerError(
                "tokenizer.json: added/special tokens не поддержаны "
                "(subset R10: чистый byte-level BPE)");
    }

    ByteLevelBpeTokenizer tok;
    const JsonValue& model = root.get_object("model");
    {
        const JsonValue* unk = model.find("unk_token");
        if (unk && unk->type != JsonValue::Type::Null)
            throw TokenizerError("tokenizer.json: unk_token не поддержан");
        const JsonValue* bf = model.find("byte_fallback");
        if (bf && bf->bool_v)
            throw TokenizerError("tokenizer.json: byte_fallback не поддержан");
    }
    if (const JsonValue* drop = model.find("dropout")) {
        if (drop->type == JsonValue::Type::Double && drop->double_v != 0.0)
            throw TokenizerError("tokenizer.json: BPE dropout не поддержан");
    }

    // vocab: token → id; id2token плотный 0..N-1 (иначе — честная ошибка).
    std::size_t max_id = 0;
    for (const auto& [token, v] : model.get_object("vocab").object_v) {
        if (v.type != JsonValue::Type::Int || v.int_v < 0)
            throw TokenizerError("tokenizer.json: vocab id must be int >= 0");
        const std::int64_t id = v.int_v;
        tok.vocab_[token] = static_cast<std::int32_t>(id);
        if (static_cast<std::uint64_t>(id) > max_id) max_id = static_cast<std::uint64_t>(id);
    }
    tok.id2token_.resize(max_id + 1);
    for (const auto& [token, id] : tok.vocab_) {
        if (!tok.id2token_[static_cast<std::size_t>(id)].empty())
            throw TokenizerError("tokenizer.json: дубликат vocab id " +
                                 std::to_string(id));
        tok.id2token_[static_cast<std::size_t>(id)] = token;
    }

    // merges: пары строк (формат v2 [["a","b"], …]) либо legacy-строки "a b"
    // (пробел в byte-level алфавите невозможен — байт 0x20 маппится в символ).
    const JsonValue& merges = model.get_array("merges");
    for (std::size_t i = 0; i < merges.array_v.size(); ++i) {
        const JsonValue& m = merges.array_v[i];
        std::pair<std::string, std::string> pair;
        if (m.type == JsonValue::Type::Array && m.array_v.size() == 2 &&
            m.array_v[0].type == JsonValue::Type::String &&
            m.array_v[1].type == JsonValue::Type::String) {
            pair = {m.array_v[0].string_v, m.array_v[1].string_v};
        } else if (m.type == JsonValue::Type::String) {
            const std::string::size_type sp = m.string_v.find(' ');
            if (sp == std::string::npos)
                throw TokenizerError("tokenizer.json: bad merge entry #" +
                                     std::to_string(i));
            pair = {m.string_v.substr(0, sp), m.string_v.substr(sp + 1)};
        } else {
            throw TokenizerError("tokenizer.json: bad merge entry type #" +
                                 std::to_string(i));
        }
        if (!tok.ranks_.emplace(pair, tok.ranks_.size()).second)
            throw TokenizerError("tokenizer.json: дубликат merge #" +
                                 std::to_string(i));
    }
    return tok;
}

// --- encode / decode ----------------------------------------------------------

void ByteLevelBpeTokenizer::merge_symbols(
    std::vector<std::string>& symbols) const {
    while (symbols.size() >= 2) {
        // Пара с минимальным рангом среди смежных (ранги уникальны;
        // при равенстве невозможном — левая, детерминизм сохранён).
        std::size_t best = ranks_.size();
        std::size_t best_pos = 0;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = ranks_.find({symbols[i], symbols[i + 1]});
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                best_pos = i;
            }
        }
        if (best == ranks_.size()) break;  // смежных пар из merges нет
        const std::string first = symbols[best_pos];
        const std::string second = symbols[best_pos + 1];
        // Слияние ВСЕХ вхождений выбранной пары слева направо (классический
        // алгоритм BPE; эквивалентно heap-семантике эталонной реализации).
        std::vector<std::string> out;
        out.reserve(symbols.size());
        std::size_t i = 0;
        while (i < symbols.size()) {
            if (i + 1 < symbols.size() && symbols[i] == first &&
                symbols[i + 1] == second) {
                out.push_back(first + second);
                i += 2;
            } else {
                out.push_back(std::move(symbols[i]));
                ++i;
            }
        }
        symbols = std::move(out);
    }
}

std::vector<std::int32_t> ByteLevelBpeTokenizer::encode(
    const std::string& text) const {
    std::vector<std::int32_t> ids;
    if (text.empty()) return ids;
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i)
        symbols.push_back(byte_to_char(static_cast<std::uint8_t>(text[i])));
    merge_symbols(symbols);
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        const auto it = vocab_.find(sym);
        if (it == vocab_.end())
            throw TokenizerError("encode: symbol '" + sym +
                                 "' отсутствует в vocab (битый tokenizer.json?)");
        ids.push_back(it->second);
    }
    return ids;
}

std::string ByteLevelBpeTokenizer::decode(std::int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id2token_.size())
        return std::string();  // конвенция byte-fallback engine (ByteTokenizer)
    const std::string& token = id2token_[static_cast<std::size_t>(id)];
    std::string out;
    std::size_t pos = 0;
    while (pos < token.size()) {
        const std::int64_t cp = utf8_next(token, pos);
        if (cp < 0) return std::string();  // некорректный UTF-8 в токене
        const int b = char_to_byte(cp);
        if (b < 0) return std::string();  // символ вне алфавита byte-level
        out.push_back(static_cast<char>(b));
    }
    return out;
}

std::string ByteLevelBpeTokenizer::decode(
    const std::vector<std::int32_t>& ids) const {
    std::string out;
    for (std::int32_t id : ids) out += decode(id);
    return out;
}

}  // namespace stell::models

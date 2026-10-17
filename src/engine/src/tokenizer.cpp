#include "stell/engine/tokenizer.hpp"

namespace stell::engine {

std::vector<std::int32_t> ByteTokenizer::encode(const std::string& text) const {
    std::vector<std::int32_t> out;
    out.reserve(text.size());
    for (char c : text) out.push_back(static_cast<unsigned char>(c));
    return out;
}

std::string ByteTokenizer::decode(std::int32_t id) const {
    if (id < 0 || id > 255) return {};           // byte-level знает только байты
    const char c = static_cast<char>(static_cast<unsigned char>(id));
    return std::string(1, c);
}

std::string ByteTokenizer::decode(const std::vector<std::int32_t>& ids) const {
    std::string out;
    for (std::int32_t id : ids) out += decode(id);
    return out;
}

} // namespace stell::engine

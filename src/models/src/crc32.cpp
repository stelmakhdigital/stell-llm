#include "crc32.hpp"

namespace stell::models {
namespace {

std::uint32_t crc32_ieee_impl(const std::uint8_t* data, std::size_t n, std::uint32_t crc) {
    static std::uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i)
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

}  // namespace

std::uint32_t crc32_ieee(const void* data, std::size_t n, std::uint32_t seed) {
    return crc32_ieee_impl(static_cast<const std::uint8_t*>(data), n, seed);
}

}  // namespace stell::models

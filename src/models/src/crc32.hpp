// CRC32 (IEEE 802.3, полином 0xEDB88320) — общий для заголовка/групп stell-tier
// (ADR-12 §2.3) и для golden-фикстур. Единственная реализация в модуле models.
#pragma once

#include <cstddef>
#include <cstdint>

namespace stell::models {

std::uint32_t crc32_ieee(const void* data, std::size_t n, std::uint32_t seed = 0);

}  // namespace stell::models

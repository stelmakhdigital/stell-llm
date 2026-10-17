// Фабрика лоадеров и определение формата (ADR-09 §3, ADR-03).
#include "stell/models/weights.hpp"

#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace stell::models {
namespace {

const char* kFormatNames[] = {
    "Safetensors", "GGUF", "NVFP4W4A16", "StellTier",
};

bool read_prefix(const std::string& uri, std::size_t n, std::uint8_t* out) {
    std::FILE* f = std::fopen(uri.c_str(), "rb");
    if (!f) return false;
    const std::size_t got = std::fread(out, 1, n, f);
    std::fclose(f);
    return got == n;
}

}  // namespace

const char* weights_format_name(WeightsFormat f) {
    return kFormatNames[static_cast<int>(f)];
}

// Фабрики конкретных форматов определены в loader_*.cpp (namespace stell::models).
IWeightsLoader* make_safetensors_loader();
IWeightsLoader* make_gguf_loader();
IWeightsLoader* make_nvfp4w4a16_loader();
IWeightsLoader* make_stelltier_loader();

std::unique_ptr<IWeightsLoader> make_loader(WeightsFormat f) {
    switch (f) {
        case WeightsFormat::Safetensors:
            return std::unique_ptr<IWeightsLoader>(make_safetensors_loader());
        case WeightsFormat::GGUF:
            return std::unique_ptr<IWeightsLoader>(make_gguf_loader());
        case WeightsFormat::NVFP4W4A16:
            return std::unique_ptr<IWeightsLoader>(make_nvfp4w4a16_loader());
        case WeightsFormat::StellTier:
            return std::unique_ptr<IWeightsLoader>(make_stelltier_loader());
    }
    throw LoaderError(LoaderError::Code::IoFailure, "unknown weights format");
}

WeightsFormat detect_format(const std::string& uri) {
    const std::string ext = [&] {
        const std::size_t dot = uri.find_last_of('.');
        return dot == std::string::npos ? std::string() : uri.substr(dot);
    }();
    if (ext == ".stelltier") return WeightsFormat::StellTier;
    if (ext == ".gguf") return WeightsFormat::GGUF;
    if (ext == ".safetensors" || ext == ".sft") return WeightsFormat::Safetensors;

    // Fallback по магии файла.
    std::uint8_t magic[8] = {};
    if (!read_prefix(uri, 8, magic))
        throw LoaderError(LoaderError::Code::IoFailure, "cannot read file magic: " + uri);
    if (std::memcmp(magic, "GGUF", 4) == 0) return WeightsFormat::GGUF;
    if (std::memcmp(magic, "STLTIER", 7) == 0) return WeightsFormat::StellTier;
    return WeightsFormat::Safetensors;  // safetensors: первые 8 байт — длина заголовка
}

}  // namespace stell::models

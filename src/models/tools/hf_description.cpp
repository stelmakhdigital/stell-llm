// CLI: описание модели из HF-чекпоинта (round 4, [R4]/[R3-подготовка]).
//
//   stell_hf_description <hf_dir> [profile]
//
// hf_dir — каталог модели HF (config.json + safetensors-шарды) или одиночный
// .safetensors; profile — F16 (умолчание) | NVFP4 | W4A16.
// JSON описания печатается в stdout (STELL_R7_REAL_DESC_JSON для прогонов
// [R7] на реальных весах). Ошибки — в stderr, код возврата 1.
#include "stell/models/convert.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: stell_hf_description <hf_dir> [F16|NVFP4|W4A16]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const std::string profile = argc == 3 ? argv[2] : "F16";
    try {
        const std::string json = stell::models::hf_to_description(dir, profile);
        std::fputs(json.c_str(), stdout);
        std::fputc('\n', stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "stell_hf_description: %s\n", e.what());
        return 1;
    }
}

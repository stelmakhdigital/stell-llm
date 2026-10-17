#include "stell/models/registry.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <set>

namespace stell::models {
namespace {

// Чтение файла целиком; ошибки чтения = RegistryError.
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw RegistryError("cannot open model description: " + path);
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad()) throw RegistryError("cannot read model description: " + path);
    return data;
}

}  // namespace

const std::vector<std::string>& Registry::builtin_names() {
    // 1 dense + 2 MoE (PLAN.md §4 «Поток D», ADR-09 §2).
    static const std::vector<std::string> names = {"llama-class", "glm-class", "qwen-moe-class"};
    return names;
}

std::string Registry::registry_dir() {
    if (const char* env = std::getenv("STELL_MODELS_REGISTRY_DIR")) return env;
#ifdef STELL_REGISTRY_DIR
    return STELL_REGISTRY_DIR;
#else
    throw RegistryError("STELL_REGISTRY_DIR не задан; используйте STELL_MODELS_REGISTRY_DIR");
#endif
}

ModelDescription Registry::load(const std::string& json_path) {
    return parse_model_description(read_file(json_path), json_path);
}

ModelDescription Registry::load_builtin(const std::string& name) {
    std::set<std::string> known(builtin_names().begin(), builtin_names().end());
    if (!known.count(name))
        throw RegistryError("unknown builtin architecture '" + name + "'");
    return load(registry_dir() + "/" + name + ".json");
}

std::vector<std::string> Registry::available() const {
    return builtin_names();
}

}  // namespace stell::models

// ============================================================================
// Реестр архитектур — контракт ADR-09 §2.
//
// Реестр растит архитектуру за архитектурой (PLAN.md §6): встроенные описания —
// декларативные JSON-файлы в src/models/registry/ (1 dense + 2 MoE на раунд 1).
// Engine не имеет if'ов по архитектурам — только ops и groups (ADR-09 §2).
// ============================================================================
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "stell/models/description.hpp"

namespace stell::models {

class RegistryError : public std::runtime_error {
public:
    explicit RegistryError(const std::string& what) : std::runtime_error(what) {}
};

class IRegistry {
public:
    virtual ~IRegistry() = default;

    // Загрузка/валидация описания из JSON-файла: schema-версия, операции,
    // роли, cardinality (ADR-09 §2). Ошибки — исключениями (DescriptionError).
    virtual ModelDescription load(const std::string& json_path) = 0;

    // Список встроенных архитектур реестра.
    virtual std::vector<std::string> available() const = 0;
};

class Registry : public IRegistry {
public:
    ModelDescription load(const std::string& json_path) override;
    std::vector<std::string> available() const override;

    // Загрузка встроенной архитектуры по имени из available().
    ModelDescription load_builtin(const std::string& name);

    // Каталог встроенных описаний: env STELL_MODELS_REGISTRY_DIR перекрывает
    // компайл-тайм значение STELL_REGISTRY_DIR (задаётся в CMakeLists).
    static std::string registry_dir();

private:
    // Порядок фиксирован (1 dense + 2 MoE, PLAN.md §4 «Поток D»).
    static const std::vector<std::string>& builtin_names();
};

}  // namespace stell::models

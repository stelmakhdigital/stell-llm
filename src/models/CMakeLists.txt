# ============================================================================
# models — Поток D (PLAN.md §4): декларативный реестр архитектур (ADR-09),
# лоадеры весов safetensors/GGUF/NVFP4W4A16/stell-tier (ADR-03/09/12),
# конвертер в stell-tier layout (ADR-12 §1). Зависимости C++ — только std.
# ============================================================================

add_library(stell_models STATIC)

target_sources(stell_models PRIVATE
    src/json.cpp
    src/description.cpp
    src/registry.cpp
    src/crc32.cpp
    src/quant.cpp
    src/stelltier_format.cpp
    src/loader_safetensors.cpp
    src/loader_gguf.cpp
    src/gguf_description.cpp
    src/loader_quant.cpp
    src/loader_stelltier.cpp
    src/weights_factory.cpp
    src/converter.cpp
)

target_include_directories(stell_models
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Round 2: реальный stell/tiers/types.hpp (поток B, ADR-08 §1) — зеркало
# tiers_compat.hpp удалено, типы stell::tiers приходят из stell_tiers.
target_link_libraries(stell_models PUBLIC stell_tiers)

# Каталог встроенных описаний (реестр, ADR-09 §2): env STELL_MODELS_REGISTRY_DIR
# перекрывает это значение (см. Registry::registry_dir).
target_compile_definitions(stell_models PRIVATE
    STELL_REGISTRY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/registry"
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(stell_models PRIVATE -Wall -Wextra)
endif()

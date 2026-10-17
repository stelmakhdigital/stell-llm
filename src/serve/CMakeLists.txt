# serve: Python control-plane живёт в python/stell_serve (см. docs/design/serve.md, ADR-11).
# Здесь собирается CPU-fallback mock-движок с контрактом ADR-11
# (libstell_mock_core.so) для локальных тестов без GPU.
#
# Зеркало контракта: src/serve/mock/capi.h (авторитетная копия — src/engine).
# Зачем mock на C, а не Python-fake: см. комментарий в stell_mock_engine.c —
# проверяется реальный проход JSON через границу C ABI и владение памятью.

# C++ нужен только для stell_core_serve_check (проверка ABI реального движка).
project(serve LANGUAGES C CXX)

add_library(stell_mock_core SHARED mock/stell_mock_engine.c)
target_include_directories(stell_mock_core PRIVATE mock)
set_target_properties(stell_mock_core PROPERTIES
    OUTPUT_NAME "stell_mock_core"
    C_STANDARD 11
)
# -fvisibility=default: экспортируем ABI целиком; -Wall для чистоты
target_compile_options(stell_mock_core PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# Локальная проверка ABI РЕАЛЬНОГО движка (src/engine, поток A) — без правки
# чужого каталога src/engine: те же исходники собираются здесь в SHARED
# libstell_core_serve.so. Это НЕ таргет потока core: в master появится его
# SHARED-таргет `stell_core` (libstell_core.so в build/cpu/src/engine/);
# биндинг serve предпочитает его (STELL_ENGINE_LIB → build/cpu/src/engine/),
# а этот таргет служит для контрактных тестов serve на реальном C ABI
# (включая stell_step), пока тот не вклеен.
# ---------------------------------------------------------------------------
add_library(stell_core_serve_check SHARED
    ../engine/src/capi.cpp
    ../engine/src/scheduler.cpp
    ../engine/src/execution_plan.cpp
    ../engine/src/speculative.cpp
    ../engine/src/json.cpp)
target_include_directories(stell_core_serve_check PRIVATE ../engine/include)
set_target_properties(stell_core_serve_check PROPERTIES
    OUTPUT_NAME "stell_core_serve"
    CXX_STANDARD 20
)
target_compile_options(stell_core_serve_check PRIVATE -Wall -Wextra)

# serve: Python control-plane живёт в python/stell_serve (см. docs/design/serve.md, ADR-11).
# Здесь собирается только CPU-fallback mock-движок с контрактом ADR-11
# (libstell_mock_core.so) для локальных тестов без GPU.
#
# Зеркало контракта: src/serve/mock/capi.h (авторитетная копия — src/engine).
# Зачем mock на C, а не Python-fake: см. комментарий в stell_mock_engine.c —
# проверяется реальный проход JSON через границу C ABI и владение памятью.

project(serve LANGUAGES C)

add_library(stell_mock_core SHARED mock/stell_mock_engine.c)
target_include_directories(stell_mock_core PRIVATE mock)
set_target_properties(stell_mock_core PROPERTIES
    OUTPUT_NAME "stell_mock_core"
    C_STANDARD 11
)
# -fvisibility=default: экспортируем ABI целиком; -Wall для чистоты
target_compile_options(stell_mock_core PRIVATE -Wall -Wextra)

cmake_minimum_required(VERSION 3.24)
project(stell-llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STELL_CUDA "Build CUDA kernels for SM70" OFF)
# Tesla V100 (SM70, SXM2) — глобально ДО add_subdirectory: новый CMake (3.2x+)
# считает пустые CUDA_ARCHITECTURES ошибкой генерации для любого .cu-таргета.
if(STELL_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
option(STELL_BUILD_TESTS "Build tests" ON)
option(STELL_BUILD_SERVE "Build serve control-plane (python) integration" ON)

enable_testing()

add_subdirectory(src/engine)
add_subdirectory(src/kernels)
add_subdirectory(src/tiers)
add_subdirectory(src/models)
add_subdirectory(src/parallel)
add_subdirectory(src/serve)
add_subdirectory(src/ui)
add_subdirectory(src/calibrate)
add_subdirectory(src/bench)
add_subdirectory(tests)

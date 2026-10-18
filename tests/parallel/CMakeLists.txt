# tests/parallel: юнит + golden для src/parallel (контракт ADR-10)
add_executable(test_parallel
    test_main.cpp
    test_plan.cpp
    test_dtype_codec.cpp
    test_emulated_collectives.cpp
    test_golden_tp.cpp
    test_pp_tp_ep_combinations.cpp
    test_ep_registry_glm.cpp
    test_stress_multi_rank.cpp
    test_meta_broadcast.cpp
    test_golden_real_models.cpp
)

target_include_directories(test_parallel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    # Реестр архитектур (glm-class) — src/models (заголовки + quant.hpp).
    ${CMAKE_SOURCE_DIR}/src/models/include
    ${CMAKE_SOURCE_DIR}/src/models/src
    # CPU-референсы kernels (attention, MoE gather/scatter).
    ${CMAKE_SOURCE_DIR}/src/kernels/include
    # Писатели синтетических safetensors/GGUF (tests/models).
    ${CMAKE_SOURCE_DIR}/tests/models
)
target_link_libraries(test_parallel PRIVATE stell::parallel stell_models stell_kernels)
target_compile_definitions(test_parallel PRIVATE
    STELL_PARALLEL_REGISTRY_DIR="${CMAKE_SOURCE_DIR}/src/models/registry"
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures"
)
add_test(NAME test_parallel COMMAND test_parallel)

# --- GPU-тест NCCL-backend (только STELL_CUDA=ON + libnccl, сервер) ----------
if(STELL_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_path(_NCCL_INCLUDE_DIR nccl.h PATHS "/usr/include" "/usr/local/cuda/include")
    find_library(_NCCL_LIB nccl PATHS "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/usr/lib")
    add_executable(test_parallel_gpu test_nccl_gpu.cpp)
    target_include_directories(test_parallel_gpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${_NCCL_INCLUDE_DIR}
    )
    target_link_libraries(test_parallel_gpu PRIVATE stell::parallel CUDA::cudart)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_parallel_gpu PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME parallel_gpu COMMAND test_parallel_gpu)
endif()

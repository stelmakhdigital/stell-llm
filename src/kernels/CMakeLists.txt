# kernels: SM70-attention FP32 QK/PV, MoE gather/scatter, dequant NVFP4/W4A16,
# KV-квант E4M3, драфтер-ядра (поток A — core; replay-тесты на CPU-референсах)
add_library(stell_kernels STATIC
    src/attention.cpp
    src/moe_gather_scatter.cpp
    src/dequant.cpp
    src/kv_quant.cpp
    src/draft.cpp
)
target_include_directories(stell_kernels PUBLIC include)
target_compile_options(stell_kernels PRIVATE -Wall -Wextra)

if(STELL_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)   # Tesla V100 (SM70, SXM2)
    target_sources(stell_kernels PRIVATE
        cuda/attention_sm70.cu
        cuda/moe_sm70.cu
        cuda/dequant_sm70.cu
        cuda/kv_quant_sm70.cu
        cuda/draft_sm70.cu
    )
    target_compile_definitions(stell_kernels PUBLIC STELL_CUDA=1)
endif()

# --- операторные replay-тесты (tests/core/kernels) — CPU-фикстуры с эталонами
add_executable(test_kernels_attention
    ../../tests/core/kernels/test_attention.cpp
)
target_link_libraries(test_kernels_attention stell_kernels)
target_include_directories(test_kernels_attention PRIVATE ../../tests/core)
add_test(NAME kernels/attention_replay COMMAND test_kernels_attention)

add_executable(test_kernels_moe
    ../../tests/core/kernels/test_moe.cpp
)
target_link_libraries(test_kernels_moe stell_kernels)
target_include_directories(test_kernels_moe PRIVATE ../../tests/core)
add_test(NAME kernels/moe_replay COMMAND test_kernels_moe)

add_executable(test_kernels_dequant
    ../../tests/core/kernels/test_dequant.cpp
)
target_link_libraries(test_kernels_dequant stell_kernels)
target_include_directories(test_kernels_dequant PRIVATE ../../tests/core)
add_test(NAME kernels/dequant_replay COMMAND test_kernels_dequant)

add_executable(test_kernels_kv_quant
    ../../tests/core/kernels/test_kv_quant.cpp
)
target_link_libraries(test_kernels_kv_quant stell_kernels)
target_include_directories(test_kernels_kv_quant PRIVATE ../../tests/core)
add_test(NAME kernels/kv_quant_replay COMMAND test_kernels_kv_quant)

add_executable(test_kernels_draft
    ../../tests/core/kernels/test_draft.cpp
)
target_link_libraries(test_kernels_draft stell_kernels)
target_include_directories(test_kernels_draft PRIVATE ../../tests/core)
add_test(NAME kernels/draft_replay COMMAND test_kernels_draft)

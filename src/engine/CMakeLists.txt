# engine: планировщик continuous batching, пулы слотов, CUDA Graph + eager-фолббек,
# спекулятивный контур, KV-streaming (поток A — core; контракты ADR-08/09/10/11)
add_library(stell_engine STATIC
    src/scheduler.cpp
    src/execution_plan.cpp
    src/speculative.cpp
)
target_include_directories(stell_engine PUBLIC include)
target_compile_options(stell_engine PRIVATE -Wall -Wextra)

if(STELL_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)   # Tesla V100 (SM70, SXM2)
    target_sources(stell_engine PRIVATE cuda/graph_executor.cu)
    target_compile_definitions(stell_engine PUBLIC STELL_CUDA=1)
endif()

# --- юнит-тесты планировщика/пулов/планов (tests/core/engine) ----------------
add_executable(test_engine_slot_pool
    ../../tests/core/engine/test_slot_pool.cpp
)
target_link_libraries(test_engine_slot_pool stell_engine)
target_include_directories(test_engine_slot_pool PRIVATE ../../tests/core)
add_test(NAME engine/slot_pool COMMAND test_engine_slot_pool)

add_executable(test_engine_scheduler
    ../../tests/core/engine/test_scheduler.cpp
)
target_link_libraries(test_engine_scheduler stell_engine)
target_include_directories(test_engine_scheduler PRIVATE ../../tests/core)
add_test(NAME engine/scheduler COMMAND test_engine_scheduler)

add_executable(test_engine_plan
    ../../tests/core/engine/test_plan.cpp
)
target_link_libraries(test_engine_plan stell_engine)
target_include_directories(test_engine_plan PRIVATE ../../tests/core)
add_test(NAME engine/plan COMMAND test_engine_plan)

add_executable(test_engine_executor
    ../../tests/core/engine/test_decode_executor.cpp
)
target_link_libraries(test_engine_executor stell_engine)
target_include_directories(test_engine_executor PRIVATE ../../tests/core)
add_test(NAME engine/decode_executor COMMAND test_engine_executor)

add_executable(test_engine_speculative
    ../../tests/core/engine/test_speculative.cpp
)
target_link_libraries(test_engine_speculative stell_engine)
target_include_directories(test_engine_speculative PRIVATE ../../tests/core)
add_test(NAME engine/speculative COMMAND test_engine_speculative)

add_executable(test_engine_kv_stream
    ../../tests/core/engine/test_kv_stream.cpp
)
target_link_libraries(test_engine_kv_stream stell_engine)
target_include_directories(test_engine_kv_stream PRIVATE ../../tests/core)
add_test(NAME engine/kv_stream COMMAND test_engine_kv_stream)

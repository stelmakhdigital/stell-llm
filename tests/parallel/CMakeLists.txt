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
)

target_include_directories(test_parallel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    # Реестр архитектур (glm-class) — src/models (заголовки + quant.hpp).
    ${CMAKE_SOURCE_DIR}/src/models/include
    ${CMAKE_SOURCE_DIR}/src/models/src
)
target_link_libraries(test_parallel PRIVATE stell::parallel stell_models)
target_compile_definitions(test_parallel PRIVATE
    STELL_PARALLEL_REGISTRY_DIR="${CMAKE_SOURCE_DIR}/src/models/registry"
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures"
)
add_test(NAME test_parallel COMMAND test_parallel)

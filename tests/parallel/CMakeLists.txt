# tests/parallel: юнит + golden для src/parallel (контракт ADR-10)
add_executable(test_parallel
    test_main.cpp
    test_plan.cpp
    test_dtype_codec.cpp
    test_emulated_collectives.cpp
    test_golden_tp.cpp
    test_pp_tp_ep_combinations.cpp
)

target_include_directories(test_parallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_parallel PRIVATE stell::parallel)
add_test(NAME test_parallel COMMAND test_parallel)

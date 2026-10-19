# tests/tiers: юнит-тесты потока B (RAM/NVMe-тиры, политика, prefetch, layout, stats)
set(TIERS_TEST_SOURCES
  test_layout.cpp
  test_nvme_tier.cpp
  test_policy.cpp
  test_predictor.cpp
  test_manager.cpp
  test_stats_threads.cpp
  test_cross_arch.cpp
  test_engine_mimic.cpp
)

# Кросс-модульный тест tiers↔models (ADR-12): конвертер models (writer
# stell-tier) → читатель tiers. models-сторона — отдельная TU с плоским
# C-ABI (cross_models_shim.cpp): зеркало типов ADR-08 в models
# (tiers_compat.hpp) несовместимо с реальным types.hpp в одной TU (ODR).
add_executable(test_cross_models
  test_cross_models.cpp
  cross_models_shim.cpp
)
target_include_directories(test_cross_models PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/tests/models
  ${CMAKE_SOURCE_DIR}/src/models/include
  ${CMAKE_SOURCE_DIR}/src/models/src
)
target_link_libraries(test_cross_models PRIVATE stell_tiers stell_models)
target_compile_definitions(test_cross_models PRIVATE
  STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures"
)
add_test(NAME tiers/test_cross_models COMMAND test_cross_models)
set_tests_properties(tiers/test_cross_models PROPERTIES TIMEOUT 60)

# [R2] round 5: тест bench per-group layout (bench_tiers_pause, режим
# pergroup): спавнит бинарник bench по абсолютному пути из генератора.
add_executable(test_bench_per_group test_bench_per_group.cpp)
target_include_directories(test_bench_per_group PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/tiers/src)
target_link_libraries(test_bench_per_group PRIVATE stell_tiers)
target_compile_definitions(test_bench_per_group PRIVATE
  STELL_BENCH_PAUSE_BIN="$<TARGET_FILE:bench_tiers_pause>")
add_test(NAME tiers/test_bench_per_group COMMAND test_bench_per_group)
set_tests_properties(tiers/test_bench_per_group PROPERTIES TIMEOUT 120)

# Отчётный бенч: паузы слоёв (CPU-путь tier-контур, io_uring, NVMe-файл).
# Не add_test — измерение, а не ассершн-тест.
add_executable(bench_tiers_pause bench_pause.cpp)
target_include_directories(bench_tiers_pause PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/tiers/src)
target_link_libraries(bench_tiers_pause PRIVATE stell_tiers)

foreach(src IN LISTS TIERS_TEST_SOURCES)
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/tiers/src)
  target_link_libraries(${name} PRIVATE stell_tiers)
  add_test(NAME tiers/${name} COMMAND ${name})
  set_tests_properties(tiers/${name} PROPERTIES TIMEOUT 60)
endforeach()

# GPU-smoke DMA-копирований (cudaMemcpyAsync H↔D, pinned→device) — только
# при STELL_CUDA; честный отчёт co-tenancy (свободная VRAM) — см. bench.
if(STELL_CUDA)
  # CUDA::cudart — directory-scoped imported target; find_package в этой
  # директории (наследование из src/tiers не видно — sibling subdirectory).
  find_package(CUDAToolkit REQUIRED)
  add_executable(bench_tiers_cuda bench_cuda.cpp)
  target_link_libraries(bench_tiers_cuda PRIVATE CUDA::cudart)
endif()

# tests/tiers: юнит-тесты потока B (RAM/NVMe-тиры, политика, prefetch, layout, stats)
set(TIERS_TEST_SOURCES
  test_layout.cpp
  test_nvme_tier.cpp
  test_policy.cpp
  test_predictor.cpp
  test_manager.cpp
  test_stats_threads.cpp
  test_cross_arch.cpp
)

foreach(src IN LISTS TIERS_TEST_SOURCES)
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/tiers/src)
  target_link_libraries(${name} PRIVATE stell_tiers)
  add_test(NAME tiers/${name} COMMAND ${name})
  set_tests_properties(tiers/${name} PROPERTIES TIMEOUT 60)
endforeach()

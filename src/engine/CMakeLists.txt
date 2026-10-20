# engine: планировщик continuous batching, пулы слотов, CUDA Graph + eager-фолббек,
# спекулятивный контур, KV-streaming, CPU-исполнение модели (round 2, интеграция
# A×D), C ABI (libstell_core.so — ADR-11 §1); контракты ADR-08/09/10/11
#
# Round 2: зависимости C++ — stell_tiers (ADR-08, реальный TierManager),
# stell_models (ADR-09/03/12, лоадеры/реестр — через TU-мост model_spec_bridge),
# stell_parallel (ADR-10, CPU-эмулятор), stell_kernels (CPU-референс attention/
# MoE/dequant).
set(STELL_ENGINE_SOURCES
    src/scheduler.cpp
    src/execution_plan.cpp
    src/speculative.cpp
    src/json.cpp
    src/tokenizer.cpp
    src/ops_runner.cpp
    src/model_spec_bridge.cpp
    src/capi.cpp
)

find_package(Threads REQUIRED)   # parallel: pthread (CPU-эмулятор, ADR-10 §4)

add_library(stell_engine STATIC ${STELL_ENGINE_SOURCES})
target_include_directories(stell_engine PUBLIC include)
target_link_libraries(stell_engine
    PUBLIC stell_tiers stell_models stell_parallel stell_kernels)
target_compile_options(stell_engine PRIVATE -Wall -Wextra)

# --- C ABI для serve (ADR-11 §1): libstell_core.so через dlopen ---------------
# Символы экспортированы целиком (visibility default); smoke-тест dlopen —
# tests/core/engine/test_capi_dlopen.cpp.
#
# Зависимости (stell_tiers/models/parallel/kernels — статические, round 1,
# потоки B/C/D) собраны БЕЗ -fPIC и их CMakeLists менять запрещено правилами
# раунда — поэтому dep-исходники включены в SHARED-таргет как собственные
# исходники с их include-каталогами (честно: дублирует компиляцию dep-модулей
# в .so, не трогая чужие цели).
add_library(stell_core SHARED
    ${STELL_ENGINE_SOURCES}
    ${CMAKE_SOURCE_DIR}/src/kernels/src/attention.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/src/moe_gather_scatter.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/src/dequant.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/src/kv_quant.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/src/draft.cpp
    ${CMAKE_SOURCE_DIR}/src/tiers/src/io_uring.cpp
    ${CMAKE_SOURCE_DIR}/src/tiers/src/layout.cpp
    ${CMAKE_SOURCE_DIR}/src/tiers/src/nvme_file.cpp
    ${CMAKE_SOURCE_DIR}/src/tiers/src/manager.cpp
    ${CMAKE_SOURCE_DIR}/src/tiers/src/policy.cpp
    ${CMAKE_SOURCE_DIR}/src/tiers/src/predictor.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/json.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/description.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/registry.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/crc32.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/quant.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/stelltier_format.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/loader_safetensors.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/loader_gguf.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/loader_quant.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/loader_stelltier.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/weights_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/models/src/converter.cpp
    ${CMAKE_SOURCE_DIR}/src/parallel/src/comm.cpp
    ${CMAKE_SOURCE_DIR}/src/parallel/src/emulated_communicator.cpp
    ${CMAKE_SOURCE_DIR}/src/parallel/src/plan.cpp
)
target_include_directories(stell_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/kernels/include
    ${CMAKE_SOURCE_DIR}/src/tiers/include
    ${CMAKE_SOURCE_DIR}/src/tiers/src
    ${CMAKE_SOURCE_DIR}/src/models/include
    ${CMAKE_SOURCE_DIR}/src/models/src
    ${CMAKE_SOURCE_DIR}/src/parallel/include)
target_compile_definitions(stell_core PRIVATE
    STELL_REGISTRY_DIR="${CMAKE_SOURCE_DIR}/src/models/registry")
target_link_libraries(stell_core PUBLIC Threads::Threads)
target_compile_options(stell_core PRIVATE -Wall -Wextra)
if(STELL_CUDA)
    target_compile_definitions(stell_core PRIVATE STELL_CUDA=1)
endif()

if(STELL_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)   # Tesla V100 (SM70, SXM2)
    # dep-исходники tiers/parallel с STELL_CUDA=1 тянут cuda_runtime.h —
    # CUDA-include нужен и в перекомпиляции внутри SHARED stell_core
    # (серверная сборка; аналогично include-строкам в чужих CMakeLists).
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(stell_core PRIVATE SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    target_sources(stell_engine PRIVATE cuda/graph_executor.cu)
    target_sources(stell_core PRIVATE cuda/graph_executor.cu)
    target_compile_definitions(stell_engine PUBLIC STELL_CUDA=1)
    target_compile_definitions(stell_core PUBLIC STELL_CUDA=1)
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

add_executable(test_engine_comm
    ../../tests/core/engine/test_comm.cpp
)
target_link_libraries(test_engine_comm stell_engine)
target_include_directories(test_engine_comm PRIVATE ../../tests/core)
add_test(NAME engine/comm COMMAND test_engine_comm)

add_executable(test_engine_capi
    ../../tests/core/engine/test_capi.cpp
)
target_link_libraries(test_engine_capi stell_engine)
target_include_directories(test_engine_capi PRIVATE ../../tests/core)
target_compile_definitions(test_engine_capi PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/capi COMMAND test_engine_capi)

# --- round 2: интеграционные тесты срезов 1 (CPU-вариант) --------------------
# Фикстуры моделей (поток D, tests/models/fixtures) и golden-каталог —
# read-only использование чужих каталогов; путь задаётся компайл-тайм.

add_executable(test_engine_ops_dense
    ../../tests/core/engine/test_ops_dense.cpp
)
target_link_libraries(test_engine_ops_dense stell_engine)
target_include_directories(test_engine_ops_dense PRIVATE ../../tests/core)
target_compile_definitions(test_engine_ops_dense PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/ops_dense COMMAND test_engine_ops_dense)

add_executable(test_engine_ops_moe
    ../../tests/core/engine/test_ops_moe.cpp
)
target_link_libraries(test_engine_ops_moe stell_engine)
target_include_directories(test_engine_ops_moe PRIVATE ../../tests/core)
target_compile_definitions(test_engine_ops_moe PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/ops_moe COMMAND test_engine_ops_moe)

add_executable(test_engine_golden
    ../../tests/core/engine/test_golden.cpp
)
target_link_libraries(test_engine_golden stell_engine)
target_include_directories(test_engine_golden PRIVATE ../../tests/core)
target_compile_definitions(test_engine_golden PRIVATE
    STELL_GOLDEN_FIXTURE="${CMAKE_SOURCE_DIR}/tests/models/golden"
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/golden COMMAND test_engine_golden)

add_executable(test_engine_tier_integration
    ../../tests/core/engine/test_tier_integration.cpp
)
target_link_libraries(test_engine_tier_integration stell_engine)
target_include_directories(test_engine_tier_integration PRIVATE ../../tests/core)
target_compile_definitions(test_engine_tier_integration PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/tier_integration COMMAND test_engine_tier_integration)

add_executable(test_engine_parallel_integration
    ../../tests/core/engine/test_parallel_integration.cpp
)
target_link_libraries(test_engine_parallel_integration stell_engine)
target_include_directories(test_engine_parallel_integration PRIVATE ../../tests/core)
add_test(NAME engine/parallel_integration COMMAND test_engine_parallel_integration)

add_executable(test_engine_tokenizer
    ../../tests/core/engine/test_tokenizer.cpp
)
target_link_libraries(test_engine_tokenizer stell_engine)
target_include_directories(test_engine_tokenizer PRIVATE ../../tests/core)
add_test(NAME engine/tokenizer COMMAND test_engine_tokenizer)

add_executable(test_engine_capi_dlopen
    ../../tests/core/engine/test_capi_dlopen.cpp
)
# Только dlopen: stell_engine НЕ линкуется (проверяется именно загрузка .so).
target_compile_definitions(test_engine_capi_dlopen PRIVATE
    STELL_CORE_LIB="$<TARGET_FILE:stell_core>")
target_link_libraries(test_engine_capi_dlopen ${CMAKE_DL_LIBS})
target_include_directories(test_engine_capi_dlopen PRIVATE ../../tests/core)
add_test(NAME engine/capi_dlopen COMMAND test_engine_capi_dlopen)

# --- round 2 (T-20261017200354): ABI stell_logits + физический путь весов ----

add_executable(test_engine_capi_logits
    ../../tests/core/engine/test_capi_logits.cpp
)
target_link_libraries(test_engine_capi_logits stell_engine)
target_include_directories(test_engine_capi_logits PRIVATE ../../tests/core)
target_compile_definitions(test_engine_capi_logits PRIVATE
    STELL_GOLDEN_FIXTURE="${CMAKE_SOURCE_DIR}/tests/models/golden"
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/capi_logits COMMAND test_engine_capi_logits)

add_executable(test_engine_weight_path
    ../../tests/core/engine/test_weight_path.cpp
)
target_link_libraries(test_engine_weight_path stell_engine)
target_include_directories(test_engine_weight_path PRIVATE ../../tests/core)
target_compile_definitions(test_engine_weight_path PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/weight_path COMMAND test_engine_weight_path)

# --- round 5 (T-20261017200354): [R1] last-only projection -------------------

add_executable(test_engine_prompt_last
    ../../tests/core/engine/test_prompt_last.cpp
)
target_link_libraries(test_engine_prompt_last stell_engine)
target_include_directories(test_engine_prompt_last PRIVATE ../../tests/core)
target_compile_definitions(test_engine_prompt_last PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/prompt_last COMMAND test_engine_prompt_last)

# --- round 5: [R3] per-layer план (смешанные dense/MoE слои) ------------------

add_executable(test_engine_ops_mixed
    ../../tests/core/engine/test_ops_mixed.cpp
)
target_link_libraries(test_engine_ops_mixed stell_engine)
target_include_directories(test_engine_ops_mixed PRIVATE ../../tests/core)
target_compile_definitions(test_engine_ops_mixed PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/ops_mixed COMMAND test_engine_ops_mixed)

# --- round 5: [R4] shared-expert (ffn.shared, dense-FFN + сумма) --------------

add_executable(test_engine_ops_shared
    ../../tests/core/engine/test_ops_shared.cpp
)
target_link_libraries(test_engine_ops_shared stell_engine)
target_include_directories(test_engine_ops_shared PRIVATE ../../tests/core)
target_compile_definitions(test_engine_ops_shared PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/ops_shared COMMAND test_engine_ops_shared)

# --- round 6: [R1] скалярный сигмоид-гейт ffn.shared.gate_scalar --------------

add_executable(test_engine_ops_gate
    ../../tests/core/engine/test_ops_gate.cpp
)
target_link_libraries(test_engine_ops_gate stell_engine)
target_include_directories(test_engine_ops_gate PRIVATE ../../tests/core)
target_compile_definitions(test_engine_ops_gate PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/ops_gate COMMAND test_engine_ops_gate)

# --- round 7: [R1]/[R2] per-head RMSNorm q/k (qk_eps) + router_renorm ---------

add_executable(test_engine_ops_qk_norm
    ../../tests/core/engine/test_ops_qk_norm.cpp
)
target_link_libraries(test_engine_ops_qk_norm stell_engine)
target_include_directories(test_engine_ops_qk_norm PRIVATE ../../tests/core)
target_compile_definitions(test_engine_ops_qk_norm PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/ops_qk_norm COMMAND test_engine_ops_qk_norm)

# --- round 5: [R6] source_slice через мост (fused qkv, одно открытие) ---------

add_executable(test_engine_weights_slice
    ../../tests/core/engine/test_weights_slice.cpp
)
target_link_libraries(test_engine_weights_slice stell_engine)
target_include_directories(test_engine_weights_slice PRIVATE ../../tests/core)
target_compile_definitions(test_engine_weights_slice PRIVATE
    STELL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/models/fixtures")
add_test(NAME engine/weights_slice COMMAND test_engine_weights_slice)

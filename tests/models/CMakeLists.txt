# ============================================================================
# tests/models — юнит-тесты модуля models (Поток D, ADR-09/03/12).
# Синтетические входные файлы (safetensors/GGUF) генерируются самими тестами;
# golden-логиты — референсы раунда 2 (независимый Python-интерпретатор +
# HF-кроссчек, см. golden/README.md); прогоны engine (core-2) дописывают
# logits_tested.
# ============================================================================

# Внутренние заголовки модуля (quant.hpp, tensor_file_formats.hpp) и каркасы тестов.
set(_models_src_dir ${CMAKE_SOURCE_DIR}/src/models/src)

add_executable(test_models_json test_json.cpp)
add_executable(test_models_description test_description.cpp)
add_executable(test_models_registry test_registry.cpp)
add_executable(test_models_weights_safetensors test_weights_safetensors.cpp)
add_executable(test_models_weights_gguf test_weights_gguf.cpp)
add_executable(test_models_weights_quant test_weights_quant.cpp)
add_executable(test_models_convert_stelltier test_convert.cpp)
add_executable(test_models_convert_streaming test_convert_streaming.cpp)
add_executable(test_models_golden_logits golden/golden_logits.cpp)
add_executable(test_models_golden_fixtures test_golden_fixtures.cpp)
add_executable(test_models_gguf_description test_gguf_description.cpp)
add_executable(test_models_gguf_mixtral test_gguf_mixtral.cpp)
add_executable(test_models_tokenizer test_tokenizer.cpp)

foreach(t
    test_models_json
    test_models_description
    test_models_registry
    test_models_weights_safetensors
    test_models_weights_gguf
    test_models_weights_quant
    test_models_convert_stelltier
    test_models_convert_streaming
    test_models_golden_logits
    test_models_golden_fixtures
    test_models_gguf_description
    test_models_gguf_mixtral
    test_models_tokenizer)
    target_include_directories(${t} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/models/include
        ${_models_src_dir}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/golden
    )
    target_link_libraries(${t} PRIVATE stell_models)
    target_compile_definitions(${t} PRIVATE
        STELL_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
        STELL_GOLDEN_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/golden"
    )
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${t} PRIVATE -Wall -Wextra)
    endif()
endforeach()

# Потоковая конвертация (round 3, R7): тест линкует engine — интерпретатор
# ops-графа (ADR-09 §4: логиты по физическому пути stell-tier == референсу
# того же плана). Мост TU-изолирован в engine (plain_spec.hpp без типов tiers).
target_link_libraries(test_models_convert_streaming PRIVATE stell_engine)

add_test(NAME models_json COMMAND test_models_json)
add_test(NAME models_description COMMAND test_models_description)
add_test(NAME models_registry COMMAND test_models_registry)
add_test(NAME models_weights_safetensors COMMAND test_models_weights_safetensors)
add_test(NAME models_weights_gguf COMMAND test_models_weights_gguf)
add_test(NAME models_weights_quant COMMAND test_models_weights_quant)
add_test(NAME models_convert_stelltier COMMAND test_models_convert_stelltier)
add_test(NAME models_convert_streaming COMMAND test_models_convert_streaming)
add_test(NAME models_golden_logits COMMAND test_models_golden_logits)
add_test(NAME models_golden_fixtures COMMAND test_models_golden_fixtures)
add_test(NAME models_gguf_description COMMAND test_models_gguf_description)
add_test(NAME models_gguf_mixtral COMMAND test_models_gguf_mixtral)
add_test(NAME models_tokenizer COMMAND test_models_tokenizer)

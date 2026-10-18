# parallel: комм-паттерны TP/EP/PP + KV-streaming (контракт ADR-10)
add_library(stell_parallel
    src/comm.cpp
    src/emulated_communicator.cpp
    src/plan.cpp
)

target_include_directories(stell_parallel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(stell_parallel PUBLIC cxx_std_20)

find_package(Threads REQUIRED)
target_link_libraries(stell_parallel PUBLIC Threads::Threads)

# NCCL-backend (CUDA/NVLink): только при STELL_CUDA=ON и наличии libnccl.
# Локальная CPU-конфигурация (STELL_CUDA=OFF) его не собирает (ADR-07, ADR-10 §4).
if(STELL_CUDA)
    set(STELL_NCCL_INCLUDE "" CACHE PATH "каталог с nccl.h (напр. /usr/include)")
    if(STELL_NCCL_INCLUDE STREQUAL "")
        if(EXISTS "/usr/include/nccl.h")
            set(STELL_NCCL_INCLUDE "/usr/include")
        elseif(EXISTS "/usr/local/cuda/include/nccl.h")
            set(STELL_NCCL_INCLUDE "/usr/local/cuda/include")
        endif()
    endif()
    if(STELL_NCCL_INCLUDE STREQUAL "")
        message(FATAL_ERROR "STELL_CUDA=ON: nccl.h не найден; задайте -DSTELL_NCCL_INCLUDE=...")
    endif()
    target_include_directories(stell_parallel PRIVATE SYSTEM "${STELL_NCCL_INCLUDE}")
    target_sources(stell_parallel PRIVATE src/nccl_communicator.cpp)
    target_compile_definitions(stell_parallel PRIVATE STELL_HAVE_NCCL=1)
    find_library(STELL_NCCL_LIB nccl PATHS "/usr/lib" "/usr/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/local/lib" "/usr/local/cuda/lib64")
    if(NOT STELL_NCCL_LIB)
        message(FATAL_ERROR "STELL_CUDA=ON: libnccl.so не найдена")
    endif()
    target_link_libraries(stell_parallel PUBLIC cuda ${STELL_NCCL_LIB})
endif()

add_library(stell::parallel ALIAS stell_parallel)

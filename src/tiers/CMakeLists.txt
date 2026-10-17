# tiers: Tier Manager (ADR-08) + stell-tier layout (ADR-12)
add_library(stell_tiers STATIC
  src/io_uring.cpp
  src/layout.cpp
  src/nvme_file.cpp
  src/manager.cpp
  src/policy.cpp
  src/predictor.cpp
)
target_include_directories(stell_tiers
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(stell_tiers PRIVATE -Wall -Wextra)
# io_uring — прямой syscall-интерфейс ядра Linux; внешних зависимостей нет.
if(STELL_CUDA)
  # CUDA/DMA-путь (device-слоты, cudaMemcpyAsync) — серверная сборка SM70.
  # Дефиниция СТЕЛОВА (PUBLIC): любой TU, включающий tier_manager.hpp, обязан
  # видеть ту же раскладку класса (иначе ODR/ABI-рассинхронизация → OOB).
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(stell_tiers PUBLIC CUDA::cudart)
  target_compile_definitions(stell_tiers PUBLIC STELL_CUDA=1)
endif()

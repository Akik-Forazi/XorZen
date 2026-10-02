# ============================================================
#  XorzenCpp - LibTorch C++ Implementation
#  FRAZIYM TECH & AI / Akik Faraji
# ============================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(XorzenCpp VERSION 0.2.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(XORZEN_ENABLE_IPO "Enable interprocedural optimization/LTO for release builds" ON)
option(XORZEN_ENABLE_AVX2 "Enable AVX2 code generation on MSVC builds" ON)
option(XORZEN_ENABLE_AVX512 "Enable AVX-512 code generation" OFF)
option(XORZEN_USE_GGML "Enable GGML optimizations (quantization, SIMD kernels)" ON)
option(XORZEN_ENABLE_MMAP "Enable memory-mapped expert loading" ON)
option(XORZEN_ENABLE_FLASH_ATTN "Enable Flash Attention optimizations" ON)

# Download LibTorch from https://pytorch.org/get-started/locally/
# and configure with -DCMAKE_PREFIX_PATH=<path-to-libtorch>.
find_package(Torch REQUIRED)
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  add_compile_options(/MP /bigobj)
  if (XORZEN_ENABLE_AVX2)
    add_compile_options(/arch:AVX2)
  endif ()
  if (XORZEN_ENABLE_AVX512)
    add_compile_options(/arch:AVX512)
  endif ()
else ()
  # GCC/Clang optimizations
  add_compile_options(-march=native -mtune=native)
  if (XORZEN_ENABLE_AVX2)
    add_compile_options(-mavx2 -mfma -mf16c)
  endif ()
  if (XORZEN_ENABLE_AVX512)
    add_compile_options(-mavx512f -mavx512dq -mavx512bw -mavx512vl)
  endif ()
endif ()
if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Python wheels for Torch on Windows are MSVC-built and their CMake package
  # may append MSVC-only flags. Strip those so MinGW can at least compile the
  # source; for final linking, prefer an ABI-matched LibTorch toolchain.
  foreach(flag "/permissive-" "/EHsc" "/bigobj")
    string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  endforeach()
  set(CMAKE_CXX_FLAGS "")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern)

# ============================================================
# GGML LIBRARY (Quantization + SIMD Kernels)
# ============================================================
if (XORZEN_USE_GGML)
  message(STATUS "Building with GGML optimizations")
  
  # Note: Copy GGML files from llama.cpp to extern/ggml/ first
  # See OPTIMIZATION_INTEGRATION_PLAN.md for instructions
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/ggml/ggml.c")
    set(GGML_SOURCES
      extern/ggml/ggml.c
      extern/ggml/ggml-quants.c
      extern/ggml/ggml-alloc.c
      extern/ggml/ggml-backend.cpp
    )
    
    # Platform-specific sources
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/ggml/ggml-cpu.c")
      list(APPEND GGML_SOURCES extern/ggml/ggml-cpu.c)
    endif ()
    
    add_library(ggml STATIC ${GGML_SOURCES})
    target_include_directories(ggml PUBLIC extern/ggml)
    
    # GGML compiler flags
    if (MSVC)
      target_compile_options(ggml PRIVATE /W3)
      if (XORZEN_ENABLE_AVX2)
        target_compile_options(ggml PRIVATE /arch:AVX2)
      endif ()
    else ()
      target_compile_options(ggml PRIVATE -march=native -mtune=native)
      if (XORZEN_ENABLE_AVX2)
        target_compile_options(ggml PRIVATE -mavx2 -mfma -mf16c)
      endif ()
    endif ()
    
    set(XORZEN_HAS_GGML TRUE)
  else ()
    message(WARNING "GGML source files not found in extern/ggml/. Disabling GGML optimizations.")
    message(WARNING "To enable GGML, copy files from llama.cpp/ggml/src/ to extern/ggml/")
    set(XORZEN_HAS_GGML FALSE)
  endif ()
else ()
  set(XORZEN_HAS_GGML FALSE)
endif ()

# ============================================================
# OPTIMIZED KERNELS (SIMD, Flash Attention, etc.)
# ============================================================
set(XORZEN_OPTIMIZED_SOURCES
  src/optimized/simd_ops.cpp
  src/optimized/flash_attn.cpp
  src/optimized/expert_mmap.cpp
  src/optimized/thread_pool.cpp
)

if (XORZEN_HAS_GGML)
  list(APPEND XORZEN_OPTIMIZED_SOURCES
    src/optimized/ggml_bridge.cpp
    src/optimized/quantize.cpp
  )
endif ()

add_library(xorzen_optimized STATIC ${XORZEN_OPTIMIZED_SOURCES})
target_include_directories(xorzen_optimized PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(xorzen_optimized PUBLIC "${TORCH_LIBRARIES}")

if (XORZEN_HAS_GGML)
  target_link_libraries(xorzen_optimized PUBLIC ggml)
  target_compile_definitions(xorzen_optimized PUBLIC XORZEN_USE_GGML)
endif ()

if (XORZEN_ENABLE_MMAP)
  target_compile_definitions(xorzen_optimized PUBLIC XORZEN_ENABLE_MMAP)
endif ()

if (XORZEN_ENABLE_FLASH_ATTN)
  target_compile_definitions(xorzen_optimized PUBLIC XORZEN_ENABLE_FLASH_ATTN)
endif ()

set(XORZEN_CORE_SOURCES
    # Core utilities
    src/core/tensor_utils.cpp
    # Model components
    src/model/routing.cpp
    src/model/hass_block.cpp
    src/model/expert.cpp
    src/model/cot_vector.cpp
    src/model/ssm.cpp
    src/model/merger.cpp
    src/model/zmoe.cpp
    src/model/coherence_field.cpp
    src/model/igris.cpp
    src/model/variants.cpp
    src/model/xorzen_model.cpp
    # Tokenizer
    src/tokenizer/bebpe_tokenizer.cpp
    # Data pipeline
    src/data/text_dataset.cpp
    src/data/data_converter.cpp
    src/data/mmap_dataset.cpp
    # Math utilities
    src/utils/math_utils.cpp
    src/utils/logger.cpp
    # Training
    src/training/checkpoint.cpp
    src/training/trainer.cpp
    # Storage (.xorzen format + infinite context KV)
    src/storage/project.cpp
    src/storage/kv_cache.cpp
    # AniXO Intelligent Architecture
    src/model/anime/anime_model.cpp
    src/model/anime/chunk_memory.cpp
    src/model/anime/multimodal.cpp
)

add_library(xorzen_core STATIC ${XORZEN_CORE_SOURCES})
target_link_libraries(xorzen_core PUBLIC "${TORCH_LIBRARIES}" xorzen_optimized)
set_target_properties(xorzen_core PROPERTIES CXX_STANDARD 20)

add_library(xorzen_backend SHARED src/api.cpp)
target_compile_definitions(xorzen_backend PRIVATE XORZEN_BUILD_DLL)
target_link_libraries(xorzen_backend PRIVATE xorzen_core "${TORCH_LIBRARIES}")
set_target_properties(xorzen_backend PROPERTIES
    OUTPUT_NAME "XorZen_BETA-v-0.2.5_win64_backend"
    CXX_STANDARD 20)

add_executable(xorzen_train src/main.cpp)
target_link_libraries(xorzen_train PRIVATE xorzen_core "${TORCH_LIBRARIES}")
set_property(TARGET xorzen_train PROPERTY CXX_STANDARD 20)

add_executable(xorzen_infer src/infer.cpp)
target_link_libraries(xorzen_infer PRIVATE xorzen_core "${TORCH_LIBRARIES}")
set_property(TARGET xorzen_infer PROPERTY CXX_STANDARD 20)

add_executable(xorzen_tokenizer_smoke src/tokenizer_smoke.cpp)
target_link_libraries(xorzen_tokenizer_smoke PRIVATE xorzen_core "${TORCH_LIBRARIES}")
set_property(TARGET xorzen_tokenizer_smoke PROPERTY CXX_STANDARD 20)

add_executable(xorzen_bench src/bench.cpp)
target_link_libraries(xorzen_bench PRIVATE xorzen_core "${TORCH_LIBRARIES}")
set_property(TARGET xorzen_bench PROPERTY CXX_STANDARD 20)

add_dependencies(xorzen_tokenizer_smoke xorzen_train xorzen_infer xorzen_bench)

if (XORZEN_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT xorzen_ipo_supported OUTPUT xorzen_ipo_error)
  if (xorzen_ipo_supported)
    set_property(TARGET xorzen_core xorzen_backend xorzen_train xorzen_infer xorzen_tokenizer_smoke xorzen_bench
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  else ()
    message(WARNING "IPO/LTO requested but not supported: ${xorzen_ipo_error}")
  endif ()
endif ()

include(CTest)
if (BUILD_TESTING)
  enable_testing()
  find_path(CATCH2_INCLUDE_DIR catch2/catch.hpp)
  if (CATCH2_INCLUDE_DIR)
    add_executable(unit_tests tests/test_logger.cpp)
    target_include_directories(unit_tests PRIVATE ${CATCH2_INCLUDE_DIR} include)
    target_link_libraries(unit_tests PRIVATE xorzen_core)
    add_test(NAME unit_tests COMMAND unit_tests)
  endif()

  add_test(NAME xorzen_train_smoke COMMAND xorzen_train)
  add_test(NAME xorzen_infer_smoke COMMAND xorzen_infer)
  add_test(NAME xorzen_bebpe_tokenizer_smoke COMMAND xorzen_tokenizer_smoke)
  add_test(NAME xorzen_micro_bench COMMAND xorzen_bench 2 1 8)
  set_tests_properties(xorzen_train_smoke xorzen_infer_smoke xorzen_bebpe_tokenizer_smoke
    xorzen_micro_bench
    PROPERTIES WORKING_DIRECTORY $<TARGET_FILE_DIR:xorzen_train>)
endif ()

install(TARGETS xorzen_backend xorzen_train xorzen_infer xorzen_tokenizer_smoke xorzen_bench
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
# ---------------------------------------------------------------------------
# C++ test suite
# ---------------------------------------------------------------------------

enable_testing()

add_executable(xorzen_cpp_tests
  tests/cpp_tests.cpp
)

target_link_libraries(xorzen_cpp_tests PRIVATE
  xorzen_core
  xorzen_optimized
  ${TORCH_LIBRARIES}
)

target_include_directories(xorzen_cpp_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)

add_executable(test_parity_rmsnorm
  tests/cpp/test_parity_rmsnorm.cpp
)

target_link_libraries(test_parity_rmsnorm PRIVATE
  xorzen_core
  xorzen_optimized
  ${TORCH_LIBRARIES}
)

target_include_directories(test_parity_rmsnorm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)

add_executable(test_parity_router
  tests/cpp/test_parity_router.cpp
)

target_link_libraries(test_parity_router PRIVATE
  xorzen_core
  xorzen_optimized
  ${TORCH_LIBRARIES}
)

target_include_directories(test_parity_router PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)

add_test(NAME test_parity_rmsnorm COMMAND test_parity_rmsnorm)
add_executable(test_mini_parity
  tests/cpp/test_mini_parity.cpp
)

target_link_libraries(test_mini_parity PRIVATE
  xorzen_core
  xorzen_optimized
  ${TORCH_LIBRARIES}
)

target_include_directories(test_mini_parity PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)

add_test(NAME test_mini_parity COMMAND test_mini_parity)

add_dependencies(xorzen_cpp_tests xorzen_tokenizer_smoke)
if (WIN32)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")

  function(xorzen_copy_torch_dlls target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TORCH_DLLS}
      $<TARGET_FILE_DIR:${target_name}>)
  endfunction()

  xorzen_copy_torch_dlls(xorzen_backend)
  xorzen_copy_torch_dlls(xorzen_train)
  xorzen_copy_torch_dlls(xorzen_infer)
  xorzen_copy_torch_dlls(xorzen_tokenizer_smoke)
  xorzen_copy_torch_dlls(xorzen_bench)
  xorzen_copy_torch_dlls(xorzen_cpp_tests)
endif ()

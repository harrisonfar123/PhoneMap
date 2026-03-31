cmake_minimum_required(VERSION 3.20)
project(slipstream C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ─── Options ───────────────────────────────────────────────────────────────────

option(SLIPSTREAM_METAL    "Enable Metal GPU backend (macOS/iOS)"  OFF)
option(SLIPSTREAM_NEON     "Enable ARM NEON SIMD (auto-detected)"  ON)
option(BUILD_TESTS         "Build unit tests"                       OFF)
option(BUILD_EXAMPLES      "Build example programs"                 OFF)

# ─── Auto-detect platform ─────────────────────────────────────────────────────

if(APPLE)
    set(SLIPSTREAM_METAL ON CACHE BOOL "" FORCE)
    message(STATUS "SlipStream: Apple platform detected — Metal enabled")
endif()

# Check for ARM NEON
include(CheckCSourceCompiles)
check_c_source_compiles("
    #include <arm_neon.h>
    int main() { float32x4_t v = vdupq_n_f32(0.0f); return 0; }
" HAS_NEON)

if(HAS_NEON)
    add_compile_definitions(SS_HAS_NEON=1)
    message(STATUS "SlipStream: ARM NEON detected")
else()
    set(SLIPSTREAM_NEON OFF)
    message(STATUS "SlipStream: ARM NEON not available — CPU fallback only")
endif()

# ─── Core library sources ─────────────────────────────────────────────────────

set(SLIPSTREAM_SOURCES
    src/core/engine.c
    src/core/gguf_loader.c
    src/core/layer_scheduler.c
    src/core/tensor.c
    src/core/quantize.c
    src/core/tokenizer.c
    src/utils/memory.c
    src/utils/thread_pool.c
    src/utils/profiler.c
)

# Add NEON backend
if(SLIPSTREAM_NEON)
    list(APPEND SLIPSTREAM_SOURCES src/backends/cpu_neon.c)
else()
    list(APPEND SLIPSTREAM_SOURCES src/backends/cpu_neon.c)  # fallback scalar
endif()

# Add Metal backend
if(SLIPSTREAM_METAL)
    enable_language(OBJC)
    list(APPEND SLIPSTREAM_SOURCES src/backends/metal_backend.m)
    add_compile_definitions(SS_HAS_METAL=1)
endif()

# ─── Build library ────────────────────────────────────────────────────────────

add_library(slipstream STATIC ${SLIPSTREAM_SOURCES})

target_include_directories(slipstream PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(slipstream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link frameworks
if(SLIPSTREAM_METAL)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(slipstream ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# Math library
target_link_libraries(slipstream m)

# Threading
find_package(Threads REQUIRED)
target_link_libraries(slipstream Threads::Threads)

# Compiler warnings
target_compile_options(slipstream PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -O2
)

# ─── CLI Example ───────────────────────────────────────────────────────────────

if(BUILD_EXAMPLES)
    add_executable(slipstream-cli examples/cli/main.c)
    target_link_libraries(slipstream-cli slipstream)
endif()

# ─── Tests ─────────────────────────────────────────────────────────────────────

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_gguf_loader  tests/test_gguf_loader.c)
    add_executable(test_tensor_ops   tests/test_tensor_ops.c)
    add_executable(test_quantize     tests/test_quantize.c)
    add_executable(test_layer_scheduler tests/test_layer_scheduler.c)

    target_link_libraries(test_gguf_loader slipstream)
    target_link_libraries(test_tensor_ops slipstream)
    target_link_libraries(test_quantize slipstream)
    target_link_libraries(test_layer_scheduler slipstream)

    add_test(NAME gguf_loader     COMMAND test_gguf_loader)
    add_test(NAME tensor_ops      COMMAND test_tensor_ops)
    add_test(NAME quantize        COMMAND test_quantize)
    add_test(NAME layer_scheduler COMMAND test_layer_scheduler)
endif()

# ─── Install ───────────────────────────────────────────────────────────────────

install(TARGETS slipstream
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES include/slipstream.h
    DESTINATION include
)

message(STATUS "")
message(STATUS "SlipStream Configuration:")
message(STATUS "  Metal GPU:   ${SLIPSTREAM_METAL}")
message(STATUS "  ARM NEON:    ${SLIPSTREAM_NEON}")
message(STATUS "  Tests:       ${BUILD_TESTS}")
message(STATUS "  Examples:    ${BUILD_EXAMPLES}")
message(STATUS "")

# tokenizer.cmake - Setup for tokenizers-cpp with vcpkg sentencepiece/msgpack

# Path configuration
set(tokenizersLibPath "external/tokenizers-cpp")
set(TokenizersLibName "TokenizersCpp")

# Save original build type
set(_saved_build_type "${CMAKE_BUILD_TYPE}")

# Always build in Release mode to disable assertions and enable optimizations
set(CMAKE_BUILD_TYPE "Release")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DNDEBUG)

# Disable tests when building as subdirectory
set(BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)

# Stamp file for tracking builds
set(TOKENIZERS_STAMP_FILE "${BUILD_ROOT}/external_builds/${TokenizersLibName}.stamp")

# =============================================================================
# Find vcpkg-provided dependencies
# =============================================================================
find_package(msgpack-cxx CONFIG REQUIRED)

# sentencepiece from vcpkg doesn't provide a CMake config, find it manually
find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)
find_library(SENTENCEPIECE_LIBRARY sentencepiece)

if(NOT SENTENCEPIECE_INCLUDE_DIR OR NOT SENTENCEPIECE_LIBRARY)
    message(FATAL_ERROR "Could not find sentencepiece. Make sure it's installed via vcpkg.")
endif()

# Create an imported target for sentencepiece to match what tokenizers-cpp expects
# tokenizers-cpp links against "sentencepiece-static" target name
if(NOT TARGET sentencepiece-static)
    add_library(sentencepiece-static STATIC IMPORTED)
    set_target_properties(sentencepiece-static PROPERTIES
        IMPORTED_LOCATION "${SENTENCEPIECE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SENTENCEPIECE_INCLUDE_DIR}"
    )
endif()

message(STATUS "Using vcpkg msgpack-cxx")
message(STATUS "Using vcpkg sentencepiece: ${SENTENCEPIECE_LIBRARY}")

# =============================================================================
# Monkey-patch tokenizers-cpp CMakeLists.txt to support external sentencepiece
# Based on unmerged PR: https://github.com/mlc-ai/tokenizers-cpp/pull/85
# =============================================================================
set(TOKENIZERS_CMAKELISTS "${CMAKE_SOURCE_DIR}/${tokenizersLibPath}/CMakeLists.txt")
file(READ "${TOKENIZERS_CMAKELISTS}" TOKENIZERS_CMAKE_CONTENT)

# Check if already patched (look for our marker)
string(FIND "${TOKENIZERS_CMAKE_CONTENT}" "TOKENIZERS_USE_EXTERNAL_SENTENCEPIECE" PATCH_FOUND)

if(PATCH_FOUND EQUAL -1)
    message(STATUS "Monkey-patching tokenizers-cpp CMakeLists.txt for external sentencepiece/msgpack support...")
    
    # Patch 1: Add option for external sentencepiece after MLC_ENABLE_SENTENCEPIECE_TOKENIZER
    string(REPLACE 
        "option(MLC_ENABLE_SENTENCEPIECE_TOKENIZER \"Enable SentencePiece tokenizer\" ON)"
        "option(MLC_ENABLE_SENTENCEPIECE_TOKENIZER \"Enable SentencePiece tokenizer\" ON)\noption(TOKENIZERS_USE_EXTERNAL_SENTENCEPIECE \"Use external sentencepiece target\" OFF)\noption(TOKENIZERS_USE_EXTERNAL_MSGPACK \"Use external msgpack target\" OFF)"
        TOKENIZERS_CMAKE_CONTENT "${TOKENIZERS_CMAKE_CONTENT}")
    
    # Patch 2: Guard the add_subdirectory(sentencepiece) call
    string(REPLACE 
        "add_subdirectory(sentencepiece sentencepiece EXCLUDE_FROM_ALL)"
        "if(NOT TOKENIZERS_USE_EXTERNAL_SENTENCEPIECE)\n  add_subdirectory(sentencepiece sentencepiece EXCLUDE_FROM_ALL)\nendif()"
        TOKENIZERS_CMAKE_CONTENT "${TOKENIZERS_CMAKE_CONTENT}")
    
    # Patch 3: Guard the add_subdirectory(msgpack) call
    string(REPLACE 
        "add_subdirectory(msgpack)"
        "if(NOT TOKENIZERS_USE_EXTERNAL_MSGPACK)\n  add_subdirectory(msgpack)\nendif()"
        TOKENIZERS_CMAKE_CONTENT "${TOKENIZERS_CMAKE_CONTENT}")
    
    # Write patched file
    file(WRITE "${TOKENIZERS_CMAKELISTS}" "${TOKENIZERS_CMAKE_CONTENT}")
    message(STATUS "Monkey-patch applied successfully")
endif()

# =============================================================================
# Configure tokenizers-cpp to use vcpkg dependencies
# =============================================================================

# Tell tokenizers-cpp to use external sentencepiece and msgpack
set(TOKENIZERS_USE_EXTERNAL_SENTENCEPIECE ON CACHE BOOL "Use external sentencepiece" FORCE)
set(TOKENIZERS_USE_EXTERNAL_MSGPACK ON CACHE BOOL "Use external msgpack" FORCE)

# Add tokenizers-cpp (it will skip its own sentencepiece/msgpack due to the patch)
add_subdirectory(${tokenizersLibPath} "${BUILD_ROOT}/external_builds/${TokenizersLibName}")

# Create stamp file after configuration
if(NOT EXISTS ${TOKENIZERS_STAMP_FILE})
    file(WRITE ${TOKENIZERS_STAMP_FILE} "Configured on ${CMAKE_SYSTEM_NAME}")
endif()

# Restore original build type for the main project
set(CMAKE_BUILD_TYPE "${_saved_build_type}")

# CMakeLists.txt - Chatterbox TTS ONNX Demo
cmake_minimum_required(VERSION 3.20)


# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# vcpkg triplet
set(VCPKG_BUILD_TYPE release)  # This skips Debug builds
set(VCPKG_TARGET_TRIPLET "x64-windows-static-release" CACHE STRING "")

project(chatterbox_tts_demo VERSION 1.0.0 LANGUAGES CXX)

# Include custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(BUILD_ROOT "${CMAKE_SOURCE_DIR}/build")
include(CompilerFlags)




# ============================================================================
# Find Dependencies (vcpkg)
# ============================================================================


find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)


include(FindONNXRuntime)
include(FindFFmpeg)

include(ExternalDependencies)
# ============================================================================
# Include Directories
# ============================================================================

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/tts)
include_directories(${CMAKE_SOURCE_DIR}/include/common)
include_directories(${FFMPEG_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})


# ============================================================================
# TTS Demo Executable
# ============================================================================

set(TTS_SOURCES
    # Main entry point
    src/tts/main.cpp
    
    # Phase 2: ONNX Infrastructure
    src/tts/OnnxSessionManager.cpp
    src/tts/ModelDownloader.cpp
    src/tts/TensorUtils.cpp
    
    # Phase 3: Audio Processing
    src/tts/AudioLoader.cpp
    src/tts/WavWriter.cpp
    src/common/audio/FFmpegUtils.cpp
    
    # Phase 4: Tokenizer
    src/tts/Tokenizer.cpp
    
    # Phase 5: Main TTS
    src/tts/ChatterboxTTS.cpp
    
    # Phase 6: Voice Caching
    src/tts/VoiceConditionalsCache.cpp
)

add_executable(${PROJECT_NAME} ${TTS_SOURCES})

# ============================================================================
# Link Libraries
# ============================================================================

target_link_libraries(${PROJECT_NAME} PRIVATE ${FFMPEG_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog)
target_link_libraries(${PROJECT_NAME} PRIVATE CURL::libcurl)
target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp)

# Windows-specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        UNICODE
        _UNICODE
    )
endif()

# ============================================================================
# Output Configuration
# ============================================================================

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)

# Copy assets folder for easy testing
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
    COMMENT "Copying assets to output directory"
)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(DIRECTORY assets/
    DESTINATION bin/assets
    FILES_MATCHING PATTERN "*.wav"
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== Chatterbox TTS Demo Configuration ===")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  ONNX Runtime:     ${ONNXRUNTIME_ROOT_PATH}")
message(STATUS "  Output:           ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}")
message(STATUS "")

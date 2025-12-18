# FindONNXRuntime.cmake
# Finds and configures ONNX Runtime static libraries
#
# Sets the following variables:
#   ONNXRUNTIME_FOUND - TRUE if ONNX Runtime was found
#   ONNXRUNTIME_INCLUDE_DIRS - Include directories for ONNX Runtime
#   ONNXRUNTIME_LIBRARIES - Libraries to link against
#   ONNXRUNTIME_ROOT_PATH - Root path to ONNX Runtime installation
#
# Creates imported target:
#   ONNXRuntime::ONNXRuntime

include(FindPackageHandleStandardArgs)

set(ONNXRUNTIME_VERSION "1.23.2" CACHE STRING "ONNXRUNTIME version")
# Set ONNX Runtime location
set(ONNXRUNTIME_STATIC_BASE "onnxruntime-win-x64-static_lib-${ONNXRUNTIME_VERSION}") 


# ONNX Runtime setup (manually installed)
set(ONNXRUNTIME_ROOT_PATH "${CMAKE_SOURCE_DIR}/external/${ONNXRUNTIME_STATIC_BASE}")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_PATH}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_PATH}/lib")

# Find the main library
find_library(ONNXRUNTIME_LIBRARY 
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_LIB_DIR}"
    NO_DEFAULT_PATH
)

# Use standard CMake package handler
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRUNTIME_LIBRARY ONNXRUNTIME_INCLUDE_DIR
    VERSION_VAR ONNXRUNTIME_VERSION
    FAIL_MESSAGE "ONNX Runtime is required for Phase 1.2 vector memory system. Please extract ${ONNXRUNTIME_STATIC_BASE} to external/"
)

# Only proceed if found
if(ONNXRuntime_FOUND)
    # Set up include directories
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_INCLUDE_DIR}")
    
    # Set up libraries
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_LIBRARY}")
    
    # Add Windows system libraries required by ONNX Runtime
    if(WIN32)
        list(APPEND ONNXRUNTIME_LIBRARIES dxcore.lib)
    endif()
    
    # Create imported target
    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime STATIC IMPORTED)
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${ONNXRUNTIME_LIBRARIES}"
        )
    endif()
endif()

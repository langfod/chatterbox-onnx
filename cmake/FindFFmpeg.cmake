# FindFFmpeg.cmake
# Finds FFmpeg libraries with multiple fallback strategies
#
# Sets the following variables:
#   FFMPEG_FOUND - TRUE if FFmpeg was found
#   FFMPEG_INCLUDE_DIRS - Include directories for FFmpeg
#   FFMPEG_LIBRARIES - FFmpeg libraries to link against

# First, try to use vcpkg's FFmpeg package if available
if(DEFINED VCPKG_TARGET_TRIPLET)
    # Try to find FFmpeg components using vcpkg naming
    find_package(FFMPEG QUIET CONFIG)
    
    # If CONFIG mode succeeded, normalize variables for downstream consumers
    if(FFMPEG_FOUND)
        # Check if targets were created (common for vcpkg CONFIG packages)
        if(TARGET FFMPEG::FFMPEG)
            set(FFMPEG_LIBRARIES FFMPEG::FFMPEG)
            message(STATUS "Found FFmpeg via CONFIG mode (using FFMPEG::FFMPEG target)")
        elseif(TARGET ffmpeg)
            set(FFMPEG_LIBRARIES ffmpeg)
            message(STATUS "Found FFmpeg via CONFIG mode (using ffmpeg target)")
        elseif(DEFINED FFMPEG_LIBRARIES)
            # Variables already set by CONFIG, use them
            message(STATUS "Found FFmpeg via CONFIG mode (using provided variables)")
        else()
            # CONFIG found it but didn't provide usable targets or variables
            message(FATAL_ERROR "FFmpeg CONFIG mode succeeded but no targets or library variables were provided. Please check your FFmpeg installation.")
        endif()
        
        # Try to get include directories if not already set
        if(NOT DEFINED FFMPEG_INCLUDE_DIRS)
            if(TARGET FFMPEG::FFMPEG)
                get_target_property(FFMPEG_INCLUDE_DIRS FFMPEG::FFMPEG INTERFACE_INCLUDE_DIRECTORIES)
            elseif(TARGET ffmpeg)
                get_target_property(FFMPEG_INCLUDE_DIRS ffmpeg INTERFACE_INCLUDE_DIRECTORIES)
            endif()
        endif()
    endif()
endif()

# Skip pkg-config on Windows with paths containing spaces - go directly to find_library
# pkg-config has issues with spaces in paths
if(NOT FFMPEG_FOUND)
    find_library(AVCODEC_LIBRARY avcodec)
    find_library(AVFORMAT_LIBRARY avformat)
    find_library(AVUTIL_LIBRARY avutil)
    find_library(SWRESAMPLE_LIBRARY swresample)
    find_library(SWSCALE_LIBRARY swscale)
    
    find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h)
    
    if(AVCODEC_LIBRARY AND AVFORMAT_LIBRARY AND AVUTIL_LIBRARY AND 
       SWRESAMPLE_LIBRARY AND SWSCALE_LIBRARY AND FFMPEG_INCLUDE_DIR)
        set(FFMPEG_LIBRARIES 
            ${AVCODEC_LIBRARY}
            ${AVFORMAT_LIBRARY}
            ${AVUTIL_LIBRARY}
            ${SWRESAMPLE_LIBRARY}
            ${SWSCALE_LIBRARY}
        )
        # Add Windows system libraries required by FFmpeg
        if(WIN32)
            list(APPEND FFMPEG_LIBRARIES bcrypt mfuuid strmiids)
        endif()
        set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})
        set(FFMPEG_FOUND TRUE)
        message(STATUS "Found FFmpeg libraries directly")
    else()
        message(FATAL_ERROR "FFmpeg libraries not found. Please install FFmpeg development libraries.")
    endif()
endif()

# Create an interface library for FFmpeg-related compile definitions
if(FFMPEG_FOUND AND NOT TARGET FFmpeg::Config)
    add_library(FFmpeg::Config INTERFACE IMPORTED)
    target_compile_definitions(FFmpeg::Config INTERFACE
        CONFIG_XWMA_DECODER=1
        CONFIG_XWMA_DEMUXER=1
        CONFIG_ASF_DEMUXER=1
    )
endif()

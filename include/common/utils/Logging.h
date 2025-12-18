/**
 * @file Logging.h
 * @brief Logging compatibility layer for standalone Chatterbox TTS
 * 
 * Maps SkyrimNet LOG_* macros to spdlog for standalone builds.
 * When integrating into SkyrimNet, replace this with the full Logging.h
 */

#pragma once

#include <spdlog/spdlog.h>

// Map LOG_* macros to spdlog
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#define LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
#define LOG_INFO(...) spdlog::info(__VA_ARGS__)
#define LOG_WARN(...) spdlog::warn(__VA_ARGS__)
#define LOG_ERROR(...) spdlog::error(__VA_ARGS__)
#define LOG_CRITICAL(...) spdlog::critical(__VA_ARGS__)
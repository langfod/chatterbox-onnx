#pragma once
#include <string>

#ifndef TEST_ENVIRONMENT
#define TEST_ENVIRONMENT 0
#endif

/* #if !TEST_ENVIRONMENT
#include <RE/Skyrim.h>
#include <SKSE/SKSE.h>
#else
*/
#include <cstdarg>
#include <cstdio>

// Test environment logging namespace
namespace SKSE {
    namespace log {
        inline void info(const char* fmt, ...) {
            va_list args;
            va_start(args, fmt);
            vprintf(fmt, args);
            printf("\n");
            va_end(args);
        }
    }
}
#endif

namespace logger = SKSE::log;
extern "C++" {
    namespace SkyrimNet {
        // Unique Save ID functionality
        // Get the unique ID for the current save game
        // Will generate a new ID if one doesn't exist
        std::string GetSaveUniqueID();
    }
}
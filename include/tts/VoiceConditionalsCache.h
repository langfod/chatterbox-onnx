/**
 * @file VoiceConditionalsCache.h
 * @brief In-memory and disk cache for Chatterbox voice conditionals
 * 
 * Provides caching of pre-computed voice conditionals to avoid
 * re-running the speech encoder for frequently used voices.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <optional>
#include <vector>
#include <mutex>
#include <filesystem>

#include "tts/ChatterboxTTS.h"

namespace ChatterboxTTS {

/**
 * @brief Cache for voice conditionals with disk persistence
 * 
 * Thread-safety notes for future integration:
 * - Get/Has are read operations (need shared lock)
 * - Put/Remove/Clear are write operations (need exclusive lock)
 * - SaveToDisk is designed to work with data copies for async I/O
 */
class VoiceConditionalsCache {
public:
    /**
     * @brief Construct cache with specified directory
     * @param cacheDir Directory for .cond files (default: "cache")
     */
    explicit VoiceConditionalsCache(const std::string& cacheDir = "cache");
    ~VoiceConditionalsCache() = default;
    
    // Non-copyable, moveable
    VoiceConditionalsCache(const VoiceConditionalsCache&) = delete;
    VoiceConditionalsCache& operator=(const VoiceConditionalsCache&) = delete;
    VoiceConditionalsCache(VoiceConditionalsCache&&) = default;
    VoiceConditionalsCache& operator=(VoiceConditionalsCache&&) = default;
    
    /**
     * @brief Check if a voice is in the memory cache
     */
    bool Has(const std::string& key) const;
    
    /**
     * @brief Check if a voice exists on disk (not necessarily loaded)
     */
    bool ExistsOnDisk(const std::string& key) const;
    
    /**
     * @brief Get voice conditionals from cache
     * @param key Voice identifier (e.g., "malebrute")
     * @return Pointer to conditionals or nullptr if not found
     */
    const VoiceConditionals* Get(const std::string& key) const;
    
    /**
     * @brief Add voice conditionals to cache
     * @param key Voice identifier
     * @param conds Voice conditionals to cache
     * @param saveToDisk Whether to persist to disk (default: true)
     * @return true on success
     */
    bool Put(const std::string& key, const VoiceConditionals& conds, bool saveToDisk = true);
    
    /**
     * @brief Load a single voice from disk into memory cache
     * @param key Voice identifier
     * @return true if loaded successfully
     */
    bool LoadFromDisk(const std::string& key);
    
    /**
     * @brief Load all .cond files from cache directory into memory
     * @return Number of files loaded
     */
    size_t LoadAllFromDisk();
    
    /**
     * @brief Save conditionals to disk (thread-safe design - takes data by value)
     * 
     * This function is designed for async I/O:
     * - Takes copies of data, not references
     * - Does not access any shared state
     * - Can be safely called from a worker thread
     * 
     * @param key Voice identifier
     * @param conds Voice conditionals (passed by value for thread safety)
     * @return true on success
     */
    bool SaveToDisk(const std::string& key, VoiceConditionals conds) const;
    
    /**
     * @brief Remove a voice from memory and disk cache
     * @param key Voice identifier
     * @return true if removed (false if not found)
     */
    bool Remove(const std::string& key);
    
    /**
     * @brief Clear entire cache (memory and disk)
     */
    void Clear();
    
    /**
     * @brief Clear only memory cache (keep disk files)
     */
    void ClearMemory();
    
    /**
     * @brief Get number of voices in memory cache
     */
    size_t Size() const { return m_cache.size(); }
    
    /**
     * @brief Get list of all cached voice keys
     */
    std::vector<std::string> GetKeys() const;
    
    /**
     * @brief Get the cache directory path
     */
    const std::string& GetCacheDir() const { return m_cacheDir; }
    
    /**
     * @brief Extract voice key from file path
     * 
     * Examples:
     *   "assets/malebrute.wav" -> "malebrute"
     *   "malebrute" -> "malebrute"
     *   "C:/voices/female_elf.wav" -> "female_elf"
     */
    static std::string ExtractKey(const std::string& pathOrKey);
    
    /**
     * @brief Get disk path for a cache key
     */
    std::string GetCachePath(const std::string& key) const;
    
private:
    std::string m_cacheDir;
    std::unordered_map<std::string, VoiceConditionals> m_cache;
    
    // Placeholder for future thread safety
    // mutable std::shared_mutex m_mutex;
    
    /**
     * @brief Ensure cache directory exists
     */
    bool EnsureCacheDir() const;
};

} // namespace ChatterboxTTS

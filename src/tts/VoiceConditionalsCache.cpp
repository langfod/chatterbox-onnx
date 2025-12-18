/**
 * @file VoiceConditionalsCache.cpp
 * @brief Implementation of voice conditionals caching
 */

#include "tts/VoiceConditionalsCache.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace fs = std::filesystem;

namespace ChatterboxTTS {

VoiceConditionalsCache::VoiceConditionalsCache(const std::string& cacheDir)
    : m_cacheDir(cacheDir)
{
}

bool VoiceConditionalsCache::Has(const std::string& key) const {
    return m_cache.find(key) != m_cache.end();
}

bool VoiceConditionalsCache::ExistsOnDisk(const std::string& key) const {
    return fs::exists(GetCachePath(key));
}

const VoiceConditionals* VoiceConditionalsCache::Get(const std::string& key) const {
    auto it = m_cache.find(key);
    if (it != m_cache.end()) {
        return &it->second;
    }
    return nullptr;
}

bool VoiceConditionalsCache::Put(const std::string& key, const VoiceConditionals& conds, bool saveToDisk) {
    if (!conds.IsValid()) {
        spdlog::warn("VoiceConditionalsCache::Put - invalid conditionals for key '{}'", key);
        return false;
    }
    
    m_cache[key] = conds;
    spdlog::info("Cached voice conditionals: '{}'", key);
    
    if (saveToDisk) {
        // Note: In threaded version, this would queue an async save
        // For now, save synchronously with a copy
        return SaveToDisk(key, conds);
    }
    
    return true;
}

bool VoiceConditionalsCache::LoadFromDisk(const std::string& key) {
    std::string path = GetCachePath(key);
    
    auto conds = VoiceConditionals::Load(path);
    if (!conds) {
        spdlog::warn("Failed to load cache file: {}", path);
        return false;
    }
    
    m_cache[key] = std::move(*conds);
    spdlog::info("Loaded voice conditionals from disk: '{}'", key);
    return true;
}

size_t VoiceConditionalsCache::LoadAllFromDisk() {
    if (!fs::exists(m_cacheDir)) {
        spdlog::info("Cache directory does not exist: {}", m_cacheDir);
        return 0;
    }
    
    size_t loaded = 0;
    
    for (const auto& entry : fs::directory_iterator(m_cacheDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cond") {
            std::string key = entry.path().stem().string();
            
            if (LoadFromDisk(key)) {
                loaded++;
            }
        }
    }
    
    spdlog::info("Loaded {} voice conditionals from cache directory", loaded);
    return loaded;
}

bool VoiceConditionalsCache::SaveToDisk(const std::string& key, VoiceConditionals conds) const {
    // This function is designed for async I/O:
    // - 'conds' is passed by value (copy) so it's safe to use from another thread
    // - Only uses const member m_cacheDir
    // - No shared state is accessed
    
    if (!EnsureCacheDir()) {
        return false;
    }
    
    std::string path = GetCachePath(key);
    
    if (!conds.Save(path)) {
        spdlog::error("Failed to save cache file: {}", path);
        return false;
    }
    
    spdlog::info("Saved voice conditionals to disk: {}", path);
    return true;
}

bool VoiceConditionalsCache::Remove(const std::string& key) {
    bool removedFromMemory = false;
    bool removedFromDisk = false;
    
    // Remove from memory
    auto it = m_cache.find(key);
    if (it != m_cache.end()) {
        m_cache.erase(it);
        removedFromMemory = true;
        spdlog::info("Removed '{}' from memory cache", key);
    }
    
    // Remove from disk
    std::string path = GetCachePath(key);
    if (fs::exists(path)) {
        std::error_code ec;
        if (fs::remove(path, ec)) {
            removedFromDisk = true;
            spdlog::info("Removed cache file: {}", path);
        } else {
            spdlog::warn("Failed to remove cache file {}: {}", path, ec.message());
        }
    }
    
    return removedFromMemory || removedFromDisk;
}

void VoiceConditionalsCache::Clear() {
    // Clear memory
    ClearMemory();
    
    // Clear disk
    if (fs::exists(m_cacheDir)) {
        size_t removed = 0;
        std::error_code ec;
        
        for (const auto& entry : fs::directory_iterator(m_cacheDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".cond") {
                if (fs::remove(entry.path(), ec)) {
                    removed++;
                } else {
                    spdlog::warn("Failed to remove {}: {}", entry.path().string(), ec.message());
                }
            }
        }
        
        spdlog::info("Cleared {} cache files from disk", removed);
    }
}

void VoiceConditionalsCache::ClearMemory() {
    size_t count = m_cache.size();
    m_cache.clear();
    spdlog::info("Cleared {} entries from memory cache", count);
}

std::vector<std::string> VoiceConditionalsCache::GetKeys() const {
    std::vector<std::string> keys;
    keys.reserve(m_cache.size());
    
    for (const auto& [key, _] : m_cache) {
        keys.push_back(key);
    }
    
    return keys;
}

std::string VoiceConditionalsCache::ExtractKey(const std::string& pathOrKey) {
    // Use filesystem to extract stem (filename without extension)
    fs::path p(pathOrKey);
    
    // If it has an extension, get the stem
    if (p.has_extension()) {
        return p.stem().string();
    }
    
    // If it has a parent path but no extension, get the filename
    if (p.has_parent_path()) {
        return p.filename().string();
    }
    
    // Otherwise, it's already just a key
    return pathOrKey;
}

std::string VoiceConditionalsCache::GetCachePath(const std::string& key) const {
    return (fs::path(m_cacheDir) / (key + ".cond")).string();
}

bool VoiceConditionalsCache::EnsureCacheDir() const {
    if (fs::exists(m_cacheDir)) {
        return true;
    }
    
    std::error_code ec;
    if (!fs::create_directories(m_cacheDir, ec)) {
        spdlog::error("Failed to create cache directory {}: {}", m_cacheDir, ec.message());
        return false;
    }
    
    return true;
}

} // namespace ChatterboxTTS

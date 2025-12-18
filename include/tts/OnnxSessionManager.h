/**
 * @file OnnxSessionManager.h
 * @brief ONNX Runtime session manager for loading and managing multiple models
 * 
 * This class handles the lifecycle of ONNX Runtime sessions, including:
 * - Environment initialization
 * - Model loading with external data file support
 * - Execution provider configuration (CPU/CUDA)
 * - Session retrieval by name
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ChatterboxTTS {

/**
 * @brief Supported execution providers
 */
enum class ExecutionProvider {
    CPU,        ///< CPU execution (always available)
    CUDA,       ///< NVIDIA CUDA (requires onnxruntime-gpu)
    AUTO        ///< Auto-detect best available provider
};

/**
 * @brief ONNX Runtime session manager
 * 
 * Manages multiple ONNX model sessions with a shared environment.
 * Thread-safe for reading sessions after loading.
 */
class OnnxSessionManager {
public:
    /**
     * @brief Construct session manager with specified provider
     * @param provider Execution provider to use (default: AUTO)
     */
    explicit OnnxSessionManager(ExecutionProvider provider = ExecutionProvider::AUTO);
    
    /**
     * @brief Destructor - cleans up all sessions
     */
    ~OnnxSessionManager();

    // Non-copyable, movable
    OnnxSessionManager(const OnnxSessionManager&) = delete;
    OnnxSessionManager& operator=(const OnnxSessionManager&) = delete;
    OnnxSessionManager(OnnxSessionManager&&) = default;
    OnnxSessionManager& operator=(OnnxSessionManager&&) = default;

    /**
     * @brief Load an ONNX model from file
     * @param modelPath Path to .onnx file (handles .onnx_data automatically)
     * @param name Identifier for retrieving session later
     * @return true if loaded successfully
     * @throws std::runtime_error on load failure
     */
    bool LoadModel(const std::string& modelPath, const std::string& name);

    /**
     * @brief Get a loaded session by name
     * @param name Session identifier used in LoadModel
     * @return Pointer to session, or nullptr if not found
     */
    Ort::Session* GetSession(const std::string& name);

    /**
     * @brief Check if a model is loaded
     * @param name Session identifier
     * @return true if session exists
     */
    bool IsModelLoaded(const std::string& name) const;

    /**
     * @brief Get list of available execution providers
     * @return Vector of provider names (e.g., "CPUExecutionProvider")
     */
    std::vector<std::string> GetAvailableProviders() const;

    /**
     * @brief Get the active execution provider name
     * @return Provider name string
     */
    std::string GetActiveProvider() const;

    /**
     * @brief Get memory info for tensor allocation
     * @return Ort::MemoryInfo for CPU allocations
     */
    Ort::MemoryInfo& GetMemoryInfo();

    /**
     * @brief Get the ONNX Runtime allocator
     * @return Reference to allocator
     */
    Ort::AllocatorWithDefaultOptions& GetAllocator();

    /**
     * @brief Get pre-configured RunOptions for inference
     * @return Reference to RunOptions (reuse to avoid allocation overhead)
     */
    Ort::RunOptions& GetRunOptions();

    /**
     * @brief Enable profiling for all subsequent Run() calls
     * @param profilePrefix File prefix for profile output (e.g., "onnx_profile")
     * @note Must be called BEFORE loading models to take effect
     */
    void EnableProfiling(const std::wstring& profilePrefix = L"onnx_profile");

    /**
     * @brief Disable profiling
     * @note Must be called BEFORE loading models to take effect
     */
    void DisableProfiling();

    /**
     * @brief End profiling and write results to file
     * @return Path to the generated profile file, or empty string if profiling wasn't enabled
     * @note Call this to flush profiling data after the code you want to profile
     */
    std::string EndProfiling();

    /**
     * @brief Check if profiling is enabled
     */
    bool IsProfilingEnabled() const { return m_profilingEnabled; }

private:
    /**
     * @brief Configure session options based on provider
     */
    void ConfigureSessionOptions();

    /**
     * @brief Detect best available execution provider
     * @return Detected provider type
     */
    ExecutionProvider DetectBestProvider() const;

    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
    std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;
    Ort::AllocatorWithDefaultOptions m_allocator;
    
    std::unordered_map<std::string, std::unique_ptr<Ort::Session>> m_sessions;
    Ort::RunOptions m_runOptions;
    
    ExecutionProvider m_provider;
    std::string m_activeProviderName;
    bool m_profilingEnabled = false;
};

} // namespace ChatterboxTTS

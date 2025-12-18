/**
 * @file OnnxSessionManager.cpp
 * @brief Implementation of ONNX Runtime session manager
 */

#include "tts/OnnxSessionManager.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <thread>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace fs = std::filesystem;

namespace ChatterboxTTS {

OnnxSessionManager::OnnxSessionManager(ExecutionProvider provider)
    : m_provider(provider)
{
    // Initialize ONNX Runtime environment
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ChatterboxTTS");
    
    // Create session options
    m_sessionOptions = std::make_unique<Ort::SessionOptions>();
    
    // Create memory info for CPU allocations (use arena allocator for better performance)
    m_memoryInfo = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
    
    // Configure RunOptions for reuse (avoids allocation per Run call)
    // Default options are fine, but having a reusable instance helps
    m_runOptions = Ort::RunOptions();
    
    // Auto-detect provider if requested
    if (m_provider == ExecutionProvider::AUTO) {
        m_provider = DetectBestProvider();
    }
    
    // Configure session options
    ConfigureSessionOptions();
    
    std::cout << "[OnnxSessionManager] Initialized with provider: " << m_activeProviderName << std::endl;
}

OnnxSessionManager::~OnnxSessionManager() {
    // Sessions must be cleared before env is destroyed
    m_sessions.clear();
}

void OnnxSessionManager::ConfigureSessionOptions() {

    // Note: Profiling is NOT enabled by default. Call EnableProfiling() before LoadModel() if needed.

    // Set graph optimization level
    m_sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // CPU Threading Configuration:
    // IntraOp: parallelism within a single operator (e.g., matrix multiply)
    // InterOp: parallelism between independent operators in the graph
    // 0 = use all available cores
    int numThreads = std::thread::hardware_concurrency() / 4; 
    if (numThreads <= 1) numThreads = 2; 
    std::cout << "[OnnxSessionManager] Using " << numThreads << " threads for CPU execution" << std::endl;

    m_sessionOptions->SetIntraOpNumThreads(numThreads);
    m_sessionOptions->SetInterOpNumThreads(numThreads);
    
    // Enable parallel execution of independent nodes in the graph
    m_sessionOptions->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    
    // Enable memory pattern optimization - reuses memory allocations across runs
    // This significantly reduces allocation overhead for repeated inference
    m_sessionOptions->EnableMemPattern();
    
    // Enable CPU memory arena - pools memory to reduce system allocation calls
    m_sessionOptions->EnableCpuMemArena();
    
    // Configure execution provider
    switch (m_provider) {
        case ExecutionProvider::CUDA: {
            #ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            m_sessionOptions->AppendExecutionProvider_CUDA(cuda_options);
            m_activeProviderName = "CUDAExecutionProvider";
            #else
            std::cout << "[OnnxSessionManager] CUDA requested but not available, falling back to CPU" << std::endl;
            m_activeProviderName = "CPUExecutionProvider";
            m_provider = ExecutionProvider::CPU;
            #endif
            break;
        }
        case ExecutionProvider::CPU:
        default:
            m_activeProviderName = "CPUExecutionProvider";
            break;
    }
}

ExecutionProvider OnnxSessionManager::DetectBestProvider() const {
    try {
        auto providers = Ort::GetAvailableProviders();
        for (const auto& provider : providers) {
            if (provider == "CUDAExecutionProvider") {
                std::cout << "[OnnxSessionManager] Auto-detected CUDA provider" << std::endl;
                return ExecutionProvider::CUDA;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OnnxSessionManager] Provider detection failed: " << e.what() << std::endl;
    }
    
    std::cout << "[OnnxSessionManager] Using CPU provider" << std::endl;
    return ExecutionProvider::CPU;
}

bool OnnxSessionManager::LoadModel(const std::string& modelPath, const std::string& name) {
    // Check if already loaded
    if (IsModelLoaded(name)) {
        std::cout << "[OnnxSessionManager] Model '" << name << "' already loaded" << std::endl;
        return true;
    }
    
    // Check file exists
    if (!fs::exists(modelPath)) {
        throw std::runtime_error("Model file not found: " + modelPath);
    }
    
    std::cout << "[OnnxSessionManager] Loading model: " << name << " from " << modelPath << std::endl;
    
    try {
        // Convert path to wide string for Windows compatibility
        #ifdef _WIN32
        std::wstring wpath(modelPath.begin(), modelPath.end());
        // Handle UTF-8 properly
        int wlen = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0);
        if (wlen > 0) {
            wpath.resize(wlen - 1);
            MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wpath.data(), wlen);
        }
        auto session = std::make_unique<Ort::Session>(*m_env, wpath.c_str(), *m_sessionOptions);
        #else
        auto session = std::make_unique<Ort::Session>(*m_env, modelPath.c_str(), *m_sessionOptions);
        #endif
        
        // Log model info
        size_t numInputs = session->GetInputCount();
        size_t numOutputs = session->GetOutputCount();
        std::cout << "[OnnxSessionManager] Loaded '" << name << "': " 
                  << numInputs << " inputs, " << numOutputs << " outputs" << std::endl;
        
        // Store session
        m_sessions[name] = std::move(session);
        return true;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load model '" + name + "': " + e.what());
    }
}

Ort::Session* OnnxSessionManager::GetSession(const std::string& name) {
    auto it = m_sessions.find(name);
    if (it != m_sessions.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool OnnxSessionManager::IsModelLoaded(const std::string& name) const {
    return m_sessions.find(name) != m_sessions.end();
}

std::vector<std::string> OnnxSessionManager::GetAvailableProviders() const {
    return Ort::GetAvailableProviders();
}

std::string OnnxSessionManager::GetActiveProvider() const {
    return m_activeProviderName;
}

Ort::MemoryInfo& OnnxSessionManager::GetMemoryInfo() {
    return *m_memoryInfo;
}

Ort::AllocatorWithDefaultOptions& OnnxSessionManager::GetAllocator() {
    return m_allocator;
}

Ort::RunOptions& OnnxSessionManager::GetRunOptions() {
    return m_runOptions;
}

void OnnxSessionManager::EnableProfiling(const std::wstring& profilePrefix) {
    if (!m_sessions.empty()) {
        std::cerr << "[OnnxSessionManager] Warning: EnableProfiling() called after models loaded. "
                  << "Profiling must be enabled before LoadModel()." << std::endl;
    }
    m_sessionOptions->EnableProfiling(profilePrefix.c_str());
    m_profilingEnabled = true;
    std::cout << "[OnnxSessionManager] Profiling enabled" << std::endl;
}

void OnnxSessionManager::DisableProfiling() {
    m_sessionOptions->DisableProfiling();
    m_profilingEnabled = false;
}

std::string OnnxSessionManager::EndProfiling() {
    if (!m_profilingEnabled) {
        return "";
    }
    
    // EndProfilingAllocated returns the path to the profile file
    // We need to call it on each session
    std::string lastProfilePath;
    for (auto& [name, session] : m_sessions) {
        if (session) {
            auto profilePath = session->EndProfilingAllocated(m_allocator);
            lastProfilePath = profilePath.get();
            std::cout << "[OnnxSessionManager] Profile written for " << name << ": " << lastProfilePath << std::endl;
        }
    }
    
    return lastProfilePath;
}

} // namespace ChatterboxTTS

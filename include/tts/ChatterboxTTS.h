/**
 * @file ChatterboxTTS.h
 * @brief Main Chatterbox TTS inference engine for C++
 * 
 * This is the core TTS class that orchestrates the ONNX models to
 * generate speech from text using voice cloning.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <random>

#include "tts/OnnxSessionManager.h"
#include "tts/AudioLoader.h"
#include "tts/Tokenizer.h"

namespace ChatterboxTTS {

// Constants from the model
constexpr int SAMPLE_RATE = 24000;
constexpr int S3_SR = 16000;           // Speech tokenizer sample rate
constexpr int64_t START_SPEECH_TOKEN = 6561;
constexpr int64_t STOP_SPEECH_TOKEN = 6562;
constexpr int64_t SILENCE_TOKEN = 4299;
constexpr int NUM_KV_HEADS = 16;
constexpr int HEAD_DIM = 64;

/**
 * @brief Voice conditionals computed from reference audio
 */
struct VoiceConditionals {
    std::vector<float> condEmb;           ///< Conditioning embedding [1, seq, hidden]
    std::vector<int64_t> promptToken;     ///< Prompt tokens [1, seq]
    std::vector<float> speakerEmbeddings; ///< Speaker embedding [1, dim]
    std::vector<float> speakerFeatures;   ///< Speaker features [1, seq, dim]
    
    // Shape information
    std::vector<int64_t> condEmbShape;
    std::vector<int64_t> promptTokenShape;
    std::vector<int64_t> speakerEmbeddingsShape;
    std::vector<int64_t> speakerFeaturesShape;
    
    bool IsValid() const { return !condEmb.empty() && !promptToken.empty(); }
    
    /**
     * @brief Save conditionals to .npz-like binary file
     */
    bool Save(const std::string& path) const;
    
    /**
     * @brief Load conditionals from binary file
     */
    static std::optional<VoiceConditionals> Load(const std::string& path);
};

/**
 * @brief Generation parameters for TTS
 */
struct GenerationConfig {
    int maxNewTokens = 1024;         ///< Maximum speech tokens to generate
    float repetitionPenalty = 1.2f;  ///< Penalty for repeated tokens (>1 reduces repetition)
    float temperature = 0.8f;        ///< Sampling temperature (higher = more random)
    int topK = 1000;                 ///< Top-k sampling parameter
    float topP = 0.95f;              ///< Top-p (nucleus) sampling parameter
    bool normalizeAudio = true;      ///< Whether to normalize reference audio
    unsigned int seed = 0;           ///< Random seed (0 = random)
};

/**
 * @brief Progress callback for generation
 */
using GenerationCallback = std::function<void(int currentStep, int totalSteps)>;

/**
 * @brief Main Chatterbox TTS inference class
 * 
 * Manages ONNX model sessions and provides text-to-speech generation
 * with voice cloning capabilities.
 */
class ChatterboxTTS {
public:
    ChatterboxTTS();
    ~ChatterboxTTS();
    
    // Non-copyable
    ChatterboxTTS(const ChatterboxTTS&) = delete;
    ChatterboxTTS& operator=(const ChatterboxTTS&) = delete;
    
    /**
     * @brief Load models from a directory
     * @param modelDir Directory containing ONNX files
     * @param dtype Model dtype suffix (e.g., "q4", "fp32")
     * @param provider Execution provider (CPU/CUDA/AUTO)
     * @return true on success
     */
    /**
     * @brief Load models from a directory
     * @param modelDir Directory containing ONNX files
     * @param dtype Model dtype suffix (e.g., "q4", "fp32")
     * @param provider Execution provider (CPU/CUDA/AUTO)
     * @param enableProfiling Enable ONNX Runtime profiling (writes onnx_profile_*.json)
     * @return true on success
     */
    bool LoadModels(const std::string& modelDir,
                    const std::string& dtype = "q4",
                    ExecutionProvider provider = ExecutionProvider::CPU,
                    bool enableProfiling = false);
    
    /**
     * @brief Unload all models and tokenizer from memory
     * 
     * Releases all ONNX sessions, tokenizer, and voice conditionals.
     * After calling this, IsReady() will return false.
     */
    void UnloadModels();
    
    /**
     * @brief Check if models are loaded and ready
     */
    bool IsReady() const;
    
    /**
     * @brief Check if tokenizer is loaded
     */
    bool HasTokenizer() const;
    
    /**
     * @brief Tokenize text using the loaded HuggingFace tokenizer
     * @param text Input text (will be normalized automatically)
     * @return TokenData with token IDs, or empty TokenData on failure
     */
    TokenData Tokenize(const std::string& text);
    
    /**
     * @brief Prepare voice conditionals from reference audio
     * @param audioPath Path to reference audio file (>5 seconds)
     * @return true on success
     */
    bool PrepareConditionals(const std::string& audioPath);
    
    /**
     * @brief Set pre-computed voice conditionals
     */
    void SetConditionals(const VoiceConditionals& conds);
    
    /**
     * @brief Get current voice conditionals
     */
    const VoiceConditionals& GetConditionals() const { return m_conds; }
    
    /**
     * @brief Check if voice conditionals are ready
     */
    bool HasConditionals() const { return m_conds.IsValid(); }
    
    /**
     * @brief Generate speech from pre-tokenized text
     * @param tokens Pre-tokenized text tokens
     * @param config Generation parameters
     * @param callback Optional progress callback
     * @return Generated audio samples at 24kHz, or empty on failure
     */
    std::vector<float> Generate(const TokenData& tokens,
                                 const GenerationConfig& config = GenerationConfig(),
                                 GenerationCallback callback = nullptr);
    
    /**
     * @brief Generate speech from raw token IDs
     * @param tokenIds Token ID vector
     * @param config Generation parameters
     * @param callback Optional progress callback
     * @return Generated audio samples at 24kHz, or empty on failure
     */
    std::vector<float> Generate(const std::vector<int64_t>& tokenIds,
                                 const GenerationConfig& config = GenerationConfig(),
                                 GenerationCallback callback = nullptr);
    
    /**
     * @brief Get sample rate of output audio
     */
    int GetSampleRate() const { return SAMPLE_RATE; }
    
    /**
     * @brief Get last error message
     */
    const std::string& GetLastError() const { return m_lastError; }

    /**
     * @brief Enable ONNX Runtime profiling
     * @note Must be called AFTER LoadModels() but BEFORE Generate()
     */
    void EnableProfiling();

    /**
     * @brief End profiling and write results to file
     * @return Path to the generated profile file
     */
    std::string EndProfiling();
    
private:
    // ONNX sessions
    std::unique_ptr<OnnxSessionManager> m_sessionManager;
    
    // Model sessions (managed by session manager)
    bool m_modelsLoaded = false;
    
    // Model dtype for fp16 handling
    std::string m_dtype;
    
    // Voice conditionals
    VoiceConditionals m_conds;
    
    // Audio loader for reference audio
    AudioLoader m_audioLoader;
    
    // HuggingFace tokenizer for text encoding
    std::unique_ptr<HFTokenizer> m_tokenizer;
    
    // Random number generator
    std::mt19937 m_rng;
    
    // Error tracking
    std::string m_lastError;
    
    // Model names
    static constexpr const char* SPEECH_ENCODER = "speech_encoder";
    static constexpr const char* EMBED_TOKENS = "embed_tokens";
    static constexpr const char* LANGUAGE_MODEL = "language_model";
    static constexpr const char* COND_DECODER = "conditional_decoder";
    
    /**
     * @brief Get model filename for given name and dtype
     */
    std::string GetModelFilename(const std::string& name, const std::string& dtype);
    
    /**
     * @brief Run speech encoder on audio
     */
    bool RunSpeechEncoder(const std::vector<float>& audio);
    
    /**
     * @brief Run embed_tokens model to get text embeddings
     */
    std::vector<float> RunEmbedTokens(const std::vector<int64_t>& tokenIds);
    
    /**
     * @brief Softmax over last axis
     */
    void Softmax(std::vector<float>& logits, size_t vocabSize);
    
    /**
     * @brief Sample from probability distribution
     */
    int64_t SampleToken(const std::vector<float>& probs);
    
    /**
     * @brief Apply repetition penalty to logits
     */
    void ApplyRepetitionPenalty(std::vector<float>& logits,
                                 const std::vector<int64_t>& generatedTokens,
                                 float penalty);
    
    /**
     * @brief Apply top-k filtering
     */
    void ApplyTopK(std::vector<float>& logits, int k);
    
    /**
     * @brief Apply top-p (nucleus) filtering
     */
    void ApplyTopP(std::vector<float>& logits, float p);
};

/**
 * @brief Utility: Clean and normalize text for TTS
 * @param text Input text
 * @return Cleaned text
 */
std::string NormalizeText(const std::string& text);

} // namespace ChatterboxTTS

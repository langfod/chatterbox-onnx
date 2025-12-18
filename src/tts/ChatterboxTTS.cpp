/**
 * @file ChatterboxTTS.cpp
 * @brief Implementation of Chatterbox TTS inference engine
 */

#include "tts/ChatterboxTTS.h"
#include "tts/TensorUtils.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>

// Set to 1 to enable detailed timing profiling of the generation loop
#define PROFILE_GENERATION_LOOP 1

#if PROFILE_GENERATION_LOOP
#define PROFILE_START(name) auto _profile_##name##_start = std::chrono::high_resolution_clock::now()
#define PROFILE_END(name, accum) accum += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - _profile_##name##_start).count()
#else
#define PROFILE_START(name) (void)0
#define PROFILE_END(name, accum) (void)0
#endif

namespace fs = std::filesystem;

namespace ChatterboxTTS {

// ============================================================================
// VoiceConditionals
// ============================================================================

bool VoiceConditionals::Save(const std::string& path) const {
    if (!IsValid()) return false;
    
    // Ensure parent directory exists
    fs::path filePath(path);
    if (filePath.has_parent_path()) {
        fs::create_directories(filePath.parent_path());
    }
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Write magic number
    uint32_t magic = 0x434F4E44;  // "COND"
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    
    // Write version
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Helper to write array
    auto writeArray = [&](const auto& data, const std::vector<int64_t>& shape) {
        // Write shape
        uint32_t numDims = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&numDims), sizeof(numDims));
        file.write(reinterpret_cast<const char*>(shape.data()), shape.size() * sizeof(int64_t));
        
        // Write data
        uint64_t dataSize = data.size() * sizeof(typename std::decay_t<decltype(data)>::value_type);
        file.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
        file.write(reinterpret_cast<const char*>(data.data()), dataSize);
    };
    
    writeArray(condEmb, condEmbShape);
    writeArray(promptToken, promptTokenShape);
    writeArray(speakerEmbeddings, speakerEmbeddingsShape);
    writeArray(speakerFeatures, speakerFeaturesShape);
    
    return file.good();
}

std::optional<VoiceConditionals> VoiceConditionals::Load(const std::string& path) {
    if (!fs::exists(path)) return std::nullopt;
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return std::nullopt;
    
    // Read magic
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x434F4E44) return std::nullopt;
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) return std::nullopt;
    
    VoiceConditionals conds;
    
    // Helper to read array
    auto readFloatArray = [&](std::vector<float>& data, std::vector<int64_t>& shape) {
        uint32_t numDims;
        file.read(reinterpret_cast<char*>(&numDims), sizeof(numDims));
        shape.resize(numDims);
        file.read(reinterpret_cast<char*>(shape.data()), numDims * sizeof(int64_t));
        
        uint64_t dataSize;
        file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        data.resize(dataSize / sizeof(float));
        file.read(reinterpret_cast<char*>(data.data()), dataSize);
    };
    
    auto readInt64Array = [&](std::vector<int64_t>& data, std::vector<int64_t>& shape) {
        uint32_t numDims;
        file.read(reinterpret_cast<char*>(&numDims), sizeof(numDims));
        shape.resize(numDims);
        file.read(reinterpret_cast<char*>(shape.data()), numDims * sizeof(int64_t));
        
        uint64_t dataSize;
        file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        data.resize(dataSize / sizeof(int64_t));
        file.read(reinterpret_cast<char*>(data.data()), dataSize);
    };
    
    readFloatArray(conds.condEmb, conds.condEmbShape);
    readInt64Array(conds.promptToken, conds.promptTokenShape);
    readFloatArray(conds.speakerEmbeddings, conds.speakerEmbeddingsShape);
    readFloatArray(conds.speakerFeatures, conds.speakerFeaturesShape);
    
    if (!file) return std::nullopt;
    
    return conds;
}

// ============================================================================
// ChatterboxTTS
// ============================================================================

ChatterboxTTS::ChatterboxTTS()
    : m_sessionManager(std::make_unique<OnnxSessionManager>(ExecutionProvider::CPU))
{
    // Initialize RNG with random seed
    std::random_device rd;
    m_rng.seed(rd());
}

ChatterboxTTS::~ChatterboxTTS() = default;

std::string ChatterboxTTS::GetModelFilename(const std::string& name, const std::string& dtype) {
    if (dtype == "fp32") {
        return name + ".onnx";
    } else if (dtype == "q8") {
        return name + "_quantized.onnx";
    } else if (dtype == "q4") {
        return name + "_q4.onnx";
    } else if (dtype == "q4f16") {
        return name + "_q4f16.onnx";
    } else {
        return name + "_" + dtype + ".onnx";
    }
}

bool ChatterboxTTS::LoadModels(const std::string& modelDir,
                                const std::string& dtype,
                                ExecutionProvider provider,
                                bool enableProfiling) {
    m_lastError.clear();
    m_modelsLoaded = false;
    
    // Recreate session manager with requested provider
    m_sessionManager = std::make_unique<OnnxSessionManager>(provider);
    
    // Enable profiling if requested (must be done before loading models)
    if (enableProfiling) {
        m_sessionManager->EnableProfiling(L"onnx_profile");
    }
    
    // Try multiple path patterns to find ONNX models
    fs::path onnxDir;
    
    // Pattern 1: Direct path with /onnx subfolder
    fs::path pattern1 = fs::path(modelDir) / "onnx";
    
    // Pattern 2: HuggingFace cache structure: models--<repo>/snapshots/<hash>/onnx
    fs::path pattern2 = fs::path(modelDir) / "models--ResembleAI--chatterbox-turbo-ONNX" / "snapshots";
    
    // Pattern 3: Direct path (modelDir is already the onnx folder)
    fs::path pattern3 = modelDir;
    
    if (fs::exists(pattern1)) {
        onnxDir = pattern1;
    } else if (fs::exists(pattern2)) {
        // Find the snapshot hash directory
        for (const auto& entry : fs::directory_iterator(pattern2)) {
            if (entry.is_directory()) {
                fs::path snapshotOnnx = entry.path() / "onnx";
                if (fs::exists(snapshotOnnx)) {
                    onnxDir = snapshotOnnx;
                    break;
                }
            }
        }
    } else if (fs::exists(pattern3)) {
        onnxDir = pattern3;
    }
    
    if (onnxDir.empty() || !fs::exists(onnxDir)) {
        m_lastError = "Could not find ONNX models directory. Tried:\n"
                      "  " + pattern1.string() + "\n"
                      "  " + pattern2.string() + "/<hash>/onnx\n"
                      "  " + pattern3.string();
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    spdlog::info("Loading ONNX models from: {} (dtype={})", onnxDir.string(), dtype);
    
    // Load each model
    const char* modelNames[] = { SPEECH_ENCODER, EMBED_TOKENS, LANGUAGE_MODEL, COND_DECODER };
    
    for (const char* name : modelNames) {
        std::string filename = GetModelFilename(name, dtype);
        fs::path modelPath = onnxDir / filename;
        
        if (!fs::exists(modelPath)) {
            m_lastError = "Model file not found: " + modelPath.string();
            spdlog::error("{}", m_lastError);
            return false;
        }
        
        if (!m_sessionManager->LoadModel(modelPath.string(), name)) {
            m_lastError = "Failed to load model: " + std::string(name);
            spdlog::error("{}", m_lastError);
            return false;
        }
    }
    
    m_modelsLoaded = true;
    m_dtype = dtype;  // Store dtype for fp16 handling
    spdlog::info("All ONNX models loaded successfully");
    
    return true;
}

bool ChatterboxTTS::IsReady() const {
    return m_modelsLoaded && 
           m_sessionManager->IsModelLoaded(SPEECH_ENCODER) &&
           m_sessionManager->IsModelLoaded(EMBED_TOKENS) &&
           m_sessionManager->IsModelLoaded(LANGUAGE_MODEL) &&
           m_sessionManager->IsModelLoaded(COND_DECODER);
}

bool ChatterboxTTS::PrepareConditionals(const std::string& audioPath) {
    m_lastError.clear();
    
    if (!IsReady()) {
        m_lastError = "Models not loaded";
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Load audio at SAMPLE_RATE (24kHz)
    AudioLoadConfig config;
    config.targetSampleRate = SAMPLE_RATE;
    config.normalize = true;
    
    auto audioData = m_audioLoader.LoadFile(audioPath, config);
    if (!audioData) {
        m_lastError = "Failed to load audio: " + m_audioLoader.GetLastError();
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Check duration (must be >5 seconds)
    float duration = audioData->GetDuration();
    if (duration < 5.0f) {
        m_lastError = "Audio prompt must be longer than 5 seconds (got " + 
                      std::to_string(duration) + "s)";
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    spdlog::info("Running speech encoder on {:.2f}s audio", duration);
    
    return RunSpeechEncoder(audioData->samples);
}

void ChatterboxTTS::SetConditionals(const VoiceConditionals& conds) {
    m_conds = conds;
}

bool ChatterboxTTS::RunSpeechEncoder(const std::vector<float>& audio) {
    auto* session = m_sessionManager->GetSession(SPEECH_ENCODER);
    if (!session) {
        m_lastError = "Speech encoder session not loaded";
        return false;
    }
    
    auto& memInfo = m_sessionManager->GetMemoryInfo();
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Prepare input: [1, num_samples]
    std::vector<float> audioData = audio;  // Copy for non-const
    std::vector<int64_t> audioShape = {1, static_cast<int64_t>(audio.size())};
    
    auto audioTensor = TensorUtils::CreateFloatTensor(memInfo, audioData, audioShape);
    
    // Get input/output names
    auto inputNames = TensorUtils::GetInputNames(*session, allocator);
    auto outputNames = TensorUtils::GetOutputNames(*session, allocator);
    
    // Create C-style name arrays
    std::vector<const char*> inputNamePtrs, outputNamePtrs;
    for (const auto& name : inputNames) inputNamePtrs.push_back(name.c_str());
    for (const auto& name : outputNames) outputNamePtrs.push_back(name.c_str());
    
    // Run inference
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(audioTensor));
    
    try {
        auto outputs = session->Run(
            m_sessionManager->GetRunOptions(),
            inputNamePtrs.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNamePtrs.data(),
            outputNamePtrs.size()
        );
        
        // Extract outputs
        // Expected outputs: cond_emb, prompt_token, speaker_embeddings, speaker_features
        if (outputs.size() < 4) {
            m_lastError = "Speech encoder returned unexpected number of outputs";
            return false;
        }
        
        m_conds.condEmb = TensorUtils::ExtractFloatData(outputs[0]);
        m_conds.condEmbShape = TensorUtils::GetShape(outputs[0]);
        
        m_conds.promptToken = TensorUtils::ExtractInt64Data(outputs[1]);
        m_conds.promptTokenShape = TensorUtils::GetShape(outputs[1]);
        
        m_conds.speakerEmbeddings = TensorUtils::ExtractFloatData(outputs[2]);
        m_conds.speakerEmbeddingsShape = TensorUtils::GetShape(outputs[2]);
        
        m_conds.speakerFeatures = TensorUtils::ExtractFloatData(outputs[3]);
        m_conds.speakerFeaturesShape = TensorUtils::GetShape(outputs[3]);
        
        spdlog::info("Voice conditionals prepared: condEmb={}, promptToken={}, speakerEmb={}, speakerFeat={}", 
                     TensorUtils::ShapeToString(m_conds.condEmbShape),
                     TensorUtils::ShapeToString(m_conds.promptTokenShape),
                     TensorUtils::ShapeToString(m_conds.speakerEmbeddingsShape),
                     TensorUtils::ShapeToString(m_conds.speakerFeaturesShape));
        
        return true;
    } catch (const Ort::Exception& e) {
        m_lastError = "ONNX Runtime error in speech encoder: " + std::string(e.what());
        spdlog::error("{}", m_lastError);
        return false;
    }
}

std::vector<float> ChatterboxTTS::RunEmbedTokens(const std::vector<int64_t>& tokenIds) {
    auto* session = m_sessionManager->GetSession(EMBED_TOKENS);
    if (!session) return {};
    
    auto& memInfo = m_sessionManager->GetMemoryInfo();
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Prepare input: [1, seq_len]
    std::vector<int64_t> tokens = tokenIds;  // Copy
    std::vector<int64_t> shape = {1, static_cast<int64_t>(tokenIds.size())};
    
    auto tokenTensor = TensorUtils::CreateInt64Tensor(memInfo, tokens, shape);
    
    auto inputNames = TensorUtils::GetInputNames(*session, allocator);
    auto outputNames = TensorUtils::GetOutputNames(*session, allocator);
    
    std::vector<const char*> inputNamePtrs, outputNamePtrs;
    for (const auto& name : inputNames) inputNamePtrs.push_back(name.c_str());
    for (const auto& name : outputNames) outputNamePtrs.push_back(name.c_str());
    
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(tokenTensor));
    
    try {
        auto outputs = session->Run(
            m_sessionManager->GetRunOptions(),
            inputNamePtrs.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNamePtrs.data(),
            outputNamePtrs.size()
        );
        
        return TensorUtils::ExtractFloatData(outputs[0]);
    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime error in embed_tokens: {}", e.what());
        return {};
    }
}

std::vector<float> ChatterboxTTS::Generate(const TokenData& tokens,
                                            const GenerationConfig& config,
                                            GenerationCallback callback) {
    return Generate(tokens.tokenIds, config, callback);
}

std::vector<float> ChatterboxTTS::Generate(const std::vector<int64_t>& tokenIds,
                                            const GenerationConfig& config,
                                            GenerationCallback callback) {
    m_lastError.clear();
    
    if (!IsReady()) {
        m_lastError = "Models not loaded";
        spdlog::error("{}", m_lastError);
        return {};
    }
    
    if (!m_conds.IsValid()) {
        m_lastError = "Voice conditionals not prepared";
        spdlog::error("{}", m_lastError);
        return {};
    }
    
    if (tokenIds.empty()) {
        m_lastError = "No input tokens";
        spdlog::error("{}", m_lastError);
        return {};
    }
    
    // Seed RNG
    if (config.seed != 0) {
        m_rng.seed(config.seed);
    }
    
    auto& memInfo = m_sessionManager->GetMemoryInfo();
    Ort::AllocatorWithDefaultOptions allocator;
    
    auto* lmSession = m_sessionManager->GetSession(LANGUAGE_MODEL);
    auto* embedSession = m_sessionManager->GetSession(EMBED_TOKENS);
    auto* decoderSession = m_sessionManager->GetSession(COND_DECODER);
    
    if (!lmSession || !embedSession || !decoderSession) {
        m_lastError = "Required session not loaded";
        return {};
    }
    
    // Get LM input/output info
    auto lmInputNames = TensorUtils::GetInputNames(*lmSession, allocator);
    auto lmOutputNames = TensorUtils::GetOutputNames(*lmSession, allocator);
    
    // Find past_key_values inputs for KV cache
    std::vector<std::string> kvCacheNames;
    for (const auto& name : lmInputNames) {
        if (name.find("past_key_values") != std::string::npos) {
            kvCacheNames.push_back(name);
        }
    }
    
    spdlog::info("Starting generation with {} input tokens, max {} new tokens",
                 tokenIds.size(), config.maxNewTokens);
    
    // Check if KV cache should be fp16 for this model
    // For q4f16 models: inputs_embeds is fp32, but KV cache is fp16
    bool kvCacheFp16 = (m_dtype == "q4f16" || m_dtype == "q4fp16" || m_dtype == "fp16");
    if (kvCacheFp16) {
        spdlog::info("Using FP16 KV cache for language model");
    }
    
    // Initialize generated tokens with START_SPEECH_TOKEN
    std::vector<int64_t> generatedTokens = {START_SPEECH_TOKEN};
    
    // KV cache storage - store Ort::Value objects directly to avoid copying
    // The output tensors from each Run() become inputs to the next Run()
    std::vector<Ort::Value> kvCacheValues(kvCacheNames.size());
    std::vector<std::vector<int64_t>> kvCacheShapes(kvCacheNames.size());
    
    // Initialize empty KV cache shapes [batch, num_heads, 0, head_dim]
    // For first iteration, we need empty tensors
    std::vector<std::vector<float>> emptyKvCache(kvCacheNames.size());
    std::vector<std::vector<Float16>> emptyKvCacheFp16(kvCacheNames.size());
    for (size_t i = 0; i < kvCacheNames.size(); ++i) {
        kvCacheShapes[i] = {1, NUM_KV_HEADS, 0, HEAD_DIM};
    }
    
    // Get cond_emb shape info
    int64_t condSeqLen = 0;
    if (m_conds.condEmbShape.size() >= 2) {
        condSeqLen = m_conds.condEmbShape[1];
    }
    int64_t hiddenSize = 0;
    if (m_conds.condEmbShape.size() >= 3) {
        hiddenSize = m_conds.condEmbShape[2];
    }
    
    // Pre-compute mapping from input name index to KV cache index (avoids per-step linear search)
    std::vector<int> inputToKvCacheIdx(lmInputNames.size(), -1);
    for (size_t i = 0; i < lmInputNames.size(); ++i) {
        if (lmInputNames[i].find("past_key_values") != std::string::npos) {
            for (size_t k = 0; k < kvCacheNames.size(); ++k) {
                if (kvCacheNames[k] == lmInputNames[i]) {
                    inputToKvCacheIdx[i] = static_cast<int>(k);
                    break;
                }
            }
        }
    }
    
    // Pre-allocate vectors that will be reused each iteration
    // Reserve for max expected sequence length to avoid reallocations
    const int64_t maxTotalSeqLen = condSeqLen + static_cast<int64_t>(tokenIds.size()) + config.maxNewTokens;
    std::vector<int64_t> attentionMask;
    attentionMask.reserve(maxTotalSeqLen);
    std::vector<int64_t> positionIds;
    positionIds.reserve(std::max(condSeqLen + static_cast<int64_t>(tokenIds.size()), int64_t(1)));
    std::vector<float> inputsEmbeds;
    inputsEmbeds.reserve((condSeqLen + static_cast<int64_t>(tokenIds.size())) * hiddenSize);
    
    // Pre-allocate output name pointers (constant across iterations)
    std::vector<const char*> outputNamePtrs;
    outputNamePtrs.reserve(lmOutputNames.size());
    for (const auto& name : lmOutputNames) {
        outputNamePtrs.push_back(name.c_str());
    }
    
    // Pre-allocate input name pointers (constant across iterations)
    std::vector<const char*> inputNamePtrs;
    inputNamePtrs.reserve(lmInputNames.size());
    for (const auto& name : lmInputNames) {
        inputNamePtrs.push_back(name.c_str());
    }
    
    // Pre-compute text token embeddings BEFORE the loop (saves one session call)
    std::vector<float> textEmbeddings = RunEmbedTokens(tokenIds);
    if (textEmbeddings.empty()) {
        m_lastError = "Failed to get text embeddings";
        return {};
    }
    int64_t textSeqLen = static_cast<int64_t>(tokenIds.size());
    
    // Generation loop
    int64_t currentPosition = 0;
    
#if PROFILE_GENERATION_LOOP
    // Timing accumulators (microseconds)
    int64_t time_embedTokens = 0;
    int64_t time_prepareInputs = 0;
    int64_t time_createTensors = 0;
    int64_t time_lmRun = 0;
    int64_t time_extractLogits = 0;
    int64_t time_sampling = 0;  // repetition penalty + temperature + topk + topp + softmax + sample
    int64_t time_kvCacheUpdate = 0;
    int totalSteps = 0;
#endif
    
    for (int step = 0; step < config.maxNewTokens; ++step) {
        if (callback) {
            callback(step, config.maxNewTokens);
        }
        
#if PROFILE_GENERATION_LOOP
        totalSteps++;
#endif
        
        PROFILE_START(prepareInputs);
        // Prepare inputs_embeds (reuse pre-allocated vector)
        inputsEmbeds.clear();
        int64_t seqLen;
        
        if (step == 0) {
            // First iteration: concatenate cond_emb + pre-computed text_embeds
            seqLen = condSeqLen + textSeqLen;
            inputsEmbeds.insert(inputsEmbeds.end(), m_conds.condEmb.begin(), m_conds.condEmb.end());
            inputsEmbeds.insert(inputsEmbeds.end(), textEmbeddings.begin(), textEmbeddings.end());
            // Clear text embeddings to free memory (no longer needed)
            textEmbeddings.clear();
            textEmbeddings.shrink_to_fit();
        } else {
            // Subsequent iterations: embed single generated token
            seqLen = 1;
            inputsEmbeds = RunEmbedTokens({generatedTokens.back()});
            if (inputsEmbeds.empty()) {
                m_lastError = "Failed to get token embedding";
                return {};
            }
        }
        
        // Prepare LM inputs
        std::vector<int64_t> embedsShape = {1, seqLen, hiddenSize};
        
        // Attention mask - extend existing mask instead of reallocating
        int64_t totalSeqLen = (step == 0) ? seqLen : (currentPosition + seqLen);
        if (step == 0) {
            attentionMask.assign(totalSeqLen, 1);
        } else {
            // Just extend by seqLen (which is 1 for subsequent iterations)
            attentionMask.resize(totalSeqLen, 1);
        }
        std::vector<int64_t> maskShape = {1, totalSeqLen};
        
        // Position IDs (reuse pre-allocated vector)
        positionIds.resize(seqLen);
        for (int64_t i = 0; i < seqLen; ++i) {
            positionIds[i] = currentPosition + i;
        }
        std::vector<int64_t> posShape = {1, seqLen};
        
        PROFILE_END(prepareInputs, time_prepareInputs);
        PROFILE_START(createTensors);
        
        // Build input tensors
        std::vector<Ort::Value> lmInputs;
        lmInputs.reserve(lmInputNames.size());
        
        // Create tensors in order expected by model (use pre-computed KV cache mapping)
        for (size_t i = 0; i < lmInputNames.size(); ++i) {
            if (lmInputNames[i] == "inputs_embeds") {
                // inputs_embeds is always fp32, even for q4f16 models
                lmInputs.push_back(TensorUtils::CreateFloatTensor(memInfo, inputsEmbeds, embedsShape));
            } else if (lmInputNames[i] == "attention_mask") {
                lmInputs.push_back(TensorUtils::CreateInt64Tensor(memInfo, attentionMask, maskShape));
            } else if (lmInputNames[i] == "position_ids") {
                lmInputs.push_back(TensorUtils::CreateInt64Tensor(memInfo, positionIds, posShape));
            } else if (inputToKvCacheIdx[i] >= 0) {
                // Use pre-computed KV cache index (avoids per-step linear search)
                int k = inputToKvCacheIdx[i];
                if (kvCacheValues[k]) {
                    // Reuse the Ort::Value from previous iteration (zero-copy!)
                    lmInputs.push_back(std::move(kvCacheValues[k]));
                } else {
                    // First iteration: create empty tensor
                    if (kvCacheFp16) {
                        lmInputs.push_back(TensorUtils::CreateFloat16Tensor(
                            memInfo, emptyKvCacheFp16[k], kvCacheShapes[k]));
                    } else {
                        lmInputs.push_back(TensorUtils::CreateFloatTensor(
                            memInfo, emptyKvCache[k], kvCacheShapes[k]));
                    }
                }
            }
        }
        
        PROFILE_END(createTensors, time_createTensors);
        PROFILE_START(lmRun);
        
        // Run LM (use reusable RunOptions from session manager)
        std::vector<Ort::Value> lmOutputs;
        try {
            lmOutputs = lmSession->Run(
                m_sessionManager->GetRunOptions(),
                inputNamePtrs.data(),
                lmInputs.data(),
                lmInputs.size(),
                outputNamePtrs.data(),
                outputNamePtrs.size()
            );
        } catch (const Ort::Exception& e) {
            m_lastError = "ONNX Runtime error in language model: " + std::string(e.what());
            spdlog::error("{}", m_lastError);
            return {};
        }
        
        PROFILE_END(lmRun, time_lmRun);
        PROFILE_START(extractLogits);
        
        // Get logits (first output) and present KV (remaining outputs)
        auto logitsShape = TensorUtils::GetShape(lmOutputs[0]);
        size_t vocabSize = static_cast<size_t>(logitsShape.back());
        size_t lastPosOffset = (logitsShape[1] - 1) * vocabSize;
        
        // Extract only the last position logits (avoid copying entire tensor)
        std::vector<float> nextLogits = TensorUtils::ExtractFloatSlice(
            lmOutputs[0], lastPosOffset, vocabSize);
        
        PROFILE_END(extractLogits, time_extractLogits);
        
        PROFILE_START(sampling);
        
        // Apply repetition penalty
        ApplyRepetitionPenalty(nextLogits, generatedTokens, config.repetitionPenalty);
        
        // Apply temperature
        if (config.temperature != 1.0f) {
            for (float& logit : nextLogits) {
                logit /= config.temperature;
            }
        }
        
        // Apply top-k
        if (config.topK > 0 && config.topK < static_cast<int>(vocabSize)) {
            ApplyTopK(nextLogits, config.topK);
        }
        
        // Apply top-p
        if (config.topP < 1.0f) {
            ApplyTopP(nextLogits, config.topP);
        }
        
        // Convert to probabilities
        Softmax(nextLogits, vocabSize);
        
        // Sample next token
        int64_t nextToken = SampleToken(nextLogits);
        
        PROFILE_END(sampling, time_sampling);
        generatedTokens.push_back(nextToken);
        
        // Check for stop token
        if (nextToken == STOP_SPEECH_TOKEN) {
            spdlog::info("Stop token detected at step {}", step + 1);
            break;
        }
        
        // Update position for next iteration
        currentPosition += seqLen;
        
        PROFILE_START(kvCacheUpdate);
        
        // Update KV cache from present_key_values outputs
        // Zero-copy: just move the Ort::Value objects to be reused as inputs next iteration
        for (size_t k = 0; k < kvCacheNames.size() && (k + 1) < lmOutputs.size(); ++k) {
            kvCacheShapes[k] = TensorUtils::GetShape(lmOutputs[k + 1]);
            kvCacheValues[k] = std::move(lmOutputs[k + 1]);
        }
        
        PROFILE_END(kvCacheUpdate, time_kvCacheUpdate);
        
        if ((step + 1) % 100 == 0) {
            spdlog::debug("Generated {} tokens...", step + 1);
        }
    }
    
#if PROFILE_GENERATION_LOOP
    // Print profiling results
    spdlog::info("\n=== Generation Loop Profiling ({} steps) ===", totalSteps);
    spdlog::info("  Embed Tokens:    {:>8.2f} ms ({:>5.2f} ms/step)", time_embedTokens / 1000.0, time_embedTokens / 1000.0 / totalSteps);
    spdlog::info("  Prepare Inputs:  {:>8.2f} ms ({:>5.2f} ms/step)", time_prepareInputs / 1000.0, time_prepareInputs / 1000.0 / totalSteps);
    spdlog::info("  Create Tensors:  {:>8.2f} ms ({:>5.2f} ms/step)", time_createTensors / 1000.0, time_createTensors / 1000.0 / totalSteps);
    spdlog::info("  LM Run:          {:>8.2f} ms ({:>5.2f} ms/step)", time_lmRun / 1000.0, time_lmRun / 1000.0 / totalSteps);
    spdlog::info("  Extract Logits:  {:>8.2f} ms ({:>5.2f} ms/step)", time_extractLogits / 1000.0, time_extractLogits / 1000.0 / totalSteps);
    spdlog::info("  Sampling:        {:>8.2f} ms ({:>5.2f} ms/step)", time_sampling / 1000.0, time_sampling / 1000.0 / totalSteps);
    spdlog::info("  KV Cache Update: {:>8.2f} ms ({:>5.2f} ms/step)", time_kvCacheUpdate / 1000.0, time_kvCacheUpdate / 1000.0 / totalSteps);
    int64_t totalCpp = time_embedTokens + time_prepareInputs + time_createTensors + time_extractLogits + time_sampling + time_kvCacheUpdate;
    spdlog::info("  ----------------------------------------");
    spdlog::info("  C++ Overhead:    {:>8.2f} ms ({:>5.2f} ms/step)", totalCpp / 1000.0, totalCpp / 1000.0 / totalSteps);
    spdlog::info("  ONNX LM Run:     {:>8.2f} ms ({:>5.2f} ms/step)", time_lmRun / 1000.0, time_lmRun / 1000.0 / totalSteps);
#endif
    
    spdlog::info("Generated {} speech tokens", generatedTokens.size());
    
    // Process generated tokens for decoder
    // Remove start token (first) and stop token (last, if present)
    // Python: speech_tokens = generate_tokens[:, 1:-1]
    std::vector<int64_t> speechTokens;
    size_t endIdx = generatedTokens.size();
    // If last token is stop token, exclude it
    if (endIdx > 0 && generatedTokens[endIdx - 1] == STOP_SPEECH_TOKEN) {
        endIdx--;
    }
    // Start from index 1 to skip START token
    for (size_t i = 1; i < endIdx; ++i) {
        speechTokens.push_back(generatedTokens[i]);
    }
    
    // Add silence tokens at end
    for (int i = 0; i < 3; ++i) {
        speechTokens.push_back(SILENCE_TOKEN);
    }
    
    // Concatenate: prompt_token + speech_tokens
    std::vector<int64_t> decoderTokens;
    decoderTokens.reserve(m_conds.promptToken.size() + speechTokens.size());
    decoderTokens.insert(decoderTokens.end(), m_conds.promptToken.begin(), m_conds.promptToken.end());
    decoderTokens.insert(decoderTokens.end(), speechTokens.begin(), speechTokens.end());
    
    spdlog::info("Running conditional decoder with {} tokens", decoderTokens.size());
    
    // Debug: Check speech tokens for validity
    int64_t minToken = *std::min_element(decoderTokens.begin(), decoderTokens.end());
    int64_t maxToken = *std::max_element(decoderTokens.begin(), decoderTokens.end());
    spdlog::info("Speech tokens: min={}, max={}, count={}", minToken, maxToken, decoderTokens.size());
    
    // Log first few generated speech tokens (after prompt)
    if (speechTokens.size() > 0) {
        std::string tokenStr;
        for (size_t i = 0; i < std::min(speechTokens.size(), size_t(20)); ++i) {
            tokenStr += std::to_string(speechTokens[i]) + " ";
        }
        spdlog::info("First speech tokens: {}", tokenStr);
    }
    
    // Run conditional decoder
    auto decoderInputNames = TensorUtils::GetInputNames(*decoderSession, allocator);
    auto decoderOutputNames = TensorUtils::GetOutputNames(*decoderSession, allocator);
    
    std::vector<int64_t> tokensShape = {1, static_cast<int64_t>(decoderTokens.size())};
    
    // IMPORTANT: These vectors must stay alive until after Run() because 
    // CreateTensor wraps the data pointer without copying
    std::vector<float> speakerEmbCopy = m_conds.speakerEmbeddings;
    std::vector<float> speakerFeatCopy = m_conds.speakerFeatures;
    
    // Build inputs in the correct order expected by the model
    std::vector<Ort::Value> decoderInputs;
    std::vector<const char*> decInputPtrs, decOutputPtrs;
    
    for (const auto& name : decoderInputNames) {
        decInputPtrs.push_back(name.c_str());
        
        if (name == "speech_tokens") {
            decoderInputs.push_back(TensorUtils::CreateInt64Tensor(memInfo, decoderTokens, tokensShape));
        } else if (name == "speaker_embeddings") {
            decoderInputs.push_back(TensorUtils::CreateFloatTensor(
                memInfo, speakerEmbCopy, m_conds.speakerEmbeddingsShape));
        } else if (name == "speaker_features") {
            decoderInputs.push_back(TensorUtils::CreateFloatTensor(
                memInfo, speakerFeatCopy, m_conds.speakerFeaturesShape));
        } else {
            spdlog::error("Unknown decoder input: {}", name);
            m_lastError = "Unknown decoder input: " + name;
            return {};
        }
    }
    
    // Verify we have the right number of inputs
    if (decoderInputs.size() != decoderInputNames.size()) {
        spdlog::error("Decoder input count mismatch: {} vs {}", decoderInputs.size(), decoderInputNames.size());
        return {};
    }
    
    for (const auto& name : decoderOutputNames) {
        decOutputPtrs.push_back(name.c_str());
    }
    
    std::vector<float> audioOutput;
    try {
        // Log decoder input shapes before running
        spdlog::debug("Decoder inputs: tokens={}, speakerEmb={}, speakerFeat={}",
                      TensorUtils::ShapeToString(tokensShape),
                      TensorUtils::ShapeToString(m_conds.speakerEmbeddingsShape),
                      TensorUtils::ShapeToString(m_conds.speakerFeaturesShape));
        
        // Check for NaN in speaker embeddings
        int nanCountEmb = 0, nanCountFeat = 0;
        for (float v : m_conds.speakerEmbeddings) if (!std::isfinite(v)) nanCountEmb++;
        for (float v : m_conds.speakerFeatures) if (!std::isfinite(v)) nanCountFeat++;
        if (nanCountEmb > 0 || nanCountFeat > 0) {
            spdlog::warn("NaN in decoder inputs: speakerEmb={}, speakerFeat={}", nanCountEmb, nanCountFeat);
        }
        
        auto outputs = decoderSession->Run(
            m_sessionManager->GetRunOptions(),
            decInputPtrs.data(),
            decoderInputs.data(),
            decoderInputs.size(),
            decOutputPtrs.data(),
            decOutputPtrs.size()
        );
        
        audioOutput = TensorUtils::ExtractFloatData(outputs[0]);
        
        // Debug: check audio output stats
        if (!audioOutput.empty()) {
            float minVal = *std::min_element(audioOutput.begin(), audioOutput.end());
            float maxVal = *std::max_element(audioOutput.begin(), audioOutput.end());
            float sum = 0;
            for (float v : audioOutput) sum += std::abs(v);
            float avgAbs = sum / audioOutput.size();
            spdlog::info("Audio stats: min={:.6f}, max={:.6f}, avgAbs={:.6f}, samples={}", 
                        minVal, maxVal, avgAbs, audioOutput.size());
        }
        
    } catch (const Ort::Exception& e) {
        m_lastError = "ONNX Runtime error in conditional decoder: " + std::string(e.what());
        spdlog::error("{}", m_lastError);
        return {};
    }
    
    spdlog::info("Generated {:.2f}s of audio", 
                 static_cast<float>(audioOutput.size()) / SAMPLE_RATE);
    
    return audioOutput;
}

void ChatterboxTTS::Softmax(std::vector<float>& logits, size_t vocabSize) {
    // Fast path: find max in single pass
    float maxLogit = logits[0];
    for (size_t i = 1; i < vocabSize; ++i) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    
    // Handle edge case of all -inf
    if (!std::isfinite(maxLogit)) {
        spdlog::warn("Softmax: all logits are -inf, using fallback");
        logits[0] = 1.0f;
        for (size_t i = 1; i < vocabSize; ++i) logits[i] = 0.0f;
        return;
    }
    
    // Fused exp and sum in single pass (better cache locality)
    float sum = 0.0f;
    for (size_t i = 0; i < vocabSize; ++i) {
        // exp(-inf - max) = 0, so no special handling needed
        float val = std::exp(logits[i] - maxLogit);
        logits[i] = val;
        sum += val;
    }
    
    // Normalize (sum is guaranteed > 0 since maxLogit is finite)
    if (sum > 0.0f) {
        float invSum = 1.0f / sum;
        for (size_t i = 0; i < vocabSize; ++i) {
            logits[i] *= invSum;
        }
    }
}

int64_t ChatterboxTTS::SampleToken(const std::vector<float>& probs) {
    // After softmax, probabilities should sum to ~1.0 and be valid
    // Use uniform random sampling with cumulative sum
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(m_rng);
    
    float cumSum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumSum += probs[i];
        if (r <= cumSum) {
            return static_cast<int64_t>(i);
        }
    }
    
    // Fallback: return last token (numerical precision edge case)
    return static_cast<int64_t>(probs.size() - 1);
}

void ChatterboxTTS::ApplyRepetitionPenalty(std::vector<float>& logits,
                                            const std::vector<int64_t>& generatedTokens,
                                            float penalty) {
    if (penalty == 1.0f) return;
    
    for (int64_t token : generatedTokens) {
        if (token >= 0 && token < static_cast<int64_t>(logits.size())) {
            if (logits[token] < 0) {
                logits[token] *= penalty;
            } else {
                logits[token] /= penalty;
            }
        }
    }
}

void ChatterboxTTS::ApplyTopK(std::vector<float>& logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) return;
    
    // Use partial_sort to find k largest - more efficient than full sort
    // Only need indices, so create index array
    std::vector<size_t> indices(logits.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    
    // Partial sort to get top-k indices
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&logits](size_t a, size_t b) { return logits[a] > logits[b]; });
    
    float threshold = logits[indices[k - 1]];
    
    // Set everything below threshold to -inf
    const float negInf = -std::numeric_limits<float>::infinity();
    for (float& logit : logits) {
        if (logit < threshold) {
            logit = negInf;
        }
    }
}

void ChatterboxTTS::ApplyTopP(std::vector<float>& logits, float p) {
    if (p >= 1.0f) return;
    
    size_t vocabSize = logits.size();
    
    // Create indices sorted by logit value (descending)
    std::vector<size_t> indices(vocabSize);
    for (size_t i = 0; i < vocabSize; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&logits](size_t a, size_t b) { return logits[a] > logits[b]; });
    
    // Find max for stable softmax
    float maxLogit = logits[indices[0]];
    
    // Compute cumulative probability and find cutoff in one pass
    float sum = 0.0f;
    float cumSum = 0.0f;
    size_t cutoffIdx = vocabSize;
    
    // First pass: compute total sum for normalization
    for (size_t i = 0; i < vocabSize; ++i) {
        sum += std::exp(logits[indices[i]] - maxLogit);
    }
    
    // Second pass: find cutoff
    float invSum = 1.0f / sum;
    for (size_t i = 0; i < vocabSize; ++i) {
        float prob = std::exp(logits[indices[i]] - maxLogit) * invSum;
        cumSum += prob;
        if (cumSum > p) {
            cutoffIdx = i + 1;  // Include the one that crossed threshold
            break;
        }
    }
    
    // Zero out tokens beyond cutoff
    const float negInf = -std::numeric_limits<float>::infinity();
    for (size_t i = cutoffIdx; i < vocabSize; ++i) {
        logits[indices[i]] = negInf;
    }
}

// ============================================================================
// Utilities
// ============================================================================

std::string NormalizeText(const std::string& text) {
    std::string result = text;
    
    if (result.empty()) {
        return "You need to add some text for me to talk.";
    }
    
    // Capitalize first letter
    if (!result.empty() && std::islower(static_cast<unsigned char>(result[0]))) {
        result[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(result[0])));
    }
    
    // Replace uncommon punctuation
    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"\xe2\x80\xa6", ", "},      // … (ellipsis)
        {":", ","},
        {"\xe2\x80\x94", "-"},       // — (em dash)
        {"\xe2\x80\x93", "-"},       // – (en dash)
        {" ,", ","},
        {"\xe2\x80\x9c", "\""},      // " (left double quote)
        {"\xe2\x80\x9d", "\""},      // " (right double quote)
        {"\xe2\x80\x98", "'"},       // ' (left single quote)
        {"\xe2\x80\x99", "'"},       // ' (right single quote)
    };
    
    for (const auto& [old_str, new_str] : replacements) {
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
    }
    
    // Remove trailing whitespace
    while (!result.empty() && std::isspace(result.back())) {
        result.pop_back();
    }
    
    // Add period if no ending punctuation
    if (!result.empty()) {
        char last = result.back();
        if (last != '.' && last != '!' && last != '?' && last != '-' && last != ',') {
            result += '.';
        }
    }
    
    return result;
}

void ChatterboxTTS::EnableProfiling() {
    if (m_sessionManager) {
        // Enable profiling on the session manager
        // Note: This requires models to be reloaded to take effect
        m_sessionManager->EnableProfiling(L"onnx_profile");
    }
}

std::string ChatterboxTTS::EndProfiling() {
    if (m_sessionManager) {
        return m_sessionManager->EndProfiling();
    }
    return "";
}

} // namespace ChatterboxTTS

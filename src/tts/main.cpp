/**
 * @file main.cpp
 * @brief Chatterbox TTS ONNX Demo - Main Entry Point
 * 
 * This demo application performs text-to-speech synthesis using ONNX Runtime
 * with quantized Chatterbox Turbo models from HuggingFace.
 * 
 * Usage:
 *   chatterbox_tts_demo -t <tokens_file> [-v <voice_file>] [-o <output.wav>]
 * 
 * Workflow:
 *   1. Pre-tokenize text using Python: python tools/pretokenize.py "Hello" -o input.tokens
 *   2. Run TTS: chatterbox_tts_demo -t input.tokens -o output.wav
 */

#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <optional>
#include <algorithm>

#include "tts/ChatterboxTTS.h"
#include "tts/Tokenizer.h"
#include "tts/WavWriter.h"
#include "tts/ModelDownloader.h"
#include "tts/VoiceConditionalsCache.h"

#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

// Hardcoded cache directory
constexpr const char* CACHE_DIR = "cache";
constexpr const char* ASSETS_DIR = "assets";

/**
 * @brief Print usage information
 */
void PrintUsage(const char* programName) {
    std::cout << "Chatterbox TTS ONNX Demo\n"
              << "========================\n\n"
              << "Usage: " << programName << " [options]\n\n"
              << "Required (one of):\n"
              << "  -t, --text <text>       Text to synthesize (direct input)\n"
              << "  -f, --file <path>       Path to .tokens file (pre-tokenized)\n"
              << "  --precache              Pre-cache all voices in assets folder\n\n"
              << "Options:\n"
              << "  -v, --voice <name>      Voice name or path to WAV file\n"
              << "                          (e.g., 'malebrute' or 'assets/malebrute.wav')\n"
              << "                          (default: femaleelfhaughty)\n"
              << "  -o, --output <path>     Output WAV file path (default: output.wav)\n"
              << "  -m, --models <dir>      Models directory (default: models/)\n"
              << "  --dtype <type>          Model dtype: fp32, q8, q4 (default: q4)\n"
              << "  --download              Download models if not present\n"
              << "  --clearcache            Clear voice conditionals cache before running\n"
              << "  -h, --help              Show this help message\n\n"
              << "Examples:\n"
              << "  # Direct text input (uses HuggingFace tokenizer)\n"
              << "  " << programName << " -t \"Hello, how are you today?\"\n\n"
              << "  # With custom voice (by name, uses cache)\n"
              << "  " << programName << " -t \"Hello world\" -v malebrute -o speech.wav\n\n"
              << "  # Pre-cache all voices in assets folder\n"
              << "  " << programName << " --precache\n\n"
              << "  # Clear cache and regenerate\n"
              << "  " << programName << " --clearcache --precache\n";
}

/**
 * @brief Command line configuration
 */
struct Config {
    std::string inputText;                                 // Direct text input
    std::string tokensPath;                               // Path to .tokens file (alternative)
    std::string voicePath = "femaleelfhaughty";           // Voice name or path
    std::string outputPath = "output.wav";                 // Output WAV file
    std::string modelsDir = "models";                      // ONNX models directory
    std::string dtype = "q4";                              // Model quantization type
    bool downloadModels = false;                           // Download models if missing
    bool showHelp = false;                                 // Show help and exit
    bool enableProfiling = false;                          // Enable ONNX profiling
    bool precache = false;                                 // Pre-cache all voices
    bool clearCache = false;                               // Clear cache before running
};

/**
 * @brief Parse command line arguments
 * @return Config struct with parsed values, or std::nullopt on error
 */
std::optional<Config> ParseArgs(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-t" || arg == "--text") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return std::nullopt;
            }
            config.inputText = argv[++i];
        }
        else if (arg == "-f" || arg == "--file") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return std::nullopt;
            }
            config.tokensPath = argv[++i];
        }
        else if (arg == "-v" || arg == "--voice") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return std::nullopt;
            }
            config.voicePath = argv[++i];
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return std::nullopt;
            }
            config.outputPath = argv[++i];
        }
        else if (arg == "-m" || arg == "--models") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return std::nullopt;
            }
            config.modelsDir = argv[++i];
        }
        else if (arg == "--dtype") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                return std::nullopt;
            }
            config.dtype = argv[++i];
            // Validate dtype
            if (config.dtype != "fp32" && config.dtype != "q8" && config.dtype != "q4" && config.dtype != "q4f16") {
                std::cerr << "Error: Invalid dtype '" << config.dtype << "'. Use fp32, q8, q4, or q4f16.\n";
                return std::nullopt;
            }
        }
        else if (arg == "--download") {
            config.downloadModels = true;
        }
        else if (arg == "--profile") {
            config.enableProfiling = true;
        }
        else if (arg == "--precache") {
            config.precache = true;
        }
        else if (arg == "--clearcache") {
            config.clearCache = true;
        }
        else if (arg == "-h" || arg == "--help") {
            config.showHelp = true;
        }
        else {
            std::cerr << "Error: Unknown option '" << arg << "'\n";
            return std::nullopt;
        }
    }
    
    return config;
}

/**
 * @brief Validate configuration and check file existence
 */
bool ValidateConfig(const Config& config) {
    // Precache mode doesn't need text input
    if (config.precache) {
        // Just need models directory
        if (!fs::exists(config.modelsDir) && !config.downloadModels) {
            std::cerr << "Error: Models directory not found: " << config.modelsDir << "\n";
            return false;
        }
        return true;
    }
    
    // Check that at least one input method is provided
    if (config.inputText.empty() && config.tokensPath.empty()) {
        std::cerr << "Error: Either -t (text), -f (tokens file), or --precache is required.\n";
        return false;
    }
    
    // Check tokens file exists (if using file input)
    if (!config.tokensPath.empty() && !fs::exists(config.tokensPath)) {
        std::cerr << "Error: Tokens file not found: " << config.tokensPath << "\n";
        return false;
    }
    
    // Voice path is validated later (may be cache key or file path)
    
    // Check models directory (warn if missing, allow --download to fix)
    if (!fs::exists(config.modelsDir)) {
        if (config.downloadModels) {
            std::cout << "Models directory not found. Will download models.\n";
        } else {
            std::cerr << "Error: Models directory not found: " << config.modelsDir << "\n";
            std::cerr << "Hint: Use --download to download models, or run:\n";
            std::cerr << "      python tools/download_models.py --output-dir " << config.modelsDir << "\n";
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Pre-cache all voice files in assets directory
 */
int RunPrecache(ChatterboxTTS::ChatterboxTTS& tts, ChatterboxTTS::VoiceConditionalsCache& cache, bool clearFirst) {
    std::cout << "\n=== Pre-caching Voice Conditionals ===\n\n";
    
    // Clear cache if requested
    if (clearFirst) {
        std::cout << "Clearing existing cache...\n";
        cache.Clear();
    }
    
    // Find all .wav files in assets folder
    if (!fs::exists(ASSETS_DIR)) {
        std::cerr << "Assets directory not found: " << ASSETS_DIR << "\n";
        return 1;
    }
    
    std::vector<fs::path> wavFiles;
    for (const auto& entry : fs::directory_iterator(ASSETS_DIR)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".xwm") {
                wavFiles.push_back(entry.path());
            }
        }
    }
    
    if (wavFiles.empty()) {
        std::cout << "No .wav or .xwm files found in " << ASSETS_DIR << "\n";
        return 0;
    }
    
    std::cout << "Found " << wavFiles.size() << " voice files to process\n\n";
    
    int processed = 0;
    int skipped = 0;
    int failed = 0;
    
    for (const auto& wavPath : wavFiles) {
        std::string key = ChatterboxTTS::VoiceConditionalsCache::ExtractKey(wavPath.string());
        std::cout << "Processing: " << key << " (" << wavPath.filename().string() << ")...\n";
        
        // Check if already in memory cache
        if (cache.Has(key)) {
            std::cout << "  -> Already in memory cache, skipping\n";
            skipped++;
            continue;
        }
        
        // Check if exists on disk (and not --clearcache)
        if (!clearFirst && cache.ExistsOnDisk(key)) {
            if (cache.LoadFromDisk(key)) {
                std::cout << "  -> Loaded from disk cache\n";
                skipped++;
                continue;
            }
        }
        
        // Need to generate conditionals
        auto startTime = std::chrono::high_resolution_clock::now();
        
        if (!tts.PrepareConditionals(wavPath.string())) {
            std::cerr << "  -> FAILED: " << tts.GetLastError() << "\n";
            failed++;
            continue;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        // Save to cache (memory + disk)
        if (cache.Put(key, tts.GetConditionals())) {
            std::cout << "  -> Cached in " << duration.count() << "ms\n";
            processed++;
        } else {
            std::cerr << "  -> Failed to cache\n";
            failed++;
        }
    }
    
    std::cout << "\n=== Pre-cache Summary ===\n";
    std::cout << "  Processed: " << processed << "\n";
    std::cout << "  Skipped:   " << skipped << " (already cached)\n";
    std::cout << "  Failed:    " << failed << "\n";
    std::cout << "  Total cached: " << cache.Size() << " voices\n";
    
    return (failed > 0) ? 1 : 0;
}

/**
 * @brief Resolve voice path to conditionals (from cache or file)
 */
bool ResolveVoice(const std::string& voicePath, 
                  ChatterboxTTS::ChatterboxTTS& tts,
                  ChatterboxTTS::VoiceConditionalsCache& cache) {
    
    std::string key = ChatterboxTTS::VoiceConditionalsCache::ExtractKey(voicePath);
    
    // 1. Check memory cache
    const auto* cachedConds = cache.Get(key);
    if (cachedConds) {
        std::cout << "Using cached voice conditionals: " << key << "\n";
        tts.SetConditionals(*cachedConds);
        return true;
    }
    
    // 2. Check disk cache
    if (cache.ExistsOnDisk(key)) {
        if (cache.LoadFromDisk(key)) {
            cachedConds = cache.Get(key);
            if (cachedConds) {
                std::cout << "Loaded voice conditionals from disk cache: " << key << "\n";
                tts.SetConditionals(*cachedConds);
                return true;
            }
        }
    }
    
    // 3. Try to find and process the voice file
    std::string actualPath;
    
    // If voicePath looks like a file path
    if (fs::exists(voicePath)) {
        actualPath = voicePath;
    } else {
        // Try to find in assets folder
        for (const auto& ext : {".wav", ".xwm"}) {
            std::string tryPath = std::string(ASSETS_DIR) + "/" + key + ext;
            if (fs::exists(tryPath)) {
                actualPath = tryPath;
                break;
            }
        }
    }
    
    if (actualPath.empty()) {
        std::cerr << "Error: Voice not found in cache and no matching file found for: " << voicePath << "\n";
        std::cerr << "  Looked for: " << key << ".wav/.xwm in " << ASSETS_DIR << "/\n";
        return false;
    }
    
    // Process the voice file
    std::cout << "Processing voice file: " << actualPath << "\n";
    if (!tts.PrepareConditionals(actualPath)) {
        return false;
    }
    
    // Cache for future use
    cache.Put(key, tts.GetConditionals());
    
    return true;
}

int main(int argc, char* argv[]) {
    // Set up logging
    spdlog::set_level(spdlog::level::info);
    
    std::cout << "Chatterbox TTS ONNX Demo v1.0\n";
    std::cout << "==============================\n\n";
    
    // Parse command line arguments
    auto configOpt = ParseArgs(argc, argv);
    if (!configOpt) {
        std::cerr << "\nUse -h or --help for usage information.\n";
        return 1;
    }
    
    Config config = *configOpt;
    
    // Show help if requested
    if (config.showHelp) {
        PrintUsage(argv[0]);
        return 0;
    }
    
    // Check required arguments (precache mode doesn't need text)
    if (!config.precache && config.inputText.empty() && config.tokensPath.empty()) {
        std::cerr << "Error: Either text (-t), tokens file (-f), or --precache is required.\n\n";
        PrintUsage(argv[0]);
        return 1;
    }
    
    // Validate configuration
    if (!ValidateConfig(config)) {
        return 1;
    }
    
    // Initialize voice cache
    ChatterboxTTS::VoiceConditionalsCache voiceCache(CACHE_DIR);
    
    // Handle --clearcache without precache (just clear and exit)
    if (config.clearCache && !config.precache && config.inputText.empty() && config.tokensPath.empty()) {
        std::cout << "Clearing voice conditionals cache...\n";
        voiceCache.Clear();
        std::cout << "Cache cleared.\n";
        return 0;
    }
    
    // Print configuration
    std::cout << "Configuration:\n";
    if (config.precache) {
        std::cout << "  Mode:         Pre-cache voices\n";
    } else if (!config.inputText.empty()) {
        std::cout << "  Input text:   \"" << config.inputText << "\"\n";
    } else {
        std::cout << "  Tokens file:  " << config.tokensPath << "\n";
    }
    if (!config.precache) {
        std::cout << "  Voice:        " << config.voicePath << "\n";
        std::cout << "  Output file:  " << config.outputPath << "\n";
    }
    std::cout << "  Models dir:   " << config.modelsDir << "\n";
    std::cout << "  Model dtype:  " << config.dtype << "\n";
    std::cout << "  Cache dir:    " << CACHE_DIR << "\n";
    std::cout << "\n";
    
    // Start timing
    std::chrono::milliseconds prepareConditionalsEndTime = std::chrono::milliseconds(0);
    std::chrono::milliseconds generation0EndTime = std::chrono::milliseconds(0);
    std::chrono::milliseconds generationEndTime = std::chrono::milliseconds(0);
    std::vector<float> audio;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // ====================================================================
        // Step 1: Download models if requested
        // ====================================================================
        if (config.downloadModels) {
            std::cout << "Downloading ONNX models...\n";
            ChatterboxTTS::ModelDownloader downloader;
            if (!downloader.DownloadChatterboxModels(config.modelsDir, config.dtype)) {
                std::cerr << "Error: Failed to download models\n";
                return 1;
            }
        }
        
        // ====================================================================
        // Step 2: Create TTS engine and load models
        // ====================================================================
        std::cout << "Loading ONNX models...\n";
        if (config.enableProfiling) {
            std::cout << "ONNX profiling enabled - profile will be written after generation\n";
        }
        ChatterboxTTS::ChatterboxTTS tts;
        
        if (!tts.LoadModels(config.modelsDir, config.dtype, 
                           ChatterboxTTS::ExecutionProvider::CPU,
                           config.enableProfiling)) {
            std::cerr << "Error: " << tts.GetLastError() << "\n";
            return 1;
        }
        
        // ====================================================================
        // Handle --precache mode
        // ====================================================================
        if (config.precache) {
            return RunPrecache(tts, voiceCache, config.clearCache);
        }
        
        // ====================================================================
        // Step 3: Prepare voice conditionals (from cache or file)
        // ====================================================================
        auto prepareConditionalsStartTime = std::chrono::high_resolution_clock::now();
        if (!ResolveVoice(config.voicePath, tts, voiceCache)) {
            std::cerr << "Error: " << tts.GetLastError() << "\n";
            return 1;
        }
        prepareConditionalsEndTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - prepareConditionalsStartTime);
        // ====================================================================
        // Step 4: Get tokens (either from text or file)
        // ====================================================================
        ChatterboxTTS::TokenData tokenData;
        
        if (!config.inputText.empty()) {
            // Direct text input - use the tokenizer loaded with models
            std::cout << "Tokenizing text...\n";
            
            if (!tts.HasTokenizer()) {
                std::cerr << "Error: Tokenizer not loaded. Ensure tokenizer.json is in the models directory.\n";
                return 1;
            }
            
            // Tokenize (normalization is done internally)
            tokenData = tts.Tokenize(config.inputText);
            
            if (!tokenData.IsValid()) {
                std::cerr << "Error: " << tts.GetLastError() << "\n";
                return 1;
            }
            
            std::cout << "Normalized text: \"" << tokenData.originalText << "\"\n";
            std::cout << "Tokenized to " << tokenData.tokenIds.size() << " tokens\n";
        } else {
            // Load from pre-tokenized file
            std::cout << "Loading tokens from: " << config.tokensPath << "\n";
            ChatterboxTTS::Tokenizer tokenizer;
            auto loadedData = tokenizer.LoadTokenFile(config.tokensPath);
            if (!loadedData) {
                std::cerr << "Error: " << tokenizer.GetLastError() << "\n";
                return 1;
            }
            tokenData = *loadedData;
            std::cout << "Loaded " << tokenData.tokenIds.size() << " tokens\n";
        }

        // ====================================================================
        // Step 5: Generate speech
        // ====================================================================
     
        std::cout << "(Warmup) Generating speech...\n";
        auto generation0StartTime = std::chrono::high_resolution_clock::now();
        ChatterboxTTS::GenerationConfig genConfig;
        genConfig.maxNewTokens = 1024;
        genConfig.temperature = 0.8f;
        genConfig.topK = 1000;
        genConfig.topP = 0.95f;
        genConfig.repetitionPenalty = 1.2f;
        genConfig.seed = 42;  // Fixed seed for reproducible generation
      
        audio = tts.Generate(tokenData, genConfig, nullptr);
        
        if (audio.empty()) {
            std::cerr << "\nError: " << tts.GetLastError() << "\n";
            return 1;
        }
        audio.clear(); // Discard warmup audio
        generation0EndTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - generation0StartTime);

        std::cout << "Generating speech...\n";
        auto generationStartTime = std::chrono::high_resolution_clock::now();       
        audio = tts.Generate(tokenData, genConfig, nullptr);
       /* audio = tts.Generate(tokenData, genConfig, 
            [](int step, int total) {
                if (step % 50 == 0) {
                    std::cout << "  Step " << step << "/" << total << "\r" << std::flush;
                }
            });
        */
        if (audio.empty()) {
            std::cerr << "\nError: " << tts.GetLastError() << "\n";
            return 1;
        }
        generationEndTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - generationStartTime);

        // End profiling if enabled (writes profile file)
        if (config.enableProfiling) {
            std::string profilePath = tts.EndProfiling();
            if (!profilePath.empty()) {
                std::cout << "Profile written to: " << profilePath << "\n";
            }
        }

        std::cout << "\nGenerated " << audio.size() << " samples ("
                  << (audio.size() / 24000.0f) << " seconds)\n";
        
        // ====================================================================
        // Step 6: Save output WAV
        // ====================================================================
        std::cout << "Saving output to: " << config.outputPath << "\n";
        
        ChatterboxTTS::WavWriter wavWriter;
        ChatterboxTTS::WavFormat format;
        format.sampleRate = 24000;
        format.channels = 1;
        format.bitsPerSample = 16;
        
        if (!wavWriter.WriteFile(config.outputPath, audio, format)) {
            std::cerr << "Error: " << wavWriter.GetLastError() << "\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\nConditionals in " << (prepareConditionalsEndTime.count() / 1000.0) << " seconds\n";
    std::cout << "Warmup generation in " << (generation0EndTime.count() / 1000.0) << " seconds\n";
    std::cout << "Generation in " << (generationEndTime.count() / 1000.0) << " seconds\n";
    std::cout << "Completed in " << (duration.count() / 1000.0) << " seconds\n";
    std::cout << "Output saved to: " << config.outputPath << "\n";
    
    return 0;
}

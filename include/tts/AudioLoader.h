/**
 * @file AudioLoader.h
 * @brief Audio loading utilities using FFmpeg
 * 
 * Loads audio files (WAV, XWM, etc.) and resamples them to the required
 * sample rate for voice encoding.
 */

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <functional>

namespace ChatterboxTTS {

/**
 * @brief Audio data container
 */
struct AudioData {
    std::vector<float> samples;  ///< Audio samples (mono, normalized to [-1, 1])
    int sampleRate;              ///< Sample rate in Hz
    int channels;                ///< Number of channels (always 1 after processing)
    
    /// Get duration in seconds
    float GetDuration() const {
        if (sampleRate <= 0) return 0.0f;
        return static_cast<float>(samples.size()) / sampleRate;
    }
    
    /// Check if audio is valid
    bool IsValid() const {
        return !samples.empty() && sampleRate > 0;
    }
};

/**
 * @brief Configuration for audio loading
 */
struct AudioLoadConfig {
    int targetSampleRate = 16000;  ///< Target sample rate for voice encoder
    bool normalize = true;          ///< Normalize audio to [-1, 1]
    float maxDurationSeconds = 60.0f; ///< Maximum duration to load (0 = no limit)
    bool convertToMono = true;      ///< Convert stereo to mono
};

/**
 * @brief Progress callback for audio loading
 */
using AudioProgressCallback = std::function<void(float progress)>;

/**
 * @brief Audio loading class using FFmpeg
 * 
 * Supports loading various audio formats including WAV, XWM, MP3, etc.
 * Handles resampling to target sample rate and conversion to mono.
 */
class AudioLoader {
public:
    AudioLoader();
    ~AudioLoader();
    
    // Non-copyable
    AudioLoader(const AudioLoader&) = delete;
    AudioLoader& operator=(const AudioLoader&) = delete;
    
    /**
     * @brief Load audio from file
     * @param path Path to audio file
     * @param config Loading configuration
     * @return AudioData or std::nullopt on failure
     */
    std::optional<AudioData> LoadFile(const std::string& path, 
                                       const AudioLoadConfig& config = AudioLoadConfig());
    
    /**
     * @brief Load audio from memory buffer
     * @param data Raw audio data
     * @param size Size of data in bytes
     * @param format Format hint (e.g., "wav", "mp3")
     * @param config Loading configuration
     * @return AudioData or std::nullopt on failure
     */
    std::optional<AudioData> LoadMemory(const uint8_t* data, size_t size,
                                         const std::string& format,
                                         const AudioLoadConfig& config = AudioLoadConfig());
    
    /**
     * @brief Get last error message
     */
    const std::string& GetLastError() const { return m_lastError; }
    
    /**
     * @brief Get supported file extensions
     */
    static std::vector<std::string> GetSupportedExtensions();
    
    /**
     * @brief Check if a file format is supported
     */
    static bool IsFormatSupported(const std::string& extension);
    
private:
    std::string m_lastError;
    
    /**
     * @brief Initialize FFmpeg (called once)
     */
    static void InitFFmpeg();
    
    /**
     * @brief Resample audio to target sample rate
     */
    std::vector<float> Resample(const std::vector<float>& input,
                                int inputSampleRate,
                                int outputSampleRate);
    
    /**
     * @brief Convert multi-channel audio to mono
     */
    std::vector<float> ConvertToMono(const std::vector<float>& input, int channels);
    
    /**
     * @brief Normalize audio samples to [-1, 1]
     */
    void NormalizeSamples(std::vector<float>& samples);
};

/**
 * @brief Utility functions for audio processing
 */
namespace AudioUtils {
    
/**
 * @brief Pad or trim audio to a specific length
 * @param audio Audio samples
 * @param targetLength Target number of samples
 * @return Padded/trimmed audio
 */
std::vector<float> PadOrTrim(const std::vector<float>& audio, size_t targetLength);

/**
 * @brief Apply simple high-pass filter to remove DC offset
 */
void RemoveDCOffset(std::vector<float>& samples);

/**
 * @brief Calculate RMS (root mean square) of audio
 */
float CalculateRMS(const std::vector<float>& samples);

/**
 * @brief Check if audio is mostly silent
 * @param samples Audio samples
 * @param threshold RMS threshold for silence (default 0.01)
 */
bool IsSilent(const std::vector<float>& samples, float threshold = 0.01f);

} // namespace AudioUtils

} // namespace ChatterboxTTS

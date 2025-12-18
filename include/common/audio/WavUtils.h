#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SkyrimNet::Audio {

// WAV format limits: dataSize is uint32_t, and wavSize = 36 + dataSize must also fit
constexpr uint32_t kMaxWavDataSize = std::numeric_limits<uint32_t>::max() - 36;

/**
 * @brief WAV file header structure for 16-bit mono PCM audio
 * 
 * Standard RIFF WAV format header. Use configure() to set sample rate
 * and data size before writing.
 */
#pragma pack(push, 1)
struct WAVHeader {
    char riffHeader[4] = {'R', 'I', 'F', 'F'};
    uint32_t wavSize = 0;
    char waveHeader[4] = {'W', 'A', 'V', 'E'};
    char fmtHeader[4] = {'f', 'm', 't', ' '};
    uint32_t fmtChunkSize = 16;
    uint16_t audioFormat = 1;       // PCM
    uint16_t numChannels = 1;       // Mono
    uint32_t sampleRate = 16000;
    uint32_t byteRate = 16000 * 2;  // SampleRate * NumChannels * BitsPerSample/8
    uint16_t blockAlign = 2;        // NumChannels * BitsPerSample/8
    uint16_t bitsPerSample = 16;
    char dataHeader[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize = 0;
    
    /**
     * @brief Configure header for specific sample rate and data size
     * @param rate Sample rate in Hz (e.g., 16000, 44100)
     * @param dataBytes Size of PCM data in bytes (must be <= kMaxWavDataSize)
     * @throws std::overflow_error if dataBytes exceeds WAV format limits
     * @throws std::invalid_argument if rate is zero
     */
    void configure(uint32_t rate, uint32_t dataBytes) {
        if (rate == 0) {
            throw std::invalid_argument("WAV sample rate cannot be zero");
        }
        if (dataBytes > kMaxWavDataSize) {
            throw std::overflow_error("PCM data size exceeds WAV format limit");
        }
        sampleRate = rate;
        byteRate = rate * 2;  // 16-bit mono = 2 bytes per sample
        dataSize = dataBytes;
        wavSize = 36 + dataSize;
    }
};
#pragma pack(pop)

/**
 * @brief Create WAV file data from raw PCM bytes (16-bit mono)
 * @param pcmData Raw PCM audio data (16-bit samples as bytes)
 * @param sampleRate Sample rate in Hz (default: 16000)
 * @return Complete WAV file as byte vector
 * @throws std::overflow_error if pcmData exceeds WAV format limits
 * @throws std::invalid_argument if sampleRate is zero
 */
inline std::vector<uint8_t> CreateWAVFromPCM(const std::vector<uint8_t>& pcmData, uint32_t sampleRate = 16000) {
    if (pcmData.size() > kMaxWavDataSize) {
        throw std::overflow_error("PCM data size exceeds WAV format limit");
    }
    
    WAVHeader header;
    header.configure(sampleRate, static_cast<uint32_t>(pcmData.size()));

    std::vector<uint8_t> wavData;
    wavData.reserve(sizeof(header) + pcmData.size());
    wavData.insert(wavData.end(), 
                   reinterpret_cast<const uint8_t*>(&header),
                   reinterpret_cast<const uint8_t*>(&header) + sizeof(header));
    wavData.insert(wavData.end(), pcmData.begin(), pcmData.end());

    return wavData;
}

/**
 * @brief Create WAV file data from 16-bit sample vector
 * @param samples PCM audio samples (16-bit signed)
 * @param sampleRate Sample rate in Hz (default: 16000)
 * @return Complete WAV file as byte vector
 * @throws std::overflow_error if samples data exceeds WAV format limits
 * @throws std::invalid_argument if sampleRate is zero
 */
inline std::vector<uint8_t> CreateWAVFromSamples(const std::vector<int16_t>& samples, uint32_t sampleRate = 16000) {
    // Check for overflow: samples.size() * sizeof(int16_t) must fit in uint32_t
    // and must not exceed kMaxWavDataSize
    constexpr size_t kMaxSamples = kMaxWavDataSize / sizeof(int16_t);
    if (samples.size() > kMaxSamples) {
        throw std::overflow_error("Sample count exceeds WAV format limit");
    }
    
    const uint32_t dataBytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    
    WAVHeader header;
    header.configure(sampleRate, dataBytes);

    std::vector<uint8_t> wavData(sizeof(header) + header.dataSize);
    std::memcpy(wavData.data(), &header, sizeof(header));
    std::memcpy(wavData.data() + sizeof(header), samples.data(), header.dataSize);

    return wavData;
}

/**
 * @brief Create WAV file data from float sample vector (-1.0 to 1.0 range)
 * @param samples PCM audio samples as normalized floats
 * @param sampleRate Sample rate in Hz
 * @param volume Volume multiplier applied before conversion (default: 1.0)
 * @return Complete WAV file as byte vector
 * @throws std::overflow_error if samples data exceeds WAV format limits
 * @throws std::invalid_argument if sampleRate is zero
 */
inline std::vector<uint8_t> CreateWAVFromSamples(const std::vector<float>& samples, 
                                                  uint32_t sampleRate,
                                                  float volume = 1.0f) {
    constexpr size_t kMaxSamples = kMaxWavDataSize / sizeof(int16_t);
    if (samples.size() > kMaxSamples) {
        throw std::overflow_error("Sample count exceeds WAV format limit");
    }
    
    std::vector<int16_t> pcm16;
    pcm16.reserve(samples.size());
    for (float x : samples) {
        x *= volume;
        x = std::clamp(x, -1.0f, 1.0f);
        pcm16.push_back(static_cast<int16_t>(x * 32767));
    }
    
    return CreateWAVFromSamples(pcm16, sampleRate);
}

} // namespace SkyrimNet::Audio

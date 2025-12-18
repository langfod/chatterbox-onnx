/**
 * @file WavWriter.cpp
 * @brief Implementation of WAV file writer
 */

#include "tts/WavWriter.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

namespace ChatterboxTTS {

// WAV file format constants
constexpr uint32_t RIFF_HEADER = 0x46464952;  // "RIFF"
constexpr uint32_t WAVE_HEADER = 0x45564157;  // "WAVE"
constexpr uint32_t FMT_HEADER  = 0x20746d66;  // "fmt "
constexpr uint32_t DATA_HEADER = 0x61746164;  // "data"
constexpr uint16_t WAVE_FORMAT_PCM = 1;

WavWriter::~WavWriter() {
    Close();
}

bool WavWriter::WriteFile(const std::string& path,
                           const std::vector<float>& samples,
                           const WavFormat& format) {
    m_lastError.clear();
    
    // Convert to int16
    auto int16Samples = ConvertFloatToInt16(samples);
    return WriteFile(path, int16Samples, format);
}

bool WavWriter::WriteFile(const std::string& path,
                           const std::vector<int16_t>& samples,
                           const WavFormat& format) {
    m_lastError.clear();
    
    if (samples.empty()) {
        m_lastError = "No samples to write";
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Ensure parent directory exists
    fs::path filePath(path);
    if (filePath.has_parent_path()) {
        fs::create_directories(filePath.parent_path());
    }
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        m_lastError = "Failed to open file for writing: " + path;
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Calculate data size
    uint32_t dataSize = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    
    // Write header
    WriteHeader(file, dataSize, format);
    
    // Write audio data
    file.write(reinterpret_cast<const char*>(samples.data()), dataSize);
    
    if (!file) {
        m_lastError = "Failed to write audio data";
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    file.close();
    
    float duration = static_cast<float>(samples.size()) / format.sampleRate;
    spdlog::info("Wrote WAV file: {} ({:.2f}s)", path, duration);
    
    return true;
}

bool WavWriter::Open(const std::string& path, const WavFormat& format) {
    m_lastError.clear();
    
    if (m_file.is_open()) {
        Close();
    }
    
    // Ensure parent directory exists
    fs::path filePath(path);
    if (filePath.has_parent_path()) {
        fs::create_directories(filePath.parent_path());
    }
    
    m_file.open(path, std::ios::binary);
    if (!m_file.is_open()) {
        m_lastError = "Failed to open file for streaming: " + path;
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    m_format = format;
    m_samplesWritten = 0;
    
    // Write placeholder header (will be updated on close)
    WriteHeader(m_file, 0, m_format);
    
    return true;
}

bool WavWriter::WriteSamples(const std::vector<float>& samples) {
    if (!m_file.is_open()) {
        m_lastError = "File not open for streaming";
        return false;
    }
    
    // Convert and write
    for (float sample : samples) {
        int16_t pcm = FloatToInt16(sample);
        m_file.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
    }
    
    m_samplesWritten += samples.size();
    return m_file.good();
}

void WavWriter::Close() {
    if (!m_file.is_open()) {
        return;
    }
    
    // Update header with final size
    UpdateHeader();
    
    m_file.close();
    spdlog::debug("Closed streaming WAV file ({} samples)", m_samplesWritten);
}

void WavWriter::WriteHeader(std::ostream& stream, uint32_t dataSize, const WavFormat& format) {
    // RIFF header
    stream.write(reinterpret_cast<const char*>(&RIFF_HEADER), 4);
    
    // File size (RIFF chunk size = file size - 8)
    uint32_t fileSize = 36 + dataSize;
    stream.write(reinterpret_cast<const char*>(&fileSize), 4);
    
    // WAVE format
    stream.write(reinterpret_cast<const char*>(&WAVE_HEADER), 4);
    
    // fmt chunk
    stream.write(reinterpret_cast<const char*>(&FMT_HEADER), 4);
    
    uint32_t fmtChunkSize = 16;
    stream.write(reinterpret_cast<const char*>(&fmtChunkSize), 4);
    
    uint16_t audioFormat = WAVE_FORMAT_PCM;
    stream.write(reinterpret_cast<const char*>(&audioFormat), 2);
    
    uint16_t numChannels = static_cast<uint16_t>(format.channels);
    stream.write(reinterpret_cast<const char*>(&numChannels), 2);
    
    uint32_t sampleRate = static_cast<uint32_t>(format.sampleRate);
    stream.write(reinterpret_cast<const char*>(&sampleRate), 4);
    
    uint32_t byteRate = sampleRate * numChannels * (format.bitsPerSample / 8);
    stream.write(reinterpret_cast<const char*>(&byteRate), 4);
    
    uint16_t blockAlign = static_cast<uint16_t>(numChannels * (format.bitsPerSample / 8));
    stream.write(reinterpret_cast<const char*>(&blockAlign), 2);
    
    uint16_t bitsPerSample = static_cast<uint16_t>(format.bitsPerSample);
    stream.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
    
    // data chunk
    stream.write(reinterpret_cast<const char*>(&DATA_HEADER), 4);
    stream.write(reinterpret_cast<const char*>(&dataSize), 4);
}

void WavWriter::UpdateHeader() {
    if (!m_file.is_open()) {
        return;
    }
    
    // Calculate final data size
    uint32_t dataSize = static_cast<uint32_t>(m_samplesWritten * sizeof(int16_t));
    uint32_t fileSize = 36 + dataSize;
    
    // Update RIFF chunk size (at offset 4)
    m_file.seekp(4);
    m_file.write(reinterpret_cast<const char*>(&fileSize), 4);
    
    // Update data chunk size (at offset 40)
    m_file.seekp(40);
    m_file.write(reinterpret_cast<const char*>(&dataSize), 4);
}

int16_t WavWriter::FloatToInt16(float sample) {
    // Clamp to [-1, 1]
    sample = std::clamp(sample, -1.0f, 1.0f);
    
    // Convert to int16 range
    return static_cast<int16_t>(sample * 32767.0f);
}

// ============================================================================
// Utility functions
// ============================================================================

std::vector<int16_t> ConvertFloatToInt16(const std::vector<float>& floatSamples) {
    std::vector<int16_t> result;
    result.reserve(floatSamples.size());
    
    for (float sample : floatSamples) {
        // Clamp and convert
        sample = std::clamp(sample, -1.0f, 1.0f);
        result.push_back(static_cast<int16_t>(sample * 32767.0f));
    }
    
    return result;
}

std::vector<float> ConvertInt16ToFloat(const std::vector<int16_t>& int16Samples) {
    std::vector<float> result;
    result.reserve(int16Samples.size());
    
    constexpr float scale = 1.0f / 32768.0f;
    
    for (int16_t sample : int16Samples) {
        result.push_back(static_cast<float>(sample) * scale);
    }
    
    return result;
}

} // namespace ChatterboxTTS

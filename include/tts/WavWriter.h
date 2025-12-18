/**
 * @file WavWriter.h
 * @brief Simple WAV file writer for TTS output
 * 
 * Writes 16-bit PCM WAV files at 24kHz (TTS output format).
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <optional>

namespace ChatterboxTTS {

/**
 * @brief WAV file format configuration
 */
struct WavFormat {
    int sampleRate = 24000;     ///< Sample rate in Hz (TTS outputs at 24kHz)
    int channels = 1;            ///< Number of channels (mono for TTS)
    int bitsPerSample = 16;      ///< Bits per sample (16-bit PCM)
};

/**
 * @brief Simple WAV file writer
 * 
 * Writes audio data to WAV format. Can write all at once or stream
 * samples incrementally.
 */
class WavWriter {
public:
    WavWriter() = default;
    ~WavWriter();
    
    // Non-copyable
    WavWriter(const WavWriter&) = delete;
    WavWriter& operator=(const WavWriter&) = delete;
    
    /**
     * @brief Write float samples to a WAV file
     * @param path Output file path
     * @param samples Audio samples (float, -1.0 to 1.0)
     * @param format WAV format configuration
     * @return true on success
     */
    bool WriteFile(const std::string& path,
                   const std::vector<float>& samples,
                   const WavFormat& format = WavFormat());
    
    /**
     * @brief Write int16 samples to a WAV file
     * @param path Output file path
     * @param samples Audio samples (16-bit PCM)
     * @param format WAV format configuration
     * @return true on success
     */
    bool WriteFile(const std::string& path,
                   const std::vector<int16_t>& samples,
                   const WavFormat& format = WavFormat());
    
    /**
     * @brief Open file for streaming writes
     * @param path Output file path
     * @param format WAV format configuration
     * @return true on success
     */
    bool Open(const std::string& path, const WavFormat& format = WavFormat());
    
    /**
     * @brief Write samples to open stream
     * @param samples Audio samples (float, -1.0 to 1.0)
     * @return true on success
     */
    bool WriteSamples(const std::vector<float>& samples);
    
    /**
     * @brief Close streaming file and finalize header
     */
    void Close();
    
    /**
     * @brief Check if file is open for streaming
     */
    bool IsOpen() const { return m_file.is_open(); }
    
    /**
     * @brief Get total samples written (streaming mode)
     */
    size_t GetSamplesWritten() const { return m_samplesWritten; }
    
    /**
     * @brief Get last error message
     */
    const std::string& GetLastError() const { return m_lastError; }
    
private:
    /**
     * @brief Write WAV header to stream
     */
    void WriteHeader(std::ostream& stream, uint32_t dataSize, const WavFormat& format);
    
    /**
     * @brief Update header with final size (for streaming)
     */
    void UpdateHeader();
    
    /**
     * @brief Convert float sample to int16
     */
    static int16_t FloatToInt16(float sample);
    
    std::ofstream m_file;
    WavFormat m_format;
    size_t m_samplesWritten = 0;
    std::string m_lastError;
};

/**
 * @brief Utility to convert float audio to int16
 */
std::vector<int16_t> ConvertFloatToInt16(const std::vector<float>& floatSamples);

/**
 * @brief Utility to convert int16 audio to float
 */
std::vector<float> ConvertInt16ToFloat(const std::vector<int16_t>& int16Samples);

} // namespace ChatterboxTTS

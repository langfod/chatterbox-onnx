#pragma once

#include <memory>
#include <string>
#include <vector>


extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace SkyrimNet {
namespace Audio {

/**
 * Converts audio data from Fuz format to WAV
 * 
 * @param inputBuffer The input audio buffer
 * @param inputSize Size of the input buffer
 * @param outputBuffer The output WAV buffer (will be allocated)
 * @param outputFilename Optional filename to set in metadata
 * @return True if conversion was successful
 */
bool ConvertFuzToWav(const char* inputBuffer, size_t inputSize, 
                 std::vector<uint8_t>& outputBuffer,
                 const std::string& outputFilename = "");

/**
 * Converts audio data from XWM format to WAV
 * 
 * @param inputBuffer The input audio buffer
 * @param inputSize Size of the input buffer
 * @param outputBuffer The output WAV buffer (will be allocated)
 * @param outputFilename Optional filename to set in metadata
 * @return True if conversion was successful
 */
bool ConvertXwmToWav(const char* inputBuffer, size_t inputSize, 
                 std::vector<uint8_t>& outputBuffer,
                 const std::string& outputFilename = "");

/**
 * Determines if the input buffer is a valid WAV file
 * 
 * @param buffer The input buffer to check
 * @param size Size of the buffer
 * @return True if the buffer contains valid WAV data
 */
bool IsWavFormat(const char* buffer, size_t size);

/**
 * Determines if the input buffer is a valid FUZ file
 * 
 * @param buffer The input buffer to check
 * @param size Size of the buffer
 * @return True if the buffer contains valid FUZ data
 */
bool IsFuzFormat(const char* buffer, size_t size);

/**
 * Determines if the input buffer is a valid XWM file
 * 
 * @param buffer The input buffer to check
 * @param size Size of the buffer
 * @return True if the buffer contains valid XWM data
 */
bool IsXwmFormat(const char* buffer, size_t size);

/**
 * Estimates the duration of an audio file from a buffer
 * 
 * @param inputBuffer The input audio buffer
 * @param inputSize Size of the input buffer
 * @return Duration in seconds, or -1.0 if estimation failed
 */
double EstimateDuration(const char* inputBuffer, size_t inputSize);

} // namespace Audio
} // namespace SkyrimNet 
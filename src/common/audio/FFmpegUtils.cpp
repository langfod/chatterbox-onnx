#include "common/audio/FFmpegUtils.h"
#include "common/utils/Logging.h"
#include "ConfigFlags.h"

#include <cstring>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace SkyrimNet {
namespace Audio {

// Forward declarations of internal functions
bool TryConvertWithFFmpeg(const uint8_t* inputBuffer, size_t inputSize, 
                          std::vector<uint8_t>& outputBuffer,
                          const std::string& outputFilename);



// Custom I/O for reading from memory buffer
struct MemoryData {
    const uint8_t* buffer;
    size_t size;
    size_t position;
};

// Custom read function for memory buffer
static int read_memory(void* opaque, uint8_t* buf, int buf_size) {
    struct MemoryData* data = (struct MemoryData*)opaque;
    size_t remaining = data->size - data->position;
    size_t to_read = (remaining < static_cast<size_t>(buf_size)) ? remaining : static_cast<size_t>(buf_size);
    
    if (to_read <= 0) {
        return AVERROR_EOF;
    }
    
    memcpy(buf, data->buffer + data->position, to_read);
    data->position += to_read;
    
    return static_cast<int>(to_read);
}

// Custom seek function for memory buffer
static int64_t seek_memory(void* opaque, int64_t offset, int whence) {
    struct MemoryData* data = (struct MemoryData*)opaque;
    
    switch (whence) {
        case AVSEEK_SIZE:
            return data->size;
        case SEEK_SET:
            data->position = offset;
            break;
        case SEEK_CUR:
            data->position += offset;
            break;
        case SEEK_END:
            data->position = data->size + offset;
            break;
        default:
            return -1;
    }
    
    data->position = std::min(data->position, data->size);
    return data->position;
}

// Custom I/O for writing to a vector
struct MemoryOutput {
    std::vector<uint8_t>* buffer;
};

// Custom write function for vector
static int write_memory(void* opaque, const uint8_t* buf, int buf_size) {
    struct MemoryOutput* output = (struct MemoryOutput*)opaque;
    size_t current_size = output->buffer->size();
    output->buffer->resize(current_size + buf_size);
    memcpy(output->buffer->data() + current_size, buf, buf_size);
    return buf_size;
}

bool IsWavFormat(const char* buffer, size_t size) {
    if (size < 12) {
        return false;
    }
    
    // Check for WAV signature "RIFF" + size + "WAVE"
    return (memcmp(buffer, "RIFF", 4) == 0 && memcmp(buffer + 8, "WAVE", 4) == 0);
}

bool IsFuzFormat(const char* buffer, size_t size) {
    if (size < 4) {
        return false;
    }
    
    // Check for FUZ signature "FUZE"
    return (memcmp(buffer, "FUZE", 4) == 0);
}

bool IsXwmFormat(const char* buffer, size_t size) {
    if (size < 4) {
        return false;
    }
    
    // Check for XWMA header
    if (memcmp(buffer, "XWMA", 4) == 0) {
        return true;
    }
    
    // Check for RIFF-based XWM (RIFF header with XWMA at bytes 8-11)
    if (size >= 12 && memcmp(buffer, "RIFF", 4) == 0 && 
        memcmp(buffer + 8, "XWMA", 4) == 0) {
        return true;
    }
    
    return false;
}

bool ConvertFuzToWav(const char* inputBuffer, size_t inputSize, 
                 std::vector<uint8_t>& outputBuffer,
                 const std::string& outputFilename) {
    LOG_INFO("Converting FUZ audio data to WAV format");
    
    // Write input FUZ file for debugging
    #ifdef DEBUG_AUDIO_FILE_WRITES
    WriteDebugFile(outputFilename.empty() ? "unknown" : outputFilename, 
                   reinterpret_cast<const uint8_t*>(inputBuffer), inputSize, "fuz");
    #endif
    
    // Clear output buffer
    outputBuffer.clear();
    
    // Check if input is valid
    if (!inputBuffer || inputSize < 12) {
        LOG_ERROR("Invalid FUZ input buffer");
        return false;
    }
    
    // Check FUZ header magic ("FUZE")
    if (memcmp(inputBuffer, "FUZE", 4) != 0) {
        LOG_ERROR("Invalid FUZ format - missing FUZE header");
        return false;
    }
    
    LOG_DEBUG("FUZ header detected");
    uint32_t lipSize = *reinterpret_cast<const uint32_t*>(inputBuffer + 8);
    const uint8_t* currentPos = reinterpret_cast<const uint8_t*>(inputBuffer + 12 + lipSize);
    
    // Calculate XWM data size
    size_t xwmSize = inputSize - (currentPos - reinterpret_cast<const uint8_t*>(inputBuffer));
    if (xwmSize <= 0) {
        LOG_ERROR("No XWM audio data found in FUZ file");
        return false;
    }
    
    LOG_DEBUG("XWM data size: {} bytes", xwmSize);
    
    // Write XWM data for debugging
    #ifdef DEBUG_AUDIO_FILE_WRITES
    WriteDebugFile(outputFilename.empty() ? "unknown" : outputFilename, 
                   currentPos, xwmSize, "xwm");
    #endif
    
    // First try the standard FFmpeg approach
    if (TryConvertWithFFmpeg(currentPos, xwmSize, outputBuffer, outputFilename)) {
        // Write final WAV output for debugging
        #ifdef DEBUG_AUDIO_FILE_WRITES
        WriteDebugFile(outputFilename.empty() ? "unknown" : outputFilename + "_final", 
                       outputBuffer.data(), outputBuffer.size(), "wav");
        #endif
        return true;
    }
    
    return false;
}

/**
 * Attempts to convert XWM to WAV using FFmpeg
 */
bool TryConvertWithFFmpeg(const uint8_t* inputBuffer, size_t inputSize, 
                          std::vector<uint8_t>& outputBuffer,
                          const std::string& outputFilename) {
    // Set up memory input for FFmpeg
    MemoryData inputData = {
        .buffer = inputBuffer,
        .size = inputSize,
        .position = 0
    };
    
    // Set up memory output for FFmpeg
    MemoryOutput outputData = {
        .buffer = &outputBuffer
    };
    
    // Initialize FFmpeg structures
    AVFormatContext* inputFormatCtx = nullptr;
    AVFormatContext* outputFormatCtx = nullptr;
    AVIOContext* inputIO = nullptr;
    AVIOContext* outputIO = nullptr;
    AVStream* inputStream = nullptr;
    AVStream* outputStream = nullptr;
    AVCodecContext* decoderCtx = nullptr;
    AVCodecContext* encoderCtx = nullptr;
    const AVCodec* decoder = nullptr;
    const AVCodec* encoder = nullptr;
    SwrContext* swrCtx = nullptr;
    AVFrame* decodedFrame = nullptr;
    AVFrame* resampledFrame = nullptr;
    AVPacket* packet = nullptr;
    bool success = false;
    
    try {
        // Create input I/O context
        constexpr size_t ioBufferSize = 4096;
        unsigned char* ioBuffer = static_cast<unsigned char*>(av_malloc(ioBufferSize));
        if (!ioBuffer) {
            throw std::runtime_error("Failed to allocate I/O buffer");
        }
        
        inputIO = avio_alloc_context(
            ioBuffer, ioBufferSize, 0, &inputData, read_memory, nullptr, seek_memory
        );
        if (!inputIO) {
            av_free(ioBuffer);
            throw std::runtime_error("Failed to create input I/O context");
        }
        
        // Create input format context
        inputFormatCtx = avformat_alloc_context();
        if (!inputFormatCtx) {
            throw std::runtime_error("Failed to allocate input format context");
        }
        
        inputFormatCtx->pb = inputIO;
        inputFormatCtx->flags |= AVFMT_FLAG_CUSTOM_IO;
        
        // Force XWM format
        const AVInputFormat* xwmFormat = av_find_input_format("xwma");
        int ret;
        
        if (!xwmFormat) {
            LOG_WARN("XWMA format not directly supported, trying alternative approach");
            
            // Try to create a dictionary with forced params for detection
            AVDictionary* options = nullptr;
            av_dict_set(&options, "audio_codec_id", "XWMA", 0);
            
            // Attempt to open without specifying format
            ret = avformat_open_input(&inputFormatCtx, nullptr, nullptr, &options);
            if (options) {
                av_dict_free(&options);
            }
            
            if (ret < 0) {
                // If that fails, try to use 'asf' format which can sometimes handle XWMA
                xwmFormat = av_find_input_format("asf");
                if (!xwmFormat) {
                    throw std::runtime_error("XWMA format not supported by FFmpeg");
                }
                
                // Try with ASF format
                ret = avformat_open_input(&inputFormatCtx, nullptr, xwmFormat, nullptr);
            }
        } else {
            // Open input with XWMA format
            ret = avformat_open_input(&inputFormatCtx, nullptr, xwmFormat, nullptr);
        }
        
        if (ret < 0) {
            char errBuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errBuf, sizeof(errBuf));
            LOG_ERROR("Failed to open input: {}", errBuf);
            throw std::runtime_error("Failed to open input");
        }
        
        // Read stream info
        if (avformat_find_stream_info(inputFormatCtx, nullptr) < 0) {
            throw std::runtime_error("Failed to find stream info");
        }
        
        // Find audio stream
        int audioStreamIndex = -1;
        for (unsigned int i = 0; i < inputFormatCtx->nb_streams; i++) {
            if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audioStreamIndex = i;
                break;
            }
        }
        
        if (audioStreamIndex == -1) {
            throw std::runtime_error("No audio stream found");
        }
        
        inputStream = inputFormatCtx->streams[audioStreamIndex];
        
        // Get decoder
        decoder = avcodec_find_decoder(inputStream->codecpar->codec_id);
        if (!decoder) {
            throw std::runtime_error("Unsupported audio codec");
        }
        
        // Create decoder context
        decoderCtx = avcodec_alloc_context3(decoder);
        if (!decoderCtx) {
            throw std::runtime_error("Failed to allocate decoder context");
        }
        
        // Copy codec parameters
        if (avcodec_parameters_to_context(decoderCtx, inputStream->codecpar) < 0) {
            throw std::runtime_error("Failed to copy codec parameters");
        }
        
        // Open decoder
        if (avcodec_open2(decoderCtx, decoder, nullptr) < 0) {
            throw std::runtime_error("Failed to open decoder");
        }
        
        // Create output I/O context for the WAV file
        unsigned char* outputIoBuffer = static_cast<unsigned char*>(av_malloc(ioBufferSize));
        if (!outputIoBuffer) {
            throw std::runtime_error("Failed to allocate output I/O buffer");
        }
        
        outputIO = avio_alloc_context(
            outputIoBuffer, ioBufferSize, 1, &outputData, nullptr, write_memory, nullptr
        );
        if (!outputIO) {
            av_free(outputIoBuffer);
            throw std::runtime_error("Failed to create output I/O context");
        }
        
        // Create output format context
        ret = avformat_alloc_output_context2(&outputFormatCtx, nullptr, "wav", nullptr);
        if (ret < 0 || !outputFormatCtx) {
            throw std::runtime_error("Failed to create output format context");
        }
        
        outputFormatCtx->pb = outputIO;
        outputFormatCtx->flags |= AVFMT_FLAG_CUSTOM_IO | AVFMT_NOFILE;
        
        // Add audio stream to output
        outputStream = avformat_new_stream(outputFormatCtx, nullptr);
        if (!outputStream) {
            throw std::runtime_error("Failed to create output stream");
        }
        
        // Find PCM encoder
        encoder = avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE);
        if (!encoder) {
            throw std::runtime_error("PCM encoder not found");
        }
        
        // Create encoder context
        encoderCtx = avcodec_alloc_context3(encoder);
        if (!encoderCtx) {
            throw std::runtime_error("Failed to allocate encoder context");
        }
        
        // Set encoder parameters (match PHP sample code: 22050 Hz, mono, s16 format)
        encoderCtx->sample_fmt = AV_SAMPLE_FMT_S16;
        encoderCtx->sample_rate = 22050;
        av_channel_layout_default(&encoderCtx->ch_layout, 1);
        encoderCtx->bit_rate = 16 * encoderCtx->sample_rate * encoderCtx->ch_layout.nb_channels;
        
        // Open encoder
        if (avcodec_open2(encoderCtx, encoder, nullptr) < 0) {
            throw std::runtime_error("Failed to open encoder");
        }
        
        // Copy encoder parameters to output stream
        if (avcodec_parameters_from_context(outputStream->codecpar, encoderCtx) < 0) {
            throw std::runtime_error("Failed to copy encoder parameters");
        }
        
        // Set metadata if filename provided
        if (!outputFilename.empty()) {
            av_dict_set(&outputFormatCtx->metadata, "title", outputFilename.c_str(), 0);
        }

        // Validate that the decoder channel layout is initialized
        if (decoderCtx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
            LOG_TRACE("Decoder channel layout is unspecified, setting to default for {} channels", decoderCtx->ch_layout.nb_channels);
            av_channel_layout_default(&decoderCtx->ch_layout, decoderCtx->ch_layout.nb_channels);
        }

        // Create resampler context
        ret = swr_alloc_set_opts2(&swrCtx,
                                  &encoderCtx->ch_layout, encoderCtx->sample_fmt, encoderCtx->sample_rate,
                                  &decoderCtx->ch_layout, decoderCtx->sample_fmt, decoderCtx->sample_rate,
                                  0, nullptr);
        if (ret < 0 || !swrCtx) {
            throw std::runtime_error("Failed to allocate resampler context");
        }
        if (swr_init(swrCtx) < 0) {
            throw std::runtime_error("Failed to initialize resampler");
        }
        
        // Allocate frames and packet
        decodedFrame = av_frame_alloc();
        resampledFrame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!decodedFrame || !resampledFrame || !packet) {
            throw std::runtime_error("Failed to allocate frames or packet");
        }
        
        // Write WAV header
        ret = avformat_write_header(outputFormatCtx, nullptr);
        if (ret < 0) {
            char errBuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errBuf, sizeof(errBuf));
            LOG_ERROR("Failed to write header: {}", errBuf);
            throw std::runtime_error("Failed to write WAV header");
        }
        
        // Decoding loop
        while (av_read_frame(inputFormatCtx, packet) >= 0) {
            if (packet->stream_index == audioStreamIndex) {
                // Send packet to decoder
                ret = avcodec_send_packet(decoderCtx, packet);
                if (ret < 0) {
                    LOG_WARN("Error sending packet to decoder");
                    continue;
                }
                
                // Receive decoded frames
                while (ret >= 0) {
                    ret = avcodec_receive_frame(decoderCtx, decodedFrame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        LOG_WARN("Error during decoding");
                        break;
                    }
                    
                    // Check if we need to reconfigure the resampler due to format changes
                    if (decodedFrame->format != decoderCtx->sample_fmt ||
                        decodedFrame->sample_rate != decoderCtx->sample_rate ||
                        av_channel_layout_compare(&decodedFrame->ch_layout, &decoderCtx->ch_layout) != 0) {
                        
                        LOG_DEBUG("Frame format differs from decoder context, reconfiguring resampler");
                        swr_free(&swrCtx);
                        
                        ret = swr_alloc_set_opts2(&swrCtx,
                                                  &encoderCtx->ch_layout, encoderCtx->sample_fmt, encoderCtx->sample_rate,
                                                  &decodedFrame->ch_layout, (AVSampleFormat)decodedFrame->format, decodedFrame->sample_rate,
                                                  0, nullptr);
                        if (ret < 0 || !swrCtx) {
                            LOG_DEBUG("Failed to allocate resampler context during reconfiguration");
                            break;
                        }
                        if (swr_init(swrCtx) < 0) {
                            LOG_DEBUG("Failed to initialize resampler during reconfiguration");
                            break;
                        }
                    }
                    
                    // Prepare resampled frame
                    av_frame_unref(resampledFrame);
                    resampledFrame->format = encoderCtx->sample_fmt;
                    resampledFrame->sample_rate = encoderCtx->sample_rate;
                    ret = av_channel_layout_copy(&resampledFrame->ch_layout, &encoderCtx->ch_layout);
                    if (ret < 0) {
                        LOG_DEBUG("Failed to copy channel layout");
                        break;
                    }
                    
                    // Calculate output sample count
                    int64_t delay = swr_get_delay(swrCtx, decodedFrame->sample_rate);
                    int64_t estimated_samples = av_rescale_rnd(delay + decodedFrame->nb_samples,
                                                                encoderCtx->sample_rate, 
                                                                decodedFrame->sample_rate,
                                                                AV_ROUND_UP);
                    if (estimated_samples > INT_MAX) {
                        LOG_ERROR("Estimated sample count exceeds INT_MAX: {}", estimated_samples);
                        break;
                    }
                    resampledFrame->nb_samples = static_cast<int>(estimated_samples);
                    
                    // Allocate resampled frame buffer
                    if (av_frame_get_buffer(resampledFrame, 0) < 0) {
                        LOG_DEBUG("Failed to allocate resampled frame buffer");
                        break;
                    }
                    
                    // Resample using lower-level API
                    int samples_out = swr_convert(swrCtx,
                                                  resampledFrame->data, resampledFrame->nb_samples,
                                                  (const uint8_t**)decodedFrame->data, decodedFrame->nb_samples);
                    if (samples_out < 0) {
                        char errBuf[AV_ERROR_MAX_STRING_SIZE];
                        av_strerror(samples_out, errBuf, sizeof(errBuf));
                        LOG_DEBUG("Error resampling audio - error code: {}, message: {}", samples_out, errBuf);
                        break;
                    }
                    
                    // Update the actual number of samples produced
                    resampledFrame->nb_samples = samples_out;
                    
                    // Skip if no samples were produced
                    if (samples_out == 0) {
                        LOG_DEBUG("No samples produced by resampler, skipping frame");
                        av_frame_unref(decodedFrame);
                        av_frame_unref(resampledFrame);
                        continue;
                    }
                    
                    // Encode resampled frame
                    if (avcodec_send_frame(encoderCtx, resampledFrame) < 0) {
                        LOG_ERROR("Error sending frame to encoder");
                        break;
                    }
                    
                    // Get encoded packets
                    while (true) {
                        ret = avcodec_receive_packet(encoderCtx, packet);
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                            break;
                        } else if (ret < 0) {
                            LOG_ERROR("Error during encoding");
                            break;
                        }
                        
                        // Write packet to output
                        if (av_write_frame(outputFormatCtx, packet) < 0) {
                            LOG_ERROR("Error writing frame");
                            break;
                        }
                    }
                    
                    av_frame_unref(decodedFrame);
                    av_frame_unref(resampledFrame);
                }
            }
            
            av_packet_unref(packet);
        }
        
        // Flush encoder
        avcodec_send_frame(encoderCtx, nullptr);
        while (true) {
            ret = avcodec_receive_packet(encoderCtx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                LOG_ERROR("Error during encoding flush");
                break;
            }
            
            if (av_write_frame(outputFormatCtx, packet) < 0) {
                LOG_ERROR("Error writing frame during flush");
                break;
            }
            
            av_packet_unref(packet);
        }
        
        // Write trailer
        ret = av_write_trailer(outputFormatCtx);
        if (ret < 0) {
            char errBuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errBuf, sizeof(errBuf));
            LOG_ERROR("Failed to write trailer: {}", errBuf);
            throw std::runtime_error("Failed to write WAV trailer");
        }
        
        success = true;
        LOG_INFO("Audio conversion completed successfully, output size: {} bytes", outputBuffer.size());
    } 
    catch (const std::exception& e) {
        LOG_ERROR("Error converting XWM to WAV with FFmpeg: {}", e.what());
        success = false;
    }
    
    // Cleanup
    LOG_TRACE("Cleaning up FFmpeg resources");
    if (packet) av_packet_free(&packet);
    if (decodedFrame) av_frame_free(&decodedFrame);
    if (resampledFrame) av_frame_free(&resampledFrame);
    if (swrCtx) swr_free(&swrCtx);
    if (decoderCtx) avcodec_free_context(&decoderCtx);
    if (encoderCtx) avcodec_free_context(&encoderCtx);
    if (inputFormatCtx) avformat_close_input(&inputFormatCtx);
    LOG_TRACE("Closing output format context");
    if (outputFormatCtx) {
        if (!(outputFormatCtx->flags & AVFMT_NOFILE) && outputFormatCtx->pb) {
            avio_close(outputFormatCtx->pb);
        } else if (outputIO) {
            if (outputIO->buffer) av_freep(&outputIO->buffer);
            avio_context_free(&outputIO);
        }
        avformat_free_context(outputFormatCtx);
    } else if (outputIO) {
        if (outputIO->buffer) av_freep(&outputIO->buffer);
        avio_context_free(&outputIO);
    }
    LOG_TRACE("Closing input format context");
    if (inputIO) {
        if (inputIO->buffer) av_freep(&inputIO->buffer);
        avio_context_free(&inputIO);
    }
    LOG_TRACE("FFmpeg resources closed");
    return success;
}

bool ConvertXwmToWav(const char* inputBuffer, size_t inputSize, 
                 std::vector<uint8_t>& outputBuffer,
                 const std::string& outputFilename) {
    LOG_INFO("Converting XWM audio data to WAV format");
    
    // Write input XWM file for debugging
    #ifdef DEBUG_AUDIO_FILE_WRITES
    WriteDebugFile(outputFilename.empty() ? "unknown" : outputFilename, 
                   reinterpret_cast<const uint8_t*>(inputBuffer), inputSize, "xwm");
    #endif
    
    // Clear output buffer
    outputBuffer.clear();
    
    // Check if input is valid
    if (!inputBuffer || inputSize < 4) {
        LOG_ERROR("Invalid XWM input buffer");
        return false;
    }
    
    // Check XWM header magic - multiple possible formats
    bool validHeader = false;
    
    // Check for XWMA header
    if (memcmp(inputBuffer, "XWMA", 4) == 0) {
        validHeader = true;
    }
    // Check for RIFF-based XWM (RIFF header with XWMA at bytes 8-11)
    else if (inputSize >= 12 && memcmp(inputBuffer, "RIFF", 4) == 0 && 
             memcmp(inputBuffer + 8, "XWMA", 4) == 0) {
        validHeader = true;
    }
    
    if (!validHeader) {
        LOG_ERROR("Invalid XWM format - missing valid header (XWMA or RIFF XWMA)");
        return false;
    }
    
    LOG_DEBUG("XWM header detected");
    
    // Use FFmpeg to convert XWM to WAV
    if (TryConvertWithFFmpeg(reinterpret_cast<const uint8_t*>(inputBuffer), 
                            inputSize, outputBuffer, outputFilename)) {
        LOG_INFO("Successfully converted XWM to WAV ({} bytes)", outputBuffer.size());
        return true;
    } else {
        LOG_ERROR("Failed to convert XWM to WAV using FFmpeg");
        return false;
    }
}

double EstimateDuration(const char* inputBuffer, size_t inputSize) {
    if (!inputBuffer || inputSize < 4) {
        LOG_ERROR("Invalid input buffer for duration estimation");
        return -1.0;
    }
    
    // Set up memory input for FFmpeg
    MemoryData inputData = {
        .buffer = reinterpret_cast<const uint8_t*>(inputBuffer),
        .size = inputSize,
        .position = 0
    };
    
    // Initialize FFmpeg structures
    AVFormatContext* formatCtx = nullptr;
    AVIOContext* ioCtx = nullptr;
    double duration = -1.0;
    
    try {
        // Create input I/O context
        constexpr size_t ioBufferSize = 4096;
        unsigned char* ioBuffer = static_cast<unsigned char*>(av_malloc(ioBufferSize));
        if (!ioBuffer) {
            throw std::runtime_error("Failed to allocate I/O buffer");
        }
        
        ioCtx = avio_alloc_context(
            ioBuffer, ioBufferSize, 0, &inputData, read_memory, nullptr, seek_memory
        );
        if (!ioCtx) {
            throw std::runtime_error("Failed to create I/O context");
        }
        
        // Create format context
        formatCtx = avformat_alloc_context();
        if (!formatCtx) {
            throw std::runtime_error("Failed to allocate format context");
        }
        
        formatCtx->pb = ioCtx;
        formatCtx->flags |= AVFMT_FLAG_CUSTOM_IO;
        
        // Handle FUZ files specially - they're containers with LIP + XWM data
        if (IsFuzFormat(inputBuffer, inputSize)) {
            if (inputSize < 12) {
                throw std::runtime_error("FUZ file too small");
            }
            
            uint32_t lipSize = *reinterpret_cast<const uint32_t*>(inputBuffer + 8);
            const char* xwmData = inputBuffer + 12 + lipSize;
            size_t xwmSize = inputSize - (12 + lipSize);
            
            if (xwmSize <= 0) {
                throw std::runtime_error("No XWM data in FUZ file");
            }
            
            // Recursively call EstimateDuration on the XWM portion
            return EstimateDuration(xwmData, xwmSize);
        }
        
        // For all other formats, let FFmpeg auto-detect
        int ret = avformat_open_input(&formatCtx, nullptr, nullptr, nullptr);
        if (ret < 0) {
            char errBuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errBuf, sizeof(errBuf));
            throw std::runtime_error(std::string("Failed to open input: ") + errBuf);
        }
        
        // Find stream info
        ret = avformat_find_stream_info(formatCtx, nullptr);
        if (ret < 0) {
            char errBuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errBuf, sizeof(errBuf));
            throw std::runtime_error(std::string("Failed to find stream info: ") + errBuf);
        }
        
        // Get duration from format context
        if (formatCtx->duration != AV_NOPTS_VALUE) {
            duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
        } else {
            // Try to get duration from the first audio stream
            for (unsigned int i = 0; i < formatCtx->nb_streams; i++) {
                AVStream* stream = formatCtx->streams[i];
                if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                    if (stream->duration != AV_NOPTS_VALUE) {
                        duration = static_cast<double>(stream->duration) * av_q2d(stream->time_base);
                        break;
                    }
                }
            }
        }
        
        if (duration <= 0.0) {
            LOG_WARN("Could not determine duration from metadata");
            duration = -1.0;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error estimating duration: {}", e.what());
        duration = -1.0;
    }
    
    // Cleanup
    if (formatCtx) {
        avformat_close_input(&formatCtx);
    }
    if (ioCtx) {
        if (ioCtx->buffer) av_freep(&ioCtx->buffer);
        avio_context_free(&ioCtx);
    }
    
    return duration;
}

} // namespace Audio
} // namespace SkyrimNet 
/**
 * @file AudioLoader.cpp
 * @brief Implementation of audio loading using FFmpeg
 * 
 * Uses shared FFmpegUtils for format detection
 */

#include "tts/AudioLoader.h"
#include "common/audio/FFmpegUtils.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

// FFmpeg headers (C linkage)
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

namespace fs = std::filesystem;

namespace ChatterboxTTS {

// Import format detection from shared FFmpegUtils
using SkyrimNet::Audio::IsWavFormat;
using SkyrimNet::Audio::IsFuzFormat;
using SkyrimNet::Audio::IsXwmFormat;

// Static initialization flag
static std::once_flag s_ffmpegInitFlag;

void AudioLoader::InitFFmpeg() {
    std::call_once(s_ffmpegInitFlag, []() {
        // Note: av_register_all() is deprecated in FFmpeg 4.0+
        // Modern FFmpeg auto-registers formats
        spdlog::debug("FFmpeg initialized");
    });
}

AudioLoader::AudioLoader() {
    InitFFmpeg();
}

AudioLoader::~AudioLoader() = default;

std::optional<AudioData> AudioLoader::LoadFile(const std::string& path,
                                                 const AudioLoadConfig& config) {
    m_lastError.clear();
    
    // Check file exists
    if (!fs::exists(path)) {
        m_lastError = "File not found: " + path;
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    spdlog::info("Loading audio: {}", path);
    
    // Open input file
    AVFormatContext* formatCtx = nullptr;
    if (avformat_open_input(&formatCtx, path.c_str(), nullptr, nullptr) < 0) {
        m_lastError = "Failed to open audio file: " + path;
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // RAII cleanup for format context
    struct FormatContextGuard {
        AVFormatContext*& ctx;
        ~FormatContextGuard() { if (ctx) avformat_close_input(&ctx); }
    } formatGuard{formatCtx};
    
    // Get stream info
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        m_lastError = "Failed to find stream info";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Find audio stream
    int audioStreamIndex = -1;
    const AVCodec* codec = nullptr;
    
    for (unsigned i = 0; i < formatCtx->nb_streams; ++i) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
            codec = avcodec_find_decoder(formatCtx->streams[i]->codecpar->codec_id);
            break;
        }
    }
    
    if (audioStreamIndex < 0 || !codec) {
        m_lastError = "No audio stream found in file";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    AVStream* audioStream = formatCtx->streams[audioStreamIndex];
    
    // Create codec context
    AVCodecContext* codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        m_lastError = "Failed to allocate codec context";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    struct CodecContextGuard {
        AVCodecContext*& ctx;
        ~CodecContextGuard() { if (ctx) avcodec_free_context(&ctx); }
    } codecGuard{codecCtx};
    
    if (avcodec_parameters_to_context(codecCtx, audioStream->codecpar) < 0) {
        m_lastError = "Failed to copy codec parameters";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        m_lastError = "Failed to open codec";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Get channel layout
    AVChannelLayout outChLayout;
    av_channel_layout_default(&outChLayout, 1); // Mono output
    
    AVChannelLayout inChLayout;
    if (codecCtx->ch_layout.nb_channels > 0) {
        av_channel_layout_copy(&inChLayout, &codecCtx->ch_layout);
    } else {
        av_channel_layout_default(&inChLayout, codecCtx->ch_layout.nb_channels > 0 ? 
                                   codecCtx->ch_layout.nb_channels : 2);
    }
    
    int inputSampleRate = codecCtx->sample_rate;
    int inputChannels = codecCtx->ch_layout.nb_channels;
    
    spdlog::debug("Input: {} Hz, {} channels, format {}", 
                  inputSampleRate, inputChannels, 
                  av_get_sample_fmt_name(codecCtx->sample_fmt));
    
    // Create resampler
    SwrContext* swrCtx = nullptr;
    if (swr_alloc_set_opts2(&swrCtx,
                            &outChLayout,
                            AV_SAMPLE_FMT_FLT,
                            config.targetSampleRate,
                            &inChLayout,
                            codecCtx->sample_fmt,
                            inputSampleRate,
                            0, nullptr) < 0) {
        m_lastError = "Failed to allocate resampler";
        spdlog::error("{}", m_lastError);
        av_channel_layout_uninit(&outChLayout);
        av_channel_layout_uninit(&inChLayout);
        return std::nullopt;
    }
    
    struct SwrContextGuard {
        SwrContext*& ctx;
        ~SwrContextGuard() { if (ctx) swr_free(&ctx); }
    } swrGuard{swrCtx};
    
    if (swr_init(swrCtx) < 0) {
        m_lastError = "Failed to initialize resampler";
        spdlog::error("{}", m_lastError);
        av_channel_layout_uninit(&outChLayout);
        av_channel_layout_uninit(&inChLayout);
        return std::nullopt;
    }
    
    // Allocate frames
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    
    struct FramePacketGuard {
        AVFrame*& frame;
        AVPacket*& packet;
        ~FramePacketGuard() {
            if (frame) av_frame_free(&frame);
            if (packet) av_packet_free(&packet);
        }
    } fpGuard{frame, packet};
    
    // Calculate max samples to read
    size_t maxSamples = 0;
    if (config.maxDurationSeconds > 0) {
        maxSamples = static_cast<size_t>(config.maxDurationSeconds * config.targetSampleRate);
    }
    
    // Try to estimate duration for better pre-allocation (uses SkyrimNet::Audio::EstimateDuration)
    size_t reserveSize = config.targetSampleRate * 30; // Default ~30 seconds
    if (formatCtx->duration != AV_NOPTS_VALUE) {
        double duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
        reserveSize = static_cast<size_t>(duration * config.targetSampleRate * 1.1); // +10% margin
    }
    
    // Read and decode frames
    std::vector<float> outputSamples;
    outputSamples.reserve(reserveSize);
    
    while (av_read_frame(formatCtx, packet) >= 0) {
        if (packet->stream_index != audioStreamIndex) {
            av_packet_unref(packet);
            continue;
        }
        
        if (avcodec_send_packet(codecCtx, packet) < 0) {
            av_packet_unref(packet);
            continue;
        }
        
        while (avcodec_receive_frame(codecCtx, frame) >= 0) {
            // Estimate output samples
            int64_t outSamples = av_rescale_rnd(
                swr_get_delay(swrCtx, inputSampleRate) + frame->nb_samples,
                config.targetSampleRate,
                inputSampleRate,
                AV_ROUND_UP
            );
            
            // Allocate temporary output buffer
            std::vector<float> tempBuffer(static_cast<size_t>(outSamples));
            uint8_t* outPtr = reinterpret_cast<uint8_t*>(tempBuffer.data());
            
            // Convert
            int converted = swr_convert(
                swrCtx,
                &outPtr,
                static_cast<int>(outSamples),
                const_cast<const uint8_t**>(frame->extended_data),
                frame->nb_samples
            );
            
            if (converted > 0) {
                outputSamples.insert(outputSamples.end(), 
                                     tempBuffer.begin(), 
                                     tempBuffer.begin() + converted);
            }
            
            av_frame_unref(frame);
            
            // Check max duration
            if (maxSamples > 0 && outputSamples.size() >= maxSamples) {
                outputSamples.resize(maxSamples);
                break;
            }
        }
        
        av_packet_unref(packet);
        
        if (maxSamples > 0 && outputSamples.size() >= maxSamples) {
            break;
        }
    }
    
    // Flush resampler
    int flushed = swr_convert(swrCtx, nullptr, 0, nullptr, 0);
    (void)flushed;
    
    // Cleanup channel layouts
    av_channel_layout_uninit(&outChLayout);
    av_channel_layout_uninit(&inChLayout);
    
    if (outputSamples.empty()) {
        m_lastError = "No audio data decoded";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Normalize if requested
    if (config.normalize) {
        NormalizeSamples(outputSamples);
    }
    
    AudioData result;
    result.samples = std::move(outputSamples);
    result.sampleRate = config.targetSampleRate;
    result.channels = 1;
    
    spdlog::info("Loaded audio: {:.2f}s @ {} Hz", result.GetDuration(), result.sampleRate);
    
    return result;
}

// Memory I/O structures (same pattern as FFmpegUtils)
namespace {
    struct MemoryReadData {
        const uint8_t* buffer;
        size_t size;
        size_t position;
    };
    
    int read_memory_buffer(void* opaque, uint8_t* buf, int buf_size) {
        MemoryReadData* data = static_cast<MemoryReadData*>(opaque);
        size_t remaining = data->size - data->position;
        size_t to_read = std::min(remaining, static_cast<size_t>(buf_size));
        
        if (to_read <= 0) {
            return AVERROR_EOF;
        }
        
        memcpy(buf, data->buffer + data->position, to_read);
        data->position += to_read;
        return static_cast<int>(to_read);
    }
    
    int64_t seek_memory_buffer(void* opaque, int64_t offset, int whence) {
        MemoryReadData* data = static_cast<MemoryReadData*>(opaque);
        
        switch (whence) {
            case AVSEEK_SIZE:
                return static_cast<int64_t>(data->size);
            case SEEK_SET:
                data->position = static_cast<size_t>(offset);
                break;
            case SEEK_CUR:
                data->position += static_cast<size_t>(offset);
                break;
            case SEEK_END:
                data->position = data->size + static_cast<size_t>(offset);
                break;
            default:
                return -1;
        }
        
        data->position = std::min(data->position, data->size);
        return static_cast<int64_t>(data->position);
    }
} // anonymous namespace

std::optional<AudioData> AudioLoader::LoadMemory(const uint8_t* data, size_t size,
                                                   const std::string& format,
                                                   const AudioLoadConfig& config) {
    m_lastError.clear();
    
    if (!data || size == 0) {
        m_lastError = "Invalid memory buffer";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    spdlog::info("Loading audio from memory ({} bytes, format hint: {})", size, format);
    
    // Detect format from magic bytes if not specified
    const char* bufferAsChar = reinterpret_cast<const char*>(data);
    std::string detectedFormat = format;
    if (detectedFormat.empty()) {
        if (IsWavFormat(bufferAsChar, size)) {
            detectedFormat = "wav";
        } else if (IsFuzFormat(bufferAsChar, size)) {
            detectedFormat = "fuz";
        } else if (IsXwmFormat(bufferAsChar, size)) {
            detectedFormat = "xwma";
        }
        if (!detectedFormat.empty()) {
            spdlog::debug("Detected format from magic bytes: {}", detectedFormat);
        }
    }
    
    // Handle FUZ files - extract XWM portion first
    const uint8_t* audioData = data;
    size_t audioSize = size;
    
    if (IsFuzFormat(bufferAsChar, size)) {
        if (size < 12) {
            m_lastError = "FUZ file too small";
            spdlog::error("{}", m_lastError);
            return std::nullopt;
        }
        
        uint32_t lipSize = *reinterpret_cast<const uint32_t*>(data + 8);
        audioData = data + 12 + lipSize;
        audioSize = size - (12 + lipSize);
        detectedFormat = "xwma";
        
        spdlog::debug("Extracted XWM from FUZ: {} bytes (lip data: {} bytes)", audioSize, lipSize);
    }
    
    // Set up memory I/O for FFmpeg
    MemoryReadData memData = {
        .buffer = audioData,
        .size = audioSize,
        .position = 0
    };
    
    constexpr size_t ioBufferSize = 4096;
    unsigned char* ioBuffer = static_cast<unsigned char*>(av_malloc(ioBufferSize));
    if (!ioBuffer) {
        m_lastError = "Failed to allocate I/O buffer";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    AVIOContext* ioCtx = avio_alloc_context(
        ioBuffer, ioBufferSize, 0, &memData, read_memory_buffer, nullptr, seek_memory_buffer
    );
    if (!ioCtx) {
        av_free(ioBuffer);
        m_lastError = "Failed to create I/O context";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // RAII cleanup for I/O context
    struct IOContextGuard {
        AVIOContext*& ctx;
        ~IOContextGuard() {
            if (ctx) {
                if (ctx->buffer) av_freep(&ctx->buffer);
                avio_context_free(&ctx);
            }
        }
    } ioGuard{ioCtx};
    
    // Create format context
    AVFormatContext* formatCtx = avformat_alloc_context();
    if (!formatCtx) {
        m_lastError = "Failed to allocate format context";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    formatCtx->pb = ioCtx;
    formatCtx->flags |= AVFMT_FLAG_CUSTOM_IO;
    
    struct FormatContextGuard {
        AVFormatContext*& ctx;
        ~FormatContextGuard() { if (ctx) avformat_close_input(&ctx); }
    } formatGuard{formatCtx};
    
    // Try to open with format hint
    const AVInputFormat* inputFormat = nullptr;
    if (!detectedFormat.empty()) {
        inputFormat = av_find_input_format(detectedFormat.c_str());
    }
    
    int ret = avformat_open_input(&formatCtx, nullptr, inputFormat, nullptr);
    if (ret < 0) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errBuf, sizeof(errBuf));
        m_lastError = "Failed to open audio: " + std::string(errBuf);
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Get stream info
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        m_lastError = "Failed to find stream info";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Find audio stream
    int audioStreamIndex = -1;
    const AVCodec* codec = nullptr;
    
    for (unsigned i = 0; i < formatCtx->nb_streams; ++i) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
            codec = avcodec_find_decoder(formatCtx->streams[i]->codecpar->codec_id);
            break;
        }
    }
    
    if (audioStreamIndex < 0 || !codec) {
        m_lastError = "No audio stream found";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    AVStream* audioStream = formatCtx->streams[audioStreamIndex];
    
    // Create codec context
    AVCodecContext* codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        m_lastError = "Failed to allocate codec context";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    struct CodecContextGuard {
        AVCodecContext*& ctx;
        ~CodecContextGuard() { if (ctx) avcodec_free_context(&ctx); }
    } codecGuard{codecCtx};
    
    if (avcodec_parameters_to_context(codecCtx, audioStream->codecpar) < 0) {
        m_lastError = "Failed to copy codec parameters";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        m_lastError = "Failed to open codec";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Get channel layout
    AVChannelLayout outChLayout;
    av_channel_layout_default(&outChLayout, 1); // Mono output
    
    AVChannelLayout inChLayout;
    if (codecCtx->ch_layout.nb_channels > 0) {
        av_channel_layout_copy(&inChLayout, &codecCtx->ch_layout);
    } else {
        av_channel_layout_default(&inChLayout, codecCtx->ch_layout.nb_channels > 0 ? 
                                   codecCtx->ch_layout.nb_channels : 2);
    }
    
    int inputSampleRate = codecCtx->sample_rate;
    int inputChannels = codecCtx->ch_layout.nb_channels;
    
    spdlog::debug("Input: {} Hz, {} channels, format {}", 
                  inputSampleRate, inputChannels, 
                  av_get_sample_fmt_name(codecCtx->sample_fmt));
    
    // Create resampler
    SwrContext* swrCtx = nullptr;
    if (swr_alloc_set_opts2(&swrCtx,
                            &outChLayout,
                            AV_SAMPLE_FMT_FLT,
                            config.targetSampleRate,
                            &inChLayout,
                            codecCtx->sample_fmt,
                            inputSampleRate,
                            0, nullptr) < 0) {
        m_lastError = "Failed to allocate resampler";
        spdlog::error("{}", m_lastError);
        av_channel_layout_uninit(&outChLayout);
        av_channel_layout_uninit(&inChLayout);
        return std::nullopt;
    }
    
    struct SwrContextGuard {
        SwrContext*& ctx;
        ~SwrContextGuard() { if (ctx) swr_free(&ctx); }
    } swrGuard{swrCtx};
    
    if (swr_init(swrCtx) < 0) {
        m_lastError = "Failed to initialize resampler";
        spdlog::error("{}", m_lastError);
        av_channel_layout_uninit(&outChLayout);
        av_channel_layout_uninit(&inChLayout);
        return std::nullopt;
    }
    
    // Allocate frames
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    
    struct FramePacketGuard {
        AVFrame*& frame;
        AVPacket*& packet;
        ~FramePacketGuard() {
            if (frame) av_frame_free(&frame);
            if (packet) av_packet_free(&packet);
        }
    } fpGuard{frame, packet};
    
    // Calculate max samples to read
    size_t maxSamples = 0;
    if (config.maxDurationSeconds > 0) {
        maxSamples = static_cast<size_t>(config.maxDurationSeconds * config.targetSampleRate);
    }
    
    // Try to estimate duration for better pre-allocation
    size_t reserveSize = config.targetSampleRate * 30; // Default ~30 seconds
    if (formatCtx->duration != AV_NOPTS_VALUE) {
        double duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
        reserveSize = static_cast<size_t>(duration * config.targetSampleRate * 1.1); // +10% margin
    }
    
    // Read and decode frames
    std::vector<float> outputSamples;
    outputSamples.reserve(reserveSize);
    
    while (av_read_frame(formatCtx, packet) >= 0) {
        if (packet->stream_index != audioStreamIndex) {
            av_packet_unref(packet);
            continue;
        }
        
        if (avcodec_send_packet(codecCtx, packet) < 0) {
            av_packet_unref(packet);
            continue;
        }
        
        while (avcodec_receive_frame(codecCtx, frame) >= 0) {
            // Estimate output samples
            int64_t outSamples = av_rescale_rnd(
                swr_get_delay(swrCtx, inputSampleRate) + frame->nb_samples,
                config.targetSampleRate,
                inputSampleRate,
                AV_ROUND_UP
            );
            
            // Allocate temporary output buffer
            std::vector<float> tempBuffer(static_cast<size_t>(outSamples));
            uint8_t* outPtr = reinterpret_cast<uint8_t*>(tempBuffer.data());
            
            // Convert
            int converted = swr_convert(
                swrCtx,
                &outPtr,
                static_cast<int>(outSamples),
                const_cast<const uint8_t**>(frame->extended_data),
                frame->nb_samples
            );
            
            if (converted > 0) {
                outputSamples.insert(outputSamples.end(), 
                                     tempBuffer.begin(), 
                                     tempBuffer.begin() + converted);
            }
            
            av_frame_unref(frame);
            
            // Check max duration
            if (maxSamples > 0 && outputSamples.size() >= maxSamples) {
                outputSamples.resize(maxSamples);
                break;
            }
        }
        
        av_packet_unref(packet);
        
        if (maxSamples > 0 && outputSamples.size() >= maxSamples) {
            break;
        }
    }
    
    // Flush resampler
    int flushed = swr_convert(swrCtx, nullptr, 0, nullptr, 0);
    (void)flushed;
    
    // Cleanup channel layouts
    av_channel_layout_uninit(&outChLayout);
    av_channel_layout_uninit(&inChLayout);
    
    if (outputSamples.empty()) {
        m_lastError = "No audio data decoded";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Normalize if requested
    if (config.normalize) {
        NormalizeSamples(outputSamples);
    }
    
    AudioData result;
    result.samples = std::move(outputSamples);
    result.sampleRate = config.targetSampleRate;
    result.channels = 1;
    
    spdlog::info("Loaded audio from memory: {:.2f}s @ {} Hz", result.GetDuration(), result.sampleRate);
    
    return result;
}

std::vector<std::string> AudioLoader::GetSupportedExtensions() {
    return {
        ".wav", ".wave",
        ".mp3",
        ".ogg", ".oga",
        ".flac",
        ".aac", ".m4a",
        ".wma",
        ".xwm",  // Skyrim audio format
        ".fuz",  // Skyrim lip-sync format (contains XWM)
    };
}

bool AudioLoader::IsFormatSupported(const std::string& extension) {
    auto exts = GetSupportedExtensions();
    std::string lower = extension;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (!lower.empty() && lower[0] != '.') {
        lower = "." + lower;
    }
    return std::find(exts.begin(), exts.end(), lower) != exts.end();
}

std::vector<float> AudioLoader::Resample(const std::vector<float>& input,
                                          int inputSampleRate,
                                          int outputSampleRate) {
    if (inputSampleRate == outputSampleRate) {
        return input;
    }
    
    // Simple linear interpolation resampling
    // For better quality, FFmpeg's swresample is used in LoadFile
    double ratio = static_cast<double>(outputSampleRate) / inputSampleRate;
    size_t outSize = static_cast<size_t>(input.size() * ratio);
    
    std::vector<float> output(outSize);
    
    for (size_t i = 0; i < outSize; ++i) {
        double srcPos = i / ratio;
        size_t srcIdx = static_cast<size_t>(srcPos);
        double frac = srcPos - srcIdx;
        
        if (srcIdx + 1 < input.size()) {
            output[i] = static_cast<float>(
                input[srcIdx] * (1.0 - frac) + input[srcIdx + 1] * frac
            );
        } else {
            output[i] = input.back();
        }
    }
    
    return output;
}

std::vector<float> AudioLoader::ConvertToMono(const std::vector<float>& input, int channels) {
    if (channels == 1) {
        return input;
    }
    
    size_t samples = input.size() / channels;
    std::vector<float> mono(samples);
    
    for (size_t i = 0; i < samples; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < channels; ++ch) {
            sum += input[i * channels + ch];
        }
        mono[i] = sum / channels;
    }
    
    return mono;
}

void AudioLoader::NormalizeSamples(std::vector<float>& samples) {
    if (samples.empty()) return;
    
    // Find max absolute value
    float maxAbs = 0.0f;
    for (float s : samples) {
        maxAbs = std::max(maxAbs, std::abs(s));
    }
    
    // Normalize if needed
    if (maxAbs > 1e-6f && maxAbs != 1.0f) {
        float scale = 1.0f / maxAbs;
        for (float& s : samples) {
            s *= scale;
        }
    }
}

// ============================================================================
// AudioUtils namespace
// ============================================================================

namespace AudioUtils {

std::vector<float> PadOrTrim(const std::vector<float>& audio, size_t targetLength) {
    if (audio.size() == targetLength) {
        return audio;
    }
    
    std::vector<float> result(targetLength, 0.0f);
    
    if (audio.size() > targetLength) {
        // Trim - take from the middle or start?
        // For voice, usually take from start
        std::copy(audio.begin(), audio.begin() + targetLength, result.begin());
    } else {
        // Pad with zeros at the end
        std::copy(audio.begin(), audio.end(), result.begin());
    }
    
    return result;
}

void RemoveDCOffset(std::vector<float>& samples) {
    if (samples.empty()) return;
    
    // Calculate mean
    double sum = 0.0;
    for (float s : samples) {
        sum += s;
    }
    float mean = static_cast<float>(sum / samples.size());
    
    // Subtract mean
    for (float& s : samples) {
        s -= mean;
    }
}

float CalculateRMS(const std::vector<float>& samples) {
    if (samples.empty()) return 0.0f;
    
    double sumSquares = 0.0;
    for (float s : samples) {
        sumSquares += s * s;
    }
    
    return std::sqrt(static_cast<float>(sumSquares / samples.size()));
}

bool IsSilent(const std::vector<float>& samples, float threshold) {
    return CalculateRMS(samples) < threshold;
}

} // namespace AudioUtils

} // namespace ChatterboxTTS

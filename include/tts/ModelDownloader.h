/**
 * @file ModelDownloader.h
 * @brief Download ONNX models from HuggingFace Hub
 * 
 * Simple curl-based downloader for fetching model files from HuggingFace.
 * Supports progress display and handles .onnx + .onnx_data file pairs.
 */

#pragma once

#include <string>
#include <functional>
#include <vector>

namespace ChatterboxTTS {

/**
 * @brief Progress callback function type
 * @param downloaded Bytes downloaded so far
 * @param total Total bytes (0 if unknown)
 * @param filename Name of file being downloaded
 */
using ProgressCallback = std::function<void(size_t downloaded, size_t total, const std::string& filename)>;

/**
 * @brief Model file descriptor
 */
struct ModelFile {
    std::string subfolder;      ///< Subfolder in repo (e.g., "onnx")
    std::string filename;       ///< Filename (e.g., "speech_encoder_q4.onnx")
    bool hasDataFile;           ///< Whether there's an associated .onnx_data file
};

/**
 * @brief Download models from HuggingFace Hub
 */
class ModelDownloader {
public:
    /**
     * @brief Construct downloader with default settings
     */
    ModelDownloader();
    
    /**
     * @brief Destructor
     */
    ~ModelDownloader();

    /**
     * @brief Set the local cache directory
     * @param path Directory to store downloaded models
     */
    void SetCacheDir(const std::string& path);

    /**
     * @brief Set HuggingFace access token
     * @param token HF_TOKEN for private repos (or empty for public)
     */
    void SetToken(const std::string& token);

    /**
     * @brief Set progress callback
     * @param callback Function to call with download progress
     */
    void SetProgressCallback(ProgressCallback callback);

    /**
     * @brief Download a single file from HuggingFace
     * @param repoId Repository ID (e.g., "ResembleAI/chatterbox-turbo-ONNX")
     * @param subfolder Subfolder in repo (e.g., "onnx") or empty
     * @param filename File to download
     * @param localPath Local destination path
     * @return true if successful
     */
    bool DownloadFile(const std::string& repoId, 
                      const std::string& subfolder,
                      const std::string& filename,
                      const std::string& localPath);

    /**
     * @brief Download file only if it doesn't exist locally
     * @param repoId Repository ID
     * @param subfolder Subfolder in repo
     * @param filename File to download
     * @param localPath Local destination path
     * @return true if file exists or download succeeded
     */
    bool DownloadIfNotExists(const std::string& repoId,
                             const std::string& subfolder,
                             const std::string& filename,
                             const std::string& localPath);

    /**
     * @brief Download all q4 Chatterbox TTS models
     * @param localDir Directory to store models
     * @param dtype Model dtype suffix ("q4", "q8", "fp32")
     * @return true if all downloads succeeded
     */
    bool DownloadChatterboxModels(const std::string& localDir, const std::string& dtype = "q4");

    /**
     * @brief Check if all required models exist locally
     * @param localDir Models directory
     * @param dtype Model dtype suffix
     * @return true if all models present
     */
    bool ModelsExist(const std::string& localDir, const std::string& dtype = "q4") const;

    /**
     * @brief Get HuggingFace download URL for a file
     * @param repoId Repository ID
     * @param subfolder Subfolder or empty
     * @param filename Filename
     * @return Full URL
     */
    static std::string GetHuggingFaceUrl(const std::string& repoId,
                                         const std::string& subfolder,
                                         const std::string& filename);

    /**
     * @brief Get list of model files for Chatterbox TTS
     * @param dtype Model dtype suffix
     * @return Vector of ModelFile descriptors
     */
    static std::vector<ModelFile> GetChatterboxModelFiles(const std::string& dtype = "q4");

    /// Default HuggingFace repo for Chatterbox ONNX models
    static constexpr const char* CHATTERBOX_REPO_ID = "ResembleAI/chatterbox-turbo-ONNX";

private:
    /**
     * @brief Internal curl download implementation
     */
    bool DownloadWithCurl(const std::string& url, const std::string& localPath);

    std::string m_cacheDir;
    std::string m_token;
    ProgressCallback m_progressCallback;
};

} // namespace ChatterboxTTS

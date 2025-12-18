#pragma once

#include "common/network/DownloadManager.h"
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>


namespace SkyrimNet {

/**
 * @brief Singleton class to manage downloads of large model files, providing feedback on progress, and keeping a list of active downloads.
 * 
 */
class ModelDownloadManager {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the ModelDownloadManager instance
     */
    static ModelDownloadManager& GetInstance();


    /**
     * @brief Start a download for any model
     * @param modelName The name of the model, as may be displayed to the user
     * @param downloadUrl The URL to download from
     * @param modelPath The local path to save the model
     * @param callback Callback function called when download completes or fails
     * @return Whether the download was started
     */
    bool StartDownload(
        const std::string& modelName,
        const std::string& downloadUrl,
        const std::string& modelPath,
        std::function<void(bool success, std::string modelPath, std::string errorMessage)> callback
    );

    struct DownloadRequest {
        std::string modelName;
        std::string downloadUrl;
        std::string modelPath;
    };

    bool StartDownload(
        const std::vector<DownloadRequest>& requests,
        std::function<void(bool success, std::string errorMessage)> callback
    );

private:
    // Private constructor for singleton
    ModelDownloadManager();
    
    // Deleted copy constructor and assignment operator
    ModelDownloadManager(const ModelDownloadManager&) = delete;
    ModelDownloadManager& operator=(const ModelDownloadManager&) = delete;
    
    // Destructor
    ~ModelDownloadManager() = default;


    // Interval for progress notifications
    static constexpr std::chrono::seconds progress_notification_interval{6};

    enum ModelDownloadStatus{
        InProgress,
        Failed,
        Complete
    };

    struct ModelDownloadInfo {
        std::string modelName;
        std::string downloadUrl;
        std::string modelPath; 

        ModelDownloadStatus status = ModelDownloadStatus::InProgress;
        std::string errorMessage;   
        
        size_t downloadedBytes;
        size_t totalBytes;
        
        // Time when the download started
        std::chrono::system_clock::time_point startTime;
        // Time when the download ended 
        std::chrono::system_clock::time_point endTime;
    };

    void ShowProgressNotification(    );

    
    /**
     * @brief Update the download progress for a model
     * @param model The model being downloaded
     * @param downloadedBytes The number of bytes downloaded so far
     * @param totalBytes The total size of the model file
     */
    void ReportProgress(
        const std::string& modelName,
        const size_t downloadedBytes,
        const size_t totalBytes
    );  

    /**
     * @brief Report an error during model download
     * @param model The model being downloaded
     * @param errorMessage The error message to report
     */
    void ReportError(
        const std::string& modelName,
        const std::string& errorMessage
    );


    void ReportSuccess(const std::string& modelName);

    /**
     * @brief Get the download URL for a model
     * @param modelName The name of the model
     * @return The full URL to download the model from
     */
    std::string GetModelDownloadUrl(const std::string& modelName) const;

    /**
     * @brief Get the filename for a model
     * @param modelName The name of the model
     * @return The filename (e.g., "ggml-large-v3-turbo.bin")
     */
    std::string GetModelFilename(const std::string& modelName) const;

    /**
     * Runtime status & history of model downloads   
     */
    std::unordered_map<std::string, ModelDownloadInfo> downloads; // modelName => ModelDownloadInfo
    std::mutex downloadsMutex;
    std::chrono::system_clock::time_point lastNotificationTime;

   
};

} // namespace SkyrimNet

#pragma once

#include "common/core/ThreadPool.h"
#include "common/network/HttpInterface.h"
#include <curl/curl.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace SkyrimNet {
namespace Network {

// Error codes specific to download operations
enum class DownloadError {
    None,
    InvalidUrl,
    ConnectionFailed,
    HttpError,
    WriteError,
    Timeout,
    Cancelled,
    Unknown
};

// Result structure for download operations
struct DownloadResult {
    bool success;
    std::string filePath;     // Path to the downloaded file
    std::string errorMessage; // Error message if success is false
    DownloadError errorCode;  // Error code if success is false
    long httpCode;            // HTTP status code (if applicable)
    size_t bytesDownloaded;   // Number of bytes downloaded

    DownloadResult() 
        : success(false)
        , errorCode(DownloadError::None)
        , httpCode(0)
        , bytesDownloaded(0) 
    {}
};

// Callback type definition for download completion
using DownloadCallback = std::function<void(const DownloadResult& result)>;

// Progress callback type definition
// Returns true to continue download, false to abort
using ProgressCallback = std::function<bool(size_t downloadedBytes, size_t totalBytes)>;

class DownloadManager {
public:
    /**
     * @brief Get the singleton instance of the DownloadManager
     * @return Reference to the DownloadManager instance
     */
    static DownloadManager& GetInstance();

    /**
     * @brief Initialize the DownloadManager
     * @param config Optional HTTP request configuration (uses defaults if not provided)
     * @return True if initialization succeeded, false otherwise
     */
    bool Initialize(const HttpRequestConfig& config = HttpRequestConfig());

    /**
     * @brief Shutdown the DownloadManager
     */
    void Shutdown();

    /**
     * @brief Download a file asynchronously
     * @param url URL to download from
     * @param targetPath Path where the file should be saved
     * @param callback Callback function to be called when download completes or fails
     * @param progressCallback Optional callback for download progress updates
     * @param timeout Timeout in milliseconds (0 means no timeout)
     * @return Unique ID for the download operation or 0 if the operation could not be started
     */
    uint64_t DownloadFileAsync(
        const std::string& url,
        const std::string& targetPath,
        DownloadCallback callback,
        ProgressCallback progressCallback = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(0)
    );

    /**
     * @brief Cancel a specific download operation
     * @param downloadId ID of the download operation to cancel
     * @return True if the operation was found and cancelled, false otherwise
     */
    bool CancelDownload(uint64_t downloadId);

    /**
     * @brief Cancel all active download operations
     */
    void CancelAllDownloads();

    /**
     * @brief Get the number of active downloads
     * @return The count of currently active downloads
     */
    size_t GetActiveDownloadCount() const;

    /**
     * @brief Download a file only if it doesn't already exist
     * @param url URL to download from
     * @param targetPath Path where the file should be saved
     * @param callback Callback function to be called when download completes or fails
     * @param progressCallback Optional callback for download progress updates
     * @param timeout Timeout in milliseconds (0 means no timeout)
     * @return Unique ID for the download operation, 0 if file already exists, or error starting download
     */
    uint64_t DownloadFileIfNotExists(
        const std::string& url,
        const std::string& targetPath,
        DownloadCallback callback,
        SkyrimNet::Network::ProgressCallback progressCallback = nullptr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(0)
    );

private:
    // Private constructor for singleton
    DownloadManager();
    // Deleted copy constructor and assignment operator
    DownloadManager(const DownloadManager&) = delete;
    DownloadManager& operator=(const DownloadManager&) = delete;
    // Destructor
    ~DownloadManager();

    // Structure to hold download operation data
    struct DownloadOperation {
        std::string url;
        std::string targetPath;
        DownloadCallback callback;
        ProgressCallback progressCallback;
        FILE* file;
        CURL* curl;
        uint64_t downloadId;
        bool isCancelled;
        size_t bytesDownloaded;
        size_t totalBytes;

        DownloadOperation() 
            : file(nullptr)
            , curl(nullptr)
            , downloadId(0)
            , isCancelled(false)
            , bytesDownloaded(0)
            , totalBytes(0)
        {}
    };

    // Helper functions for CURL operations
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    static int ProgressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow);
    void PerformDownload(std::shared_ptr<DownloadOperation> operation);

    // Member variables
    bool m_initialized;
    HttpRequestConfig m_config;
    std::atomic<uint64_t> m_nextDownloadId;
    mutable std::mutex m_activeDownloadsMutex;
    std::vector<uint64_t> m_activeDownloads;

    // Task name for the ThreadPool
    static constexpr const char* DOWNLOAD_TASK_NAME = "FileDownload";
};

} // namespace Network
} // namespace SkyrimNet
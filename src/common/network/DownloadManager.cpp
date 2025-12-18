#include "BuildConstants.h"
#include "common/network/DownloadManager.h"
#include <filesystem>
#include "common/utils/Logging.h"

namespace SkyrimNet {
namespace Network {

DownloadManager& DownloadManager::GetInstance() {
    static DownloadManager instance;
    return instance;
}

DownloadManager::DownloadManager() 
    : m_initialized(false)
    , m_nextDownloadId(1) 
{
}

DownloadManager::~DownloadManager() {
    Shutdown();
}

bool DownloadManager::Initialize(const HttpRequestConfig& config) {
    if (m_initialized) {
        return true;
    }

    LOG_INFO("[DownloadManager] Initializing");
    
    // Store the configuration
    m_config = config;
    
    // Initialize CURL globally
    CURLcode result = curl_global_init(CURL_GLOBAL_ALL);
    if (result != CURLE_OK) {
        LOG_ERROR("[DownloadManager] Failed to initialize CURL: {}", curl_easy_strerror(result));
        return false;
    }

    m_initialized = true;
    LOG_INFO("[DownloadManager] Initialized successfully");
    return true;
}

void DownloadManager::Shutdown() {
    if (!m_initialized) {
        return;
    }

    LOG_INFO("[DownloadManager] Shutting down");
    
    // Cancel any active downloads
    CancelAllDownloads();
    
    // Clean up CURL
    curl_global_cleanup();
    
    m_initialized = false;
}

size_t DownloadManager::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto operation = static_cast<DownloadOperation*>(userp);
    
    // Check if the download has been cancelled
    if (operation->isCancelled) {
        LOG_DEBUG("[DownloadManager] Download cancelled during write callback");
        return 0; // Return 0 to abort the transfer
    }
    
    // Calculate the real size
    size_t realSize = size * nmemb;
    
    // Write the data to the file
    size_t written = fwrite(contents, size, nmemb, operation->file);
    
    // Update bytes downloaded counter using realSize
    if (written > 0) {
        operation->bytesDownloaded += realSize;
    }
    
    // Return bytes actually written or 0 to abort
    return written > 0 ? written : 0;
}

int DownloadManager::ProgressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    auto operation = static_cast<DownloadOperation*>(clientp);
    
    // Mark ultotal and ulnow as unused to suppress warnings
    (void)ultotal;
    (void)ulnow;
    
    // Check if the download has been cancelled
    if (operation->isCancelled) {
        LOG_DEBUG("[DownloadManager] Download cancelled during progress callback");
        return 1; // Return non-zero to abort the transfer
    }
    
    // Update total size if available
    if (dltotal > 0) {
        operation->totalBytes = static_cast<size_t>(dltotal);
    }
    
    // Update downloaded size
    operation->bytesDownloaded = static_cast<size_t>(dlnow);
    
    // Call the progress callback if provided
    if (operation->progressCallback) {
        bool shouldContinue = operation->progressCallback(
            operation->bytesDownloaded, 
            operation->totalBytes
        );
        
        if (!shouldContinue) {
            LOG_DEBUG("[DownloadManager] Download aborted by progress callback");
            operation->isCancelled = true;
            return 1; // Return non-zero to abort the transfer
        }
    }
    
    return 0; // Return 0 to continue
}

uint64_t DownloadManager::DownloadFileAsync(
    const std::string& url,
    const std::string& targetPath,
    DownloadCallback callback,
    SkyrimNet::Network::ProgressCallback progressCallback,
    std::chrono::milliseconds timeout
) {
    if (!m_initialized) {
        LOG_ERROR("[DownloadManager] Attempted to download file before initializing");
        DownloadResult result;
        result.success = false;
        result.filePath = targetPath;
        result.errorMessage = "DownloadManager not initialized";
        result.errorCode = DownloadError::Unknown;
        if (callback) {
            callback(result);
        }
        return 0;
    }
    
    // Create a shared_ptr to the download operation
    auto operation = std::make_shared<DownloadOperation>();
    operation->url = url;
    operation->targetPath = targetPath;
    operation->callback = callback;
    operation->progressCallback = progressCallback;
    operation->downloadId = m_nextDownloadId++;
    
    // Ensure target directory exists
    std::filesystem::path filePath(targetPath);
    try {
        std::filesystem::create_directories(filePath.parent_path());
    } catch (const std::exception& e) {
        LOG_ERROR("[DownloadManager] Failed to create directory: {}", e.what());
        DownloadResult result;
        result.success = false;
        result.filePath = targetPath;
        result.errorMessage = std::string("Failed to create directory: ") + e.what();
        result.errorCode = DownloadError::WriteError;
        if (callback) {
            callback(result);
        }
        return 0;
    }
    
    // Add to active downloads
    {
        std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
        m_activeDownloads.push_back(operation->downloadId);
    }
    
    // Enqueue the download task
    try {
        ThreadPool::getInstance().enqueue(
            DOWNLOAD_TASK_NAME,
            [this, operation]() {
                this->PerformDownload(operation);
            },
            url,  // Use the URL as the task key
            timeout.count() > 0 ? timeout : std::chrono::seconds(30)  // Default 30 second timeout
        );
    } catch (const std::exception& e) {
        LOG_ERROR("[DownloadManager] Failed to enqueue download task: {}", e.what());
        DownloadResult result;
        result.success = false;
        result.filePath = targetPath;
        result.errorMessage = std::string("Failed to enqueue download task: ") + e.what();
        result.errorCode = DownloadError::Unknown;
        if (callback) {
            callback(result);
        }
        
        // Remove from active downloads
        {
            std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
            auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), operation->downloadId);
            if (it != m_activeDownloads.end()) {
                m_activeDownloads.erase(it);
            }
        }
        
        return 0;
    }
    
    LOG_DEBUG("[DownloadManager] Started download of {} to {} (ID: {})", url, targetPath, operation->downloadId);
    return operation->downloadId;
}

void DownloadManager::PerformDownload(std::shared_ptr<DownloadOperation> operation) {
    DownloadResult result;
    result.filePath = operation->targetPath;

    // Open the file for writing using platform-specific safe methods
    std::string tempPath = operation->targetPath + ".tmp";

    // Remove leftover temp file if it exists
    try {
        std::filesystem::remove(tempPath);
    } catch (...) {
        // Ignore deletion errors
    }

    FILE* file = nullptr;
#ifdef _WIN32
    // On Windows, use the safe fopen_s
    errno_t err = fopen_s(&file, tempPath.c_str(), "wb");
    if (err != 0 || file == nullptr) {
        LOG_ERROR("[DownloadManager] Failed to open file for writing - code: {}", std::to_string(err));
        file = nullptr; // Ensure file is null on error
    }
#else
    // On other platforms, use regular fopen
    file = fopen(tempPath.c_str(), "wb");
#endif

    operation->file = file;
    if (!operation->file) {
        LOG_ERROR("[DownloadManager] Failed to open file for writing: {}", tempPath);
        result.success = false;
        result.errorMessage = "Failed to open file for writing";
        result.errorCode = DownloadError::WriteError;
        
        // Remove from active downloads
        {
            std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
            auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), operation->downloadId);
            if (it != m_activeDownloads.end()) {
                m_activeDownloads.erase(it);
            }
        }
        
        if (operation->callback) {
            operation->callback(result);
        }
        return;
    }
    
    // Initialize CURL handle for this download
    operation->curl = curl_easy_init();
    if (!operation->curl) {
        LOG_ERROR("[DownloadManager] Failed to initialize CURL handle");
        fclose(operation->file);
        
        // Try to delete the partial file
        try {
            std::filesystem::remove(tempPath);
        } catch (...) {
            // Ignore deletion errors
        }
        
        result.success = false;
        result.errorMessage = "Failed to initialize CURL handle";
        result.errorCode = DownloadError::Unknown;
        
        // Remove from active downloads
        {
            std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
            auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), operation->downloadId);
            if (it != m_activeDownloads.end()) {
                m_activeDownloads.erase(it);
            }
        }
        
        if (operation->callback) {
            operation->callback(result);
        }
        return;
    }
    
    // Set up CURL options
    curl_easy_setopt(operation->curl, CURLOPT_URL, operation->url.c_str());
    curl_easy_setopt(operation->curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(operation->curl, CURLOPT_WRITEDATA, operation.get());
    curl_easy_setopt(operation->curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
    curl_easy_setopt(operation->curl, CURLOPT_XFERINFODATA, operation.get());
    curl_easy_setopt(operation->curl, CURLOPT_NOPROGRESS, 0L); // Enable progress tracking
    curl_easy_setopt(operation->curl, CURLOPT_FOLLOWLOCATION, m_config.followRedirects ? 1L : 0L);
    curl_easy_setopt(operation->curl, CURLOPT_FAILONERROR, 1L); // Fail on HTTP errors
    curl_easy_setopt(operation->curl, CURLOPT_SSL_VERIFYPEER, /*m_config.verifySSL ? 1L : */0L);
    curl_easy_setopt(operation->curl, CURLOPT_SSL_VERIFYHOST, 2L); // Verify host
    
    // Configure SSL certificate verification
    //if (m_config.verifySSL) {
    //    if (!m_config.caBundlePath.empty()) {
    //        // Use custom CA bundle if provided
    //        curl_easy_setopt(operation->curl, CURLOPT_CAINFO, m_config.caBundlePath.c_str());
    //    } else {
    //        // Fall back to Windows native certificate store
    //        curl_easy_setopt(operation->curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
    //    }
    //}
    
    std::string userAgent = "SkyrimNet/" + std::string(SkyrimNet::Build::VERSION_STRING);
    curl_easy_setopt(operation->curl, CURLOPT_USERAGENT, userAgent.c_str());
    
    // Perform the download
    LOG_DEBUG("[DownloadManager] Executing CURL download for: {}", operation->url);
    CURLcode res = curl_easy_perform(operation->curl);
    
    // Check for cancellation
    if (operation->isCancelled) {
        LOG_DEBUG("[DownloadManager] Download was cancelled: {}", operation->url);
        fclose(operation->file);
        curl_easy_cleanup(operation->curl);
        
        // Try to delete the partial file
        try {
            std::filesystem::remove(tempPath);
        } catch (...) {
            // Ignore deletion errors
        }
        
        result.success = false;
        result.errorMessage = "Download was cancelled";
        result.errorCode = DownloadError::Cancelled;
        result.bytesDownloaded = operation->bytesDownloaded;
        
        // Remove from active downloads
        {
            std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
            auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), operation->downloadId);
            if (it != m_activeDownloads.end()) {
                m_activeDownloads.erase(it);
            }
        }
        
        if (operation->callback) {
            operation->callback(result);
        }
        return;
    }
    
    // Check for CURL errors
    if (res != CURLE_OK) {
        LOG_ERROR("[DownloadManager] Download failed: {}", curl_easy_strerror(res));
        fclose(operation->file);
        curl_easy_cleanup(operation->curl);
        
        // Try to delete the partial file
        try {
            std::filesystem::remove(tempPath);
        } catch (...) {
            // Ignore deletion errors
        }
        
        result.success = false;
        result.errorMessage = std::string("Download failed: ") + curl_easy_strerror(res);
        result.bytesDownloaded = operation->bytesDownloaded;
        
        // Map CURL error to our error code
        switch (res) {
            case CURLE_URL_MALFORMAT:
                result.errorCode = DownloadError::InvalidUrl;
                break;
            case CURLE_COULDNT_CONNECT:
            case CURLE_COULDNT_RESOLVE_HOST:
            case CURLE_COULDNT_RESOLVE_PROXY:
                result.errorCode = DownloadError::ConnectionFailed;
                break;
            case CURLE_WRITE_ERROR:
                result.errorCode = DownloadError::WriteError;
                break;
            case CURLE_OPERATION_TIMEDOUT:
                result.errorCode = DownloadError::Timeout;
                break;
            case CURLE_HTTP_RETURNED_ERROR:
                result.errorCode = DownloadError::HttpError;
                break;
            default:
                result.errorCode = DownloadError::Unknown;
                break;
        }
        
        // Get HTTP code if available
        long httpCode = 0;
        curl_easy_getinfo(operation->curl, CURLINFO_RESPONSE_CODE, &httpCode);
        result.httpCode = httpCode;
        
        // Remove from active downloads
        {
            std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
            auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), operation->downloadId);
            if (it != m_activeDownloads.end()) {
                m_activeDownloads.erase(it);
            }
        }
        
        if (operation->callback) {
            operation->callback(result);
        }
        return;
    }
    
    // Success path
    fclose(operation->file);
    
    // Get HTTP code
    long httpCode = 0;
    curl_easy_getinfo(operation->curl, CURLINFO_RESPONSE_CODE, &httpCode);
    
    // Get total size
    double dlSize = 0.0;
    curl_easy_getinfo(operation->curl, CURLINFO_SIZE_DOWNLOAD, &dlSize);
    
    curl_easy_cleanup(operation->curl);

    // Rename the temporary file to the final target path
    try {
        std::filesystem::rename(tempPath, operation->targetPath);
    } catch (const std::exception& e) {
        LOG_ERROR("[DownloadManager] Failed to rename temp file: {}", e.what());
        try {
            std::filesystem::remove(tempPath);
        } catch (...) {}

        result.success = false;
        result.errorMessage = std::string("Failed to rename temp file: ") + e.what();
        result.errorCode = DownloadError::WriteError;

        // Remove from active downloads
        {
            std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
            auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), operation->downloadId);
            if (it != m_activeDownloads.end()) {
                m_activeDownloads.erase(it);
            }
        }

        if (operation->callback) {
            operation->callback(result);
        }
        return;
    }

    result.success = true;
    result.httpCode = httpCode;
    result.bytesDownloaded = static_cast<size_t>(dlSize);
    
    LOG_INFO("[DownloadManager] Download completed successfully: {} ({})", 
        operation->url, 
        result.bytesDownloaded);
    
    // Remove from active downloads
    {
        std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
        auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), operation->downloadId);
        if (it != m_activeDownloads.end()) {
            m_activeDownloads.erase(it);
        }
    }
    
    if (operation->callback) {
        operation->callback(result);
    }
}

bool DownloadManager::CancelDownload(uint64_t downloadId) {
    if (!m_initialized) {
        LOG_ERROR("[DownloadManager] Attempted to cancel download before initializing");
        return false;
    }
    
    bool found = false;
    
    // Check if the download is in the active list
    {
        std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
        auto it = std::find(m_activeDownloads.begin(), m_activeDownloads.end(), downloadId);
        if (it != m_activeDownloads.end()) {
            found = true;
        }
    }
    
    if (found) {
        // Try to cancel via ThreadPool
        // This works by setting a flag that the download operation checks
        auto& tp = ThreadPool::getInstance();
        tp.cancelTaskById(downloadId);
        
        LOG_DEBUG("[DownloadManager] Cancelled download with ID: {}", downloadId);
        return true;
    }
    
    LOG_DEBUG("[DownloadManager] Download ID not found for cancellation: {}", downloadId);
    return false;
}

void DownloadManager::CancelAllDownloads() {
    if (!m_initialized) {
        return;
    }
    
    std::vector<uint64_t> downloadsToCancel;
    
    // Get all active download IDs
    {
        std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
        downloadsToCancel = m_activeDownloads;
    }
    
    // Cancel each download
    for (uint64_t id : downloadsToCancel) {
        CancelDownload(id);
    }
    
    LOG_DEBUG("[DownloadManager] Cancelled all downloads (count: {})", downloadsToCancel.size());
}

size_t DownloadManager::GetActiveDownloadCount() const {
    if (!m_initialized) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(m_activeDownloadsMutex);
    return m_activeDownloads.size();
}

uint64_t DownloadManager::DownloadFileIfNotExists(
    const std::string& url,
    const std::string& targetPath,
    DownloadCallback callback,
    SkyrimNet::Network::ProgressCallback progressCallback,
    std::chrono::milliseconds timeout
) {
    // Check if the file already exists
    if (std::filesystem::exists(targetPath)) {
        LOG_DEBUG("[DownloadManager] File already exists, skipping download: {}", targetPath);
        
        // File exists, so prepare a success result
        DownloadResult result;
        result.success = true;
        result.filePath = targetPath;
        result.httpCode = 200; // Assuming OK since file exists
        
        // Get the file size
        try {
            result.bytesDownloaded = std::filesystem::file_size(targetPath);
        } catch (const std::exception& e) {
            LOG_WARN("[DownloadManager] Failed to get file size: {}", e.what());
        }
        
        // Call the callback with the successful result
        if (callback) {
            callback(result);
        }
        
        return 0; // Return 0 to indicate no download was started
    }
    
    // File doesn't exist, start the download
    return DownloadFileAsync(url, targetPath, callback, progressCallback, timeout);
}

} // namespace Network
} // namespace SkyrimNet 
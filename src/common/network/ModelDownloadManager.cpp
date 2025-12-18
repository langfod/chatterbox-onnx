#include "common/network/ModelDownloadManager.h"
#include "common/core/core.h"
#include "Skyrim/utils/SKSEHelpers.h"
#include <filesystem>
#include <algorithm>
#include <atomic>

namespace SkyrimNet {

ModelDownloadManager& ModelDownloadManager::GetInstance() {
    static ModelDownloadManager instance;
    return instance;
}

ModelDownloadManager::ModelDownloadManager() {
    lastNotificationTime = std::chrono::system_clock::now();
}


bool ModelDownloadManager::StartDownload(
    const std::string& modelName,
    const std::string& downloadUrl,
    const std::string& modelPath,
    std::function<void(bool success, std::string modelPath, std::string errorMessage)> callback
) {
    // Prepare ModelDownloadInfo
    ModelDownloadInfo info;
    info.modelName = modelName;
    info.downloadUrl = downloadUrl;
    info.modelPath = modelPath;
    info.downloadedBytes = 0;
    info.totalBytes = 0;
    info.status = ModelDownloadStatus::InProgress;
    info.startTime = std::chrono::system_clock::now();
    info.endTime = {};

    {
        std::lock_guard<std::mutex> lock(downloadsMutex);
        downloads[modelName] = info;
    }

    if (std::filesystem::exists(modelPath)) {
        ReportSuccess(modelName);
        if (callback) {
            callback(true, modelPath, "");
        }
        return true;
    }

    auto& downloadManager = Network::DownloadManager::GetInstance();
    downloadManager.Initialize();

    downloadManager.DownloadFileIfNotExists(
        downloadUrl,
        modelPath,
        // Completion callback
        [this, modelName, modelPath, callback](const Network::DownloadResult& result) {
            if (result.success) {
                ReportSuccess(modelName);
                if (callback) {
                    callback(true, modelPath, "");
                }
            } else {
                ReportError(modelName, result.errorMessage);
                if (callback) {
                    callback(false, "", result.errorMessage);
                }
            }
        },
        // Progress callback
        [this, modelName](size_t downloadedBytes, size_t totalBytes) {
            bool isTimeToShowNotification = false;
            { 
                std::lock_guard<std::mutex> lock(downloadsMutex);
                auto now = std::chrono::system_clock::now();
                if (now - lastNotificationTime >= progress_notification_interval) {
                    LOG_INFO("ModelDownloadManager: Downloading {} - {}/{}", modelName, downloadedBytes, totalBytes);
                    lastNotificationTime = now;
                    isTimeToShowNotification = true;
                }
            }

            if(isTimeToShowNotification){
                ReportProgress(modelName, downloadedBytes, totalBytes);
                ShowProgressNotification();
            }
         
          return true;
        }
    );
    
    return true;
}

bool ModelDownloadManager::StartDownload(
    const std::vector<DownloadRequest>& requests,
    std::function<void(bool success, std::string errorMessage)> callback) {
    if (requests.empty()) {
        if (callback) {
            callback(true, "");
        }
        return true;
    }

    struct SharedState {
        std::atomic<size_t> remaining{0};
        std::atomic<bool> success{true};
        std::mutex mu;
        std::string errorMsg;
    };

    auto state = std::make_shared<SharedState>();
    state->remaining.store(requests.size());

    bool startedAll = true;
    for (const auto& r : requests) {
        bool started = StartDownload(
            r.modelName,
            r.downloadUrl,
            r.modelPath,
            [state, callback](bool success, std::string, std::string error) {
                if (!success) {
                    state->success.store(false);
                    if (!error.empty()) {
                        std::lock_guard<std::mutex> lk(state->mu);
                        if (!state->errorMsg.empty())
                            state->errorMsg += "\n";
                        state->errorMsg += error;
                    }
                }
                if (state->remaining.fetch_sub(1) == 1) {
                    if (callback)
                        callback(state->success.load(), state->errorMsg);
                }
            });
        if (!started)
            startedAll = false;
    }
    return startedAll;
}


void ModelDownloadManager::ShowProgressNotification() {
    size_t totalDownloaded = 0;
    size_t totalSize = 0;
    size_t activeCount = 0;
    {
        // Sum up total loaded & total completed
        std::lock_guard<std::mutex> lock(downloadsMutex);
        for (const auto& [name, info] : downloads) {
            if (info.status != ModelDownloadStatus::Failed && info.totalBytes > 0 && info.downloadedBytes < info.totalBytes) {
                totalDownloaded += info.downloadedBytes;
                totalSize += info.totalBytes;
                ++activeCount;
            }
        }
    }
    if (activeCount == 0 || totalSize == 0) return;
    int percent = static_cast<int>((double)totalDownloaded / totalSize * 100.0);
    std::string msg = fmt::format("Downloading Models - {}%", percent);
   
    Skyrim::Utils::SubmitTaskToMainGameThread(
        "ShowProgressNotification",
        [msg]() {
            SHOW_NOTIFICATION(msg.c_str());
        }
    );
}

void ModelDownloadManager::ReportProgress(const std::string& modelName, const size_t downloadedBytes, const size_t totalBytes) {
    std::lock_guard<std::mutex> lock(downloadsMutex);
    auto it = downloads.find(modelName);
    if (it != downloads.end()) {
        it->second.downloadedBytes = downloadedBytes;
        it->second.totalBytes = totalBytes;
    }
}

void ModelDownloadManager::ReportError(const std::string& modelName, const std::string& errorMessage) {
    std::lock_guard<std::mutex> lock(downloadsMutex);
    auto it = downloads.find(modelName);
    if (it != downloads.end()) {
        it->second.status = ModelDownloadStatus::Failed;
        it->second.errorMessage = errorMessage;
        it->second.endTime = std::chrono::system_clock::now();
    }
}

void ModelDownloadManager::ReportSuccess(
    const std::string& modelName
)
{
    std::lock_guard<std::mutex> lock(downloadsMutex);
    auto it = downloads.find(modelName);
    if (it != downloads.end()) {
        it->second.status = ModelDownloadStatus::Complete;
        it->second.endTime = std::chrono::system_clock::now();
    }

    // Check wether all are completed
    bool allDone = true;
    for (const auto& [name, info] : downloads) {
        if (info.status == ModelDownloadStatus::InProgress) {
          allDone = false;
          break;
        }
    }

    // Clear all so the next download shows proper percentages
    if(allDone)
    {
        downloads.clear();
    }
}

} // namespace SkyrimNet

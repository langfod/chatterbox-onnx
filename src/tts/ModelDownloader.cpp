/**
 * @file ModelDownloader.cpp
 * @brief Implementation of HuggingFace model downloader
 */

#include "tts/ModelDownloader.h"
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>

namespace fs = std::filesystem;

namespace ChatterboxTTS {

// Curl write callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    size_t totalSize = size * nmemb;
    file->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Curl progress callback data
struct ProgressData {
    ProgressCallback* callback;
    std::string filename;
};

// Curl progress callback
static int ProgressCallback_Curl(void* clientp, curl_off_t dltotal, curl_off_t dlnow, 
                                  curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    ProgressData* data = static_cast<ProgressData*>(clientp);
    if (data && data->callback && *data->callback) {
        (*data->callback)(static_cast<size_t>(dlnow), static_cast<size_t>(dltotal), data->filename);
    }
    return 0;
}

ModelDownloader::ModelDownloader() 
    : m_cacheDir("models")
{
    // Check for HF_TOKEN environment variable
    const char* token = std::getenv("HF_TOKEN");
    if (token) {
        m_token = token;
    }
    
    // Initialize curl globally (should be done once per app)
    static bool curlInitialized = false;
    if (!curlInitialized) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curlInitialized = true;
    }
}

ModelDownloader::~ModelDownloader() {
    // Note: curl_global_cleanup() should be called at app exit
}

void ModelDownloader::SetCacheDir(const std::string& path) {
    m_cacheDir = path;
}

void ModelDownloader::SetToken(const std::string& token) {
    m_token = token;
}

void ModelDownloader::SetProgressCallback(ProgressCallback callback) {
    m_progressCallback = callback;
}

std::string ModelDownloader::GetHuggingFaceUrl(const std::string& repoId,
                                                const std::string& subfolder,
                                                const std::string& filename) {
    std::string url = "https://huggingface.co/" + repoId + "/resolve/main/";
    if (!subfolder.empty()) {
        url += subfolder + "/";
    }
    url += filename;
    return url;
}

bool ModelDownloader::DownloadWithCurl(const std::string& url, const std::string& localPath) {
    // Create parent directories
    fs::path path(localPath);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    
    // Open output file
    std::ofstream outFile(localPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "[ModelDownloader] Failed to open file for writing: " << localPath << std::endl;
        return false;
    }
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[ModelDownloader] Failed to initialize curl" << std::endl;
        return false;
    }
    
    // Setup progress data
    ProgressData progressData;
    progressData.callback = &m_progressCallback;
    progressData.filename = fs::path(localPath).filename().string();
    
    // Set curl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "ChatterboxTTS/1.0");
    
    // Progress callback
    if (m_progressCallback) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback_Curl);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressData);
    }
    
    // Add auth header if token is set
    struct curl_slist* headers = nullptr;
    if (!m_token.empty()) {
        std::string authHeader = "Authorization: Bearer " + m_token;
        headers = curl_slist_append(headers, authHeader.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    // Perform download
    CURLcode res = curl_easy_perform(curl);
    
    // Check HTTP response code
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    
    // Cleanup
    if (headers) {
        curl_slist_free_all(headers);
    }
    curl_easy_cleanup(curl);
    outFile.close();
    
    if (res != CURLE_OK) {
        std::cerr << "[ModelDownloader] Download failed: " << curl_easy_strerror(res) << std::endl;
        fs::remove(localPath);
        return false;
    }
    
    if (httpCode != 200) {
        std::cerr << "[ModelDownloader] HTTP error " << httpCode << " downloading " << url << std::endl;
        fs::remove(localPath);
        return false;
    }
    
    std::cout << std::endl;  // Newline after progress
    return true;
}

bool ModelDownloader::DownloadFile(const std::string& repoId,
                                    const std::string& subfolder,
                                    const std::string& filename,
                                    const std::string& localPath) {
    std::string url = GetHuggingFaceUrl(repoId, subfolder, filename);
    std::cout << "[ModelDownloader] Downloading: " << filename << std::endl;
    return DownloadWithCurl(url, localPath);
}

bool ModelDownloader::DownloadIfNotExists(const std::string& repoId,
                                           const std::string& subfolder,
                                           const std::string& filename,
                                           const std::string& localPath) {
    if (fs::exists(localPath)) {
        std::cout << "[ModelDownloader] File exists: " << localPath << std::endl;
        return true;
    }
    return DownloadFile(repoId, subfolder, filename, localPath);
}

std::vector<ModelFile> ModelDownloader::GetChatterboxModelFiles(const std::string& dtype) {
    // Map dtype to filename suffix (must match ChatterboxTTS::GetModelFilename)
    std::string suffix;
    if (dtype == "fp32") {
        suffix = "";
    } else if (dtype == "q8") {
        suffix = "_quantized";
    } else if (dtype == "q4") {
        suffix = "_q4";
    } else if (dtype == "q4f16") {
        suffix = "_q4f16";
    } else {
        suffix = "_" + dtype;
    }
    
    return {
        // ONNX model files (dtype-specific)
        {"onnx", "speech_encoder" + suffix + ".onnx", true},
        {"onnx", "embed_tokens" + suffix + ".onnx", true},
        {"onnx", "language_model" + suffix + ".onnx", true},
        {"onnx", "conditional_decoder" + suffix + ".onnx", true},
        // Common files (from repo root, same for all dtypes)
        {"", "tokenizer.json", false},
    };
}

bool ModelDownloader::ModelsExist(const std::string& localDir, const std::string& dtype) const {
    auto files = GetChatterboxModelFiles(dtype);
    
    for (const auto& file : files) {
        fs::path onnxPath = fs::path(localDir) / file.subfolder / file.filename;
        if (!fs::exists(onnxPath)) {
            return false;
        }
        
        if (file.hasDataFile) {
            fs::path dataPath = onnxPath.string() + "_data";
            // Data file is optional - some models have embedded weights
        }
    }
    
    return true;
}

bool ModelDownloader::DownloadChatterboxModels(const std::string& localDir, const std::string& dtype) {
    std::cout << "[ModelDownloader] Downloading Chatterbox TTS models (" << dtype << ")..." << std::endl;
    std::cout << "[ModelDownloader] Repository: " << CHATTERBOX_REPO_ID << std::endl;
    std::cout << "[ModelDownloader] Destination: " << localDir << std::endl;
    
    // Create directory
    fs::create_directories(localDir);
    
    auto files = GetChatterboxModelFiles(dtype);
    bool allSuccess = true;
    
    for (const auto& file : files) {
        // Download main .onnx file
        fs::path localPath = fs::path(localDir) / file.subfolder / file.filename;
        
        if (!DownloadIfNotExists(CHATTERBOX_REPO_ID, file.subfolder, file.filename, localPath.string())) {
            allSuccess = false;
            continue;
        }
        
        // Download .onnx_data file if expected
        if (file.hasDataFile) {
            std::string dataFilename = file.filename + "_data";
            fs::path dataPath = fs::path(localDir) / file.subfolder / dataFilename;
            
            // Try to download data file (may not exist for all models)
            if (!fs::exists(dataPath)) {
                std::string url = GetHuggingFaceUrl(CHATTERBOX_REPO_ID, file.subfolder, dataFilename);
                std::cout << "[ModelDownloader] Downloading: " << dataFilename << std::endl;
                // Don't fail if data file doesn't exist
                DownloadWithCurl(url, dataPath.string());
            }
        }
    }
    
    if (allSuccess) {
        std::cout << "[ModelDownloader] All models downloaded successfully!" << std::endl;
    }
    
    return allSuccess;
}

} // namespace ChatterboxTTS

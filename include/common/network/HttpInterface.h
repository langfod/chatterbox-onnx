#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace SkyrimNet {

// HTTP timeout exception
class HttpTimeoutException : public std::runtime_error {
public:
    explicit HttpTimeoutException(const std::string& message) : std::runtime_error(message) {}
};

// HTTP method enumeration
enum class HttpMethod {
    GET,
    POST,
    PUT,
    HTTP_DELETE,
    HEAD,
    PATCH
};

// HTTP response structure
struct HttpResponse {
    int statusCode;
    std::vector<uint8_t> body;
    std::map<std::string, std::string> headers;
    
    HttpResponse(int code = 200) : statusCode(code) {}
    
    // Helper to get body as string
    std::string GetBodyAsString() const {
        return std::string(body.begin(), body.end());
    }
    
    // Helper to check if request was successful
    bool IsSuccess() const {
        return statusCode >= 200 && statusCode < 300;
    }
};

// Outgoing multipart form data structure for HTTP client requests
// Note: This is distinct from cpp-httplib's MultipartFormData used for incoming server requests
struct OutgoingMultipartFormData {
    std::string name;
    std::vector<uint8_t> data;
    std::string filename;
    std::string contentType;
    
    OutgoingMultipartFormData(const std::string& n, const std::vector<uint8_t>& d, 
                     const std::string& fn = "", const std::string& ct = "application/octet-stream")
        : name(n), data(d), filename(fn), contentType(ct) {}
    
    // Convenience constructor for string data
    OutgoingMultipartFormData(const std::string& n, const std::string& value, 
                     const std::string& fn = "", const std::string& ct = "text/plain")
        : name(n), filename(fn), contentType(ct) {
        data.assign(value.begin(), value.end());
    }
};

// Request configuration
struct HttpRequestConfig {
    long timeout = 300;           // Request timeout in seconds
    long connectTimeout = 10;    // Connection timeout in seconds
    bool followRedirects = true; // Whether to follow HTTP redirects
    bool verifySSL = false;       // Whether to verify SSL certificates
    std::string caBundlePath;    // Path to CA certificate bundle (optional, uses system default if empty)
    std::string userAgent = "SkyrimNet/1.0";
    std::map<std::string, std::string> defaultHeaders;
    
    HttpRequestConfig() = default;
    
    HttpRequestConfig(long t) : timeout(t) {}
    
    HttpRequestConfig(long t, long ct) : timeout(t), connectTimeout(ct) {}
};

using StreamDataCallback = std::function<void(const std::vector<uint8_t>&)>;
using StreamErrorCallback = std::function<void(const std::string&)>;
using StreamCompleteCallback = std::function<void(int, const std::map<std::string, std::string>&)>;

// Streaming response structure with per-request cancellation support
struct StreamingResponse {
    int statusCode;
    std::map<std::string, std::string> headers;
    bool isComplete;
    bool isStarted;  // Indicates if the request has actually started
    
    // Per-request cancellation token - allows canceling specific requests
    // without affecting other concurrent streaming requests
    std::shared_ptr<std::atomic<bool>> cancellationToken;
    
    StreamingResponse(int code = 0) : statusCode(code), isComplete(false), isStarted(false) {
        cancellationToken = std::make_shared<std::atomic<bool>>(false);
    }
    
    // Helper to check if this specific request was cancelled
    bool IsCancelled() const {
        return cancellationToken && cancellationToken->load();
    }
    
    // Helper to cancel this specific request
    void Cancel() {
        if (cancellationToken) {
            cancellationToken->store(true);
        }
    }
    
    // Helper to check if the request is in progress
    bool IsInProgress() const {
        return isStarted && !isComplete && !IsCancelled();
    }
    
    // Helper to check if we have a valid status code (request completed)
    bool HasValidStatus() const {
        return isComplete && statusCode > 0;
    }
};

// Generic HTTP interface
class HttpInterface {
public:
    virtual ~HttpInterface() = default;
    
    // Basic HTTP methods
    virtual HttpResponse Get(const std::string& url, 
                           const std::map<std::string, std::string>& headers = {}) = 0;
    
    virtual HttpResponse Post(const std::string& url, 
                            const std::string& data,
                            const std::map<std::string, std::string>& headers = {}) = 0;
    
    virtual HttpResponse Put(const std::string& url, 
                           const std::string& data,
                           const std::map<std::string, std::string>& headers = {}) = 0;
    
    virtual HttpResponse Delete(const std::string& url, 
                              const std::map<std::string, std::string>& headers = {}) = 0;
    
    virtual HttpResponse Head(const std::string& url, 
                            const std::map<std::string, std::string>& headers = {}) = 0;
    
    // Multipart form upload
    virtual HttpResponse PostMultipart(const std::string& url,
                                     const std::vector<OutgoingMultipartFormData>& formData,
                                     const std::map<std::string, std::string>& headers = {}) = 0;
    
    // Generic request method
    virtual HttpResponse Request(HttpMethod method,
                               const std::string& url,
                               const std::string& data = "",
                               const std::map<std::string, std::string>& headers = {}) = 0;
    
    // Streaming method
    virtual StreamingResponse PostStream(const std::string& url,
                                       const std::string& data,
                                       StreamDataCallback dataCallback,
                                       StreamErrorCallback errorCallback,
                                       StreamCompleteCallback completeCallback,
                                       const std::map<std::string, std::string>& headers = {}) = 0;
    
    // Streaming method with external cancellation token
    // The caller can set the token to true from within the dataCallback to abort the stream
    virtual StreamingResponse PostStream(const std::string& url,
                                       const std::string& data,
                                       StreamDataCallback dataCallback,
                                       StreamErrorCallback errorCallback,
                                       StreamCompleteCallback completeCallback,
                                       const std::map<std::string, std::string>& headers,
                                       std::shared_ptr<std::atomic<bool>> cancellationToken) = 0;
    
    // Async streaming method
    virtual StreamingResponse PostStreamAsync(const std::string& url,
                                            const std::string& data,
                                            StreamDataCallback dataCallback,
                                            StreamErrorCallback errorCallback,
                                            StreamCompleteCallback completeCallback,
                                            const std::map<std::string, std::string>& headers = {}) = 0;
    
    // Stream control methods
    virtual void CancelStream(const StreamingResponse& response) = 0;
    
    // Utility methods
    virtual bool FileExists(const std::string& url) = 0;
    virtual std::vector<uint8_t> DownloadFile(const std::string& url) = 0;
};

// Service types for HTTP interface factory
enum class HttpServiceType {
    Unknown,
    OpenRouter,
    Zonos, 
    XTTS,
    ElevenLabs,
    Whisper,
    STT,
    Download,
    Memory
};

// Factory function to create HTTP interface implementation
std::unique_ptr<HttpInterface> CreateHttpInterface(const HttpRequestConfig& config = HttpRequestConfig{}, HttpServiceType serviceType = HttpServiceType::Unknown);

} // namespace SkyrimNet
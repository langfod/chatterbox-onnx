#include "BuildConstants.h"
#include "common/network/HttpInterface.h"
#include "common/core/ThreadPool.h"
#include "common/utils/Logging.h"
#include <curl/curl.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <thread>
#include <sstream>
#include <mutex>

// Include mock interface for testing
#if TEST_ENVIRONMENT
#include "MockHttpInterface.h"
#include "ConfigFlags.h"
#endif

namespace SkyrimNet {

#if TEST_ENVIRONMENT
// Determine if mock interface should be used based on the calling context
bool ShouldUseMockHttpInterface(HttpServiceType serviceType = HttpServiceType::Unknown) {
    // If no service type is provided, default to using mock interface in test environment
    if (serviceType == HttpServiceType::Unknown) {
        return true;
    }
    
    // Map service types to their external test flags for cleaner lookup
    static const std::unordered_map<HttpServiceType, int> serviceFlags = {
        {HttpServiceType::OpenRouter, RUN_EXTERNAL_OPENROUTER_TESTS},
        {HttpServiceType::Zonos, RUN_EXTERNAL_ZONOS_TESTS},
        {HttpServiceType::Whisper, RUN_EXTERNAL_WHISPER_TESTS},
        {HttpServiceType::XTTS, RUN_EXTERNAL_XTTS_TESTS},
        {HttpServiceType::STT, RUN_EXTERNAL_STT_TESTS},
        {HttpServiceType::Download, RUN_EXTERNAL_DOWNLOAD_TESTS},
        {HttpServiceType::Memory, RUN_EXTERNAL_MEMORY_TESTS}
    };
    
    auto it = serviceFlags.find(serviceType);
    if (it != serviceFlags.end()) {
        return (it->second == 0);  // Use mock if external tests are disabled (0)
    }
    
    // Default to mock interface for unknown service types in test environment
    return true;
}
#endif

class CurlHttpInterface : public HttpInterface, public std::enable_shared_from_this<CurlHttpInterface> {
private:
    // Static default headers
    static const std::map<std::string, std::string> kDefaultHeaders;
    
    // Request config - set only during construction
    const HttpRequestConfig m_config;
    
    // Helper to set common cURL options
    void SetCommonCurlOptions(CURL* curl) const {
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, m_config.timeout);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, m_config.connectTimeout);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, m_config.followRedirects ? 1L : 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        
        // Configure SSL certificate verification
        //if (m_config.verifySSL) {
        //    if (!m_config.caBundlePath.empty()) {
        //        // Use custom CA bundle if provided
        //        curl_easy_setopt(curl, CURLOPT_CAINFO, m_config.caBundlePath.c_str());
        //    } else {
        //        // Fall back to Windows native certificate store
        //        curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
        //    }
        //}
        
        std::string userAgent = "SkyrimNet/" + std::string(SkyrimNet::Build::VERSION_STRING);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, userAgent.c_str());
    }

    // Initialize CURL handle lazily
    CURL* GetCurlHandle() const {
        // Use thread_local unique_ptr for the handle
        thread_local std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> tlsHandle(nullptr, curl_easy_cleanup);
        if (!tlsHandle) {
            tlsHandle.reset(curl_easy_init());
            if (!tlsHandle) {
                throw std::runtime_error("Failed to initialize cURL handle");
            }
            // Set common options once per thread
            SetCommonCurlOptions(tlsHandle.get());
        }
        return tlsHandle.get();
    }
    
    // Reset handle for new request
    void ResetCurlHandle() const {
        CURL* curl = GetCurlHandle();
        curl_easy_reset(curl);
        // Re-apply common options after reset
        SetCommonCurlOptions(curl);
    }
    
    // Per-request context structure
    struct RequestContext {
        std::vector<uint8_t> responseBody;
        std::map<std::string, std::string> responseHeaders;
    };
    
    // Streaming context structure for SSE handling
    struct StreamingContext {
        std::shared_ptr<std::atomic<bool>> cancellationToken;
        StreamDataCallback dataCallback;
        StreamErrorCallback errorCallback;
        StreamCompleteCallback completeCallback;
        std::map<std::string, std::string> responseHeaders;
        std::mutex callbackMutex; // Protect callback execution
        int statusCode{200};
        std::vector<uint8_t> buffer; // Buffer for accumulating data
        
        StreamingContext(StreamDataCallback dc, StreamErrorCallback ec, StreamCompleteCallback cc)
            : dataCallback(std::move(dc)), errorCallback(std::move(ec)), completeCallback(std::move(cc)) {}
    };
    
    // Callback function for writing response data
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        RequestContext* context = static_cast<RequestContext*>(userp);
        
        const char* data = static_cast<const char*>(contents);
        size_t totalSize = size * nmemb;
        
        context->responseBody.insert(context->responseBody.end(), data, data + totalSize);
        
        return totalSize;
    }

    // Callback function for streaming response data
    static size_t StreamingWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        StreamingContext* context = static_cast<StreamingContext*>(userp);
        
        // Check if request was cancelled before acquiring lock
        if (context->cancellationToken && context->cancellationToken->load()) {
            return 0; // Return 0 to signal cURL to stop the transfer
        }
        
        std::lock_guard<std::mutex> lock(context->callbackMutex);
        
        // Check again after acquiring lock
        if (context->cancellationToken && context->cancellationToken->load()) {
            return 0; // Return 0 to signal cURL to stop the transfer
        }
        
        const char* data = static_cast<const char*>(contents);
        size_t totalSize = size * nmemb;
        
        // Convert the incoming data to vector<uint8_t> format
        std::vector<uint8_t> chunk(data, data + totalSize);
        
        // Call the user's data callback with the chunk
        if (context->dataCallback) {
            try {
                context->dataCallback(chunk);
            } catch (const std::exception& e) {
                // If callback throws, we could call error callback here
                // For now, we'll just log and continue
                LOG_DEBUG("Streaming data callback threw exception: {}", e.what());
            }
        }
        
        return totalSize;
    }

    // Callback function for writing response headers
    static size_t HeaderCallback(char* buffer, size_t size, size_t nitems, void* userp) {
        RequestContext* context = static_cast<RequestContext*>(userp);
        
        size_t totalSize = size * nitems;
        std::string headerLine(buffer, totalSize);
        
        // Remove trailing newline/carriage return
        while (!headerLine.empty() && (headerLine.back() == '\n' || headerLine.back() == '\r')) {
            headerLine.pop_back();
        }
        
        // Skip empty lines
        if (headerLine.empty()) {
            return totalSize;
        }
        
        // Parse header line (format: "Name: Value")
        size_t colonPos = headerLine.find(':');
        if (colonPos != std::string::npos) {
            std::string name = headerLine.substr(0, colonPos);
            std::string value = headerLine.substr(colonPos + 1);
            
            // Trim whitespace
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
                value.erase(0, 1);
            }
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
                value.pop_back();
            }
            
            context->responseHeaders[name] = value;
        }
        
        return totalSize;
    }

    // Callback function for streaming response headers
    static size_t StreamingHeaderCallback(char* buffer, size_t size, size_t nitems, void* userp) {
        StreamingContext* context = static_cast<StreamingContext*>(userp);
        std::lock_guard<std::mutex> lock(context->callbackMutex);
        
        size_t totalSize = size * nitems;
        std::string headerLine(buffer, totalSize);
        
        // Remove trailing newline/carriage return
        while (!headerLine.empty() && (headerLine.back() == '\n' || headerLine.back() == '\r')) {
            headerLine.pop_back();
        }
        
        // Skip empty lines
        if (headerLine.empty()) {
            return totalSize;
        }
        
        // Parse header line (format: "Name: Value")
        size_t colonPos = headerLine.find(':');
        if (colonPos != std::string::npos) {
            std::string name = headerLine.substr(0, colonPos);
            std::string value = headerLine.substr(colonPos + 1);
            
            // Trim whitespace
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
                value.erase(0, 1);
            }
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
                value.pop_back();
            }
            
            context->responseHeaders[name] = value;
        }
        
        return totalSize;
    }

    // Private method for async streaming with external cancellation token
    void PostStreamWithToken(const std::string& url,
                           const std::string& data,
                           StreamDataCallback dataCallback,
                           StreamErrorCallback errorCallback,
                           StreamCompleteCallback completeCallback,
                           const std::map<std::string, std::string>& headers,
                           std::shared_ptr<std::atomic<bool>> cancellationToken) {
        LOG_DEBUG("CurlHttpInterface::PostStreamWithToken called for URL: {}", url);
        
        // Create a dedicated cURL handle for this async request to avoid conflicts with synchronous requests
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> asyncCurlHandle(curl_easy_init(), curl_easy_cleanup);
        if (!asyncCurlHandle) {
            if (errorCallback) {
                errorCallback("Failed to initialize cURL handle for async request");
            }
            return;
        }

        CURL *curl = asyncCurlHandle.get();
        
        // Set common options for the async handle
        SetCommonCurlOptions(curl);
        
        // Create streaming context
        auto context = std::make_shared<StreamingContext>(dataCallback, errorCallback, completeCallback);
        
        // Use the provided cancellation token
        context->cancellationToken = cancellationToken;
        
        // Configure cURL handle for streaming
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamingWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, context.get());
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, StreamingHeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, context.get());
        
        // Set POST method
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        
        // Set post data if provided
        if (!data.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(data.length()));
        }
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Perform the streaming request
        CURLcode result = curl_easy_perform(curl);
        
        if (result != CURLE_OK) {
            std::string errorMsg = curl_easy_strerror(result);
            
            // Check for timeout-related errors and call error callback with appropriate message
            if (result == CURLE_OPERATION_TIMEDOUT) {
                if (errorCallback) {
                    errorCallback("Request timeout: " + errorMsg);
                }
            } else {
                if (errorCallback) {
                    errorCallback("cURL async streaming request failed: " + errorMsg);
                }
            }
            return;
        }
        
        // Get response code
        long statusCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
        
        // Call completion callback
        if (completeCallback) {
            completeCallback(static_cast<int>(statusCode), context->responseHeaders);
        }
    }
    
    // Helper function to set up headers for a request
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> SetupHeaders(const std::map<std::string, std::string>& headers) const {
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headerList(nullptr, curl_slist_free_all);
        
        // Build combined headers (default + request-specific)
        std::map<std::string, std::string> allHeaders = kDefaultHeaders;
        for (const auto& [key, value] : headers) {
            allHeaders[key] = value;
        }
        
        // Create header list
        curl_slist* rawHeaderList = nullptr;
        for (const auto& [key, value] : allHeaders) {
            std::string header = key + ": " + value;
            rawHeaderList = curl_slist_append(rawHeaderList, header.c_str());
        }
        
        if (rawHeaderList) {
            headerList.reset(rawHeaderList);
        }
        
        return headerList;
    }

public:
    CurlHttpInterface(const HttpRequestConfig& config = HttpRequestConfig{})
        : m_config(config) {
        // Ensure cURL is globally initialized
        static std::once_flag curl_init_flag;
        std::call_once(curl_init_flag, []() {
            CURLcode result = curl_global_init(CURL_GLOBAL_DEFAULT);
            if (result != CURLE_OK) {
                throw std::runtime_error("Failed to initialize cURL globally: " + std::string(curl_easy_strerror(result)));
            }
        });
    }
    
    ~CurlHttpInterface() {
        // Note: cURL global cleanup should be done at application shutdown
        // We don't call curl_global_cleanup() here to avoid cleanup while other threads might still be using cURL
    }
    
    HttpResponse Get(const std::string& url, 
                    const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::Get called for URL: {}", url);
        
        // Get the reusable CURL handle and reset it for this request
        CURL* curl = GetCurlHandle();
        ResetCurlHandle();
        
        // Set up request context
        RequestContext context;
        
        // Configure cURL handle for GET request
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &context);
        // GET is the default method, no additional setup needed
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Perform the request
        CURLcode result = curl_easy_perform(curl);
        
        if (result != CURLE_OK) {
            std::string errorMsg = curl_easy_strerror(result);
            
            // Check for timeout-related errors and throw appropriate HTTP exceptions
            if (result == CURLE_OPERATION_TIMEDOUT) {
                throw HttpTimeoutException("Request timeout: " + errorMsg);
            } else {
                throw std::runtime_error("cURL GET request failed: " + errorMsg);
            }
        }
        
        // Get response code
        long statusCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
        
        // Create response object
        HttpResponse response(static_cast<int>(statusCode));
        response.body = context.responseBody;
        response.headers = context.responseHeaders;
        
        return response;
    }
    
    HttpResponse Post(const std::string& url, 
                     const std::string& data,
                     const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::Post called for URL: {}", url);
        
        // Get the reusable CURL handle and reset it for this request
        CURL* curl = GetCurlHandle();
        ResetCurlHandle();
        
        // Set up request context
        RequestContext context;
        
        // Configure cURL handle for POST request
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &context);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        
        // Set post data if provided
        if (!data.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(data.length()));
        }
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Perform the request
        CURLcode result = curl_easy_perform(curl);
        
        if (result != CURLE_OK) {
            std::string errorMsg = curl_easy_strerror(result);
            
            // Check for timeout-related errors and throw appropriate HTTP exceptions
            if (result == CURLE_OPERATION_TIMEDOUT) {
                throw HttpTimeoutException("Request timeout: " + errorMsg);
            } else {
                throw std::runtime_error("cURL POST request failed: " + errorMsg);
            }
        }
        
        // Get response code
        long statusCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
        
        // Create response object
        HttpResponse response(static_cast<int>(statusCode));
        response.body = context.responseBody;
        response.headers = context.responseHeaders;
        
        return response;
    }
    
    HttpResponse Put(const std::string& url, 
                    const std::string& data,
                    const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::Put called for URL: {}", url);
        
        // Get the reusable CURL handle and reset it for this request
        CURL* curl = GetCurlHandle();
        ResetCurlHandle();
        
        // Set up request context
        RequestContext context;
        
        // Configure cURL handle for PUT request
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &context);
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        
        // Set post data if provided
        if (!data.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(data.length()));
        }
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Perform the request
        CURLcode result = curl_easy_perform(curl);
        
        if (result != CURLE_OK) {
            std::string errorMsg = curl_easy_strerror(result);
            
            // Check for timeout-related errors and throw appropriate HTTP exceptions
            if (result == CURLE_OPERATION_TIMEDOUT) {
                throw HttpTimeoutException("Request timeout: " + errorMsg);
            } else {
                throw std::runtime_error("cURL PUT request failed: " + errorMsg);
            }
        }
        
        // Get response code
        long statusCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
        
        // Create response object
        HttpResponse response(static_cast<int>(statusCode));
        response.body = context.responseBody;
        response.headers = context.responseHeaders;
        
        return response;
    }
    
    HttpResponse Delete(const std::string& url, 
                       const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::Delete called for URL: {}", url);
        
        // Get the reusable CURL handle and reset it for this request
        CURL* curl = GetCurlHandle();
        ResetCurlHandle();
        
        // Set up request context
        RequestContext context;
        
        // Configure cURL handle for DELETE request
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &context);
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Perform the request
        CURLcode result = curl_easy_perform(curl);
        
        if (result != CURLE_OK) {
            std::string errorMsg = curl_easy_strerror(result);
            
            // Check for timeout-related errors and throw appropriate HTTP exceptions
            if (result == CURLE_OPERATION_TIMEDOUT) {
                throw HttpTimeoutException("Request timeout: " + errorMsg);
            } else {
                throw std::runtime_error("cURL DELETE request failed: " + errorMsg);
            }
        }
        
        // Get response code
        long statusCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
        
        // Create response object
        HttpResponse response(static_cast<int>(statusCode));
        response.body = context.responseBody;
        response.headers = context.responseHeaders;
        
        return response;
    }
    
    HttpResponse Head(const std::string& url, 
                     const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::Head called for URL: {}", url);
        
        // Get the reusable CURL handle and reset it for this request
        CURL* curl = GetCurlHandle();
        ResetCurlHandle();
        
        // Set up request context
        RequestContext context;
        
        // Configure cURL handle for HEAD request
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &context);
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Perform the request
        CURLcode result = curl_easy_perform(curl);
        
        if (result != CURLE_OK) {
            std::string errorMsg = curl_easy_strerror(result);
            
            // Check for timeout-related errors and throw appropriate HTTP exceptions
            if (result == CURLE_OPERATION_TIMEDOUT) {
                throw HttpTimeoutException("Request timeout: " + errorMsg);
            } else {
                throw std::runtime_error("cURL HEAD request failed: " + errorMsg);
            }
        }
        
        // Get response code
        long statusCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
        
        // Create response object
        HttpResponse response(static_cast<int>(statusCode));
        response.body = context.responseBody;
        response.headers = context.responseHeaders;
        
        return response;
    }
    
    HttpResponse PostMultipart(const std::string& url,
                              const std::vector<OutgoingMultipartFormData>& formData,
                              const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::PostMultipart called for URL: {}", url);
        
        // Get the reusable CURL handle and reset it for this request
        CURL* curl = GetCurlHandle();
        ResetCurlHandle();
        
        // Set up request context
        RequestContext context;
        
        // Configure cURL handle for this specific request
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &context);
        
        // Set POST method
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        
        // Create multipart form data using curl_mime
        curl_mime* mime = curl_mime_init(curl);
        if (!mime) {
            throw std::runtime_error("Failed to initialize cURL MIME");
        }
        
        // RAII cleanup for MIME
        std::unique_ptr<curl_mime, decltype(&curl_mime_free)> mimeCleanup(mime, curl_mime_free);
        
        // Add each form field
        for (const auto& field : formData) {
            curl_mimepart* part = curl_mime_addpart(mime);
            if (!part) {
                throw std::runtime_error("Failed to add MIME part");
            }
            
            // Set field name
            curl_mime_name(part, field.name.c_str());
            
            // Set field data
            if (field.data.empty()) {
                // Empty field
                curl_mime_data(part, "", CURL_ZERO_TERMINATED);
            } else {
                // Set binary data
                curl_mime_data(part, reinterpret_cast<const char*>(field.data.data()), field.data.size());
            }
            
            // Set content type if specified
            if (!field.contentType.empty()) {
                curl_mime_type(part, field.contentType.c_str());
            }
            
            // Set filename if specified
            if (!field.filename.empty()) {
                curl_mime_filename(part, field.filename.c_str());
            }
        }
        
        // Set the MIME data
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Perform the request
        CURLcode result = curl_easy_perform(curl);
        
        if (result != CURLE_OK) {
            std::string errorMsg = curl_easy_strerror(result);
            
            // Check for timeout-related errors and throw appropriate HTTP exceptions
            if (result == CURLE_OPERATION_TIMEDOUT) {
                throw HttpTimeoutException("Request timeout: " + errorMsg);
            } else {
                throw std::runtime_error("cURL multipart request failed: " + errorMsg);
            }
        }
        
        // Get response code
        long statusCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
        
        // Create response object
        HttpResponse response(static_cast<int>(statusCode));
        response.body = context.responseBody;
        response.headers = context.responseHeaders;
        
        return response;
    }
    
    HttpResponse Request(HttpMethod method,
                        const std::string& url,
                        const std::string& data,
                        const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::Request called for URL: {}", url);
        
        switch (method) {
            case HttpMethod::GET:
                return Get(url, headers);
            case HttpMethod::POST:
                return Post(url, data, headers);
            case HttpMethod::PUT:
                return Put(url, data, headers);
            case HttpMethod::HTTP_DELETE:
                return Delete(url, headers);
            case HttpMethod::HEAD:
                return Head(url, headers);
            case HttpMethod::PATCH:
                // Implement PATCH method individually
                {
                    // Get the reusable CURL handle and reset it for this request
                    CURL* curl = GetCurlHandle();
                    ResetCurlHandle();
                    
                    // Set up request context
                    RequestContext context;
                    
                    // Configure cURL handle for PATCH request
                    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &context);
                    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
                    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &context);
                    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PATCH");
                    
                    // Set post data if provided
                    if (!data.empty()) {
                        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
                        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(data.length()));
                    }
                    
                    // Set headers
                    auto headerList = SetupHeaders(headers);
                    if (headerList) {
                        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
                    }
                    
                    // Perform the request
                    CURLcode result = curl_easy_perform(curl);
                    
                    if (result != CURLE_OK) {
                        std::string errorMsg = curl_easy_strerror(result);
                        
                        // Check for timeout-related errors and throw appropriate HTTP exceptions
                        if (result == CURLE_OPERATION_TIMEDOUT) {
                            throw HttpTimeoutException("Request timeout: " + errorMsg);
                        } else {
                            throw std::runtime_error("cURL PATCH request failed: " + errorMsg);
                        }
                    }
                    
                    // Get response code
                    long statusCode = 0;
                    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
                    
                    // Create response object
                    HttpResponse response(static_cast<int>(statusCode));
                    response.body = context.responseBody;
                    response.headers = context.responseHeaders;
                    
                    return response;
                }
            default:
                throw std::runtime_error("Unsupported HTTP method");
        }
    }
    
    StreamingResponse PostStream(const std::string& url,
                               const std::string& data,
                               StreamDataCallback dataCallback,
                               StreamErrorCallback errorCallback,
                               StreamCompleteCallback completeCallback,
                               const std::map<std::string, std::string>& headers) override {
        // Delegate to the overload with a new cancellation token
        return PostStream(url, data, dataCallback, errorCallback, completeCallback, headers,
                         std::make_shared<std::atomic<bool>>(false));
    }
    
    StreamingResponse PostStream(const std::string& url,
                               const std::string& data,
                               StreamDataCallback dataCallback,
                               StreamErrorCallback errorCallback,
                               StreamCompleteCallback completeCallback,
                               const std::map<std::string, std::string>& headers,
                               std::shared_ptr<std::atomic<bool>> cancellationToken) override {
        LOG_DEBUG("CurlHttpInterface::PostStream called for URL: {}", url);
        
        // Create a dedicated cURL handle for this streaming request
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> streamCurlHandle(curl_easy_init(), curl_easy_cleanup);
        if (!streamCurlHandle) {
            throw std::runtime_error("Failed to initialize cURL handle for streaming request");
        }

        CURL *curl = streamCurlHandle.get();
        
        // Set common options for the streaming handle
        SetCommonCurlOptions(curl);
        
        // Create streaming context with the provided cancellation token
        auto context = std::make_shared<StreamingContext>(dataCallback, errorCallback, completeCallback);
        context->cancellationToken = cancellationToken;
        
        // Configure cURL handle for streaming
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamingWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, context.get());
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, StreamingHeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, context.get());
        
        // Set POST method
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        
        // Set post data if provided
        if (!data.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(data.length()));
        }
        
        // Set headers
        auto headerList = SetupHeaders(headers);
        if (headerList) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList.get());
        }
        
        // Create streaming response object
        StreamingResponse streamingResponse(0);
        streamingResponse.cancellationToken = cancellationToken;
        streamingResponse.isStarted = true;
        
        try {
            // Perform the streaming request
            CURLcode result = curl_easy_perform(curl);
            
            // Check if cancelled (CURLE_WRITE_ERROR is returned when callback returns 0)
            if (result == CURLE_WRITE_ERROR && cancellationToken && cancellationToken->load()) {
                LOG_DEBUG("Stream cancelled via cancellation token");
                streamingResponse.statusCode = 200; // Consider it successful
                streamingResponse.headers = context->responseHeaders;
                streamingResponse.isComplete = true;
                
                if (completeCallback) {
                    completeCallback(streamingResponse.statusCode, streamingResponse.headers);
                }
                return streamingResponse;
            }
            
            if (result != CURLE_OK) {
                std::string errorMsg = curl_easy_strerror(result);
                
                if (result == CURLE_OPERATION_TIMEDOUT) {
                    if (errorCallback) {
                        errorCallback("Request timeout: " + errorMsg);
                    }
                    throw HttpTimeoutException("Request timeout: " + errorMsg);
                } else {
                    if (errorCallback) {
                        errorCallback("cURL streaming request failed: " + errorMsg);
                    }
                    throw std::runtime_error("cURL streaming request failed: " + errorMsg);
                }
            }
            
            // Get response code
            long statusCode = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
            
            // Update streaming response
            streamingResponse.statusCode = static_cast<int>(statusCode);
            streamingResponse.headers = context->responseHeaders;
            streamingResponse.isComplete = true;
            
            // Call completion callback
            if (completeCallback) {
                completeCallback(streamingResponse.statusCode, streamingResponse.headers);
            }
            
        } catch (const std::exception& e) {
            if (errorCallback) {
                errorCallback("Streaming request exception: " + std::string(e.what()));
            }
            throw;
        }
        
        return streamingResponse;
    }
    
    StreamingResponse PostStreamAsync(const std::string& url,
                                    const std::string& data,
                                    StreamDataCallback dataCallback,
                                    StreamErrorCallback errorCallback,
                                    StreamCompleteCallback completeCallback,
                                    const std::map<std::string, std::string>& headers) override {
        LOG_DEBUG("CurlHttpInterface::PostStreamAsync called for URL: {}", url);
        
        // Create streaming response object immediately for cancellation support
        // Use status code 0 to indicate the request hasn't started yet
        StreamingResponse streamingResponse(0);
        streamingResponse.cancellationToken = std::make_shared<std::atomic<bool>>(false);
        
        try {
            // Use shared_from_this() to get a proper shared_ptr to this object
            auto self = shared_from_this();
            
            // Start async operation in background thread
            ThreadPool::getInstance().enqueue("PostStreamAsync", [self, url, data, dataCallback, errorCallback, completeCallback, headers, streamingResponse]() {
                try {
                    // Create a modified PostStream that uses the provided cancellation token
                    self->PostStreamWithToken(url, data, dataCallback, errorCallback, completeCallback, headers, streamingResponse.cancellationToken);
                } catch (const std::exception& e) {
                    if (errorCallback) {
                        errorCallback("Async request failed: " + std::string(e.what()));
                    }
                }
            });
            
            return streamingResponse;
        } catch (const std::exception&) {
            // If we can't start the async operation, mark as cancelled and return
            streamingResponse.cancellationToken->store(true);
            streamingResponse.isComplete = true;
            return streamingResponse;
        }
    }
    
    void CancelStream(const StreamingResponse& response) override {
        LOG_DEBUG("CurlHttpInterface::CancelStream called");
        
        if (response.cancellationToken) {
            response.cancellationToken->store(true);
            LOG_DEBUG("Streaming request cancelled");
        }
    }
    
    bool FileExists(const std::string& url) override {
        LOG_DEBUG("CurlHttpInterface::FileExists called for URL: {}", url);
        
        try {
            auto response = Head(url, {});
            return response.statusCode == 200;
        } catch (...) {
            return false;
        }
    }
    
    std::vector<uint8_t> DownloadFile(const std::string& url) override {
        LOG_DEBUG("CurlHttpInterface::DownloadFile called for URL: {}", url);
        
        auto response = Get(url, {});
        if (!response.IsSuccess()) {
            throw std::runtime_error("Failed to download file: HTTP " + std::to_string(response.statusCode));
        }
        return response.body;
    }
};

// Define static default headers
const std::map<std::string, std::string> CurlHttpInterface::kDefaultHeaders = {
    {"Connection", "keep-alive"}
};

// Factory function for creating CurlHttpInterface
std::unique_ptr<HttpInterface> CreateHttpInterface(const HttpRequestConfig& config, HttpServiceType serviceType) {
#if TEST_ENVIRONMENT
    // Check if we should use the mock interface based on service type
    if (ShouldUseMockHttpInterface(serviceType)) {
        return std::make_unique<Tests::MockHttpInterface>(config);
    }
#endif

    return std::make_unique<CurlHttpInterface>(config);
}

} // namespace SkyrimNet 
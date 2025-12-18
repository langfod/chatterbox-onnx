/**
 * @file Tokenizer.cpp
 * @brief Implementation of HuggingFace tokenizer and pre-tokenized file loader
 */

#include "tts/Tokenizer.h"
#include <tokenizers_cpp.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace ChatterboxTTS {

// ============================================================================
// HFTokenizer Implementation
// ============================================================================

HFTokenizer::HFTokenizer() = default;

HFTokenizer::~HFTokenizer() = default;

HFTokenizer::HFTokenizer(HFTokenizer&&) noexcept = default;

HFTokenizer& HFTokenizer::operator=(HFTokenizer&&) noexcept = default;

bool HFTokenizer::LoadFromFile(const std::string& path) {
    m_lastError.clear();
    
    if (!fs::exists(path)) {
        m_lastError = "Tokenizer file not found: " + path;
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Read file content
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        m_lastError = "Failed to open tokenizer file: " + path;
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Read entire file into string
    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    
    std::string jsonBlob;
    jsonBlob.resize(size);
    file.read(jsonBlob.data(), size);
    
    if (!file) {
        m_lastError = "Failed to read tokenizer file";
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    return LoadFromJSON(jsonBlob);
}

bool HFTokenizer::LoadFromJSON(const std::string& jsonBlob) {
    m_lastError.clear();
    
    try {
        m_tokenizer = tokenizers::Tokenizer::FromBlobJSON(jsonBlob);
        
        if (!m_tokenizer) {
            m_lastError = "Failed to create tokenizer from JSON";
            spdlog::error("{}", m_lastError);
            return false;
        }
        
        spdlog::info("Loaded HuggingFace tokenizer (vocab_size={})", m_tokenizer->GetVocabSize());
        return true;
        
    } catch (const std::exception& e) {
        m_lastError = std::string("Exception loading tokenizer: ") + e.what();
        spdlog::error("{}", m_lastError);
        return false;
    }
}

// Special token IDs for Chatterbox
constexpr int64_t END_OF_TEXT_TOKEN = 50256;

std::vector<int64_t> HFTokenizer::Encode(const std::string& text) {
    if (!m_tokenizer) {
        spdlog::warn("Tokenizer not loaded, returning empty tokens");
        return {};
    }
    
    std::vector<int32_t> ids32 = m_tokenizer->Encode(text);
    
    // Convert int32 to int64
    std::vector<int64_t> ids64;
    ids64.reserve(ids32.size() + 2);  // +2 for endoftext tokens
    for (int32_t id : ids32) {
        ids64.push_back(static_cast<int64_t>(id));
    }
    
    // Append two <|endoftext|> tokens (50256) to match the Python tokenizer's 
    // post-processing. The embed_tokens.onnx model expects:
    // - input_ids[:-2] -> text tokens (vocab 50276) 
    // - input_ids[-2:] -> speech token placeholders (replaced with 6561 if == 50256)
    ids64.push_back(END_OF_TEXT_TOKEN);
    ids64.push_back(END_OF_TEXT_TOKEN);
    
    return ids64;
}

std::string HFTokenizer::Decode(const std::vector<int64_t>& ids) {
    if (!m_tokenizer) {
        return "";
    }
    
    // Convert int64 to int32
    std::vector<int32_t> ids32;
    ids32.reserve(ids.size());
    for (int64_t id : ids) {
        ids32.push_back(static_cast<int32_t>(id));
    }
    
    return m_tokenizer->Decode(ids32);
}

size_t HFTokenizer::GetVocabSize() const {
    return m_tokenizer ? m_tokenizer->GetVocabSize() : 0;
}

std::string HFTokenizer::IdToToken(int32_t id) {
    return m_tokenizer ? m_tokenizer->IdToToken(id) : "";
}

int32_t HFTokenizer::TokenToId(const std::string& token) {
    return m_tokenizer ? m_tokenizer->TokenToId(token) : -1;
}

// ============================================================================
// Text Normalization (matching Python punc_norm)
// ============================================================================

std::string NormalizeTextForTTS(const std::string& text) {
    std::string result = text;
    
    if (result.empty()) {
        return "You need to add some text for me to talk.";
    }
    
    // Capitalize first letter
    if (!result.empty() && std::islower(static_cast<unsigned char>(result[0]))) {
        result[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(result[0])));
    }
    
    // Remove multiple spaces (collapse to single space)
    std::string collapsed;
    bool lastWasSpace = false;
    for (char c : result) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!lastWasSpace) {
                collapsed += ' ';
                lastWasSpace = true;
            }
        } else {
            collapsed += c;
            lastWasSpace = false;
        }
    }
    result = collapsed;
    
    // Replace uncommon punctuation (UTF-8 sequences)
    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"\xe2\x80\xa6", ", "},      // … (ellipsis)
        {":", ","},
        {"\xe2\x80\x94", "-"},       // — (em dash)
        {"\xe2\x80\x93", "-"},       // – (en dash)
        {" ,", ","},
        {"\xe2\x80\x9c", "\""},      // " (left double quote)
        {"\xe2\x80\x9d", "\""},      // " (right double quote)
        {"\xe2\x80\x98", "'"},       // ' (left single quote)
        {"\xe2\x80\x99", "'"},       // ' (right single quote)
    };
    
    for (const auto& [old_str, new_str] : replacements) {
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
    }
    
    // Remove trailing whitespace
    while (!result.empty() && std::isspace(static_cast<unsigned char>(result.back()))) {
        result.pop_back();
    }
    
    // Add period if no ending punctuation
    if (!result.empty()) {
        char last = result.back();
        if (last != '.' && last != '!' && last != '?' && last != '-' && last != ',') {
            result += '.';
        }
    }
    
    return result;
}

// ============================================================================
// Tokenizer (file loader) Implementation
// ============================================================================

std::optional<TokenData> Tokenizer::LoadTokenFile(const std::string& path) {
    m_lastError.clear();
    
    if (!fs::exists(path)) {
        m_lastError = "Token file not found: " + path;
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        m_lastError = "Failed to open token file: " + path;
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Read header
    TokenFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (!file || header.magic != TOKEN_FILE_MAGIC) {
        // Try fallback formats
        file.clear();
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Format 2: pretokenize.py format [num_tokens: uint32] [tokens: uint32...]
        uint32_t numTokens;
        file.read(reinterpret_cast<char*>(&numTokens), sizeof(numTokens));
        
        size_t expectedSize = sizeof(uint32_t) + numTokens * sizeof(uint32_t);
        if (file && fileSize == expectedSize && numTokens > 0 && numTokens < 100000) {
            TokenData data;
            data.tokenIds.resize(numTokens);
            
            for (uint32_t i = 0; i < numTokens; ++i) {
                uint32_t tok;
                file.read(reinterpret_cast<char*>(&tok), sizeof(tok));
                data.tokenIds[i] = static_cast<int64_t>(tok);
            }
            
            if (file) {
                spdlog::info("Loaded {} tokens from pretokenize format: {}", numTokens, path);
                return data;
            }
        }
        
        // Format 3: raw int64 array without header
        file.clear();
        file.seekg(0, std::ios::beg);
        
        if (fileSize % sizeof(int64_t) == 0 && fileSize > 0) {
            size_t numTokens64 = fileSize / sizeof(int64_t);
            TokenData data;
            data.tokenIds.resize(numTokens64);
            file.read(reinterpret_cast<char*>(data.tokenIds.data()), fileSize);
            
            if (file) {
                spdlog::info("Loaded {} tokens from raw int64 file: {}", numTokens64, path);
                return data;
            }
        }
        
        m_lastError = "Invalid token file format: " + path;
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Validate version
    if (header.version != TOKEN_FILE_VERSION) {
        m_lastError = "Unsupported token file version: " + std::to_string(header.version);
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    TokenData data;
    
    // Read tokens
    data.tokenIds.resize(header.numTokens);
    file.read(reinterpret_cast<char*>(data.tokenIds.data()), 
              header.numTokens * sizeof(int64_t));
    
    if (!file) {
        m_lastError = "Failed to read token data";
        spdlog::error("{}", m_lastError);
        return std::nullopt;
    }
    
    // Read original text if present
    if (header.textLength > 0) {
        data.originalText.resize(header.textLength);
        file.read(data.originalText.data(), header.textLength);
    }
    
    spdlog::info("Loaded {} tokens from: {}", data.tokenIds.size(), path);
    
    return data;
}

std::unordered_map<int, TokenData> Tokenizer::LoadBatchTokenFile(const std::string& path) {
    m_lastError.clear();
    std::unordered_map<int, TokenData> result;
    
    if (!fs::exists(path)) {
        m_lastError = "Batch token file not found: " + path;
        spdlog::error("{}", m_lastError);
        return result;
    }
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        m_lastError = "Failed to open batch token file: " + path;
        spdlog::error("{}", m_lastError);
        return result;
    }
    
    // Read batch file header
    uint32_t magic, version, numEntries;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&numEntries), sizeof(numEntries));
    
    if (!file || magic != TOKEN_FILE_MAGIC) {
        m_lastError = "Invalid batch token file format";
        spdlog::error("{}", m_lastError);
        return result;
    }
    
    // Read entries
    for (uint32_t i = 0; i < numEntries; ++i) {
        int32_t index;
        uint32_t numTokens, textLength;
        
        file.read(reinterpret_cast<char*>(&index), sizeof(index));
        file.read(reinterpret_cast<char*>(&numTokens), sizeof(numTokens));
        file.read(reinterpret_cast<char*>(&textLength), sizeof(textLength));
        
        if (!file) break;
        
        TokenData data;
        data.tokenIds.resize(numTokens);
        file.read(reinterpret_cast<char*>(data.tokenIds.data()), 
                  numTokens * sizeof(int64_t));
        
        if (textLength > 0) {
            data.originalText.resize(textLength);
            file.read(data.originalText.data(), textLength);
        }
        
        result[index] = std::move(data);
    }
    
    spdlog::info("Loaded {} batch entries from: {}", result.size(), path);
    
    return result;
}

TokenData Tokenizer::CreateTokenData(const std::vector<int64_t>& tokens) {
    TokenData data;
    data.tokenIds = tokens;
    return data;
}

bool Tokenizer::SaveTokenFile(const std::string& path, const TokenData& data) {
    m_lastError.clear();
    
    if (data.tokenIds.empty()) {
        m_lastError = "No tokens to save";
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Ensure parent directory exists
    fs::path filePath(path);
    if (filePath.has_parent_path()) {
        fs::create_directories(filePath.parent_path());
    }
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        m_lastError = "Failed to create token file: " + path;
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    // Write header
    TokenFileHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic = TOKEN_FILE_MAGIC;
    header.version = TOKEN_FILE_VERSION;
    header.numTokens = static_cast<uint32_t>(data.tokenIds.size());
    header.textLength = static_cast<uint32_t>(data.originalText.size());
    
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Write tokens
    file.write(reinterpret_cast<const char*>(data.tokenIds.data()),
               data.tokenIds.size() * sizeof(int64_t));
    
    // Write original text if present
    if (!data.originalText.empty()) {
        file.write(data.originalText.data(), data.originalText.size());
    }
    
    if (!file) {
        m_lastError = "Failed to write token data";
        spdlog::error("{}", m_lastError);
        return false;
    }
    
    spdlog::info("Saved {} tokens to: {}", data.tokenIds.size(), path);
    return true;
}

bool Tokenizer::IsTokenFile(const std::string& path) {
    if (!fs::exists(path)) {
        return false;
    }
    
    // Check extension
    fs::path filePath(path);
    std::string ext = filePath.extension().string();
    if (ext != ".tokens" && ext != ".bin") {
        return false;
    }
    
    // Try to read magic number
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    if (file && magic == TOKEN_FILE_MAGIC) {
        return true;
    }
    
    // Also accept raw int64 arrays
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    
    return (fileSize > 0 && fileSize % sizeof(int64_t) == 0);
}

} // namespace ChatterboxTTS

/**
 * @file TensorUtils.cpp
 * @brief Implementation of ONNX tensor utilities
 */

#include "tts/TensorUtils.h"
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstring>

namespace ChatterboxTTS {
namespace TensorUtils {

// ============================================================================
// Tensor Creation
// ============================================================================

Ort::Value CreateFloatTensor(Ort::MemoryInfo& memoryInfo,
                              std::vector<float>& data,
                              const TensorShape& shape) {
    return Ort::Value::CreateTensor<float>(
        memoryInfo, 
        data.data(), 
        data.size(),
        shape.data(), 
        shape.size()
    );
}

Ort::Value CreateInt64Tensor(Ort::MemoryInfo& memoryInfo,
                              std::vector<int64_t>& data,
                              const TensorShape& shape) {
    return Ort::Value::CreateTensor<int64_t>(
        memoryInfo,
        data.data(),
        data.size(),
        shape.data(),
        shape.size()
    );
}

Ort::Value CreateInt32Tensor(Ort::MemoryInfo& memoryInfo,
                              std::vector<int32_t>& data,
                              const TensorShape& shape) {
    return Ort::Value::CreateTensor<int32_t>(
        memoryInfo,
        data.data(),
        data.size(),
        shape.data(),
        shape.size()
    );
}

Ort::Value CreateEmptyFloatTensor(Ort::MemoryInfo& memoryInfo,
                                   const TensorShape& shape) {
    size_t count = GetElementCount(shape);
    // Note: We need persistent storage, so this creates a tensor with newly allocated memory
    // For empty KV-cache with 0 in shape, count will be 0
    if (count == 0) {
        // Create a tensor with shape that has 0 elements
        return Ort::Value::CreateTensor<float>(
            memoryInfo,
            nullptr,  // No data needed for 0-element tensor
            0,
            shape.data(),
            shape.size()
        );
    }
    
    // For non-empty tensors, we need to allocate storage
    // This is a limitation - caller needs to maintain the data vector
    static std::vector<float> emptyData;
    emptyData.resize(count, 0.0f);
    return CreateFloatTensor(memoryInfo, emptyData, shape);
}

Ort::Value CreateEmptyInt64Tensor(Ort::MemoryInfo& memoryInfo,
                                   const TensorShape& shape) {
    size_t count = GetElementCount(shape);
    if (count == 0) {
        return Ort::Value::CreateTensor<int64_t>(
            memoryInfo,
            nullptr,
            0,
            shape.data(),
            shape.size()
        );
    }
    
    static std::vector<int64_t> emptyData;
    emptyData.resize(count, 0);
    return CreateInt64Tensor(memoryInfo, emptyData, shape);
}

// ============================================================================
// FP16 Tensor Creation
// ============================================================================

Ort::Value CreateFloat16Tensor(Ort::MemoryInfo& memoryInfo,
                                std::vector<Float16>& data,
                                const TensorShape& shape) {
    return Ort::Value::CreateTensor<Float16>(
        memoryInfo, 
        data.data(), 
        data.size(),
        shape.data(), 
        shape.size()
    );
}

Ort::Value CreateFloat16TensorFromFloat(Ort::MemoryInfo& memoryInfo,
                                         const std::vector<float>& data,
                                         const TensorShape& shape,
                                         std::vector<Float16>& outFp16Data) {
    outFp16Data = ConvertToFp16(data);
    return CreateFloat16Tensor(memoryInfo, outFp16Data, shape);
}

Ort::Value CreateEmptyFloat16Tensor(Ort::MemoryInfo& memoryInfo,
                                     const TensorShape& shape) {
    size_t count = GetElementCount(shape);
    if (count == 0) {
        return Ort::Value::CreateTensor<Float16>(
            memoryInfo,
            nullptr,
            0,
            shape.data(),
            shape.size()
        );
    }
    
    static std::vector<Float16> emptyData;
    emptyData.resize(count);
    // Default construct all elements
    for (auto& elem : emptyData) {
        elem = Float16(0.0f);
    }
    return CreateFloat16Tensor(memoryInfo, emptyData, shape);
}

// ============================================================================
// FP16 Conversion Utilities
// ============================================================================

Float16 FloatToFp16(float fp32) {
    // Use ONNX Runtime's built-in conversion
    return Float16(fp32);
}

float Fp16ToFloat(Float16 fp16) {
    // Use ONNX Runtime's built-in conversion
    return fp16.ToFloat();
}

std::vector<Float16> ConvertToFp16(const std::vector<float>& fp32Data) {
    std::vector<Float16> fp16Data(fp32Data.size());
    for (size_t i = 0; i < fp32Data.size(); ++i) {
        fp16Data[i] = FloatToFp16(fp32Data[i]);
    }
    return fp16Data;
}

std::vector<float> ConvertToFp32(const std::vector<Float16>& fp16Data) {
    std::vector<float> fp32Data(fp16Data.size());
    for (size_t i = 0; i < fp16Data.size(); ++i) {
        fp32Data[i] = Fp16ToFloat(fp16Data[i]);
    }
    return fp32Data;
}

// ============================================================================
// Tensor Data Extraction
// ============================================================================

std::vector<float> ExtractFloatData(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    size_t count = typeInfo.GetElementCount();
    
    const float* data = tensor.GetTensorData<float>();
    return std::vector<float>(data, data + count);
}

std::vector<int64_t> ExtractInt64Data(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    size_t count = typeInfo.GetElementCount();
    
    const int64_t* data = tensor.GetTensorData<int64_t>();
    return std::vector<int64_t>(data, data + count);
}

std::vector<int32_t> ExtractInt32Data(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    size_t count = typeInfo.GetElementCount();
    
    const int32_t* data = tensor.GetTensorData<int32_t>();
    return std::vector<int32_t>(data, data + count);
}

std::vector<Float16> ExtractFloat16Data(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    size_t count = typeInfo.GetElementCount();
    
    const Float16* data = tensor.GetTensorData<Float16>();
    return std::vector<Float16>(data, data + count);
}

std::vector<float> ExtractFloatDataAuto(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    auto elementType = typeInfo.GetElementType();
    size_t count = typeInfo.GetElementCount();
    
    if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // Extract as fp16 and convert to fp32
        const Float16* data = tensor.GetTensorData<Float16>();
        std::vector<Float16> fp16Data(data, data + count);
        return ConvertToFp32(fp16Data);
    } else {
        // Assume fp32
        const float* data = tensor.GetTensorData<float>();
        return std::vector<float>(data, data + count);
    }
}

std::vector<float> ExtractFloatSlice(const Ort::Value& tensor, size_t offset, size_t count) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    auto elementType = typeInfo.GetElementType();
    size_t totalCount = typeInfo.GetElementCount();
    
    // Bounds check
    if (offset + count > totalCount) {
        count = (offset < totalCount) ? (totalCount - offset) : 0;
    }
    if (count == 0) {
        return {};
    }
    
    if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // Extract slice as fp16 and convert to fp32
        const Float16* data = tensor.GetTensorData<Float16>();
        std::vector<Float16> fp16Slice(data + offset, data + offset + count);
        return ConvertToFp32(fp16Slice);
    } else {
        // Assume fp32 - direct slice
        const float* data = tensor.GetTensorData<float>();
        return std::vector<float>(data + offset, data + offset + count);
    }
}

// ============================================================================
// Tensor Information
// ============================================================================

TensorShape GetShape(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    return typeInfo.GetShape();
}

size_t GetElementCount(const TensorShape& shape) {
    if (shape.empty()) {
        return 0;
    }
    
    // Handle shapes with 0 dimension
    for (auto dim : shape) {
        if (dim == 0) {
            return 0;
        }
    }
    
    return std::accumulate(shape.begin(), shape.end(), size_t(1), 
                           [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
}

std::string GetElementTypeName(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    auto elementType = typeInfo.GetElementType();
    
    switch (elementType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:    return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:     return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:   return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:   return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:   return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:    return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:     return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:   return "string";
        default:                                     return "unknown";
    }
}

bool IsFloatTensor(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    auto elementType = typeInfo.GetElementType();
    return elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
           elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

bool IsIntTensor(const Ort::Value& tensor) {
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    auto elementType = typeInfo.GetElementType();
    return elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
           elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
           elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
           elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

// ============================================================================
// Debug Utilities
// ============================================================================

std::string ShapeToString(const TensorShape& shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

void PrintTensorInfo(const Ort::Value& tensor, const std::string& name) {
    auto shape = GetShape(tensor);
    auto typeName = GetElementTypeName(tensor);
    size_t count = GetElementCount(shape);
    
    std::cout << "Tensor '" << name << "': "
              << "shape=" << ShapeToString(shape)
              << ", dtype=" << typeName
              << ", elements=" << count
              << std::endl;
}

void PrintTensorSample(const Ort::Value& tensor, const std::string& name, size_t maxValues) {
    PrintTensorInfo(tensor, name);
    
    auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    auto elementType = typeInfo.GetElementType();
    size_t count = typeInfo.GetElementCount();
    size_t printCount = std::min(count, maxValues);
    
    std::cout << "  Values: [";
    
    if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = tensor.GetTensorData<float>();
        for (size_t i = 0; i < printCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << data[i];
        }
    } else if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* data = tensor.GetTensorData<int64_t>();
        for (size_t i = 0; i < printCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << data[i];
        }
    } else if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t* data = tensor.GetTensorData<int32_t>();
        for (size_t i = 0; i < printCount; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << data[i];
        }
    }
    
    if (count > maxValues) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

// ============================================================================
// Session Helpers
// ============================================================================

std::vector<std::string> GetInputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
    std::vector<std::string> names;
    size_t numInputs = session.GetInputCount();
    names.reserve(numInputs);
    
    for (size_t i = 0; i < numInputs; ++i) {
        auto namePtr = session.GetInputNameAllocated(i, allocator);
        names.push_back(namePtr.get());
    }
    
    return names;
}

std::vector<std::string> GetOutputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
    std::vector<std::string> names;
    size_t numOutputs = session.GetOutputCount();
    names.reserve(numOutputs);
    
    for (size_t i = 0; i < numOutputs; ++i) {
        auto namePtr = session.GetOutputNameAllocated(i, allocator);
        names.push_back(namePtr.get());
    }
    
    return names;
}

void PrintSessionInfo(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator, const std::string& modelName) {
    std::cout << "\n=== Model: " << modelName << " ===" << std::endl;
    
    // Print inputs
    auto inputNames = GetInputNames(session, allocator);
    std::cout << "Inputs (" << inputNames.size() << "):" << std::endl;
    for (size_t i = 0; i < inputNames.size(); ++i) {
        auto typeInfo = session.GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        auto shape = tensorInfo.GetShape();
        auto elementType = tensorInfo.GetElementType();
        
        std::string typeName;
        switch (elementType) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   typeName = "float32"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: typeName = "float16"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   typeName = "int64"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   typeName = "int32"; break;
            default:                                    typeName = "other"; break;
        }
        
        std::cout << "  [" << i << "] " << inputNames[i] 
                  << " : " << ShapeToString(shape)
                  << " (" << typeName << ")" << std::endl;
    }
    
    // Print outputs
    auto outputNames = GetOutputNames(session, allocator);
    std::cout << "Outputs (" << outputNames.size() << "):" << std::endl;
    for (size_t i = 0; i < outputNames.size(); ++i) {
        auto typeInfo = session.GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        auto shape = tensorInfo.GetShape();
        auto elementType = tensorInfo.GetElementType();
        
        std::string typeName;
        switch (elementType) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   typeName = "float32"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: typeName = "float16"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   typeName = "int64"; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   typeName = "int32"; break;
            default:                                    typeName = "other"; break;
        }
        
        std::cout << "  [" << i << "] " << outputNames[i]
                  << " : " << ShapeToString(shape)
                  << " (" << typeName << ")" << std::endl;
    }
    std::cout << std::endl;
}

} // namespace TensorUtils
} // namespace ChatterboxTTS

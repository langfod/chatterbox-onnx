/**
 * @file TensorUtils.h
 * @brief Utilities for creating and manipulating ONNX Runtime tensors
 * 
 * Helper functions for:
 * - Creating tensors from std::vector
 * - Extracting data from tensors to std::vector
 * - Getting tensor shapes and info
 * - Debug printing
 */

#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <cstdint>

namespace ChatterboxTTS {

/**
 * @brief Tensor shape as vector of int64_t
 */
using TensorShape = std::vector<int64_t>;

/**
 * @brief Type alias for fp16 values (ONNX Runtime's Float16_t)
 */
using Float16 = Ort::Float16_t;

/**
 * @brief Tensor utilities namespace
 */
namespace TensorUtils {

// ============================================================================
// Tensor Creation
// ============================================================================

/**
 * @brief Create a float tensor from vector data
 * @param memoryInfo Memory info for allocation
 * @param data Float data
 * @param shape Tensor shape
 * @return Ort::Value tensor
 */
Ort::Value CreateFloatTensor(Ort::MemoryInfo& memoryInfo,
                              std::vector<float>& data,
                              const TensorShape& shape);

/**
 * @brief Create an int64 tensor from vector data
 * @param memoryInfo Memory info for allocation
 * @param data Int64 data
 * @param shape Tensor shape
 * @return Ort::Value tensor
 */
Ort::Value CreateInt64Tensor(Ort::MemoryInfo& memoryInfo,
                              std::vector<int64_t>& data,
                              const TensorShape& shape);

/**
 * @brief Create an int32 tensor from vector data
 * @param memoryInfo Memory info for allocation
 * @param data Int32 data
 * @param shape Tensor shape
 * @return Ort::Value tensor
 */
Ort::Value CreateInt32Tensor(Ort::MemoryInfo& memoryInfo,
                              std::vector<int32_t>& data,
                              const TensorShape& shape);

/**
 * @brief Create an empty float tensor with given shape (filled with zeros)
 * @param memoryInfo Memory info for allocation
 * @param shape Tensor shape
 * @return Ort::Value tensor
 */
Ort::Value CreateEmptyFloatTensor(Ort::MemoryInfo& memoryInfo,
                                   const TensorShape& shape);

/**
 * @brief Create an empty int64 tensor with given shape (filled with zeros)
 * @param memoryInfo Memory info for allocation
 * @param shape Tensor shape
 * @return Ort::Value tensor
 */
Ort::Value CreateEmptyInt64Tensor(Ort::MemoryInfo& memoryInfo,
                                   const TensorShape& shape);

/**
 * @brief Create a float16 tensor from float16 data
 * @param memoryInfo Memory info for allocation
 * @param data Float16 data
 * @param shape Tensor shape
 * @return Ort::Value tensor
 */
Ort::Value CreateFloat16Tensor(Ort::MemoryInfo& memoryInfo,
                                std::vector<Float16>& data,
                                const TensorShape& shape);

/**
 * @brief Create a float16 tensor from float32 data (with conversion)
 * @param memoryInfo Memory info for allocation
 * @param data Float32 data to convert
 * @param shape Tensor shape
 * @param outFp16Data Output vector to hold fp16 data (must stay alive)
 * @return Ort::Value tensor
 */
Ort::Value CreateFloat16TensorFromFloat(Ort::MemoryInfo& memoryInfo,
                                         const std::vector<float>& data,
                                         const TensorShape& shape,
                                         std::vector<Float16>& outFp16Data);

/**
 * @brief Create an empty float16 tensor with given shape
 * @param memoryInfo Memory info for allocation
 * @param shape Tensor shape
 * @return Ort::Value tensor
 */
Ort::Value CreateEmptyFloat16Tensor(Ort::MemoryInfo& memoryInfo,
                                     const TensorShape& shape);

// ============================================================================
// Tensor Data Extraction
// ============================================================================

/**
 * @brief Extract float data from tensor to vector
 * @param tensor Input tensor
 * @return Vector of float values
 */
std::vector<float> ExtractFloatData(const Ort::Value& tensor);

/**
 * @brief Extract int64 data from tensor to vector
 * @param tensor Input tensor
 * @return Vector of int64 values
 */
std::vector<int64_t> ExtractInt64Data(const Ort::Value& tensor);

/**
 * @brief Extract int32 data from tensor to vector
 * @param tensor Input tensor
 * @return Vector of int32 values
 */
std::vector<int32_t> ExtractInt32Data(const Ort::Value& tensor);

/**
 * @brief Extract float16 data from tensor to vector
 * @param tensor Input tensor
 * @return Vector of Float16 values
 */
std::vector<Float16> ExtractFloat16Data(const Ort::Value& tensor);

/**
 * @brief Extract float data from tensor (auto-converts fp16 to fp32)
 * @param tensor Input tensor (can be fp32 or fp16)
 * @return Vector of float values
 */
std::vector<float> ExtractFloatDataAuto(const Ort::Value& tensor);

/**
 * @brief Extract a slice of float data from tensor (auto-converts fp16 to fp32)
 * @param tensor Input tensor (can be fp32 or fp16)
 * @param offset Start offset in elements
 * @param count Number of elements to extract
 * @return Vector of float values
 * 
 * This is more efficient than ExtractFloatDataAuto when you only need a portion
 * of a large tensor (e.g., last position logits from LM output).
 */
std::vector<float> ExtractFloatSlice(const Ort::Value& tensor, size_t offset, size_t count);

// ============================================================================
// FP16 Conversion Utilities
// ============================================================================

/**
 * @brief Convert fp32 to fp16
 * @param fp32 Float32 value
 * @return Float16 value
 */
Float16 FloatToFp16(float fp32);

/**
 * @brief Convert fp16 to fp32
 * @param fp16 Float16 value
 * @return Float32 value
 */
float Fp16ToFloat(Float16 fp16);

/**
 * @brief Convert vector of fp32 to fp16
 * @param fp32Data Input float32 data
 * @return Vector of Float16 values
 */
std::vector<Float16> ConvertToFp16(const std::vector<float>& fp32Data);

/**
 * @brief Convert vector of fp16 to fp32
 * @param fp16Data Input float16 data
 * @return Vector of float values
 */
std::vector<float> ConvertToFp32(const std::vector<Float16>& fp16Data);

// ============================================================================
// Tensor Information
// ============================================================================

/**
 * @brief Get the shape of a tensor
 * @param tensor Input tensor
 * @return Shape as vector of dimensions
 */
TensorShape GetShape(const Ort::Value& tensor);

/**
 * @brief Calculate total number of elements in shape
 * @param shape Tensor shape
 * @return Number of elements
 */
size_t GetElementCount(const TensorShape& shape);

/**
 * @brief Get tensor element type as string
 * @param tensor Input tensor
 * @return Type string (e.g., "float32", "int64")
 */
std::string GetElementTypeName(const Ort::Value& tensor);

/**
 * @brief Check if tensor is a float type
 * @param tensor Input tensor
 * @return true if float32 or float16
 */
bool IsFloatTensor(const Ort::Value& tensor);

/**
 * @brief Check if tensor is an integer type
 * @param tensor Input tensor
 * @return true if int64, int32, etc.
 */
bool IsIntTensor(const Ort::Value& tensor);

// ============================================================================
// Debug Utilities
// ============================================================================

/**
 * @brief Print tensor information to stdout
 * @param tensor Tensor to describe
 * @param name Label for the tensor
 */
void PrintTensorInfo(const Ort::Value& tensor, const std::string& name);

/**
 * @brief Format shape as string
 * @param shape Tensor shape
 * @return String like "[1, 32, 1024]"
 */
std::string ShapeToString(const TensorShape& shape);

/**
 * @brief Print first N values of tensor (for debugging)
 * @param tensor Tensor to print
 * @param name Label
 * @param maxValues Maximum values to print
 */
void PrintTensorSample(const Ort::Value& tensor, const std::string& name, size_t maxValues = 10);

// ============================================================================
// Session Helpers
// ============================================================================

/**
 * @brief Get input names from an ONNX session
 * @param session ONNX session
 * @param allocator Allocator for string allocation
 * @return Vector of input names
 */
std::vector<std::string> GetInputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator);

/**
 * @brief Get output names from an ONNX session
 * @param session ONNX session
 * @param allocator Allocator for string allocation
 * @return Vector of output names
 */
std::vector<std::string> GetOutputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator);

/**
 * @brief Print all input/output info for a session
 * @param session ONNX session
 * @param allocator Allocator
 * @param modelName Name to display
 */
void PrintSessionInfo(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator, const std::string& modelName);

} // namespace TensorUtils

} // namespace ChatterboxTTS

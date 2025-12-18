#pragma once
#include <string>
#include <vector>

namespace SkyrimNet {
namespace Utils {

    enum class GpuProvider {
        AUTO_DETECT,    // Automatically detect best available GPU
        CUDA,           // NVIDIA CUDA
        ROCM,           // AMD ROCm
        OPENVINO,       // Intel OpenVINO (cross-platform)
        CPU_FALLBACK    // CPU fallback
    };


class OnnxUtil {
public:
    static OnnxUtil& GetInstance();
    GpuProvider DetectBestProvider() const;
    std::vector<std::string> SetupProviders(GpuProvider preferred) const;

private:
    OnnxUtil() = default;
};

} // namespace Utils
} // namespace SkyrimNet

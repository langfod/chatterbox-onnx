#include "common/utils/OnnxUtil.h"
#include <onnxruntime_cxx_api.h>

namespace SkyrimNet {
namespace Utils {

OnnxUtil& OnnxUtil::GetInstance() {
    static OnnxUtil instance;
    return instance;
}

GpuProvider OnnxUtil::DetectBestProvider() const {
    try {
        auto providers = Ort::GetAvailableProviders();
        for (const auto& provider : providers) {
            if (provider == "CUDAExecutionProvider") {
                return GpuProvider::CUDA;
            }
            if (provider == "ROCMExecutionProvider") {
                return GpuProvider::ROCM;
            }
            if (provider == "OpenVINOExecutionProvider") {
                return GpuProvider::OPENVINO;
            }
        }
    } catch (...) {
        // Ignore errors in provider detection
    }
    return GpuProvider::CPU_FALLBACK;
}

std::vector<std::string> OnnxUtil::SetupProviders(GpuProvider preferred) const {
    std::vector<std::string> providers;
    auto actual = (preferred == GpuProvider::AUTO_DETECT) ? DetectBestProvider() : preferred;
    switch (actual) {
        case GpuProvider::CUDA:
            providers.push_back("CUDAExecutionProvider");
            break;
        case GpuProvider::ROCM:
            providers.push_back("ROCMExecutionProvider");
            break;
        case GpuProvider::OPENVINO:
            providers.push_back("OpenVINOExecutionProvider");
            break;
        default:
            break;
    }
    providers.push_back("CPUExecutionProvider");
    return providers;
}

} // namespace Utils
} // namespace SkyrimNet

#pragma once

#include <string>
#include <vector>

enum class EnvironmentType {
    WEB_BROWSER,
    APPLE_SILICON,
    WINDOWS,
    LINUX,
    IOS,
    ANDROID,
    UNKNOWN
};

enum class BackendType {
    CPU,
    CUDA,
    COREML,
    DIRECTML,
    TENSORRT,
    ROCM,
    WEBGL,
    WEBGPU,
    WEBNN,
    NNAPI,
    UNKNOWN
};

struct EnvironmentInfo {
    EnvironmentType type;
    std::string description;
    std::vector<BackendType> available_backends;
    BackendType optimal_backend;
    std::string device_info;
    bool has_npu;
    bool has_gpu;
};

class EnvironmentDetector {
public:
    EnvironmentDetector();
    ~EnvironmentDetector();

    EnvironmentInfo detectEnvironment();
    std::vector<BackendType> getAvailableBackends();
    BackendType getOptimalBackend();
    std::string getDeviceInfo();

private:
    EnvironmentInfo env_info_;
    bool detected_;

    // Detection methods
    EnvironmentType detectPlatform();
    std::vector<BackendType> detectBackends(EnvironmentType platform);
    BackendType selectOptimalBackend(const std::vector<BackendType>& backends);
    std::string getPlatformDescription(EnvironmentType platform);
    
    // Platform-specific detection
    bool isAppleSilicon();
    bool hasCUDA();
    bool hasROCm();
    bool hasDirectML();
    bool hasCoreML();
    bool hasNNAPI();
    bool isWebEnvironment();
    
    // Backend capability checks
    bool checkCUDAVersion();
    bool checkROCmVersion();
    bool checkCoreMLSupport();
    bool checkDirectMLSupport();
};

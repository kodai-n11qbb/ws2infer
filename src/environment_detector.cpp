#include "environment_detector.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef __APPLE__
    #include <sys/sysctl.h>
    #include <TargetConditionals.h>
#endif

#ifdef _WIN32
    #include <windows.h>
    #include <d3d12.h>
#endif

#ifdef __linux__
    #include <unistd.h>
    #include <sys/utsname.h>
#endif

EnvironmentDetector::EnvironmentDetector() : detected_(false) {
}

EnvironmentDetector::~EnvironmentDetector() {
}

EnvironmentInfo EnvironmentDetector::detectEnvironment() {
    if (detected_) {
        return env_info_;
    }

    env_info_.type = detectPlatform();
    env_info_.description = getPlatformDescription(env_info_.type);
    env_info_.available_backends = detectBackends(env_info_.type);
    env_info_.optimal_backend = selectOptimalBackend(env_info_.available_backends);
    env_info_.device_info = getDeviceInfo();
    
    // Check for NPU/GPU availability
    env_info_.has_npu = (env_info_.optimal_backend == BackendType::COREML || 
                        env_info_.optimal_backend == BackendType::NNAPI);
    env_info_.has_gpu = (env_info_.optimal_backend == BackendType::CUDA ||
                        env_info_.optimal_backend == BackendType::DIRECTML ||
                        env_info_.optimal_backend == BackendType::TENSORRT ||
                        env_info_.optimal_backend == BackendType::ROCM ||
                        env_info_.optimal_backend == BackendType::WEBGL ||
                        env_info_.optimal_backend == BackendType::WEBGPU);

    detected_ = true;
    
    std::cout << "Environment Detection Results:" << std::endl;
    std::cout << "Platform: " << env_info_.description << std::endl;
    std::cout << "Optimal Backend: ";
    switch (env_info_.optimal_backend) {
        case BackendType::CPU: std::cout << "CPU"; break;
        case BackendType::CUDA: std::cout << "CUDA"; break;
        case BackendType::COREML: std::cout << "CoreML"; break;
        case BackendType::DIRECTML: std::cout << "DirectML"; break;
        case BackendType::TENSORRT: std::cout << "TensorRT"; break;
        case BackendType::ROCM: std::cout << "ROCm"; break;
        case BackendType::WEBGL: std::cout << "WebGL"; break;
        case BackendType::WEBGPU: std::cout << "WebGPU"; break;
        case BackendType::WEBNN: std::cout << "WebNN"; break;
        case BackendType::NNAPI: std::cout << "NNAPI"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;
    std::cout << "Device Info: " << env_info_.device_info << std::endl;
    
    return env_info_;
}

EnvironmentType EnvironmentDetector::detectPlatform() {
#ifdef __APPLE__
    #if TARGET_OS_IPHONE
        return EnvironmentType::IOS;
    #elif TARGET_OS_MAC
        if (isAppleSilicon()) {
            return EnvironmentType::APPLE_SILICON;
        } else {
            return EnvironmentType::UNKNOWN; // Intel Mac
        }
    #endif
#endif

#ifdef _WIN32
    return EnvironmentType::WINDOWS;
#endif

#ifdef __linux__
    return EnvironmentType::LINUX;
#endif

#ifdef __ANDROID__
    return EnvironmentType::ANDROID;
#endif

    // Check for web environment (browser detection)
    if (isWebEnvironment()) {
        return EnvironmentType::WEB_BROWSER;
    }

    return EnvironmentType::UNKNOWN;
}

std::vector<BackendType> EnvironmentDetector::detectBackends(EnvironmentType platform) {
    std::vector<BackendType> backends;
    
    // CPU is always available
    backends.push_back(BackendType::CPU);
    
    switch (platform) {
        case EnvironmentType::APPLE_SILICON:
            if (hasCoreML()) {
                backends.push_back(BackendType::COREML);
            }
            break;
            
        case EnvironmentType::WINDOWS:
            if (hasDirectML()) {
                backends.push_back(BackendType::DIRECTML);
            }
            if (hasCUDA()) {
                backends.push_back(BackendType::CUDA);
                backends.push_back(BackendType::TENSORRT);
            }
            break;
            
        case EnvironmentType::LINUX:
            if (hasCUDA()) {
                backends.push_back(BackendType::CUDA);
                backends.push_back(BackendType::TENSORRT);
            }
            if (hasROCm()) {
                backends.push_back(BackendType::ROCM);
            }
            break;
            
        case EnvironmentType::IOS:
            if (hasCoreML()) {
                backends.push_back(BackendType::COREML);
            }
            break;
            
        case EnvironmentType::ANDROID:
            if (hasNNAPI()) {
                backends.push_back(BackendType::NNAPI);
            }
            break;
            
        case EnvironmentType::WEB_BROWSER:
            backends.push_back(BackendType::WEBGL);
            backends.push_back(BackendType::WEBGPU);
            backends.push_back(BackendType::WEBNN);
            break;
            
        default:
            break;
    }
    
    return backends;
}

BackendType EnvironmentDetector::selectOptimalBackend(const std::vector<BackendType>& backends) {
    // Priority order for optimal performance
    std::vector<BackendType> priority = {
        BackendType::COREML,      // Apple Neural Engine
        BackendType::NNAPI,       // Android NPU
        BackendType::TENSORRT,    // NVIDIA TensorRT
        BackendType::CUDA,        // NVIDIA GPU
        BackendType::ROCM,        // AMD GPU
        BackendType::DIRECTML,    // Windows GPU
        BackendType::WEBNN,       // Web Neural Network
        BackendType::WEBGPU,      // Web GPU
        BackendType::WEBGL,       // Web GL
        BackendType::CPU          // Fallback
    };
    
    for (BackendType backend : priority) {
        if (std::find(backends.begin(), backends.end(), backend) != backends.end()) {
            return backend;
        }
    }
    
    return BackendType::CPU;
}

std::string EnvironmentDetector::getPlatformDescription(EnvironmentType platform) {
    switch (platform) {
        case EnvironmentType::WEB_BROWSER: return "Web Browser";
        case EnvironmentType::APPLE_SILICON: return "Apple Silicon (M-series)";
        case EnvironmentType::WINDOWS: return "Windows";
        case EnvironmentType::LINUX: return "Linux";
        case EnvironmentType::IOS: return "iOS";
        case EnvironmentType::ANDROID: return "Android";
        default: return "Unknown Platform";
    }
}

bool EnvironmentDetector::isAppleSilicon() {
#ifdef __APPLE__
    size_t size;
    sysctlbyname("hw.machine", NULL, &size, NULL, 0);
    char* machine = new char[size];
    sysctlbyname("hw.machine", machine, &size, NULL, 0);
    std::string machine_str(machine);
    delete[] machine;
    
    // Check for Apple Silicon identifiers
    return machine_str.find("arm64") != std::string::npos ||
           machine_str.find("Apple") != std::string::npos;
#else
    return false;
#endif
}

bool EnvironmentDetector::hasCUDA() {
    // Check for CUDA installation
    std::ifstream cuda_file("/usr/local/cuda/version.txt");
    if (cuda_file.is_open()) {
        return checkCUDAVersion();
    }
    
    // Also check common CUDA paths
    std::vector<std::string> cuda_paths = {
        "/usr/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
        "/opt/cuda/bin/nvidia-smi"
    };
    
    for (const auto& path : cuda_paths) {
        if (std::ifstream(path).good()) {
            return checkCUDAVersion();
        }
    }
    
    return false;
}

bool EnvironmentDetector::checkCUDAVersion() {
    // This would typically run nvidia-smi and check version
    // For simplicity, just check if the command exists
    return system("which nvidia-smi > /dev/null 2>&1") == 0;
}

bool EnvironmentDetector::hasROCm() {
    // Check for ROCm installation
    std::vector<std::string> rocm_paths = {
        "/opt/rocm/bin/rocminfo",
        "/usr/bin/rocminfo"
    };
    
    for (const auto& path : rocm_paths) {
        if (std::ifstream(path).good()) {
            return checkROCmVersion();
        }
    }
    
    return false;
}

bool EnvironmentDetector::checkROCmVersion() {
    return system("which rocminfo > /dev/null 2>&1") == 0;
}

bool EnvironmentDetector::hasDirectML() {
#ifdef _WIN32
    // Check for DirectML availability
    // This is a simplified check
    return system("where dxdiag > nul 2>&1") == 0;
#else
    return false;
#endif
}

bool EnvironmentDetector::hasCoreML() {
#ifdef __APPLE__
    // CoreML is available on modern Apple platforms
    return true;
#else
    return false;
#endif
}

bool EnvironmentDetector::hasNNAPI() {
#ifdef __ANDROID__
    // NNAPI is available on Android 8.0+
    return true;
#else
    return false;
#endif
}

bool EnvironmentDetector::isWebEnvironment() {
    // Check for browser environment indicators
    const char* user_agent = std::getenv("USER_AGENT");
    const char* http_user_agent = std::getenv("HTTP_USER_AGENT");
    
    return (user_agent && std::string(user_agent).find("Mozilla") != std::string::npos) ||
           (http_user_agent && std::string(http_user_agent).find("Mozilla") != std::string::npos);
}

std::string EnvironmentDetector::getDeviceInfo() {
    std::ostringstream info;
    
#ifdef __APPLE__
    size_t size;
    sysctlbyname("hw.model", NULL, &size, NULL, 0);
    char* model = new char[size];
    sysctlbyname("hw.model", model, &size, NULL, 0);
    info << "Apple " << model;
    delete[] model;
    
    // Get memory info
    size_t mem_size;
    uint64_t mem_bytes;
    size = sizeof(mem_bytes);
    sysctlbyname("hw.memsize", &mem_bytes, &size, NULL, 0);
    info << ", " << (mem_bytes / (1024 * 1024 * 1024)) << "GB RAM";
    
#elif _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    info << "Windows System, " << (statex.ullTotalPhys / (1024 * 1024 * 1024)) << "GB RAM";
    
#elif __linux__
    struct utsname sysinfo;
    uname(&sysinfo);
    info << sysinfo.sysname << " " << sysinfo.release << " " << sysinfo.machine;
    
    // Get memory info
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                std::istringstream iss(line);
                std::string label;
                long mem_kb;
                iss >> label >> mem_kb;
                info << ", " << (mem_kb / (1024 * 1024)) << "GB RAM";
                break;
            }
        }
    }
#else
    info << "Unknown System";
#endif
    
    return info.str();
}

std::vector<BackendType> EnvironmentDetector::getAvailableBackends() {
    if (!detected_) {
        detectEnvironment();
    }
    return env_info_.available_backends;
}

BackendType EnvironmentDetector::getOptimalBackend() {
    if (!detected_) {
        detectEnvironment();
    }
    return env_info_.optimal_backend;
}


#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>
#include "environment_detector.h"

// Forward declarations for ONNX Runtime
#ifdef __cplusplus
extern "C" {
#endif
typedef struct OrtApi* OrtApiPtr;
typedef struct OrtSession* OrtSessionPtr;
typedef struct OrtEnv* OrtEnvPtr;
typedef struct OrtMemoryInfo* OrtMemoryInfoPtr;
typedef struct OrtSessionOptions* OrtSessionOptionsPtr;
typedef struct OrtValue* OrtValuePtr;
typedef struct OrtAllocator* OrtAllocatorPtr;
#ifdef __cplusplus
}
#endif

struct InferenceResult {
    std::vector<float> output_data;
    std::vector<int64_t> output_shape;
    std::string output_name;
    double inference_time_ms;
    bool success;
    std::string error_message;
};

class ONNXInference {
public:
    ONNXInference(const std::string& model_path, const EnvironmentInfo& env_info);
    ~ONNXInference();

    bool initialize();
    InferenceResult runInference(const cv::Mat& image);
    InferenceResult runInference(const std::vector<float>& input_data, 
                                const std::vector<int64_t>& input_shape);

    // Model information
    std::vector<std::string> getInputNames() const;
    std::vector<std::string> getOutputNames() const;
    std::vector<int64_t> getInputShape(const std::string& input_name = "") const;
    std::vector<int64_t> getOutputShape(const std::string& output_name = "") const;

    // Configuration
    void setNumThreads(int num_threads);
    void setOptimizationLevel(int level);
    void setDeviceID(int device_id);

private:
    std::string model_path_;
    EnvironmentInfo env_info_;
    
    // ONNX Runtime objects
    OrtEnvPtr env_;
    OrtSessionPtr session_;
    OrtSessionOptionsPtr session_options_;
    OrtMemoryInfoPtr memory_info_;
    OrtAllocatorPtr allocator_;
    
    // Model metadata
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::map<std::string, std::vector<int64_t>> input_shapes_;
    std::map<std::string, std::vector<int64_t>> output_shapes_;
    
    // Configuration
    int num_threads_;
    int optimization_level_;
    int device_id_;
    bool initialized_;
    
    // Initialization methods
    bool initializeEnvironment();
    bool initializeSession();
    bool loadModelMetadata();
    void configureSessionOptions();
    
    // Backend-specific configuration
    void configureCPUBackend();
    void configureCUDABackend();
    void configureCoreMLBackend();
    void configureDirectMLBackend();
    void configureTensorRTBackend();
    void configureROCmBackend();
    void configureWebBackends();
    
    // Utility methods
    std::string getBackendString(BackendType backend) const;
    bool checkBackendAvailability(BackendType backend) const;
    void* createTensor(const std::vector<float>& data, const std::vector<int64_t>& shape);
    void releaseTensor(void* tensor);
    std::vector<float> extractTensorData(void* tensor);
};

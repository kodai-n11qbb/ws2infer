#include "onnx_inference.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>

// Include ONNX Runtime headers
#include <onnxruntime_c_api.h>

ONNXInference::ONNXInference(const std::string& model_path, const EnvironmentInfo& env_info)
    : model_path_(model_path), env_info_(env_info), 
      env_(nullptr), session_(nullptr), session_options_(nullptr),
      memory_info_(nullptr), allocator_(nullptr),
      num_threads_(0), optimization_level_(1), device_id_(0), initialized_(false) {
}

ONNXInference::~ONNXInference() {
    if (session_) {
        // Release session
        OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort->ReleaseSession(session_);
    }
    
    if (session_options_) {
        OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort->ReleaseSessionOptions(session_options_);
    }
    
    if (env_) {
        OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort->ReleaseEnv(env_);
    }
}

bool ONNXInference::initialize() {
    if (initialized_) {
        return true;
    }

    std::cout << "Initializing ONNX Inference with backend: " 
              << getBackendString(env_info_.optimal_backend) << std::endl;

    if (!initializeEnvironment()) {
        std::cerr << "Failed to initialize ONNX Runtime environment" << std::endl;
        return false;
    }

    if (!initializeSession()) {
        std::cerr << "Failed to initialize ONNX session" << std::endl;
        return false;
    }

    if (!loadModelMetadata()) {
        std::cerr << "Failed to load model metadata" << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "ONNX Inference initialized successfully" << std::endl;
    
    return true;
}

bool ONNXInference::initializeEnvironment() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Create environment
    OrtStatus* status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ws2infer", &env_);
    if (status) {
        std::cerr << "Failed to create ONNX environment: " << ort->GetErrorMessage(status) << std::endl;
        return false;
    }

    // Create session options
    status = ort->CreateSessionOptions(&session_options_);
    if (status) {
        std::cerr << "Failed to create session options: " << ort->GetErrorMessage(status) << std::endl;
        return false;
    }

    // Configure session options based on backend
    configureSessionOptions();

    // Create memory info
    status = ort->CreateMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_);
    if (status) {
        std::cerr << "Failed to create memory info: " << ort->GetErrorMessage(status) << std::endl;
        return false;
    }

    return true;
}

void ONNXInference::configureSessionOptions() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // Set number of threads
    if (num_threads_ > 0) {
        ort->SetIntraOpNumThreads(session_options_, num_threads_);
        ort->SetInterOpNumThreads(session_options_, num_threads_);
    }

    // Set optimization level
    GraphOptimizationLevel opt_level;
    switch (optimization_level_) {
        case 0: opt_level = ORT_DISABLE_ALL; break;
        case 1: opt_level = ORT_ENABLE_BASIC; break;
        case 2: opt_level = ORT_ENABLE_EXTENDED; break;
        default: opt_level = ORT_ENABLE_ALL; break;
    }
    ort->SetSessionGraphOptimizationLevel(session_options_, opt_level);

    // Configure backend-specific options
    switch (env_info_.optimal_backend) {
        case BackendType::CPU:
            configureCPUBackend();
            break;
        case BackendType::CUDA:
        case BackendType::TENSORRT:
            configureCUDABackend();
            break;
        case BackendType::COREML:
            configureCoreMLBackend();
            break;
        case BackendType::DIRECTML:
            configureDirectMLBackend();
            break;
        case BackendType::ROCM:
            configureROCmBackend();
            break;
        case BackendType::WEBGL:
        case BackendType::WEBGPU:
        case BackendType::WEBNN:
            configureWebBackends();
            break;
        default:
            configureCPUBackend();
            break;
    }
}

void ONNXInference::configureCPUBackend() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // CPU-specific optimizations
    ort->AddSessionConfigEntry(session_options_, "session.use_env_allocators", "0");
    ort->AddSessionConfigEntry(session_options_, "session.use_deterministic_compute", "1");
}

void ONNXInference::configureCUDABackend() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Add CUDA execution provider
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_id_;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = SIZE_MAX;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.do_copy_in_default_stream = 1;
    
    OrtStatus* status = ort->SessionOptionsAppendExecutionProvider_CUDA(session_options_, &cuda_options);
    if (status) {
        std::cerr << "Failed to add CUDA provider: " << ort->GetErrorMessage(status) << std::endl;
        ort->ReleaseStatus(status);
        return;
    }

    // If TensorRT is available and requested
    if (env_info_.optimal_backend == BackendType::TENSORRT) {
        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = device_id_;
        trt_options.trt_max_workspace_size = 1 << 30; // 1GB
        trt_options.trt_max_partition_iterations = 1000;
        trt_options.trt_min_subgraph_size = 1;
        trt_options.trt_fp16_enable = true;
        trt_options.trt_int8_enable = false;
        trt_options.trt_int8_use_calibration = false;
        trt_options.trt_engine_cache_enable = true;
        trt_options.trt_engine_cache_path = "./trt_cache";
        
        status = ort->SessionOptionsAppendExecutionProvider_TensorRT(session_options_, &trt_options);
        if (status) {
            std::cerr << "Failed to add TensorRT provider: " << ort->GetErrorMessage(status) << std::endl;
            ort->ReleaseStatus(status);
        }
    }
}

void ONNXInference::configureCoreMLBackend() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Add CoreML execution provider
    OrtStatus* status = ort->SessionOptionsAppendExecutionProvider_CoreML(session_options_, 0);
    if (status) {
        std::cerr << "Failed to add CoreML provider: " << ort->GetErrorMessage(status) << std::endl;
        ort->ReleaseStatus(status);
    }
}

void ONNXInference::configureDirectMLBackend() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Add DirectML execution provider
    OrtStatus* status = ort->SessionOptionsAppendExecutionProvider_DML(session_options_, device_id_);
    if (status) {
        std::cerr << "Failed to add DirectML provider: " << ort->GetErrorMessage(status) << std::endl;
        ort->ReleaseStatus(status);
    }
}

void ONNXInference::configureROCmBackend() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Add ROCm execution provider
    OrtStatus* status = ort->SessionOptionsAppendExecutionProvider_ROCm(session_options_, device_id_);
    if (status) {
        std::cerr << "Failed to add ROCm provider: " << ort->GetErrorMessage(status) << std::endl;
        ort->ReleaseStatus(status);
    }
}

void ONNXInference::configureWebBackends() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Web-specific configurations
    ort->AddSessionConfigEntry(session_options_, "session.use_arena", "1");
    ort->AddSessionConfigEntry(session_options_, "session.enable_mem_pattern", "0");
}

bool ONNXInference::initializeSession() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Check if model file exists
    std::ifstream model_file(model_path_);
    if (!model_file.good()) {
        std::cerr << "Model file not found: " << model_path_ << std::endl;
        return false;
    }

    // Create session
    OrtStatus* status = ort->CreateSession(env_, model_path_.c_str(), session_options_, &session_);
    if (status) {
        std::cerr << "Failed to create session: " << ort->GetErrorMessage(status) << std::endl;
        ort->ReleaseStatus(status);
        return false;
    }

    return true;
}

bool ONNXInference::loadModelMetadata() {
    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // Get input count
    size_t num_input_nodes;
    OrtStatus* status = ort->SessionGetInputCount(session_, &num_input_nodes);
    if (status) {
        std::cerr << "Failed to get input count: " << ort->GetErrorMessage(status) << std::endl;
        ort->ReleaseStatus(status);
        return false;
    }

    // Get input names and shapes
    for (size_t i = 0; i < num_input_nodes; i++) {
        char* input_name;
        OrtTypeInfo* typeinfo;
        OrtTensorTypeAndShapeInfo* tensor_info;
        
        status = ort->SessionGetInputName(session_, i, allocator_, &input_name);
        if (status) {
            ort->ReleaseStatus(status);
            continue;
        }
        
        status = ort->SessionGetInputTypeInfo(session_, i, &typeinfo);
        if (status) {
            ort->ReleaseStatus(status);
            ort->AllocatorFree(allocator_, input_name);
            continue;
        }
        
        ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        
        size_t num_dims;
        ort->GetDimensionsCount(tensor_info, &num_dims);
        std::vector<int64_t> shape(num_dims);
        ort->GetDimensions(tensor_info, shape.data(), num_dims);
        
        input_names_.emplace_back(input_name);
        input_shapes_[input_name] = shape;
        
        ort->AllocatorFree(allocator_, input_name);
        ort->ReleaseTypeInfo(typeinfo);
    }

    // Get output count
    size_t num_output_nodes;
    status = ort->SessionGetOutputCount(session_, &num_output_nodes);
    if (status) {
        std::cerr << "Failed to get output count: " << ort->GetErrorMessage(status) << std::endl;
        ort->ReleaseStatus(status);
        return false;
    }

    // Get output names and shapes
    for (size_t i = 0; i < num_output_nodes; i++) {
        char* output_name;
        OrtTypeInfo* typeinfo;
        OrtTensorTypeAndShapeInfo* tensor_info;
        
        status = ort->SessionGetOutputName(session_, i, allocator_, &output_name);
        if (status) {
            ort->ReleaseStatus(status);
            continue;
        }
        
        status = ort->SessionGetOutputTypeInfo(session_, i, &typeinfo);
        if (status) {
            ort->ReleaseStatus(status);
            ort->AllocatorFree(allocator_, output_name);
            continue;
        }
        
        ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        
        size_t num_dims;
        ort->GetDimensionsCount(tensor_info, &num_dims);
        std::vector<int64_t> shape(num_dims);
        ort->GetDimensions(tensor_info, shape.data(), num_dims);
        
        output_names_.emplace_back(output_name);
        output_shapes_[output_name] = shape;
        
        ort->AllocatorFree(allocator_, output_name);
        ort->ReleaseTypeInfo(typeinfo);
    }

    return true;
}

InferenceResult ONNXInference::runInference(const cv::Mat& image) {
    if (!initialized_) {
        return { {}, {}, "", 0.0, false, "ONNX Inference not initialized" };
    }

    // Preprocess image to tensor
    // This is a simplified preprocessing - in practice, you'd use the Preprocessor class
    cv::Mat processed_image;
    cv::resize(image, processed_image, cv::Size(224, 224));
    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);
    processed_image.convertTo(processed_image, CV_32F, 1.0/255.0);

    // Convert to tensor (NCHW format)
    std::vector<float> input_data;
    int channels = processed_image.channels();
    int height = processed_image.rows;
    int width = processed_image.cols;
    
    input_data.reserve(channels * height * width);
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input_data.push_back(processed_image.at<cv::Vec3f>(h, w)[c]);
            }
        }
    }

    std::vector<int64_t> input_shape = {1, channels, height, width};
    return runInference(input_data, input_shape);
}

InferenceResult ONNXInference::runInference(const std::vector<float>& input_data, 
                                           const std::vector<int64_t>& input_shape) {
    InferenceResult result;
    
    if (!initialized_) {
        result.success = false;
        result.error_message = "ONNX Inference not initialized";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    try {
        // Create input tensor
        OrtValue* input_tensor = nullptr;
        OrtStatus* status = ort->CreateTensorWithDataAsOrtValue(
            memory_info_, 
            input_data.data(), 
            input_data.size() * sizeof(float),
            input_shape.data(), 
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor
        );

        if (status) {
            result.success = false;
            result.error_message = "Failed to create input tensor: " + std::string(ort->GetErrorMessage(status));
            ort->ReleaseStatus(status);
            return result;
        }

        // Run inference
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }

        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }

        std::vector<OrtValue*> output_tensors(output_names_.size(), nullptr);

        status = ort->Run(
            session_,
            nullptr,
            input_names_cstr.data(),
            &input_tensor,
            1,
            output_names_cstr.data(),
            output_names_cstr.size(),
            output_tensors.data()
        );

        if (status) {
            result.success = false;
            result.error_message = "Inference failed: " + std::string(ort->GetErrorMessage(status));
            ort->ReleaseStatus(status);
            ort->ReleaseValue(input_tensor);
            return result;
        }

        // Extract output data
        if (!output_tensors.empty()) {
            result.output_name = output_names_[0];
            
            // Get output tensor info
            OrtTensorTypeAndShapeInfo* output_info;
            status = ort->GetTensorTypeAndShape(output_tensors[0], &output_info);
            if (status) {
                ort->ReleaseStatus(status);
            } else {
                size_t num_dims;
                ort->GetDimensionsCount(output_info, &num_dims);
                result.output_shape.resize(num_dims);
                ort->GetDimensions(output_info, result.output_shape.data(), num_dims);
                ort->ReleaseTensorTypeAndShapeInfo(output_info);
            }

            // Get output data
            float* output_data;
            status = ort->GetTensorMutableData(output_tensors[0], (void**)&output_data);
            if (status) {
                result.success = false;
                result.error_message = "Failed to get output data: " + std::string(ort->GetErrorMessage(status));
                ort->ReleaseStatus(status);
            } else {
                size_t output_size = 1;
                for (auto dim : result.output_shape) {
                    output_size *= dim;
                }
                result.output_data.assign(output_data, output_data + output_size);
                result.success = true;
            }
        }

        // Clean up
        ort->ReleaseValue(input_tensor);
        for (auto tensor : output_tensors) {
            if (tensor) {
                ort->ReleaseValue(tensor);
            }
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = "Exception during inference: " + std::string(e.what());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result;
}

std::vector<std::string> ONNXInference::getInputNames() const {
    return input_names_;
}

std::vector<std::string> ONNXInference::getOutputNames() const {
    return output_names_;
}

std::vector<int64_t> ONNXInference::getInputShape(const std::string& input_name) const {
    if (input_name.empty() && !input_shapes_.empty()) {
        return input_shapes_.begin()->second;
    }
    
    auto it = input_shapes_.find(input_name);
    return (it != input_shapes_.end()) ? it->second : std::vector<int64_t>();
}

std::vector<int64_t> ONNXInference::getOutputShape(const std::string& output_name) const {
    if (output_name.empty() && !output_shapes_.empty()) {
        return output_shapes_.begin()->second;
    }
    
    auto it = output_shapes_.find(output_name);
    return (it != output_shapes_.end()) ? it->second : std::vector<int64_t>();
}

void ONNXInference::setNumThreads(int num_threads) {
    num_threads_ = num_threads;
}

void ONNXInference::setOptimizationLevel(int level) {
    optimization_level_ = level;
}

void ONNXInference::setDeviceID(int device_id) {
    device_id_ = device_id;
}

std::string ONNXInference::getBackendString(BackendType backend) const {
    switch (backend) {
        case BackendType::CPU: return "CPU";
        case BackendType::CUDA: return "CUDA";
        case BackendType::COREML: return "CoreML";
        case BackendType::DIRECTML: return "DirectML";
        case BackendType::TENSORRT: return "TensorRT";
        case BackendType::ROCM: return "ROCm";
        case BackendType::WEBGL: return "WebGL";
        case BackendType::WEBGPU: return "WebGPU";
        case BackendType::WEBNN: return "WebNN";
        case BackendType::NNAPI: return "NNAPI";
        default: return "Unknown";
    }
}

#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "onnx_inference.h"

struct DetectionResult {
    float confidence;
    int class_id;
    std::string class_name;
    cv::Rect bbox;
    std::vector<float> features;
};

struct PostprocessingConfig {
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int max_detections = 100;
    bool apply_softmax = true;
    bool normalize_output = true;
    std::vector<std::string> class_names;
    cv::Size original_size;
    cv::Size input_size;
};

class Postprocessor {
public:
    Postprocessor();
    ~Postprocessor();

    void setConfig(const PostprocessingConfig& config);
    PostprocessingConfig getConfig() const;

    // Main processing functions
    std::string formatInferenceResult(const InferenceResult& inference_result);
    std::vector<DetectionResult> processDetections(const InferenceResult& inference_result);
    std::vector<float> processClassification(const InferenceResult& inference_result);
    cv::Mat processSegmentation(const InferenceResult& inference_result);

    // Utility functions
    std::string toJSON(const std::vector<DetectionResult>& detections);
    std::string toJSON(const std::vector<float>& classification);
    std::string toJSON(const InferenceResult& result);
    std::string generateErrorResponse(const std::string& error_message);

    // Detection-specific processing
    std::vector<DetectionResult> applyNMS(const std::vector<DetectionResult>& detections);
    std::vector<DetectionResult> scaleBoundingBoxes(const std::vector<DetectionResult>& detections,
                                                   const cv::Size& input_size,
                                                   const cv::Size& original_size);
    std::vector<DetectionResult> filterByConfidence(const std::vector<DetectionResult>& detections,
                                                   float threshold);

    // Classification processing
    std::vector<float> applySoftmax(const std::vector<float>& logits);
    int getTopKIndex(const std::vector<float>& probabilities, int k = 1);
    std::vector<std::pair<int, float>> getTopKIndices(const std::vector<float>& probabilities, int k = 5);

private:
    PostprocessingConfig config_;
    
    // Helper functions
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    cv::Rect scaleBox(const cv::Rect& box, const cv::Size& input_size, const cv::Size& original_size);
    std::string escapeJsonString(const std::string& input);
    
    // Model-specific processing
    std::vector<DetectionResult> processYOLOOutput(const std::vector<float>& output_data,
                                                  const std::vector<int64_t>& output_shape);
    std::vector<DetectionResult> processSSDOutput(const std::vector<float>& output_data,
                                                 const std::vector<int64_t>& output_shape);
    std::vector<DetectionResult> processRCNNOutput(const std::vector<float>& output_data,
                                                  const std::vector<int64_t>& output_shape);
};

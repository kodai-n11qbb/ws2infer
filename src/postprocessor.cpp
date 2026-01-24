#include "postprocessor.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

Postprocessor::Postprocessor() {
    config_.confidence_threshold = 0.5f;
    config_.nms_threshold = 0.4f;
    config_.max_detections = 100;
    config_.apply_softmax = true;
    config_.normalize_output = true;
}

std::string Postprocessor::formatInferenceResult(const InferenceResult& inference_result) {
    if (!inference_result.success) {
        return generateErrorResponse(inference_result.error_message);
    }

    std::string result_json;
    if (inference_result.output_shape.size() == 1) {
        auto classification = processClassification(inference_result);
        result_json = toJSON(classification);
    } else {
        auto detections = processDetections(inference_result);
        result_json = toJSON(detections);
    }

    // Add metadata
    std::ostringstream metadata;
    metadata << std::fixed << std::setprecision(2);
    metadata << "\"inference_time_ms\":" << inference_result.inference_time_ms;
    
    if (result_json.back() == '}') {
        result_json.pop_back();
        result_json += "," + metadata.str() + "}";
    }

    return result_json;
}

std::vector<DetectionResult> Postprocessor::processDetections(const InferenceResult& inference_result) {
    std::vector<DetectionResult> detections;
    
    if (!inference_result.success || inference_result.output_data.empty()) {
        return detections;
    }

    // Process based on output shape
    if (inference_result.output_shape.size() == 4 && inference_result.output_shape[2] == 4) {
        detections = processYOLOOutput(inference_result.output_data, inference_result.output_shape);
    } else if (inference_result.output_shape.size() == 3 && inference_result.output_shape[2] == 7) {
        detections = processSSDOutput(inference_result.output_data, inference_result.output_shape);
    } else if (inference_result.output_shape.size() == 2) {
        detections = processRCNNOutput(inference_result.output_data, inference_result.output_shape);
    }

    // Apply filtering and NMS
    detections = filterByConfidence(detections, config_.confidence_threshold);
    detections = applyNMS(detections);
    
    if (detections.size() > static_cast<size_t>(config_.max_detections)) {
        detections.resize(config_.max_detections);
    }

    return detections;
}

std::vector<float> Postprocessor::processClassification(const InferenceResult& inference_result) {
    std::vector<float> probabilities;
    
    if (!inference_result.success || inference_result.output_data.empty()) {
        return probabilities;
    }

    probabilities = inference_result.output_data;
    
    if (config_.apply_softmax) {
        probabilities = applySoftmax(probabilities);
    }

    return probabilities;
}

std::string Postprocessor::toJSON(const std::vector<DetectionResult>& detections) {
    std::ostringstream json;
    json << "{\"type\":\"detections\",\"count\":" << detections.size() << ",\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        if (i > 0) json << ",";
        json << "{\"confidence\":" << std::fixed << std::setprecision(3) << detections[i].confidence
             << ",\"class_id\":" << detections[i].class_id
             << ",\"class_name\":\"" << detections[i].class_name << "\""
             << ",\"bbox\":{\"x\":" << detections[i].bbox.x
             << ",\"y\":" << detections[i].bbox.y
             << ",\"width\":" << detections[i].bbox.width
             << ",\"height\":" << detections[i].bbox.height << "}}";
    }

    json << "]}";
    return json.str();
}

std::string Postprocessor::toJSON(const std::vector<float>& classification) {
    std::ostringstream json;
    json << "{\"type\":\"classification\",\"probabilities\":[";

    for (size_t i = 0; i < classification.size(); ++i) {
        if (i > 0) json << ",";
        json << std::fixed << std::setprecision(6) << classification[i];
    }

    json << "]}";
    return json.str();
}

std::vector<DetectionResult> Postprocessor::applyNMS(const std::vector<DetectionResult>& detections) {
    std::vector<DetectionResult> nms_detections;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        nms_detections.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            if (detections[i].class_id == detections[j].class_id) {
                float iou = calculateIoU(detections[i].bbox, detections[j].bbox);
                if (iou > config_.nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return nms_detections;
}

std::vector<DetectionResult> Postprocessor::filterByConfidence(const std::vector<DetectionResult>& detections, float threshold) {
    std::vector<DetectionResult> filtered;
    
    for (const auto& detection : detections) {
        if (detection.confidence >= threshold) {
            filtered.push_back(detection);
        }
    }

    return filtered;
}

float Postprocessor::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    if (x2 <= x1 || y2 <= y1) return 0.0f;

    float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
    float area1 = static_cast<float>(box1.width * box1.height);
    float area2 = static_cast<float>(box2.width * box2.height);
    float union_area = area1 + area2 - intersection;

    return intersection / union_area;
}

std::vector<float> Postprocessor::applySoftmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    
    for (float& prob : probabilities) {
        prob /= sum;
    }
    
    return probabilities;
}

std::string Postprocessor::generateErrorResponse(const std::string& error_message) {
    return "{\"type\":\"error\",\"error\":\"" + error_message + "\"}";
}

std::vector<DetectionResult> Postprocessor::processYOLOOutput(const std::vector<float>& output_data, const std::vector<int64_t>& output_shape) {
    std::vector<DetectionResult> detections;
    // Simplified YOLO processing - would need model-specific implementation
    return detections;
}

std::vector<DetectionResult> Postprocessor::processSSDOutput(const std::vector<float>& output_data, const std::vector<int64_t>& output_shape) {
    std::vector<DetectionResult> detections;
    // Simplified SSD processing - would need model-specific implementation
    return detections;
}

std::vector<DetectionResult> Postprocessor::processRCNNOutput(const std::vector<float>& output_data, const std::vector<int64_t>& output_shape) {
    std::vector<DetectionResult> detections;
    // Simplified R-CNN processing - would need model-specific implementation
    return detections;
}

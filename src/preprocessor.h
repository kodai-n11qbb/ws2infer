#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct PreprocessingConfig {
    cv::Size target_size = cv::Size(224, 224);
    bool normalize = true;
    double mean[3] = {0.485, 0.456, 0.406};  // ImageNet means
    double std[3] = {0.229, 0.224, 0.225};    // ImageNet stds
    bool convert_to_float = true;
    bool channel_first = true;  // NCHW vs NHWC
    cv::Scalar pad_color = cv::Scalar(0, 0, 0);
    bool maintain_aspect_ratio = true;
};

class Preprocessor {
public:
    Preprocessor();
    ~Preprocessor();

    void setConfig(const PreprocessingConfig& config);
    PreprocessingConfig getConfig() const;

    cv::Mat processImage(const cv::Mat& image);
    std::vector<float> processForInference(const cv::Mat& image);
    std::vector<float> processBatch(const std::vector<cv::Mat>& images);

    // Utility functions
    cv::Mat resizeWithPadding(const cv::Mat& image, const cv::Size& target_size);
    cv::Mat normalizeImage(const cv::Mat& image);
    std::vector<float> imageToTensor(const cv::Mat& image, bool channel_first = true);

private:
    PreprocessingConfig config_;
    
    cv::Mat applyNormalization(const cv::Mat& image);
    cv::Mat convertDataType(const cv::Mat& image);
};

#include "preprocessor.h"
#include <iostream>
#include <algorithm>

Preprocessor::Preprocessor() {
    // Default configuration for ImageNet-based models
    config_.target_size = cv::Size(224, 224);
    config_.normalize = true;
    config_.convert_to_float = true;
    config_.channel_first = true;
    config_.maintain_aspect_ratio = true;
}

Preprocessor::~Preprocessor() {
}

void Preprocessor::setConfig(const PreprocessingConfig& config) {
    config_ = config;
}

PreprocessingConfig Preprocessor::getConfig() const {
    return config_;
}

cv::Mat Preprocessor::processImage(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Empty image provided to preprocessor" << std::endl;
        return cv::Mat();
    }

    cv::Mat processed = image.clone();

    // Resize image
    if (image.size() != config_.target_size) {
        if (config_.maintain_aspect_ratio) {
            processed = resizeWithPadding(processed, config_.target_size);
        } else {
            cv::resize(processed, processed, config_.target_size, 0, 0, cv::INTER_LINEAR);
        }
    }

    // Apply normalization if enabled
    if (config_.normalize) {
        processed = applyNormalization(processed);
    }

    // Convert data type if needed
    if (config_.convert_to_float) {
        processed = convertDataType(processed);
    }

    return processed;
}

std::vector<float> Preprocessor::processForInference(const cv::Mat& image) {
    cv::Mat processed = processImage(image);
    
    if (processed.empty()) {
        return std::vector<float>();
    }

    return imageToTensor(processed, config_.channel_first);
}

std::vector<float> Preprocessor::processBatch(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        return std::vector<float>();
    }

    // Process first image to get dimensions
    cv::Mat first_processed = processImage(images[0]);
    if (first_processed.empty()) {
        return std::vector<float>();
    }

    int batch_size = static_cast<int>(images.size());
    int channels = first_processed.channels();
    int height = first_processed.rows;
    int width = first_processed.cols;

    std::vector<float> batch_tensor;
    batch_tensor.reserve(batch_size * channels * height * width);

    for (const auto& image : images) {
        cv::Mat processed = processImage(image);
        if (processed.empty()) {
            continue;
        }

        std::vector<float> image_tensor = imageToTensor(processed, config_.channel_first);
        batch_tensor.insert(batch_tensor.end(), image_tensor.begin(), image_tensor.end());
    }

    return batch_tensor;
}

cv::Mat Preprocessor::resizeWithPadding(const cv::Mat& image, const cv::Size& target_size) {
    if (image.empty()) {
        return cv::Mat();
    }

    // Calculate scaling factor to maintain aspect ratio
    double scale_x = static_cast<double>(target_size.width) / image.cols;
    double scale_y = static_cast<double>(target_size.height) / image.rows;
    double scale = std::min(scale_x, scale_y);

    // Calculate new dimensions
    int new_width = static_cast<int>(image.cols * scale);
    int new_height = static_cast<int>(image.rows * scale);

    // Resize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // Create target size image with padding
    cv::Mat padded(target_size, image.type(), config_.pad_color);

    // Calculate padding offsets to center the image
    int offset_x = (target_size.width - new_width) / 2;
    int offset_y = (target_size.height - new_height) / 2;

    // Copy resized image to center of padded image
    cv::Rect roi(offset_x, offset_y, new_width, new_height);
    resized.copyTo(padded(roi));

    return padded;
}

cv::Mat Preprocessor::applyNormalization(const cv::Mat& image) {
    cv::Mat normalized;

    if (image.depth() == CV_8U) {
        // Convert to float first
        image.convertTo(normalized, CV_32F, 1.0 / 255.0);
    } else {
        normalized = image.clone();
    }

    // Apply mean and std normalization
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);

    for (int i = 0; i < channels.size() && i < 3; ++i) {
        channels[i] = (channels[i] - config_.mean[i]) / config_.std[i];
    }

    cv::merge(channels, normalized);

    return normalized;
}

cv::Mat Preprocessor::convertDataType(const cv::Mat& image) {
    cv::Mat converted;
    
    if (image.depth() == CV_8U) {
        // Convert to float and normalize to [0, 1]
        image.convertTo(converted, CV_32F, 1.0 / 255.0);
    } else if (image.depth() != CV_32F) {
        // Convert to float
        image.convertTo(converted, CV_32F);
    } else {
        converted = image.clone();
    }

    return converted;
}

std::vector<float> Preprocessor::imageToTensor(const cv::Mat& image, bool channel_first) {
    if (image.empty()) {
        return std::vector<float>();
    }

    int channels = image.channels();
    int height = image.rows;
    int width = image.cols;

    std::vector<float> tensor;
    tensor.reserve(channels * height * width);

    if (channel_first) {
        // NCHW format: Channel, Height, Width
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    if (image.depth() == CV_32F) {
                        tensor.push_back(image.at<cv::Vec3f>(h, w)[c]);
                    } else {
                        tensor.push_back(image.at<cv::Vec3b>(h, w)[c]);
                    }
                }
            }
        }
    } else {
        // NHWC format: Height, Width, Channel
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channels; ++c) {
                    if (image.depth() == CV_32F) {
                        tensor.push_back(image.at<cv::Vec3f>(h, w)[c]);
                    } else {
                        tensor.push_back(image.at<cv::Vec3b>(h, w)[c]);
                    }
                }
            }
        }
    }

    return tensor;
}

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

class FrameExtractor {
public:
    FrameExtractor();
    ~FrameExtractor();

    bool initialize(const std::string& video_source);
    cv::Mat extractFrame();
    std::vector<cv::Mat> extractFrames(int count);
    bool hasNext();
    void reset();
    void release();

    // Video properties
    int getWidth() const;
    int getHeight() const;
    double getFPS() const;
    int getTotalFrames() const;

private:
    cv::VideoCapture capture_;
    cv::Mat current_frame_;
    bool is_initialized_;
    
    // Frame extraction settings
    int target_width_;
    int target_height_;
    bool maintain_aspect_ratio_;
    
    cv::Mat preprocessFrame(const cv::Mat& frame);
    cv::Mat resizeFrame(const cv::Mat& frame);
};

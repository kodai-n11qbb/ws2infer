#include "frame_extractor.h"
#include <iostream>

FrameExtractor::FrameExtractor() 
    : is_initialized_(false), target_width_(640), target_height_(480), maintain_aspect_ratio_(true) {
}

FrameExtractor::~FrameExtractor() {
    release();
}

bool FrameExtractor::initialize(const std::string& video_source) {
    release();
    
    // Try to open video source (file, camera index, or stream URL)
    if (video_source.find("http") == 0 || video_source.find("rtsp") == 0) {
        // Network stream
        capture_.open(video_source, cv::CAP_FFMPEG);
    } else if (video_source.find("/") == std::string::npos && video_source.length() == 1) {
        // Camera index (0, 1, 2, etc.)
        int camera_index = std::stoi(video_source);
        capture_.open(camera_index);
    } else {
        // Video file
        capture_.open(video_source);
    }
    
    if (!capture_.isOpened()) {
        std::cerr << "Failed to open video source: " << video_source << std::endl;
        return false;
    }
    
    is_initialized_ = true;
    
    // Read first frame to validate
    if (!capture_.read(current_frame_)) {
        std::cerr << "Failed to read first frame from video source" << std::endl;
        release();
        return false;
    }
    
    std::cout << "Frame extractor initialized successfully" << std::endl;
    std::cout << "Resolution: " << getWidth() << "x" << getHeight() << std::endl;
    std::cout << "FPS: " << getFPS() << std::endl;
    
    return true;
}

cv::Mat FrameExtractor::extractFrame() {
    if (!is_initialized_) {
        std::cerr << "Frame extractor not initialized" << std::endl;
        return cv::Mat();
    }
    
    if (capture_.read(current_frame_)) {
        return preprocessFrame(current_frame_);
    }
    
    return cv::Mat();
}

std::vector<cv::Mat> FrameExtractor::extractFrames(int count) {
    std::vector<cv::Mat> frames;
    
    for (int i = 0; i < count && hasNext(); ++i) {
        cv::Mat frame = extractFrame();
        if (!frame.empty()) {
            frames.push_back(frame);
        }
    }
    
    return frames;
}

bool FrameExtractor::hasNext() {
    if (!is_initialized_) {
        return false;
    }
    
    return !current_frame_.empty();
}

void FrameExtractor::reset() {
    if (is_initialized_) {
        capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
}

void FrameExtractor::release() {
    if (capture_.isOpened()) {
        capture_.release();
    }
    current_frame_.release();
    is_initialized_ = false;
}

int FrameExtractor::getWidth() const {
    if (!is_initialized_) return 0;
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int FrameExtractor::getHeight() const {
    if (!is_initialized_) return 0;
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double FrameExtractor::getFPS() const {
    if (!is_initialized_) return 0.0;
    return capture_.get(cv::CAP_PROP_FPS);
}

int FrameExtractor::getTotalFrames() const {
    if (!is_initialized_) return 0;
    return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
}

cv::Mat FrameExtractor::preprocessFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        return frame;
    }
    
    cv::Mat processed = frame.clone();
    
    // Resize frame if needed
    if (frame.cols != target_width_ || frame.rows != target_height_) {
        processed = resizeFrame(frame);
    }
    
    // Convert BGR to RGB (common for deep learning models)
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    
    return processed;
}

cv::Mat FrameExtractor::resizeFrame(const cv::Mat& frame) {
    cv::Mat resized;
    
    if (maintain_aspect_ratio_) {
        // Calculate scaling factor
        double scale_x = static_cast<double>(target_width_) / frame.cols;
        double scale_y = static_cast<double>(target_height_) / frame.rows;
        double scale = std::min(scale_x, scale_y);
        
        // Calculate new dimensions
        int new_width = static_cast<int>(frame.cols * scale);
        int new_height = static_cast<int>(frame.rows * scale);
        
        // Resize with aspect ratio preserved
        cv::resize(frame, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
        
        // Create target size image with padding
        cv::Mat padded(target_height_, target_width_, frame.type(), cv::Scalar(0, 0, 0));
        
        // Calculate padding offsets
        int offset_x = (target_width_ - new_width) / 2;
        int offset_y = (target_height_ - new_height) / 2;
        
        // Copy resized frame to center of padded image
        cv::Rect roi(offset_x, offset_y, new_width, new_height);
        resized.copyTo(padded(roi));
        
        resized = padded;
    } else {
        // Resize without maintaining aspect ratio
        cv::resize(frame, resized, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_LINEAR);
    }
    
    return resized;
}

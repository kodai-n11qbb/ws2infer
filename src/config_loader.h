#pragma once
#include <string>
#include <fstream>
#include "json.hpp"

struct ServerConfig {
    std::string host = "localhost";
    int port = 8080;
    int max_connections = 10;
    int buffer_size = 4096;
};

struct ModelConfig {
    std::string path = "model.onnx";
    std::string backend = "auto";
    std::vector<int> input_size = {224, 224};
    int batch_size = 1;
};

struct InferenceConfig {
    std::string device = "auto";
    int num_threads = 4;
    int optimization_level = 1;
};

struct PreprocessingConfig {
    bool normalize = true;
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    std::string resize_method = "bilinear";
};

struct LoggingConfig {
    std::string level = "info";
    std::string file = "ws2infer.log";
    bool console = true;
};

struct Config {
    ServerConfig server;
    ModelConfig model;
    InferenceConfig inference;
    PreprocessingConfig preprocessing;
    LoggingConfig logging;
};

class ConfigLoader {
public:
    static bool load_config(const std::string& config_path, Config& config);
    static void print_config(const Config& config);
};

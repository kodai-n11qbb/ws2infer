#include "config_loader.h"
#include <iostream>
#include <filesystem>

bool ConfigLoader::load_config(const std::string& config_path, Config& config) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << config_path << std::endl;
        std::cerr << "Using default configuration." << std::endl;
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;

        // Parse server config
        if (j.contains("server")) {
            const auto& server = j["server"];
            if (server.contains("host")) config.server.host = server["host"];
            if (server.contains("port")) config.server.port = server["port"];
            if (server.contains("max_connections")) config.server.max_connections = server["max_connections"];
            if (server.contains("buffer_size")) config.server.buffer_size = server["buffer_size"];
        }

        // Parse model config
        if (j.contains("model")) {
            const auto& model = j["model"];
            if (model.contains("path")) config.model.path = model["path"];
            if (model.contains("backend")) config.model.backend = model["backend"];
            if (model.contains("input_size")) config.model.input_size = model["input_size"].get<std::vector<int>>();
            if (model.contains("batch_size")) config.model.batch_size = model["batch_size"];
        }

        // Parse inference config
        if (j.contains("inference")) {
            const auto& inference = j["inference"];
            if (inference.contains("device")) config.inference.device = inference["device"];
            if (inference.contains("num_threads")) config.inference.num_threads = inference["num_threads"];
            if (inference.contains("optimization_level")) config.inference.optimization_level = inference["optimization_level"];
        }

        // Parse preprocessing config
        if (j.contains("preprocessing")) {
            const auto& preprocessing = j["preprocessing"];
            if (preprocessing.contains("normalize")) config.preprocessing.normalize = preprocessing["normalize"];
            if (preprocessing.contains("mean")) config.preprocessing.mean = preprocessing["mean"].get<std::vector<float>>();
            if (preprocessing.contains("std")) config.preprocessing.std = preprocessing["std"].get<std::vector<float>>();
            if (preprocessing.contains("resize_method")) config.preprocessing.resize_method = preprocessing["resize_method"];
        }

        // Parse logging config
        if (j.contains("logging")) {
            const auto& logging = j["logging"];
            if (logging.contains("level")) config.logging.level = logging["level"];
            if (logging.contains("file")) config.logging.file = logging["file"];
            if (logging.contains("console")) config.logging.console = logging["console"];
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing config file: " << e.what() << std::endl;
        return false;
    }
}

void ConfigLoader::print_config(const Config& config) {
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Server:" << std::endl;
    std::cout << "  Host: " << config.server.host << std::endl;
    std::cout << "  Port: " << config.server.port << std::endl;
    std::cout << "  Max connections: " << config.server.max_connections << std::endl;
    std::cout << "Model:" << std::endl;
    std::cout << "  Path: " << config.model.path << std::endl;
    std::cout << "  Backend: " << config.model.backend << std::endl;
    std::cout << "  Input size: [" << config.model.input_size[0] << ", " << config.model.input_size[1] << "]" << std::endl;
    std::cout << "Inference:" << std::endl;
    std::cout << "  Device: " << config.inference.device << std::endl;
    std::cout << "  Threads: " << config.inference.num_threads << std::endl;
    std::cout << "Logging:" << std::endl;
    std::cout << "  Level: " << config.logging.level << std::endl;
    std::cout << "  Console: " << (config.logging.console ? "enabled" : "disabled") << std::endl;
    std::cout << "===================" << std::endl;
}

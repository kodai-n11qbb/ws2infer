#include <iostream>
#include <memory>
#include <string>
#include <signal.h>
#include "websocket_server.h"
#include "onnx_inference.h"
#include "environment_detector.h"
#include "config_loader.h"

std::unique_ptr<WebSocketServer> server;
std::unique_ptr<ONNXInference> inference;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (server) {
        server->stop();
    }
    exit(0);
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Load configuration
    Config config;
    std::string config_path = "config.json";
    
    // Check for custom config path
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--config") {
            config_path = argv[i + 1];
            break;
        }
    }
    
    bool config_loaded = ConfigLoader::load_config(config_path, config);
    ConfigLoader::print_config(config);

    // Override with command line arguments if provided
    std::string ws_url = "ws://" + config.server.host + ":" + std::to_string(config.server.port);
    std::string model_path = config.model.path;
    
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--ws-url" && i + 1 < argc) {
            ws_url = argv[i + 1];
        } else if (std::string(argv[i]) == "--model" && i + 1 < argc) {
            model_path = argv[i + 1];
        }
    }

    std::cout << "Starting WebSocket ONNX Inference Server" << std::endl;
    std::cout << "WebSocket URL: " << ws_url << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    try {
        // Detect environment and initialize ONNX inference
        EnvironmentDetector detector;
        auto env_info = detector.detectEnvironment();
        std::cout << "Detected environment: " << env_info.description << std::endl;

        // Initialize ONNX inference engine
        inference = std::make_unique<ONNXInference>(model_path, env_info);
        if (!inference->initialize()) {
            std::cerr << "Failed to initialize ONNX inference engine" << std::endl;
            return 1;
        }

        // Initialize WebSocket server
        server = std::make_unique<WebSocketServer>(ws_url, *inference);
        if (!server->start()) {
            std::cerr << "Failed to start WebSocket server" << std::endl;
            return 1;
        }

        std::cout << "Server started successfully. Press Ctrl+C to stop." << std::endl;

        // Run the server
        server->run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

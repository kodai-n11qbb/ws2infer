#include <iostream>
#include <memory>
#include <string>
#include <signal.h>
#include "websocket_server.h"
#include "onnx_inference.h"
#include "environment_detector.h"

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

    // Parse command line arguments
    std::string ws_url = "ws://localhost:8080";
    std::string model_path = "model.onnx";
    
    if (argc > 1) {
        ws_url = argv[1];
    }
    if (argc > 2) {
        model_path = argv[2];
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

#!/bin/bash

# Build script for ws2infer WebSocket ONNX Inference Server

set -e

echo "Building ws2infer WebSocket ONNX Inference Server..."

# Check if build directory exists
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed successfully!"
echo "Executable location: build/ws2infer"
echo ""
echo "To run the server:"
echo "  ./build/ws2infer [ws_url] [model_path]"
echo ""
echo "Example:"
echo "  ./build/ws2infer ws://localhost:8080 model.onnx"

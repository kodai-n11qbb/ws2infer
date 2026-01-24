cmake_minimum_required(VERSION 3.16)
project(ws2infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
    PATHS
    /opt/homebrew/include
    /usr/local/include
    /usr/include
    PATH_SUFFIXES onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
    /opt/homebrew/lib
    /usr/local/lib
    /usr/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Please install onnxruntime: brew install onnxruntime")
endif()

message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Find nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, using header-only version")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

# Add executable
add_executable(ws2infer
    src/main.cpp
    src/websocket_server.cpp
    src/frame_extractor.cpp
    src/preprocessor.cpp
    src/onnx_inference.cpp
    src/postprocessor.cpp
    src/environment_detector.cpp
    src/config_loader.cpp
)

# Link libraries
target_link_libraries(ws2infer
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    pthread
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ws2infer PRIVATE -Wall -Wextra -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ws2infer PRIVATE -Wall -Wextra -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(ws2infer PRIVATE /W3 /O2)
endif()

# Install targets
install(TARGETS ws2infer DESTINATION bin)

#pragma once

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <vector>
#include <map>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
#endif

#include "onnx_inference.h"

struct WebSocketConnection {
    int socket_fd;
    std::string endpoint;
    bool is_connected;
    
    WebSocketConnection(int fd, const std::string& ep) 
        : socket_fd(fd), endpoint(ep), is_connected(true) {}
};

class WebSocketServer {
public:
    WebSocketServer(const std::string& url, ONNXInference& inference);
    ~WebSocketServer();

    bool start();
    void stop();
    void run();

private:
    std::string url_;
    std::string host_;
    int port_;
    int server_socket_;
    bool running_;
    ONNXInference& inference_;
    
    std::vector<std::unique_ptr<WebSocketConnection>> connections_;
    std::thread accept_thread_;
    std::map<int, std::thread> client_threads_;

    bool parseUrl(const std::string& url);
    bool createServerSocket();
    void acceptConnections();
    void handleClient(int client_socket);
    bool performWebSocketHandshake(int client_socket, const std::string& handshake);
    std::string extractWebSocketKey(const std::string& handshake);
    std::string generateWebSocketResponse(const std::string& key);
    std::string decodeWebSocketFrame(const std::vector<uint8_t>& frame);
    std::vector<uint8_t> encodeWebSocketFrame(const std::string& message);
    void sendToClient(int client_socket, const std::string& message);
    void closeConnection(int client_socket);
    
    // Message handlers
    void handleVideoFrame(const std::string& frame_data, int client_socket);
    void handleConfigMessage(const std::string& config, int client_socket);
};

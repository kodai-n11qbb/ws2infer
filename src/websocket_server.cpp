#include "websocket_server.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <opencv2/opencv.hpp>

// Simple SHA1 implementation for WebSocket handshake
#include <vector>
#include <iomanip>

struct SHA1_CTX {
    unsigned long state[5];
    unsigned long count[2];
    unsigned char buffer[64];
};

void SHA1Transform(unsigned long state[5], const unsigned char buffer[64]);
void SHA1Init(SHA1_CTX* context);
void SHA1Update(SHA1_CTX* context, const unsigned char* data, unsigned int len);
void SHA1Final(unsigned char digest[20], SHA1_CTX* context);

void SHA1Init(SHA1_CTX* context) {
    context->state[0] = 0x67452301;
    context->state[1] = 0xEFCDAB89;
    context->state[2] = 0x98BADCFE;
    context->state[3] = 0x10325476;
    context->state[4] = 0xC3D2E1F0;
    context->count[0] = context->count[1] = 0;
}

void SHA1Update(SHA1_CTX* context, const unsigned char* data, unsigned int len) {
    unsigned int i, j;
    
    j = (context->count[0] >> 3) & 63;
    if ((context->count[0] += len << 3) < (len << 3)) context->count[1]++;
    context->count[1] += (len >> 29);
    
    if ((j + len) > 63) {
        memcpy(&context->buffer[j], data, (i = 64-j));
        SHA1Transform(context->state, context->buffer);
        for ( ; i + 63 < len; i += 64) {
            SHA1Transform(context->state, &data[i]);
        }
        j = 0;
    }
    else i = 0;
    memcpy(&context->buffer[j], &data[i], len - i);
}

void SHA1Final(unsigned char digest[20], SHA1_CTX* context) {
    unsigned long i, j;
    unsigned char finalcount[8];
    
    for (i = 0; i < 8; i++) {
        finalcount[i] = (unsigned char)((context->count[(i >= 4) ? 0 : 1] >> ((3-(i & 3)) * 8) ) & 255);
    }
    
    SHA1Update(context, (unsigned char *)"\200", 1);
    while ((context->count[0] & 504) != 448) {
        SHA1Update(context, (unsigned char *)"\0", 1);
    }
    SHA1Update(context, finalcount, 8);
    
    for (i = 0; i < 20; i++) {
        digest[i] = (unsigned char)((context->state[i>>2] >> ((3-(i & 3)) * 8) ) & 255);
    }
    
    memset(context, 0, sizeof(*context));
    memset(&finalcount, 0, sizeof(finalcount));
}

#define rol(value, bits) (((value) << (bits)) | ((value) >> (32-(bits))))

#define blk0(i) (block->l[i] = htonl(block->l[i]))
#define blk(i) (block->l[i&15] = rol(block->l[(i+13)&15]^block->l[(i+8)&15] ^ block->l[(i+2)&15]^block->l[i&15],1))

#define R0(v,w,x,y,z,i) z+=((w&(x^y))^y)+blk0(i)+0x5A827999+rol(v,5);w=rol(w,30);
#define R1(v,w,x,y,z,i) z+=((w&(x^y))^y)+blk(i)+0x5A827999+rol(v,5);w=rol(w,30);
#define R2(v,w,x,y,z,i) z+=(w^x^y)+blk(i)+0x6ED9EBA1+rol(v,5);w=rol(w,30);
#define R3(v,w,x,y,z,i) z+=(((w|x)&y)|(w&x))+blk(i)+0x8F1BBCDC+rol(v,5);w=rol(w,30);
#define R4(v,w,x,y,z,i) z+=(w^x^y)+blk(i)+0xCA62C1D6+rol(v,5);w=rol(w,30);

typedef struct {
    unsigned long l[16];
} CHAR64LONG16;

void SHA1Transform(unsigned long state[5], const unsigned char buffer[64]) {
    unsigned long a, b, c, d, e;
    CHAR64LONG16 block[1];
    int i;
    
    memcpy(block, buffer, 64);
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    
    R0(a,b,c,d,e, 0); R0(e,a,b,c,d, 1); R0(d,e,a,b,c, 2); R0(c,d,e,a,b, 3);
    R0(b,c,d,e,a, 4); R0(a,b,c,d,e, 5); R0(e,a,b,c,d, 6); R0(d,e,a,b,c, 7);
    R0(c,d,e,a,b, 8); R0(b,c,d,e,a, 9); R0(a,b,c,d,e,10); R0(e,a,b,c,d,11);
    R0(d,e,a,b,c,12); R0(c,d,e,a,b,13); R0(b,c,d,e,a,14); R0(a,b,c,d,e,15);
    R1(e,a,b,c,d,16); R1(d,e,a,b,c,17); R1(c,d,e,a,b,18); R1(b,c,d,e,a,19);
    R2(a,b,c,d,e,20); R2(e,a,b,c,d,21); R2(d,e,a,b,c,22); R2(c,d,e,a,b,23);
    R2(b,c,d,e,a,24); R2(a,b,c,d,e,25); R2(e,a,b,c,d,26); R2(d,e,a,b,c,27);
    R2(c,d,e,a,b,28); R2(b,c,d,e,a,29); R2(a,b,c,d,e,30); R2(e,a,b,c,d,31);
    R2(d,e,a,b,c,32); R2(c,d,e,a,b,33); R2(b,c,d,e,a,34); R2(a,b,c,d,e,35);
    R2(e,a,b,c,d,36); R2(d,e,a,b,c,37); R2(c,d,e,a,b,38); R2(b,c,d,e,a,39);
    R3(a,b,c,d,e,40); R3(e,a,b,c,d,41); R3(d,e,a,b,c,42); R3(c,d,e,a,b,43);
    R3(b,c,d,e,a,44); R3(a,b,c,d,e,45); R3(e,a,b,c,d,46); R3(d,e,a,b,c,47);
    R3(c,d,e,a,b,48); R3(b,c,d,e,a,49); R3(a,b,c,d,e,50); R3(e,a,b,c,d,51);
    R3(d,e,a,b,c,52); R3(c,d,e,a,b,53); R3(b,c,d,e,a,54); R3(a,b,c,d,e,55);
    R3(e,a,b,c,d,56); R3(d,e,a,b,c,57); R3(c,d,e,a,b,58); R3(b,c,d,e,a,59);
    R4(a,b,c,d,e,60); R4(e,a,b,c,d,61); R4(d,e,a,b,c,62); R4(c,d,e,a,b,63);
    R4(b,c,d,e,a,64); R4(a,b,c,d,e,65); R4(e,a,b,c,d,66); R4(d,e,a,b,c,67);
    R4(c,d,e,a,b,68); R4(b,c,d,e,a,69); R4(a,b,c,d,e,70); R4(e,a,b,c,d,71);
    R4(d,e,a,b,c,72); R4(c,d,e,a,b,73); R4(b,c,d,e,a,74); R4(a,b,c,d,e,75);
    R4(e,a,b,c,d,76); R4(d,e,a,b,c,77); R4(c,d,e,a,b,78); R4(b,c,d,e,a,79);
    
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    
    a = b = c = d = e = 0;
}

#undef R0
#undef R1
#undef R2
#undef R3
#undef R4

// Simple SHA1 function
void SHA1(const unsigned char* data, size_t len, unsigned char* hash) {
    SHA1_CTX ctx;
    SHA1Init(&ctx);
    SHA1Update(&ctx, data, len);
    SHA1Final(hash, &ctx);
}

// Simple base64 encoding function
std::string base64_encode(const std::string& input) {
    const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    int val = 0, valb = -6;
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) encoded.push_back(chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (encoded.size() % 4) encoded.push_back('=');
    return encoded;
}

std::string base64_decode(const std::string& encoded) {
    const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[chars[i]] = i;
    
    std::string decoded;
    int val = 0, valb = -8;
    for (unsigned char c : encoded) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}

WebSocketServer::WebSocketServer(const std::string& url, ONNXInference& inference)
    : url_(url), inference_(inference), server_socket_(-1), running_(false) {
    parseUrl(url);
}

WebSocketServer::~WebSocketServer() {
    stop();
}

bool WebSocketServer::parseUrl(const std::string& url) {
    size_t host_start = url.find("://");
    if (host_start == std::string::npos) {
        host_start = 0;
    } else {
        host_start += 3;
    }
    
    size_t port_start = url.find(":", host_start);
    if (port_start != std::string::npos) {
        host_ = url.substr(host_start, port_start - host_start);
        port_ = std::stoi(url.substr(port_start + 1));
    } else {
        host_ = url.substr(host_start);
        port_ = (url.find("wss://") == 0) ? 443 : 8080;
    }
    
    return true;
}

bool WebSocketServer::createServerSocket() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return false;
    }
#endif

    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return false;
    }

    int opt = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, 
                   reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "Failed to set socket options" << std::endl;
        return false;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_);

    if (bind(server_socket_, reinterpret_cast<struct sockaddr*>(&server_addr), 
             sizeof(server_addr)) < 0) {
        std::cerr << "Failed to bind socket" << std::endl;
        return false;
    }

    if (listen(server_socket_, 10) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
        return false;
    }

    return true;
}

bool WebSocketServer::start() {
    if (!createServerSocket()) {
        return false;
    }

    running_ = true;
    accept_thread_ = std::thread(&WebSocketServer::acceptConnections, this);
    
    std::cout << "WebSocket server listening on " << host_ << ":" << port_ << std::endl;
    return true;
}

void WebSocketServer::stop() {
    running_ = false;
    
    if (server_socket_ >= 0) {
#ifdef _WIN32
        closesocket(server_socket_);
#else
        close(server_socket_);
#endif
        server_socket_ = -1;
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    for (auto& pair : client_threads_) {
        if (pair.second.joinable()) {
            pair.second.join();
        }
    }

    connections_.clear();
    client_threads_.clear();

#ifdef _WIN32
    WSACleanup();
#endif
}

void WebSocketServer::run() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void WebSocketServer::acceptConnections() {
    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket_, 
                                  reinterpret_cast<struct sockaddr*>(&client_addr), 
                                  &client_len);
        
        if (client_socket < 0) {
            if (running_) {
                std::cerr << "Failed to accept connection" << std::endl;
            }
            continue;
        }

        std::cout << "New connection from " << inet_ntoa(client_addr.sin_addr) << std::endl;
        
        auto connection = std::make_unique<WebSocketConnection>(client_socket, "");
        
        // Perform WebSocket handshake
        char buffer[4096];
        int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            std::string handshake(buffer);
            
            if (performWebSocketHandshake(client_socket, handshake)) {
                connections_.push_back(std::move(connection));
                
                // Start client handler thread
                client_threads_[client_socket] = std::thread(&WebSocketServer::handleClient, 
                                                           this, client_socket);
            } else {
                std::cerr << "WebSocket handshake failed" << std::endl;
#ifdef _WIN32
                closesocket(client_socket);
#else
                close(client_socket);
#endif
            }
        }
    }
}

bool WebSocketServer::performWebSocketHandshake(int client_socket, const std::string& handshake) {
    std::string key = extractWebSocketKey(handshake);
    if (key.empty()) {
        return false;
    }

    std::string response = generateWebSocketResponse(key);
    send(client_socket, response.c_str(), response.length(), 0);
    
    return true;
}

std::string WebSocketServer::extractWebSocketKey(const std::string& handshake) {
    size_t key_pos = handshake.find("Sec-WebSocket-Key:");
    if (key_pos == std::string::npos) {
        return "";
    }
    
    size_t key_start = handshake.find(" ", key_pos) + 1;
    size_t key_end = handshake.find("\r\n", key_start);
    
    return handshake.substr(key_start, key_end - key_start);
}

std::string WebSocketServer::generateWebSocketResponse(const std::string& key) {
    std::string magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string combined = key + magic_string;
    
    unsigned char hash[20];
    SHA1(reinterpret_cast<const unsigned char*>(combined.c_str()), combined.length(), hash);
    
    std::string response = "HTTP/1.1 101 Switching Protocols\r\n";
    response += "Upgrade: websocket\r\n";
    response += "Connection: Upgrade\r\n";
    response += "Sec-WebSocket-Accept: ";
    
    // Base64 encode the hash
    std::string hash_str(reinterpret_cast<char*>(hash), 20);
    std::string encoded_key = base64_encode(hash_str);
    
    response += encoded_key + "\r\n\r\n";
    
    return response;
}

void WebSocketServer::handleClient(int client_socket) {
    std::vector<uint8_t> buffer(4096);
    
    while (running_) {
        int bytes_received = recv(client_socket, 
                                  reinterpret_cast<char*>(buffer.data()), 
                                  buffer.size(), 0);
        
        if (bytes_received <= 0) {
            break;
        }
        
        buffer.resize(bytes_received);
        std::string message = decodeWebSocketFrame(buffer);
        
        if (!message.empty()) {
            // Parse message type
            if (message.find("\"type\":\"frame\"") != std::string::npos) {
                handleVideoFrame(message, client_socket);
            } else if (message.find("\"type\":\"config\"") != std::string::npos) {
                handleConfigMessage(message, client_socket);
            }
        }
    }
    
    closeConnection(client_socket);
}

std::string WebSocketServer::decodeWebSocketFrame(const std::vector<uint8_t>& frame) {
    if (frame.size() < 2) {
        return "";
    }
    
    uint8_t fin = frame[0] & 0x80;
    uint8_t opcode = frame[0] & 0x0F;
    uint8_t masked = frame[1] & 0x80;
    uint64_t payload_length = frame[1] & 0x7F;
    
    size_t header_length = 2;
    
    if (payload_length == 126) {
        if (frame.size() < 4) return "";
        payload_length = (frame[2] << 8) | frame[3];
        header_length = 4;
    } else if (payload_length == 127) {
        if (frame.size() < 10) return "";
        payload_length = 0;
        for (int i = 0; i < 8; i++) {
            payload_length = (payload_length << 8) | frame[2 + i];
        }
        header_length = 10;
    }
    
    if (masked) {
        header_length += 4;
    }
    
    if (frame.size() < header_length + payload_length) {
        return "";
    }
    
    std::string payload;
    if (masked) {
        uint8_t mask[4];
        for (int i = 0; i < 4; i++) {
            mask[i] = frame[header_length - 4 + i];
        }
        
        for (uint64_t i = 0; i < payload_length; i++) {
            payload += frame[header_length + i] ^ mask[i % 4];
        }
    } else {
        payload = std::string(reinterpret_cast<const char*>(frame.data() + header_length), 
                            payload_length);
    }
    
    return payload;
}

std::vector<uint8_t> WebSocketServer::encodeWebSocketFrame(const std::string& message) {
    std::vector<uint8_t> frame;
    
    uint8_t first_byte = 0x80; // FIN = 1
    first_byte |= 0x01; // Text frame
    frame.push_back(first_byte);
    
    uint64_t payload_length = message.length();
    if (payload_length < 126) {
        frame.push_back(static_cast<uint8_t>(payload_length));
    } else if (payload_length < 65536) {
        frame.push_back(126);
        frame.push_back(static_cast<uint8_t>(payload_length >> 8));
        frame.push_back(static_cast<uint8_t>(payload_length & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; i--) {
            frame.push_back(static_cast<uint8_t>(payload_length >> (i * 8)));
        }
    }
    
    for (char c : message) {
        frame.push_back(static_cast<uint8_t>(c));
    }
    
    return frame;
}

void WebSocketServer::sendToClient(int client_socket, const std::string& message) {
    auto frame = encodeWebSocketFrame(message);
    send(client_socket, reinterpret_cast<const char*>(frame.data()), frame.size(), 0);
}

void WebSocketServer::closeConnection(int client_socket) {
    auto it = std::find_if(connections_.begin(), connections_.end(),
                          [client_socket](const std::unique_ptr<WebSocketConnection>& conn) {
                              return conn->socket_fd == client_socket;
                          });
    
    if (it != connections_.end()) {
        (*it)->is_connected = false;
        connections_.erase(it);
    }
    
#ifdef _WIN32
    closesocket(client_socket);
#else
    close(client_socket);
#endif
    
    if (client_threads_.find(client_socket) != client_threads_.end()) {
        client_threads_.erase(client_socket);
    }
    
    std::cout << "Connection closed" << std::endl;
}

void WebSocketServer::handleVideoFrame(const std::string& frame_data, int client_socket) {
    try {
        // Parse JSON to extract base64 image data
        // For simplicity, assuming frame_data contains base64 encoded image
        std::string base64_data = frame_data; // Extract from JSON in real implementation
        
        // Decode base64 to image
        std::string decoded_str = base64_decode(base64_data);
        std::vector<uchar> decoded_data(decoded_str.begin(), decoded_str.end());
        cv::Mat image = cv::imdecode(decoded_data, cv::IMREAD_COLOR);
        
        if (image.empty()) {
            std::cerr << "Failed to decode image" << std::endl;
            return;
        }
        
        // Run inference
        auto results = inference_.runInference(image);
        
        // Send results back to client
        std::string response = "{\"type\":\"inference_result\",\"success\":" + 
                               std::string(results.success ? "true" : "false") + 
                               ",\"inference_time_ms\":" + std::to_string(results.inference_time_ms) + "}";
        sendToClient(client_socket, response);
        
    } catch (const std::exception& e) {
        std::cerr << "Error handling video frame: " << e.what() << std::endl;
        std::string error_response = "{\"type\":\"error\",\"message\":\"" + std::string(e.what()) + "\"}";
        sendToClient(client_socket, error_response);
    }
}

void WebSocketServer::handleConfigMessage(const std::string& config, int client_socket) {
    try {
        // Parse configuration and update inference settings
        // For simplicity, just acknowledge
        std::string response = "{\"type\":\"config_ack\",\"status\":\"success\"}";
        sendToClient(client_socket, response);
        
    } catch (const std::exception& e) {
        std::cerr << "Error handling config message: " << e.what() << std::endl;
        std::string error_response = "{\"type\":\"error\",\"message\":\"" + std::string(e.what()) + "\"}";
        sendToClient(client_socket, error_response);
    }
}

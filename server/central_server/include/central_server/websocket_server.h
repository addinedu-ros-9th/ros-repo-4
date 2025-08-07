#ifndef WEBSOCKET_SERVER_H
#define WEBSOCKET_SERVER_H

#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <openssl/sha.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <iostream>
#include <algorithm>
#include <json/json.h>
#include <ctime>

class WebSocketServer {
public:
    // 클라이언트 정보 구조체
    struct ClientInfo {
        int socket_fd;
        std::string ip_address;
        std::string client_id;
        std::string client_type;  // "gui", "admin", "unknown"
        
        ClientInfo(int fd, const std::string& ip) 
            : socket_fd(fd), ip_address(ip), client_id(""), client_type("unknown") {}
    };
    
    WebSocketServer(int port = 3000);
    ~WebSocketServer();
    
    // 서버 시작/정지
    bool start();
    void stop();
    bool isRunning() const { return running_; }
    
    // 메시지 전송 함수들
    void broadcastMessage(const std::string& message);
    bool sendMessageToIP(const std::string& ip_address, const std::string& message);
    bool sendMessageToClient(int client_socket, const std::string& message);
    
    // 클라이언트 타입별 메시지 전송
    void broadcastMessageToType(const std::string& client_type, const std::string& message);
    std::vector<std::string> getClientsByType(const std::string& client_type) const;
    
    // 클라이언트 타입 설정
    bool setClientType(const std::string& ip_address, const std::string& client_type);
    
    // 로봇 알림 메시지 전송 함수들
    void sendAlertOccupied(int robot_id, std::string type);  // 지정된 타입의 클라이언트에게
    void sendAlertIdle(int robot_id);                        // 모든 클라이언트에게
    void sendNavigatingComplete(int robot_id);               // GUI 클라이언트에게만
    
    // 클라이언트 관리
    size_t getClientCount() const;
    std::vector<std::string> getConnectedIPs() const;
    bool isClientConnected(const std::string& ip_address) const;
    
private:
    int port_;
    int server_socket_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    // 클라이언트 관리 (IP 주소 기반)
    std::map<std::string, ClientInfo> clients_by_ip_;
    std::map<int, std::string> clients_by_socket_;

    
    // 서버 메인 루프
    void serverLoop();
    
    // 클라이언트 처리
    void handleClient(int client_socket, const std::string& client_ip);
    
    // WebSocket 헬퍼 함수들
    bool isWebSocketRequest(const std::string& request);
    std::string generateWebSocketAcceptKey(const std::string& client_key);
    std::string createWebSocketHandshakeResponse(const std::string& client_key);
    bool sendWebSocketFrame(int client_socket, const std::string& message);
    void removeClient(int client_socket);
    std::string extractWebSocketKey(const std::string& request);
    std::string extractClientTypeFromRequest(const std::string& request);
    
    // 메시지 파싱 및 처리 함수들
    std::string parseWebSocketMessage(const char* buffer, ssize_t bytes_received);
    bool handleClientTypeMessage(int client_socket, const std::string& client_ip, const std::string& message);
};

#endif // WEBSOCKET_SERVER_H 
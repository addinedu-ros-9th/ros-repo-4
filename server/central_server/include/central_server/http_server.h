#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <map>
#include <vector>
#include <mutex>
#include <json/json.h>
#include "central_server/database_manager.h"
#include "central_server/robot_navigation_manager.h"
#include "central_server/admin_request_handler.h"
#include "central_server/user_request_handler.h"
#include "central_server/robot_request_handler.h"

// WebSocket 관련 헤더
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

class HttpServer {
public:
    // HTTP 요청 구조체를 public으로 이동
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string body;
        std::map<std::string, std::string> headers;
    };
    
    HttpServer(std::shared_ptr<DatabaseManager> db_manager, int port = 8080);
    ~HttpServer();
    
    // 서버 시작/정지
    void start();
    void stop();
    bool isRunning() const;
    
    // 실시간 통신을 위한 브로드캐스트 기능
    void broadcastToClients(const std::string& message);
    
    // 로봇 네비게이션 관리자 설정
    void setRobotNavigationManager(std::shared_ptr<RobotNavigationManager> nav_manager);
    
private:
    std::shared_ptr<DatabaseManager> db_manager_;
    int port_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    // WebSocket 클라이언트 관리
    std::vector<int> websocket_clients_;
    std::mutex websocket_clients_mutex_;
    
    // 로봇 네비게이션 관리자
    std::shared_ptr<RobotNavigationManager> nav_manager_;
    
    // 요청 핸들러들
    std::unique_ptr<AdminRequestHandler> admin_handler_;
    std::unique_ptr<UserRequestHandler> user_handler_;
    std::unique_ptr<RobotRequestHandler> robot_handler_;
    
    // 로봇 현재 위치 정보
    struct RobotPosition {
        double x;
        double y;
        double yaw;
        bool valid;
    };
    RobotPosition current_robot_position_;
    std::mutex robot_position_mutex_;
    
    // HTTP 서버 메인 루프
    void serverLoop();
    
    // 요청 처리
    std::string processRequest(const HttpRequest& request);
    
    // WebSocket 관련 함수들
    std::string handleWebSocketUpgrade(const HttpRequest& request, int client_socket);
    
    // WebSocket 관련 함수들
    void handleWebSocketClient(int client_socket);
    bool isWebSocketRequest(const HttpRequest& request);
    std::string generateWebSocketAcceptKey(const std::string& client_key);
    void sendWebSocketFrame(int client_socket, const std::string& message);
    void removeWebSocketClient(int client_socket);
    
    // 유틸리티 함수들
    Json::Value parseJson(const std::string& jsonStr);
    
    // HTTP 요청 파싱
    HttpRequest parseHttpRequest(const std::string& request);
    std::string createHttpResponse(int status_code, 
                                  const std::string& content_type,
                                  const std::string& body,
                                  const std::string& additional_headers = "");
};

#endif // HTTP_SERVER_H 
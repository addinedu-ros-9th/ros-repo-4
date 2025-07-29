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
    
private:
    std::shared_ptr<DatabaseManager> db_manager_;
    int port_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    // WebSocket 클라이언트 관리
    std::vector<int> websocket_clients_;
    std::mutex websocket_clients_mutex_;
    
    // HTTP 서버 메인 루프
    void serverLoop();
    
    // 요청 처리
    std::string processRequest(const HttpRequest& request);
    
    // API 엔드포인트 핸들러들
    std::string handleAuthSSN(const Json::Value& request);
    std::string handleAuthPatientId(const Json::Value& request);
    std::string handleAuthRFID(const Json::Value& request);
    std::string handleAuthDirection(const Json::Value& request);
    std::string handleRobotReturn(const Json::Value& request);
    std::string handleWithoutAuthDirection(const Json::Value& request);
    std::string handleRobotStatus(const Json::Value& request);
    std::string handleWebSocketUpgrade(const HttpRequest& request, int client_socket);
    std::string handleGetLLMConfig(const Json::Value& request);
    
    // WebSocket 관련 함수들
    void handleWebSocketClient(int client_socket);
    bool isWebSocketRequest(const HttpRequest& request);
    std::string generateWebSocketAcceptKey(const std::string& client_key);
    void sendWebSocketFrame(int client_socket, const std::string& message);
    void removeWebSocketClient(int client_socket);
    
    // 유틸리티 함수들
    Json::Value parseJson(const std::string& jsonStr);
    std::string createSuccessResponse(const std::string& name, 
                                    const std::string& time_hhmm, 
                                    const std::string& reservation);
    std::string createErrorResponse(const std::string& message);
    std::string createStatusResponse(int status_code);
    
    // HTTP 요청 파싱
    HttpRequest parseHttpRequest(const std::string& request);
    std::string createHttpResponse(int status_code, 
                                  const std::string& content_type,
                                  const std::string& body,
                                  const std::string& additional_headers = "");
};

#endif // HTTP_SERVER_H 
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
    
    // 로봇 네비게이션 관리자 설정
    void setRobotNavigationManager(std::shared_ptr<RobotNavigationManager> nav_manager);
    
private:
    std::shared_ptr<DatabaseManager> db_manager_;
    int port_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    // 로봇 네비게이션 관리자
    std::shared_ptr<RobotNavigationManager> nav_manager_;
    
    // 요청 핸들러들
    std::unique_ptr<AdminRequestHandler> admin_handler_;
    std::unique_ptr<UserRequestHandler> user_handler_;
    
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
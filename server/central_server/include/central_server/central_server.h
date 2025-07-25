#ifndef CENTRAL_SERVER_H
#define CENTRAL_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <atomic>
#include <memory>

#include "central_server/database_manager.h"
#include "central_server/http_server.h"

class CentralServer : public rclcpp::Node
{
public:
    CentralServer();
    ~CentralServer();
    
    void start();
    void stop();

private:
    // 실행 상태
    std::atomic<bool> running_;
    
    // 데이터베이스 관리자
    std::unique_ptr<DatabaseManager> db_manager_;
    
    // HTTP 서버
    std::unique_ptr<HttpServer> http_server_;
    
    // 스레드들
    std::thread db_thread_;
    std::thread http_thread_;
    
    // 각 스레드 실행 함수들
    void runDatabaseThread();
    void runHttpThread();
};

#endif // CENTRAL_SERVER_H 
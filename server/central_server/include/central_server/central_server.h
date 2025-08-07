#ifndef CENTRAL_SERVER_H
#define CENTRAL_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <control_interfaces/srv/event_handle.h>
#include <control_interfaces/srv/track_handle.h>

#include "database_manager.h"
#include "http_server.h"
#include "robot_navigation_manager.h"

#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <mutex>

class CentralServer : public rclcpp::Node
{
public:
    CentralServer();
    ~CentralServer();
    
    void init();
    void start();
    void stop();

private:
    // 기존 함수들
    void runDatabaseThread();
    void runHttpThread();
    void eventHandleCallback(
        const std::shared_ptr<control_interfaces::srv::EventHandle::Request> request,
        std::shared_ptr<control_interfaces::srv::EventHandle::Response> response);
    void trackHandleCallback(
        const std::shared_ptr<control_interfaces::srv::TrackHandle::Request> request,
        std::shared_ptr<control_interfaces::srv::TrackHandle::Response> response);
    
    // HTTP 서버 설정
    void setupHttpServer();
    
    // 기존 멤버 변수들
    std::atomic<bool> running_;
    
    std::unique_ptr<DatabaseManager> db_manager_;
    std::unique_ptr<HttpServer> http_server_;
    std::unique_ptr<RobotNavigationManager> nav_manager_;
    
    rclcpp::Service<control_interfaces::srv::EventHandle>::SharedPtr event_service_;
    rclcpp::Service<control_interfaces::srv::TrackHandle>::SharedPtr track_service_;
    
    std::thread db_thread_;
    std::thread http_thread_;
    
    // HTTP 서버 설정
    int http_port_;
    std::string http_host_;
};

#endif // CENTRAL_SERVER_H
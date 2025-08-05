#ifndef CENTRAL_SERVER_H
#define CENTRAL_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <robot_interfaces/msg/robot_status.hpp>
#include <robot_interfaces/srv/change_robot_status.hpp>

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
    void statusCallback(const robot_interfaces::msg::RobotStatus::SharedPtr msg);
    void changeStatusCallback(
        const std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Request> request,
        std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Response> response);
    
    // HTTP 기반 실시간 통신 함수들
    void setupHttpServer();
    void sendRobotLocationToGui(int robot_id, float location_x, float location_y);
    void sendRobotStatusToGui(int robot_id, const std::string& status, const std::string& source);
    void sendArrivalNotificationToGui(int robot_id);
    void broadcastToGuiClients(const std::string& message);
    
    // 기존 멤버 변수들
    std::atomic<bool> running_;
    
    std::unique_ptr<DatabaseManager> db_manager_;
    std::unique_ptr<HttpServer> http_server_;
    std::unique_ptr<RobotNavigationManager> nav_manager_;
    
    rclcpp::Subscription<robot_interfaces::msg::RobotStatus>::SharedPtr status_subscriber_;
    rclcpp::Service<robot_interfaces::srv::ChangeRobotStatus>::SharedPtr status_service_;
    
    std::thread db_thread_;
    std::thread http_thread_;
    
    // HTTP 서버 설정
    int http_port_;
    std::string http_host_;
};

#endif // CENTRAL_SERVER_H
#ifndef CENTRAL_SERVER_H
#define CENTRAL_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <robot_interfaces/msg/robot_status.hpp>
#include <robot_interfaces/srv/change_robot_status.srv>
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
    
    // ROS2 구독자/서비스
    image_transport::Subscriber image_subscriber_;
    rclcpp::Subscription<robot_interfaces::msg::RobotStatus>::SharedPtr status_subscriber_;
    rclcpp::Service<robot_interfaces::srv::ChangeRobotStatus>::SharedPtr status_service_;
    
    // Image Transport
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    
    // 스레드들
    std::thread db_thread_;
    std::thread http_thread_;
    
    // 콜백 함수들
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void statusCallback(const robot_interfaces::msg::RobotStatus::SharedPtr msg);
    void changeStatusCallback(
        const std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Request> request,
        std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Response> response);
    
    // 각 스레드 실행 함수들
    void runDatabaseThread();
    void runHttpThread();
};

#endif // CENTRAL_SERVER_H 
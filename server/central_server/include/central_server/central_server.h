#ifndef CENTRAL_SERVER_H
#define CENTRAL_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <robot_interfaces/msg/robot_status.hpp>
#include <robot_interfaces/srv/change_robot_status.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include "database_manager.h"
#include "http_server.h"

#include <thread>
#include <atomic>
#include <memory>

// UDP 관련 헤더 추가
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

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
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void statusCallback(const robot_interfaces::msg::RobotStatus::SharedPtr msg);
    void changeStatusCallback(
        const std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Request> request,
        std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Response> response);
    
    // UDP 관련 새로운 함수들 추가
    void setupUdpRelay();
    void runUdpReceiverThread();
    void runGuiSenderThread();
    void processReceivedImage(const cv::Mat& image);
    bool initializeUdpReceiver();
    bool initializeGuiSender();
    void sendImageToGui(const cv::Mat& image);
    
    // 기존 멤버 변수들
    std::atomic<bool> running_;
    
    std::unique_ptr<DatabaseManager> db_manager_;
    std::unique_ptr<HttpServer> http_server_;
    
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Subscriber image_subscriber_;
    rclcpp::Subscription<robot_interfaces::msg::RobotStatus>::SharedPtr status_subscriber_;
    rclcpp::Service<robot_interfaces::srv::ChangeRobotStatus>::SharedPtr status_service_;
    
    std::thread db_thread_;
    std::thread http_thread_;
    
    // UDP 관련 새로운 멤버 변수들 추가
    std::thread udp_receiver_thread_;
    std::thread gui_sender_thread_;
    
    // UDP 소켓 관련
    int udp_receiver_socket_;
    int gui_sender_socket_;
    struct sockaddr_in ai_server_addr_;
    struct sockaddr_in gui_client_addr_;
    
    // 설정 값들
    std::string ai_server_ip_;
    int ai_udp_receive_port_;
    std::string gui_client_ip_;
    int gui_client_port_;
    int max_packet_size_;
    
    // 이미지 버퍼 (스레드 간 공유)
    std::mutex image_buffer_mutex_;
    cv::Mat latest_image_;
    bool new_image_available_;
};

#endif // CENTRAL_SERVER_H
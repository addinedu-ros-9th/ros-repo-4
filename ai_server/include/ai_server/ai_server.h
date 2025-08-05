#ifndef AI_SERVER_H
#define AI_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <robot_interfaces/msg/robot_status.hpp>
#include <robot_interfaces/srv/change_robot_status.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>

#include "ai_server/webcam_streamer.h"
#include "ai_server/udp_image_sender.h"

class AIServer : public rclcpp::Node
{
public:
    AIServer();
    ~AIServer();
    
    void initialize(); // 초기화 함수 추가
    void start();
    void stop();

private:
    // 실행 상태
    std::atomic<bool> running_;
    
    std::string gui_client_ip_;
    int gui_client_port_;
    int max_packet_size_;

    // 웹캠 스트리머 (전면/후면 카메라)
    std::unique_ptr<WebcamStreamer> front_camera_;
    std::unique_ptr<WebcamStreamer> back_camera_;
    std::atomic<int> current_camera_; // 0: front, 1: back
    
    // UDP 이미지 전송기
    std::unique_ptr<UdpImageSender> udp_sender_;
    
    // HTTP 서버 관련
    int http_server_fd_;
    int http_port_;
    std::thread http_server_thread_;
    
    // ROS2 퍼블리셔/서브스크라이버
    image_transport::Publisher image_publisher_;
    rclcpp::Publisher<robot_interfaces::msg::RobotStatus>::SharedPtr status_publisher_;
    rclcpp::Client<robot_interfaces::srv::ChangeRobotStatus>::SharedPtr status_client_;
    
    // 스레드들
    std::thread webcam_thread_;
    std::thread processing_thread_;
    
    void loadConfig();
    
    // 콜백 함수들
    void publishWebcamFrame(const cv::Mat& frame);
    void processFrame(const cv::Mat& frame);
    void sendStatusToCentralServer(const std::string& status_msg);
    void sendImageViaUDP(const cv::Mat& frame);
    
    // 스레드 실행 함수들
    void runWebcamThread();
    void runProcessingThread();
    void runHttpServerThread();
    
    // HTTP 서버 함수들
    bool initializeHttpServer();
    void handleHttpRequest(int client_fd);
    void handleCameraChangeRequest(int client_fd, const std::string& request);
    std::string createHttpResponse(const cv::Mat& image);
    cv::Mat getCurrentCameraFrame();
    
    // 카메라 전환 함수
    void switchCamera(int camera_id);
    
    // Image Transport
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
};

#endif // AI_SERVER_H

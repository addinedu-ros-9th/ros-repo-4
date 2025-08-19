#ifndef AI_SERVER_H
#define AI_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <control_interfaces/msg/detected_obstacle.hpp>
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
#include "ai_server/frame_shared_memory.h"

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
    
    // 프레임 공유 메모리 (딥러닝 시스템과 공유)
    std::unique_ptr<FrameSharedMemory> front_frame_shm_;
    std::unique_ptr<FrameSharedMemory> back_frame_shm_;
    
    // HTTP 서버 관련
    int http_server_fd_;
    int http_port_;
    std::thread http_server_thread_;
    
    // ROS2 퍼블리셔/서브스크라이버
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Publisher image_publisher_;
    rclcpp::Publisher<control_interfaces::msg::DetectedObstacle>::SharedPtr obstacle_publisher_;
    
    // 딥러닝 처리 관련
    bool enable_deeplearning_;
    std::thread deeplearning_thread_;
    std::atomic<bool> deeplearning_running_;
    
    // 딥러닝 결과 저장
    struct DeepLearningResult {
        std::vector<std::map<std::string, cv::Rect>> person_detections;  // front, back
        std::vector<std::string> gestures;  // front, back
        std::vector<float> confidences;     // front, back
        std::mutex result_mutex;
    };
    std::unique_ptr<DeepLearningResult> dl_result_;
    
    // 스레드들
    std::thread webcam_thread_;
    std::thread processing_thread_;
    
    void loadConfig();
    
    // 콜백 함수들
    void publishWebcamFrame(const cv::Mat& frame);
    void processFrame(const cv::Mat& frame);
    void sendImageViaUDP(const cv::Mat& frame);
    void sendStatusToCentralServer(const std::string& status_msg);
    
    // HTTP 서버 관련 함수들
    void runHttpServerThread();
    bool initializeHttpServer();
    void handleHttpRequest(int client_fd);
    void handleCameraChangeRequest(int client_fd, const std::string& request);
    std::string createHttpResponse(const cv::Mat& image);
    cv::Mat getCurrentCameraFrame();
    void switchCamera(int camera_id);
    
    // 웹캠 스레드 함수
    void runWebcamThread();
    void runProcessingThread();
    
    // 딥러닝 처리 함수들
    void runDeepLearningThread();
    void startDeepLearning();
    void stopDeepLearning();
    void processDeepLearning();
    std::string executePythonScript(const std::string& script_path, const std::string& input_data);
};

#endif // AI_SERVER_H

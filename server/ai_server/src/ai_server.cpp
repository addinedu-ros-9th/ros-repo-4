#include "ai_server/ai_server.h"
#include <chrono>
#include <yaml-cpp/yaml.h>  


AIServer::AIServer() 
    : Node("ai_server"), 
      running_(false),
      gui_client_ip_("127.0.0.1"),   
      gui_client_port_(8888),        
      max_packet_size_(60000),
      webcam_streamer_(std::make_unique<WebcamStreamer>(2))
{
    RCLCPP_INFO(this->get_logger(), "AI Server 노드 생성중...");

    loadConfig();
    udp_sender_ = std::make_unique<UdpImageSender>(gui_client_ip_, gui_client_port_);
}

void AIServer::loadConfig()
{
    try {
        // 설정 파일 경로 (프로젝트 루트 기준)
        std::string config_path = "../server/config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        
        // GUI 클라이언트 설정 읽기
        if (config["ai_server"]["target_central_server"]) {
            gui_client_ip_ = config["ai_server"]["target_central_server"]["ip"].as<std::string>();
            gui_client_port_ = config["ai_server"]["target_central_server"]["port"].as<int>();
        }
        
        // 최대 패킷 크기 읽기
        if (config["ai_server"]["max_packet_size"]) {
            max_packet_size_ = config["ai_server"]["max_packet_size"].as<int>();
        }
        
        RCLCPP_INFO(this->get_logger(), "설정 파일 로드 완료:");
        RCLCPP_INFO(this->get_logger(), "  - GUI 클라이언트: %s:%d", 
                   gui_client_ip_.c_str(), gui_client_port_);
        RCLCPP_INFO(this->get_logger(), "  - 최대 패킷 크기: %d", max_packet_size_);
        
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "설정 파일 로드 실패: %s", e.what());
        RCLCPP_WARN(this->get_logger(), "기본값 사용: %s:%d", 
                   gui_client_ip_.c_str(), gui_client_port_);
    }
}

void AIServer::initialize()
{
    RCLCPP_INFO(this->get_logger(), "AI Server 초기화 중...");
    
    // Image Transport 초기화 (shared_from_this() 사용)
    image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // 퍼블리셔 생성
    image_publisher_ = image_transport_->advertise("webcam/image_raw", 1);
    status_publisher_ = this->create_publisher<robot_interfaces::msg::RobotStatus>(
        "robot_status", 10);
    
    // 서비스 클라이언트 생성 (central_server와 통신용)
    status_client_ = this->create_client<robot_interfaces::srv::ChangeRobotStatus>(
        "change_robot_status");
    
    RCLCPP_INFO(this->get_logger(), "AI Server 초기화 완료");
}

AIServer::~AIServer()
{
    stop();
}

void AIServer::start()
{
    if (running_) {
        RCLCPP_WARN(this->get_logger(), "AI Server가 이미 실행중입니다.");
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "AI Server 시작중...");
    
    // 웹캠 초기화
    if (!webcam_streamer_->initialize()) {
        RCLCPP_ERROR(this->get_logger(), "웹캠 초기화 실패!");
        return;
    }
    
    // UDP 전송기 초기화
    if (!udp_sender_->initialize()) {
        RCLCPP_ERROR(this->get_logger(), "UDP 전송기 초기화 실패!");
        return;
    }
    
    // UDP 설정 (옵션)
    udp_sender_->setCompressionQuality(70); // JPEG 압축 품질 70%
    udp_sender_->setMaxPacketSize(60000);   // 최대 패킷 크기 60KB
    
    running_ = true;
    
    // 스레드 시작
    webcam_thread_ = std::thread(&AIServer::runWebcamThread, this);
    processing_thread_ = std::thread(&AIServer::runProcessingThread, this);
    
    RCLCPP_INFO(this->get_logger(), "AI Server 시작 완료!");
}

void AIServer::stop()
{
    if (!running_) {
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "AI Server 종료중...");
    
    running_ = false;
    webcam_streamer_->stop();
    udp_sender_->stop();
    
    // 스레드 종료 대기
    if (webcam_thread_.joinable()) {
        webcam_thread_.join();
    }
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    RCLCPP_INFO(this->get_logger(), "AI Server 종료됨");
}

void AIServer::runWebcamThread()
{
    RCLCPP_INFO(this->get_logger(), "웹캠 스레드 시작");
    
    webcam_streamer_->start([this](const cv::Mat& frame) {
        this->publishWebcamFrame(frame);
        this->processFrame(frame);
        this->sendImageViaUDP(frame);
    });
    
    RCLCPP_INFO(this->get_logger(), "웹캠 스레드 종료");
}

void AIServer::runProcessingThread()
{
    RCLCPP_INFO(this->get_logger(), "처리 스레드 시작");
    
    while (running_) {
        // 주기적으로 상태 전송
        sendStatusToCentralServer("AI Server 정상 작동중");
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    RCLCPP_INFO(this->get_logger(), "처리 스레드 종료");
}

void AIServer::publishWebcamFrame(const cv::Mat& frame)
{
    if (frame.empty()) {
        return;
    }
    
    try {
        // OpenCV Mat을 ROS2 Image 메시지로 변환
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = this->get_clock()->now();
        cv_image.header.frame_id = "webcam_frame";
        cv_image.encoding = "bgr8";
        cv_image.image = frame;
        
        // 퍼블리시
        image_publisher_.publish(cv_image.toImageMsg());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "이미지 퍼블리시 실패: %s", e.what());
    }
}

void AIServer::processFrame(const cv::Mat& frame)
{
    // 여기에 AI 처리 로직을 추가할 수 있습니다
    // 예: 객체 감지, 얼굴 인식 등
    
    // 간단한 예시: 프레임 크기 정보
    static int frame_count = 0;
    frame_count++;
    
    if (frame_count % 30 == 0) { // 30프레임마다 로그
        RCLCPP_INFO(this->get_logger(), 
                   "프레임 처리중 - 크기: %dx%d, 프레임 #%d", 
                   frame.cols, frame.rows, frame_count);
    }
}

void AIServer::sendStatusToCentralServer(const std::string& status_msg)
{
    // RobotStatus 메시지 생성 및 퍼블리시
    auto status_message = robot_interfaces::msg::RobotStatus();
    status_message.robot_id = 999; // AI Server ID (숫자)
    status_message.status = status_msg;
    // timestamp 필드가 없으므로 제거
    
    status_publisher_->publish(status_message);
    
    // 서비스 요청 (선택적)
    if (status_client_->wait_for_service(std::chrono::milliseconds(100))) {
        auto request = std::make_shared<robot_interfaces::srv::ChangeRobotStatus::Request>();
        request->robot_id = 999; // AI Server ID (숫자)
        request->new_status = status_msg;
        
        auto future = status_client_->async_send_request(request);
        // 비동기로 처리하므로 결과를 기다리지 않음
    }
    RCLCPP_INFO(this->get_logger(), "상태 변경 서비스 응답 완료");
}

void AIServer::sendImageViaUDP(const cv::Mat& frame)
{
    if (frame.empty()) {
        return;
    }
    
    try {
        // UDP로 이미지 전송
        udp_sender_->sendImage(frame);
        
        // 로그 (너무 자주 출력하지 않도록 제한)
        static int udp_frame_count = 0;
        udp_frame_count++;
        
        if (udp_frame_count % 60 == 0) { // 60프레임마다 로그
            RCLCPP_INFO(this->get_logger(), 
                       "UDP 이미지 전송 - 프레임 #%d", udp_frame_count);
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "UDP 이미지 전송 실패: %s", e.what());
    }
}

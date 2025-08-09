#include "ai_server/ai_server.h"
#include <chrono>
#include <yaml-cpp/yaml.h>  


AIServer::AIServer() 
    : Node("ai_server"), 
      running_(false),
      gui_client_ip_("127.0.0.1"),   
      gui_client_port_(8888),        
      max_packet_size_(60000),
      http_port_(5005),  // 기본 HTTP 포트 설정
      current_camera_(0),  // 기본값: 전면 카메라
      enable_deeplearning_(true),  // 딥러닝 활성화
      deeplearning_running_(false)
{
    RCLCPP_INFO(this->get_logger(), "AI Server 노드 생성중...");

    loadConfig();
    
    // 딥러닝 결과 초기화
    dl_result_ = std::make_unique<DeepLearningResult>();
    dl_result_->person_detections.resize(2);  // front, back
    dl_result_->gestures.resize(2);
    dl_result_->confidences.resize(2);
    
    // 전면/후면 카메라 초기화
    front_camera_ = std::make_unique<WebcamStreamer>(0);  // /dev/video0
    back_camera_ = std::make_unique<WebcamStreamer>(2);   // /dev/video2
    
    udp_sender_ = std::make_unique<UdpImageSender>(gui_client_ip_, gui_client_port_);
    
    // 프레임 공유 메모리 초기화 (딥러닝 시스템과 공유)
    front_frame_shm_ = std::make_unique<FrameSharedMemory>("/front_camera_frame", 640, 480);
    back_frame_shm_ = std::make_unique<FrameSharedMemory>("/back_camera_frame", 640, 480);
}

void AIServer::loadConfig()
{
    try {
        // 설정 파일 경로 (절대 경로 사용)
        std::string config_path = "/home/ckim/ros-repo-4/config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        
        // AI 서버 설정 읽기
        if (config["ai_server"]) {
            // HTTP 서버 포트
            if (config["ai_server"]["port"]) {
                http_port_ = config["ai_server"]["port"].as<int>();
            }
            
            // UDP 타겟 설정
            if (config["ai_server"]["udp_target"]) {
                gui_client_ip_ = config["ai_server"]["udp_target"]["ip"].as<std::string>();
                gui_client_port_ = config["ai_server"]["udp_target"]["port"].as<int>();
            }
            
            // 최대 패킷 크기
            if (config["ai_server"]["max_packet_size"]) {
                max_packet_size_ = config["ai_server"]["max_packet_size"].as<int>();
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "설정 파일 로드 완료:");
        RCLCPP_INFO(this->get_logger(), "  - HTTP 서버 포트: %d", http_port_);
        RCLCPP_INFO(this->get_logger(), "  - UDP 타겟: %s:%d", 
                   gui_client_ip_.c_str(), gui_client_port_);
        RCLCPP_INFO(this->get_logger(), "  - 최대 패킷 크기: %d", max_packet_size_);
        
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "설정 파일 로드 실패: %s", e.what());
        RCLCPP_WARN(this->get_logger(), "기본값 사용: HTTP 포트 %d, UDP 타겟 %s:%d", 
                   http_port_, gui_client_ip_.c_str(), gui_client_port_);
    }
}

void AIServer::initialize()
{
    RCLCPP_INFO(this->get_logger(), "AI Server 초기화 중...");
    
    // Image Transport 초기화 (shared_from_this() 사용)
    image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // 퍼블리셔 생성
    image_publisher_ = image_transport_->advertise("webcam/image_raw", 1);
    obstacle_publisher_ = this->create_publisher<control_interfaces::msg::DetectedObstacle>(
        "detected_obstacles", 10);
    
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
    
    // 전면 카메라 초기화
    if (!front_camera_->initialize()) {
        RCLCPP_ERROR(this->get_logger(), "전면 카메라 초기화 실패!");
        return;
    }
    
    // 후면 카메라 초기화
    if (!back_camera_->initialize()) {
        RCLCPP_ERROR(this->get_logger(), "후면 카메라 초기화 실패!");
        return;
    }
    
    // UDP 전송기 초기화
    if (!udp_sender_->initialize()) {
        RCLCPP_ERROR(this->get_logger(), "UDP 전송기 초기화 실패!");
        return;
    }
    
    // HTTP 서버 초기화
    if (!initializeHttpServer()) {
        RCLCPP_ERROR(this->get_logger(), "HTTP 서버 초기화 실패!");
        return;
    }
    
    // UDP 설정 (옵션)
    udp_sender_->setCompressionQuality(70); // JPEG 압축 품질 70%
    udp_sender_->setMaxPacketSize(60000);   // 최대 패킷 크기 60KB
    
    running_ = true;
    
    // 스레드 시작
    webcam_thread_ = std::thread(&AIServer::runWebcamThread, this);
    processing_thread_ = std::thread(&AIServer::runProcessingThread, this);
    http_server_thread_ = std::thread(&AIServer::runHttpServerThread, this);
    
    // 딥러닝 처리 시작
    if (enable_deeplearning_) {
        startDeepLearning();
    }
    
    RCLCPP_INFO(this->get_logger(), "AI Server 시작 완료!");
    RCLCPP_INFO(this->get_logger(), "HTTP 서버: http://localhost:%d", http_port_);
    RCLCPP_INFO(this->get_logger(), "카메라 전환: GET /front 또는 GET /back");
}

void AIServer::stop()
{
    if (!running_) {
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "AI Server 종료중...");
    
    running_ = false;
    front_camera_->stop();
    back_camera_->stop();
    udp_sender_->stop();
    
    // 딥러닝 처리 종료
    if (enable_deeplearning_) {
        stopDeepLearning();
    }
    
    // HTTP 서버 소켓 닫기
    if (http_server_fd_ >= 0) {
        close(http_server_fd_);
        http_server_fd_ = -1;
    }
    
    // 스레드 종료 대기
    if (webcam_thread_.joinable()) {
        webcam_thread_.join();
    }
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    if (http_server_thread_.joinable()) {
        http_server_thread_.join();
    }
    if (deeplearning_thread_.joinable()) {
        deeplearning_thread_.join();
    }
    
    RCLCPP_INFO(this->get_logger(), "AI Server 종료됨");
}

void AIServer::runWebcamThread()
{
    RCLCPP_INFO(this->get_logger(), "웹캠 스레드 시작");
    
    // 현재 선택된 카메라에 따라 콜백 설정
    auto front_callback = [this](const cv::Mat& frame) {
        // 전면 카메라는 항상 공유 메모리에 저장 (딥러닝 시스템용)
        if (front_frame_shm_ && front_frame_shm_->isAvailable()) {
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(640, 480));
            if (front_frame_shm_->writeFrame(resized_frame)) {
                static int frame_count = 0;
                frame_count++;
                if (frame_count % 30 == 0) { // 30프레임마다 로그
                    RCLCPP_INFO(this->get_logger(), "전면 카메라 프레임 공유 메모리 저장 성공");
                }
            }
        }
        
        if (current_camera_ == 0) {  // 전면 카메라가 선택된 경우만 처리
            this->publishWebcamFrame(frame);
            this->processFrame(frame);
            this->sendImageViaUDP(frame);
        }
    };
    
    auto back_callback = [this](const cv::Mat& frame) {
        // 후면 카메라는 항상 공유 메모리에 저장 (딥러닝 시스템용)
        if (back_frame_shm_ && back_frame_shm_->isAvailable()) {
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(640, 480));
            if (back_frame_shm_->writeFrame(resized_frame)) {
                static int frame_count = 0;
                frame_count++;
                if (frame_count % 30 == 0) { // 30프레임마다 로그
                    RCLCPP_INFO(this->get_logger(), "후면 카메라 프레임 공유 메모리 저장 성공");
                }
            }
        }
        
        if (current_camera_ == 1) {  // 후면 카메라가 선택된 경우만 처리
            this->publishWebcamFrame(frame);
            this->processFrame(frame);
            this->sendImageViaUDP(frame);
        }
    };
    
    // 두 카메라 모두 시작
    front_camera_->start(front_callback);
    back_camera_->start(back_callback);
    
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
    // 간단한 로그 출력으로 대체 (서비스 호출 제거)
    RCLCPP_INFO(this->get_logger(), "AI Server 상태: %s", status_msg.c_str());
    
    // 필요시 장애물 감지 정보를 퍼블리시할 수 있음
    // auto obstacle_msg = control_interfaces::msg::DetectedObstacle();
    // obstacle_msg.x = 0.0;
    // obstacle_msg.y = 0.0;
    // obstacle_msg.yaw = 0.0;
    // obstacle_publisher_->publish(obstacle_msg);
}

void AIServer::sendImageViaUDP(const cv::Mat& frame)
{
    if (frame.empty()) {
        return;
    }
    
    try {
        // 현재 카메라 타입으로 UDP 전송
        udp_sender_->sendImage(frame, current_camera_);
        
        // 로그 (너무 자주 출력하지 않도록 제한)
        static int udp_frame_count = 0;
        udp_frame_count++;
        
        if (udp_frame_count % 60 == 0) { // 60프레임마다 로그
            std::string camera_name = (current_camera_ == 0) ? "전면" : "후면";
            RCLCPP_INFO(this->get_logger(), 
                       "UDP 이미지 전송 - %s 카메라, 프레임 #%d", 
                       camera_name.c_str(), udp_frame_count);
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "UDP 이미지 전송 실패: %s", e.what());
    }
}

void AIServer::runHttpServerThread()
{
    RCLCPP_INFO(this->get_logger(), "HTTP 서버 스레드 시작");
    
    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(http_server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (running_) {
                RCLCPP_WARN(this->get_logger(), "HTTP 클라이언트 연결 실패");
            }
            continue;
        }
        
        RCLCPP_INFO(this->get_logger(), "HTTP 클라이언트 연결됨: %s", 
                   inet_ntoa(client_addr.sin_addr));
        
        handleHttpRequest(client_fd);
        close(client_fd);
    }
    
    RCLCPP_INFO(this->get_logger(), "HTTP 서버 스레드 종료");
}

bool AIServer::initializeHttpServer()
{
    http_server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (http_server_fd_ < 0) {
        RCLCPP_ERROR(this->get_logger(), "HTTP 서버 소켓 생성 실패");
        return false;
    }
    
    // 소켓 옵션 설정 (재사용 주소)
    int opt = 1;
    setsockopt(http_server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(http_port_);
    
    if (bind(http_server_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        RCLCPP_ERROR(this->get_logger(), "HTTP 서버 바인드 실패");
        close(http_server_fd_);
        return false;
    }
    
    if (listen(http_server_fd_, 5) < 0) {
        RCLCPP_ERROR(this->get_logger(), "HTTP 서버 리스닝 실패");
        close(http_server_fd_);
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "HTTP 서버 초기화 완료 (포트: %d)", http_port_);
    return true;
}

void AIServer::handleHttpRequest(int client_fd)
{
    char buffer[4096];  // 더 큰 버퍼로 증가
    int bytes_received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received <= 0) {
        return;
    }
    
    buffer[bytes_received] = '\0';
    std::string request(buffer);
    
    RCLCPP_INFO(this->get_logger(), "HTTP 요청: %s", request.substr(0, request.find('\n')).c_str());
    
    // POST /change/camera 요청 처리
    if (request.find("POST /change/camera") != std::string::npos) {
        handleCameraChangeRequest(client_fd, request);
        return;
    }
    
    // 기존 GET 요청들 (테스트용)
    if (request.find("GET /front") != std::string::npos) {
        switchCamera(0);  // 전면 카메라로 전환
    } else if (request.find("GET /back") != std::string::npos) {
        switchCamera(1);  // 후면 카메라로 전환
    }
    
    // 현재 카메라 프레임 가져오기
    cv::Mat frame = getCurrentCameraFrame();
    if (frame.empty()) {
        std::string error_response = "HTTP/1.1 500 Internal Server Error\r\n"
                                   "Content-Type: text/plain\r\n"
                                   "Content-Length: 25\r\n\r\n"
                                   "Camera not available";
        send(client_fd, error_response.c_str(), error_response.length(), 0);
        return;
    }
    
    // HTTP 응답 생성 및 전송
    std::string response = createHttpResponse(frame);
    send(client_fd, response.c_str(), response.length(), 0);
}

void AIServer::handleCameraChangeRequest(int client_fd, const std::string& request)
{
    RCLCPP_INFO(this->get_logger(), "전체 HTTP 요청 내용:");
    RCLCPP_INFO(this->get_logger(), "%s", request.c_str());
    
    // JSON 파싱 (개선된 구현)
    std::string camera_type = "front";  // 기본값
    
    // 여러 가지 패턴으로 시도
    std::vector<std::string> patterns = {
        "\"camera\":\"",
        "\"camera\": \"",
        "\"camera\" : \"",
        "\"camera\" :\""
    };
    
    for (const auto& pattern : patterns) {
        size_t camera_pos = request.find(pattern);
        if (camera_pos != std::string::npos) {
            size_t start = camera_pos + pattern.length();
            size_t end = request.find("\"", start);
            if (end != std::string::npos) {
                camera_type = request.substr(start, end - start);
                RCLCPP_INFO(this->get_logger(), "패턴 '%s'로 파싱 성공", pattern.c_str());
                break;
            }
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "파싱된 카메라 타입: '%s'", camera_type.c_str());
    
    // 카메라 전환
    if (camera_type == "front") {
        switchCamera(0);
        RCLCPP_INFO(this->get_logger(), "HTTP 요청: 전면 카메라로 전환");
    } else if (camera_type == "back") {
        switchCamera(1);
        RCLCPP_INFO(this->get_logger(), "HTTP 요청: 후면 카메라로 전환");
    } else {
        RCLCPP_WARN(this->get_logger(), "알 수 없는 카메라 타입: '%s'", camera_type.c_str());
    }
    
    // 성공 응답
    std::string response = "HTTP/1.1 200 OK\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: 25\r\n"
                          "Access-Control-Allow-Origin: *\r\n"
                          "\r\n"
                          "{\"status\":\"success\"}";
    
    send(client_fd, response.c_str(), response.length(), 0);
}

std::string AIServer::createHttpResponse(const cv::Mat& image)
{
    // 이미지를 JPEG로 인코딩
    std::vector<uchar> jpeg_data;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
    cv::imencode(".jpg", image, jpeg_data, params);
    
    std::string response = "HTTP/1.1 200 OK\r\n"
                          "Content-Type: image/jpeg\r\n"
                          "Content-Length: " + std::to_string(jpeg_data.size()) + "\r\n"
                          "Access-Control-Allow-Origin: *\r\n"
                          "\r\n";
    
    // 헤더와 이미지 데이터 결합
    std::string full_response = response;
    full_response.append(jpeg_data.begin(), jpeg_data.end());
    
    return full_response;
}

cv::Mat AIServer::getCurrentCameraFrame()
{
    cv::Mat frame;
    
    if (current_camera_ == 0) {
        // 전면 카메라에서 프레임 가져오기
        if (front_camera_) {
            // WebcamStreamer에서 현재 프레임 가져오기
            // (실제 구현은 WebcamStreamer 클래스에 따라 달라질 수 있음)
            frame = cv::Mat::zeros(480, 640, CV_8UC3);  // 임시 구현
        }
    } else {
        // 후면 카메라에서 프레임 가져오기
        if (back_camera_) {
            frame = cv::Mat::zeros(480, 640, CV_8UC3);  // 임시 구현
        }
    }
    
    return frame;
}

void AIServer::switchCamera(int camera_id)
{
    if (camera_id == 0) {
        current_camera_ = 0;
        RCLCPP_INFO(this->get_logger(), "전면 카메라로 전환");
    } else if (camera_id == 1) {
        current_camera_ = 1;
        RCLCPP_INFO(this->get_logger(), "후면 카메라로 전환");
    }
}

// 딥러닝 처리 함수들
void AIServer::startDeepLearning()
{
    if (deeplearning_running_) {
        RCLCPP_WARN(this->get_logger(), "딥러닝 처리가 이미 실행중입니다.");
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "딥러닝 처리 시작...");
    deeplearning_running_ = true;
    deeplearning_thread_ = std::thread(&AIServer::runDeepLearningThread, this);
}

void AIServer::stopDeepLearning()
{
    if (!deeplearning_running_) {
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "딥러닝 처리 종료중...");
    deeplearning_running_ = false;
    
    if (deeplearning_thread_.joinable()) {
        deeplearning_thread_.join();
    }
    
    RCLCPP_INFO(this->get_logger(), "딥러닝 처리 종료됨");
}

void AIServer::runDeepLearningThread()
{
    RCLCPP_INFO(this->get_logger(), "딥러닝 처리 스레드 시작");
    
    while (deeplearning_running_) {
        // 주기적으로 딥러닝 처리 실행
        processDeepLearning();
        
        // 1초마다 실행 (실시간 처리 대신 주기적 처리)
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    RCLCPP_INFO(this->get_logger(), "딥러닝 처리 스레드 종료");
}

void AIServer::processDeepLearning()
{
    // 현재 카메라 프레임 가져오기
    cv::Mat front_frame, back_frame;
    
    // 전면 카메라 프레임 (임시로 더미 데이터 사용)
    front_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // 후면 카메라 프레임 (임시로 더미 데이터 사용)
    back_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // Python 스크립트 실행을 위한 데이터 준비
    // 실제로는 카메라에서 읽은 프레임을 사용해야 함
    std::string input_data = "front_frame_data";  // 실제로는 이미지 데이터
    
    // 딥러닝 Python 스크립트 실행
    std::string script_path = "/home/ckim/ros-repo-4/deeplearning/src/dual_camera_system.py";
    std::string result = executePythonScript(script_path, input_data);
    
    // 결과 파싱 및 저장 (실제 구현 필요)
    {
        std::lock_guard<std::mutex> lock(dl_result_->result_mutex);
        // 결과를 dl_result_에 저장
        dl_result_->gestures[0] = "NORMAL";  // 전면 카메라
        dl_result_->gestures[1] = "NORMAL";  // 후면 카메라
        dl_result_->confidences[0] = 0.5;
        dl_result_->confidences[1] = 0.5;
    }
    
    RCLCPP_DEBUG(this->get_logger(), "딥러닝 처리 완료");
}

std::string AIServer::executePythonScript(const std::string& script_path, const std::string& input_data)
{
    // Python 스크립트 실행을 위한 명령어 구성
    std::string command = "python3 " + script_path + " --input " + input_data;
    
    // 실제로는 popen을 사용하여 Python 스크립트 실행
    // 여기서는 간단한 예시만 제공
    RCLCPP_DEBUG(this->get_logger(), "Python 스크립트 실행: %s", command.c_str());
    
    return "dummy_result";  // 실제로는 Python 스크립트의 출력을 반환
}

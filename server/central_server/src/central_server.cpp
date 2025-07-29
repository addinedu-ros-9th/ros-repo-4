#include "central_server/central_server.h"
#include <chrono>
#include <yaml-cpp/yaml.h>

CentralServer::CentralServer() : Node("central_server") {
    RCLCPP_INFO(this->get_logger(), "CentralServer 생성자 호출됨");
    
    udp_receiver_socket_ = -1;
    gui_sender_socket_ = -1;
    new_image_available_ = false;
    max_packet_size_ = 60000;

    setupUdpRelay();

    // DatabaseManager 생성
    db_manager_ = std::make_unique<DatabaseManager>();
    
    // HttpServer 생성 (DatabaseManager를 shared_ptr로 전달)
    auto shared_db_manager = std::shared_ptr<DatabaseManager>(db_manager_.get(), [](DatabaseManager*){});
    http_server_ = std::make_unique<HttpServer>(shared_db_manager, 8080);
    
    RCLCPP_INFO(this->get_logger(), "ROS2 토픽 및 서비스 설정 완료");
}

CentralServer::~CentralServer() {
    stop();
}

void CentralServer::setupUdpRelay()
{
    try {
        // 설정 파일 로드
        const char* config_env = std::getenv("CENTRAL_SERVER_CONFIG");
        std::string config_path = config_env ? config_env : "/home/wonho/ros-repo-4/server/config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        
        // AI Server로부터 수신할 설정
        ai_udp_receive_port_ = config["central_server"]["udp_receive_port"].as<int>();
        
        // GUI로 전송할 설정
        gui_client_ip_ = config["central_server"]["target_ros_gui"]["ip"].as<std::string>();
        gui_client_port_ = config["central_server"]["target_ros_gui"]["port"].as<int>();
        
        RCLCPP_INFO(this->get_logger(), "UDP 중계 설정 로드 완료:");
        RCLCPP_INFO(this->get_logger(), "  - AI Server 수신 포트: %d", ai_udp_receive_port_);
        RCLCPP_INFO(this->get_logger(), "  - GUI 전송 대상: %s:%d", 
                   gui_client_ip_.c_str(), gui_client_port_);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "설정 파일 로드 실패: %s", e.what());
        // 기본값 사용
        ai_udp_receive_port_ = 5005;
        gui_client_ip_ = "127.0.0.1";
        gui_client_port_ = 8888;
        
        RCLCPP_WARN(this->get_logger(), "기본값 사용: 수신포트=%d, GUI=%s:%d",
                   ai_udp_receive_port_, gui_client_ip_.c_str(), gui_client_port_);
    }
}

void CentralServer::start() {
    if (running_.load()) {
        RCLCPP_WARN(this->get_logger(), "서버가 이미 실행중입니다");
        return;
    }
    
    running_.store(true);
    
    RCLCPP_INFO(this->get_logger(), "1단계: UDP 소켓 초기화중...");
    if (!initializeUdpReceiver() || !initializeGuiSender()) {
        RCLCPP_ERROR(this->get_logger(), "UDP 소켓 초기화 실패!");
        running_.store(false);
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "2단계: UDP 수신 스레드 시작중...");
    udp_receiver_thread_ = std::thread(&CentralServer::runUdpReceiverThread, this);
    
    RCLCPP_INFO(this->get_logger(), "3단계: GUI 전송 스레드 시작중...");
    gui_sender_thread_ = std::thread(&CentralServer::runGuiSenderThread, this);
    
    RCLCPP_INFO(this->get_logger(), "4단계: DB 스레드 시작중...");
    db_thread_ = std::thread(&CentralServer::runDatabaseThread, this);
    
    RCLCPP_INFO(this->get_logger(), "5단계: HTTP 스레드 시작중...");
    http_thread_ = std::thread(&CentralServer::runHttpThread, this);
    
    // 잠시 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    RCLCPP_INFO(this->get_logger(), "Central Server 시작 완료!");
}

void CentralServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "서버 종료중...");
    running_.store(false);
    
    // UDP 스레드 종료 대기
    if (udp_receiver_thread_.joinable()) {
        udp_receiver_thread_.join();
        RCLCPP_INFO(this->get_logger(), "UDP 수신 스레드 종료됨");
    }
    
    if (gui_sender_thread_.joinable()) {
        gui_sender_thread_.join();
        RCLCPP_INFO(this->get_logger(), "GUI 전송 스레드 종료됨");
    }
    
    // 기존 스레드들 종료 대기
    if (db_thread_.joinable()) {
        db_thread_.join();
        RCLCPP_INFO(this->get_logger(), "DB 스레드 종료됨");
    }
    
    if (http_thread_.joinable()) {
        http_thread_.join();
        RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료됨");
    }
    
    // 소켓 정리
    if (udp_receiver_socket_ >= 0) {
        close(udp_receiver_socket_);
        udp_receiver_socket_ = -1;
    }
    
    if (gui_sender_socket_ >= 0) {
        close(gui_sender_socket_);
        gui_sender_socket_ = -1;
    }
    
    RCLCPP_INFO(this->get_logger(), "서버 종료 완료");
}

void CentralServer::runDatabaseThread() {
    RCLCPP_INFO(this->get_logger(), "DB 스레드 시작됨");
    
    // DB 연결 시도
    if (db_manager_->connect()) {
        RCLCPP_INFO(this->get_logger(), "MySQL 데이터베이스 연결 성공!");
    } else {
        RCLCPP_ERROR(this->get_logger(), "MySQL 데이터베이스 연결 실패!");
        RCLCPP_WARN(this->get_logger(), "DB 없이 계속 실행합니다...");
    }
    
    while (running_.load() && rclcpp::ok()) {
        // DB 연결 상태 확인 (5초마다)
        if (!db_manager_->isConnected()) {
            RCLCPP_WARN(this->get_logger(), "DB 연결 끊어짐. 재연결 시도...");
            db_manager_->connect();
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    RCLCPP_INFO(this->get_logger(), "DB 스레드 종료중...");
}

void CentralServer::runHttpThread() {
    RCLCPP_INFO(this->get_logger(), "HTTP 스레드 시작됨");
    
    // HTTP 서버 시작
    http_server_->start();
    RCLCPP_INFO(this->get_logger(), "HTTP 서버 시작 완료 (포트: 8080)");
    
    while (running_.load() && rclcpp::ok()) {
        // HTTP 서버 상태 확인
        if (!http_server_->isRunning()) {
            RCLCPP_ERROR(this->get_logger(), "HTTP 서버가 중지됨!");
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // HTTP 서버 정지
    http_server_->stop();
    RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료중...");
}

void CentralServer::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        // ROS2 Image 메시지를 OpenCV Mat으로 변환
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        
        // 이미지 정보 로그 (너무 자주 출력하지 않도록 제한)
        static int frame_count = 0;
        frame_count++;
        
        if (frame_count % 60 == 0) { // 60프레임마다 로그
            RCLCPP_INFO(this->get_logger(), 
                       "AI 서버로부터 이미지 수신 - 크기: %dx%d, 프레임 #%d", 
                       cv_ptr->image.cols, cv_ptr->image.rows, frame_count);
        }
        
        // 여기에 이미지 처리 로직을 추가할 수 있습니다
        // 예: 이미지 저장, 분석, 웹으로 스트리밍 등
        
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "이미지 변환 실패: %s", e.what());
    }
}

void CentralServer::statusCallback(const robot_interfaces::msg::RobotStatus::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), 
               "AI 서버 상태 수신 - Robot ID: %d, Status: %s", 
               msg->robot_id, msg->status.c_str());
    
    if (db_manager_->isConnected()) {
        RCLCPP_DEBUG(this->get_logger(), "상태를 데이터베이스에 저장 중...");
    }
}

void CentralServer::changeStatusCallback(
    const std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Request> request,
    std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Response> response)
{
    RCLCPP_INFO(this->get_logger(), 
               "상태 변경 요청 - Robot ID: %d, New Status: %s", 
               request->robot_id, request->new_status.c_str());
    
    response->success = true;
    response->message = "상태 변경 완료";
    
    RCLCPP_INFO(this->get_logger(), "상태 변경 서비스 응답 완료");
}

bool CentralServer::initializeUdpReceiver()
{
    // AI Server로부터 수신할 UDP 소켓 생성
    udp_receiver_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_receiver_socket_ < 0) {
        RCLCPP_ERROR(this->get_logger(), "UDP 수신 소켓 생성 실패");
        return false;
    }
    
    // 소켓 옵션 설정
    int opt = 1;
    if (setsockopt(udp_receiver_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        RCLCPP_WARN(this->get_logger(), "SO_REUSEADDR 설정 실패");
    }
    
    // 주소 설정
    memset(&ai_server_addr_, 0, sizeof(ai_server_addr_));
    ai_server_addr_.sin_family = AF_INET;
    ai_server_addr_.sin_addr.s_addr = INADDR_ANY;  // 모든 IP에서 수신
    ai_server_addr_.sin_port = htons(ai_udp_receive_port_);
    
    // 바인딩
    if (bind(udp_receiver_socket_, (struct sockaddr*)&ai_server_addr_, sizeof(ai_server_addr_)) < 0) {
        RCLCPP_ERROR(this->get_logger(), "UDP 수신 소켓 바인딩 실패 (포트: %d)", ai_udp_receive_port_);
        close(udp_receiver_socket_);
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "UDP 수신 소켓 초기화 완료 (포트: %d)", ai_udp_receive_port_);
    return true;
}

bool CentralServer::initializeGuiSender()
{
    // GUI로 전송할 UDP 소켓 생성
    gui_sender_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (gui_sender_socket_ < 0) {
        RCLCPP_ERROR(this->get_logger(), "GUI 전송 소켓 생성 실패");
        return false;
    }
    
    // GUI 주소 설정
    memset(&gui_client_addr_, 0, sizeof(gui_client_addr_));
    gui_client_addr_.sin_family = AF_INET;
    gui_client_addr_.sin_port = htons(gui_client_port_);
    
    if (inet_pton(AF_INET, gui_client_ip_.c_str(), &gui_client_addr_.sin_addr) <= 0) {
        RCLCPP_ERROR(this->get_logger(), "GUI IP 주소 변환 실패: %s", gui_client_ip_.c_str());
        close(gui_sender_socket_);
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "GUI 전송 소켓 초기화 완료 (%s:%d)", 
               gui_client_ip_.c_str(), gui_client_port_);
    return true;
}

void CentralServer::runUdpReceiverThread()
{
    RCLCPP_INFO(this->get_logger(), "UDP 수신 스레드 시작됨");
    
    std::vector<uint8_t> buffer(max_packet_size_);
    struct sockaddr_in sender_addr;
    socklen_t sender_len = sizeof(sender_addr);
    
    while (running_.load() && rclcpp::ok()) {
        // AI Server로부터 UDP 패킷 수신
        ssize_t received = recvfrom(udp_receiver_socket_, buffer.data(), buffer.size(), 0,
                                   (struct sockaddr*)&sender_addr, &sender_len);
        
        if (received > 0) {
            try {
                // 받은 데이터를 이미지로 디코딩
                std::vector<uint8_t> jpeg_data(buffer.begin(), buffer.begin() + received);
                cv::Mat image = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
                
                if (!image.empty()) {
                    processReceivedImage(image);
                    
                    // 로그 (너무 자주 출력하지 않도록)
                    static int packet_count = 0;
                    packet_count++;
                    if (packet_count % 60 == 0) {
                        RCLCPP_INFO(this->get_logger(), 
                                   "AI Server로부터 이미지 수신 - 크기: %dx%d, 패킷#%d", 
                                   image.cols, image.rows, packet_count);
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), "이미지 디코딩 실패");
                }
                
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "이미지 처리 오류: %s", e.what());
            }
        } else if (received < 0) {
            if (running_.load()) {
                RCLCPP_ERROR(this->get_logger(), "UDP 수신 오류: %s", strerror(errno));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "UDP 수신 스레드 종료중...");
}

void CentralServer::processReceivedImage(const cv::Mat& image)
{
    std::lock_guard<std::mutex> lock(image_buffer_mutex_);
    latest_image_ = image.clone();
    new_image_available_ = true;
}

void CentralServer::runGuiSenderThread()
{
    RCLCPP_INFO(this->get_logger(), "GUI 전송 스레드 시작됨");
    
    while (running_.load() && rclcpp::ok()) {
        cv::Mat image_to_send;
        bool has_new_image = false;
        
        // 새로운 이미지가 있는지 확인
        {
            std::lock_guard<std::mutex> lock(image_buffer_mutex_);
            if (new_image_available_) {
                image_to_send = latest_image_.clone();
                new_image_available_ = false;
                has_new_image = true;
            }
        }
        
        // 새로운 이미지가 있으면 GUI로 전송
        if (has_new_image && !image_to_send.empty()) {
            sendImageToGui(image_to_send);
        }
        
        // 30fps로 제한 (33ms)
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    
    RCLCPP_INFO(this->get_logger(), "GUI 전송 스레드 종료중...");
}

void CentralServer::sendImageToGui(const cv::Mat& image)
{
    try {
        // 이미지를 JPEG로 압축
        std::vector<uint8_t> jpeg_buffer;
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 70};
        
        if (!cv::imencode(".jpg", image, jpeg_buffer, compression_params)) {
            RCLCPP_ERROR(this->get_logger(), "이미지 JPEG 인코딩 실패");
            return;
        }
        
        // GUI로 UDP 전송
        ssize_t sent = sendto(gui_sender_socket_, jpeg_buffer.data(), jpeg_buffer.size(), 0,
                             (struct sockaddr*)&gui_client_addr_, sizeof(gui_client_addr_));
        
        if (sent < 0) {
            RCLCPP_ERROR(this->get_logger(), "GUI로 전송 실패: %s", strerror(errno));
        } else {
            // 로그 (너무 자주 출력하지 않도록)
            static int send_count = 0;
            send_count++;
            if (send_count % 60 == 0) {
                RCLCPP_DEBUG(this->get_logger(), "GUI로 이미지 전송 완료 - %ld bytes", sent);
            }
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "GUI 전송 오류: %s", e.what());
    }
}

void CentralServer::init() {
    RCLCPP_INFO(this->get_logger(), "[init] this ptr: %p", (void*)this);
    try {
        auto self = rclcpp::Node::shared_from_this();
        RCLCPP_INFO(this->get_logger(), "[init] shared_from_this() ptr: %p", (void*)self.get());
        image_transport_ = std::make_shared<image_transport::ImageTransport>(self);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[init] Exception during shared_from_this(): %s", e.what());
        throw;
    }
    image_subscriber_ = image_transport_->subscribe(
        "webcam/image_raw", 1, 
        std::bind(&CentralServer::imageCallback, this, std::placeholders::_1));
    status_subscriber_ = this->create_subscription<robot_interfaces::msg::RobotStatus>(
        "robot_status", 10,
        std::bind(&CentralServer::statusCallback, this, std::placeholders::_1));
    status_service_ = this->create_service<robot_interfaces::srv::ChangeRobotStatus>(
        "change_robot_status",
        std::bind(&CentralServer::changeStatusCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(this->get_logger(), "ROS2 토픽 및 서비스 설정 완료");
}
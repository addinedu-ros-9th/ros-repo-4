#include "central_server/central_server.h"
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <json/json.h>

CentralServer::CentralServer() : Node("central_server") {
    RCLCPP_INFO(this->get_logger(), "CentralServer 생성자 호출됨");
    
    http_port_ = 8080;
    http_host_ = "0.0.0.0";

    setupHttpServer();

    // DatabaseManager 생성
    db_manager_ = std::make_unique<DatabaseManager>();
    
    // HttpServer 생성 (DatabaseManager를 shared_ptr로 전달)
    auto shared_db_manager = std::shared_ptr<DatabaseManager>(db_manager_.get(), [](DatabaseManager*){});
    http_server_ = std::make_unique<HttpServer>(shared_db_manager, http_port_);
    
    RCLCPP_INFO(this->get_logger(), "ROS2 토픽 및 서비스 설정 완료");
}

CentralServer::~CentralServer() {
    stop();
}

void CentralServer::setupHttpServer()
{
    try {
        // 설정 파일 로드
        const char* config_env = std::getenv("CENTRAL_SERVER_CONFIG");
        std::string config_path = config_env ? config_env : "/home/wonho/ros-repo-4/config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        
        // HTTP 서버 설정
        http_port_ = config["central_server"]["http_port"].as<int>();
        http_host_ = config["central_server"]["http_host"].as<std::string>();
        
        RCLCPP_INFO(this->get_logger(), "HTTP 서버 설정 로드 완료:");
        RCLCPP_INFO(this->get_logger(), "  - HTTP 서버 포트: %d", http_port_);
        RCLCPP_INFO(this->get_logger(), "  - HTTP 서버 호스트: %s", http_host_.c_str());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "설정 파일 로드 실패: %s", e.what());
        // 기본값 사용
        http_port_ = 8080;
        http_host_ = "0.0.0.0";
        
        RCLCPP_WARN(this->get_logger(), "기본값 사용: HTTP 포트=%d, 호스트=%s",
                   http_port_, http_host_.c_str());
    }
}

void CentralServer::start() {
    if (running_.load()) {
        RCLCPP_WARN(this->get_logger(), "서버가 이미 실행중입니다");
        return;
    }
    
    running_.store(true);
    
    RCLCPP_INFO(this->get_logger(), "1단계: DB 스레드 시작중...");
    db_thread_ = std::thread(&CentralServer::runDatabaseThread, this);
    
    RCLCPP_INFO(this->get_logger(), "2단계: HTTP 스레드 시작중...");
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
    
    // 기존 스레드들 종료 대기
    if (db_thread_.joinable()) {
        db_thread_.join();
        RCLCPP_INFO(this->get_logger(), "DB 스레드 종료됨");
    }
    
    if (http_thread_.joinable()) {
        http_thread_.join();
        RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료됨");
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
    RCLCPP_INFO(this->get_logger(), "HTTP 서버 시작 완료 (포트: %d)", http_port_);
    
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
    
    // HTTP를 통해 GUI 클라이언트들에게 로봇 상태 전송
    sendRobotStatusToGui(msg->robot_id, msg->status, "robot_controller");
    
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
    
    // HTTP를 통해 GUI 클라이언트들에게 상태 변경 알림
    sendRobotStatusToGui(request->robot_id, request->new_status, "user_gui");
    
    response->success = true;
    response->message = "상태 변경 완료";
    
    RCLCPP_INFO(this->get_logger(), "상태 변경 서비스 응답 완료");
}

void CentralServer::sendRobotLocationToGui(int robot_id, float location_x, float location_y)
{
    Json::Value message;
    message["robot_id"] = robot_id;
    message["location_x"] = location_x;
    message["location_y"] = location_y;
    
    std::string json_message = message.toStyledString();
    broadcastToGuiClients(json_message);
    
    RCLCPP_INFO(this->get_logger(), "로봇 위치 전송: Robot ID %d at (%.2f, %.2f)", 
               robot_id, location_x, location_y);
}

void CentralServer::sendRobotStatusToGui(int robot_id, const std::string& status, const std::string& source)
{
    Json::Value message;
    message["robot_id"] = robot_id;
    message["status"] = status;
    message["source"] = source;
    
    std::string json_message = message.toStyledString();
    broadcastToGuiClients(json_message);
    
    RCLCPP_INFO(this->get_logger(), "로봇 상태 전송: Robot ID %d, Status %s, Source %s", 
               robot_id, status.c_str(), source.c_str());
}

void CentralServer::sendArrivalNotificationToGui(int robot_id)
{
    Json::Value message;
    message["robot_id"] = robot_id;
    message["status"] = "arrived";
    
    std::string json_message = message.toStyledString();
    broadcastToGuiClients(json_message);
    
    RCLCPP_INFO(this->get_logger(), "도착 알림 전송: Robot ID %d", robot_id);
}

void CentralServer::broadcastToGuiClients(const std::string& message)
{
    // HttpServer를 통해 GUI 클라이언트들에게 메시지 브로드캐스트
    if (http_server_) {
        http_server_->broadcastToClients(message);
        RCLCPP_DEBUG(this->get_logger(), "GUI 클라이언트들에게 메시지 브로드캐스트: %s", message.c_str());
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
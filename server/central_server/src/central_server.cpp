#include "central_server/central_server.h"
#include <chrono>

CentralServer::CentralServer() : Node("central_server"), running_(false) {
    RCLCPP_INFO(this->get_logger(), "CentralServer 생성자 호출됨");
    
    // DatabaseManager 생성
    db_manager_ = std::make_unique<DatabaseManager>();
    
    // HttpServer 생성 (DatabaseManager를 shared_ptr로 전달)
    auto shared_db_manager = std::shared_ptr<DatabaseManager>(db_manager_.get(), [](DatabaseManager*){});
    http_server_ = std::make_unique<HttpServer>(shared_db_manager, 8080);
    
    // Image Transport 초기화
    image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // AI 서버로부터 이미지 구독
    image_subscriber_ = image_transport_->subscribe(
        "webcam/image_raw", 1, 
        std::bind(&CentralServer::imageCallback, this, std::placeholders::_1));
    
    // AI 서버로부터 상태 구독
    status_subscriber_ = this->create_subscription<robot_interfaces::msg::RobotStatus>(
        "robot_status", 10,
        std::bind(&CentralServer::statusCallback, this, std::placeholders::_1));
    
    // 상태 변경 서비스 제공
    status_service_ = this->create_service<robot_interfaces::srv::ChangeRobotStatus>(
        "change_robot_status",
        std::bind(&CentralServer::changeStatusCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    
    RCLCPP_INFO(this->get_logger(), "ROS2 토픽 및 서비스 설정 완료");
}

CentralServer::~CentralServer() {
    stop();
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
}

void CentralServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "서버 종료중...");
    running_.store(false);
    
    // 스레드 종료 대기
    if (db_thread_.joinable()) {
        db_thread_.join();
        RCLCPP_INFO(this->get_logger(), "DB 스레드 종료됨");
    }
    
    if (http_thread_.joinable()) {
        http_thread_.join();
        RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료됨");
    }
}

void CentralServer::runDatabaseThread() {
    RCLCPP_INFO(this->get_logger(), "DB 스레드 시작됨");
    
    // DB 연결 시도
    if (db_manager_->connect()) {
        RCLCPP_INFO(this->get_logger(), "MySQL 데이터베이스 연결 성공!");
        
        // 간단한 테스트
        auto stations = db_manager_->getAllStations();
        RCLCPP_INFO(this->get_logger(), "정류장 개수: %zu", stations.size());
        
    } else {
        RCLCPP_ERROR(this->get_logger(), "MySQL 데이터베이스 연결 실패!");
        RCLCPP_WARN(this->get_logger(), "DB 없이 계속 실행합니다...");
    }
    
    while (running_.load() && rclcpp::ok()) {
        // DB 연결 상태 확인
        if (db_manager_->isConnected()) {
            RCLCPP_DEBUG(this->get_logger(), "DB 연결 상태 양호");
        } else {
            RCLCPP_WARN(this->get_logger(), "DB 연결 끊어짐. 재연결 시도...");
            db_manager_->connect();
        }
        
        // 5초마다 체크
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    RCLCPP_INFO(this->get_logger(), "DB 스레드 종료중...");
}

void CentralServer::runHttpThread() {
    RCLCPP_INFO(this->get_logger(), "HTTP 스레드 시작됨");
    
    // HTTP 서버 시작
    http_server_->start();
    RCLCPP_INFO(this->get_logger(), "HTTP 서버가 포트 8080에서 시작되었습니다");
    
    while (running_.load() && rclcpp::ok()) {
        // HTTP 서버 상태 확인
        if (http_server_->isRunning()) {
            RCLCPP_DEBUG(this->get_logger(), "HTTP 서버 실행중...");
        } else {
            RCLCPP_WARN(this->get_logger(), "HTTP 서버가 중지됨!");
            break;
        }
        
        // 1초마다 체크
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
               "AI 서버 상태 수신 - Robot ID: %s, Status: %s", 
               msg->robot_id.c_str(), msg->status.c_str());
    
    // 데이터베이스에 상태 저장 (DB가 연결된 경우)
    if (db_manager_->isConnected()) {
        // 여기에 DB 저장 로직을 추가할 수 있습니다
        RCLCPP_DEBUG(this->get_logger(), "상태를 데이터베이스에 저장 중...");
    }
}

void CentralServer::changeStatusCallback(
    const std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Request> request,
    std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Response> response)
{
    RCLCPP_INFO(this->get_logger(), 
               "상태 변경 요청 - Robot ID: %s, New Status: %s", 
               request->robot_id.c_str(), request->new_status.c_str());
    
    // 상태 변경 처리
    response->success = true;
    response->message = "상태 변경 완료";
    
    RCLCPP_INFO(this->get_logger(), "상태 변경 서비스 응답 완료");
}
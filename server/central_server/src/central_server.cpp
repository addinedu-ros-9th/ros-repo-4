#include "central_server/central_server.h"
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <json/json.h>
#include <iomanip>
#include <sstream>

CentralServer::CentralServer() : Node("central_server") {
    RCLCPP_INFO(this->get_logger(), "CentralServer 생성자 호출됨");
    
    http_port_ = 8080;
    http_host_ = "0.0.0.0";

    setupHttpServer();

    // DatabaseManager 생성
    db_manager_ = std::make_unique<DatabaseManager>();
    
    // RobotNavigationManager 생성
    nav_manager_ = std::make_unique<RobotNavigationManager>();
    
    // HttpServer 생성 (DatabaseManager를 shared_ptr로 전달)
    auto shared_db_manager = std::shared_ptr<DatabaseManager>(db_manager_.get(), [](DatabaseManager*){});
    http_server_ = std::make_unique<HttpServer>(shared_db_manager, http_port_);
    
    // HttpServer와 RobotNavigationManager 연결
    auto shared_nav_manager = std::shared_ptr<RobotNavigationManager>(nav_manager_.get(), [](RobotNavigationManager*){});
    http_server_->setRobotNavigationManager(shared_nav_manager);
}

CentralServer::~CentralServer() {
    stop();
}

void CentralServer::setupHttpServer()
{
    try {
        // 설정 파일 로드
        const char* config_env = std::getenv("CENTRAL_SERVER_CONFIG");
        std::string config_path = config_env ? config_env : "config.yaml";
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
    
    bool db_connection_logged = false;
    auto last_db_check = std::chrono::steady_clock::now();
    const auto db_check_interval = std::chrono::seconds(30); // 30초마다 DB 연결 확인으로 증가
    
    // 초기 DB 연결 시도
    if (db_manager_->connect()) {
        RCLCPP_INFO(this->get_logger(), "MySQL 데이터베이스 연결 성공!");
        db_connection_logged = true;
    } else {
        RCLCPP_ERROR(this->get_logger(), "MySQL 데이터베이스 초기 연결 실패!");
        RCLCPP_WARN(this->get_logger(), "DB 없이 계속 실행합니다... (30초마다 재연결 시도)");
        RCLCPP_INFO(this->get_logger(), "DB 연결 문제 해결 방법:");
        RCLCPP_INFO(this->get_logger(), "  1. MySQL 서비스 확인: sudo systemctl status mysql");
        RCLCPP_INFO(this->get_logger(), "  2. 사용자 권한 확인: sudo mysql -u root -p");
        RCLCPP_INFO(this->get_logger(), "  3. config.yaml의 DB 설정 확인");
        db_connection_logged = true;
    }
    
    while (running_.load() && rclcpp::ok()) {
        auto now = std::chrono::steady_clock::now();
        
        // DB 연결 상태 확인 (30초마다)
        if (now - last_db_check >= db_check_interval) {
            if (!db_manager_->isConnected()) {
                // 첫 번째 연결 실패 후에는 DEBUG 레벨로만 로그 출력
                if (!db_connection_logged) {
                    RCLCPP_WARN(this->get_logger(), "DB 연결 끊어짐. 주기적으로 재연결 시도합니다...");
                    db_connection_logged = true;
                }
                
                // 재연결 시도 (실패해도 에러 로그는 DatabaseManager에서 처리)
                bool reconnect_success = db_manager_->connect();
                
                if (reconnect_success) {
                    RCLCPP_INFO(this->get_logger(), "DB 연결이 복구되었습니다!");
                    db_connection_logged = false;
                }
            } else {
                // 연결이 정상인 경우
                if (db_connection_logged) {
                    RCLCPP_INFO(this->get_logger(), "DB 연결 상태: 정상");
                    db_connection_logged = false;
                }
            }
            last_db_check = now;
        }
        
        // RobotNavigationManager의 콜백들 처리
        if (nav_manager_) {
            try {
                auto shared_nav_manager = std::shared_ptr<RobotNavigationManager>(nav_manager_.get(), [](RobotNavigationManager*){});
                rclcpp::spin_some(shared_nav_manager);
            } catch (const std::exception& e) {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "RobotNavigationManager 스핀 중 오류: %s", e.what());
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100ms 간격으로 스핀
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

void CentralServer::eventHandleCallback(
    const std::shared_ptr<control_interfaces::srv::EventHandle::Request> request,
    std::shared_ptr<control_interfaces::srv::EventHandle::Response> response)
{
    RCLCPP_INFO(this->get_logger(), 
               "이벤트 핸들 요청 - Event Type: %s", 
               request->event_type.c_str());
    
    // 이벤트 타입에 따른 처리
    if (request->event_type == "alert_occupied") {
        // 관리자 사용중 블락 알림
        response->status = "success";
    } else if (request->event_type == "alert_idle") {
        // 사용 가능한 상태 알림
        response->status = "success";
    } else if (request->event_type == "navigating_complete") {
        // 길안내 완료 알림
        response->status = "success";
    } else {
        response->status = "unknown_event";
    }
    
    RCLCPP_INFO(this->get_logger(), "이벤트 핸들 서비스 응답 완료");
}

void CentralServer::trackHandleCallback(
    const std::shared_ptr<control_interfaces::srv::TrackHandle::Request> request,
    std::shared_ptr<control_interfaces::srv::TrackHandle::Response> response)
{
    RCLCPP_INFO(this->get_logger(), 
               "트랙 핸들 요청 - Event Type: %s, Left Angle: %.2f, Right Angle: %.2f", 
               request->event_type.c_str(), request->left_angle, request->right_angle);
    
    // 트랙 핸들 처리
    response->status = "success";
    response->distance = 0.0; // 실제 거리 계산 로직 필요
    
    RCLCPP_INFO(this->get_logger(), "트랙 핸들 서비스 응답 완료");
}



void CentralServer::init() {
    RCLCPP_INFO(this->get_logger(), "[init] this ptr: %p", (void*)this);
    
    event_service_ = this->create_service<control_interfaces::srv::EventHandle>(
        "event_handle",
        std::bind(&CentralServer::eventHandleCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    track_service_ = this->create_service<control_interfaces::srv::TrackHandle>(
        "track_handle",
        std::bind(&CentralServer::trackHandleCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(this->get_logger(), "ROS2 서비스 설정 완료");
}
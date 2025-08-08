#include "central_server/central_server.h"
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <json/json.h>
#include <ctime>
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
    
    // 서비스 클라이언트들 초기화 (Central → Robot)
    control_event_client_ = this->create_client<control_interfaces::srv::EventHandle>("/control_event");
    navigate_client_ = this->create_client<control_interfaces::srv::NavigateHandle>("/navigate_event");
    tracking_event_client_ = this->create_client<control_interfaces::srv::TrackHandle>("/tracking_event");
    
    // RobotNavigationManager에 서비스 클라이언트 설정
    nav_manager_->setControlEventClient(control_event_client_);
    nav_manager_->setNavigateClient(navigate_client_);
    nav_manager_->setTrackingEventClient(tracking_event_client_);
    
    // WebSocket 서버 설정
    setupWebSocketServer();

    // 로봇 ID 로드 (config.yaml의 central_server.robot_id, 기본 0)
    try {
        const char* config_env = std::getenv("CENTRAL_SERVER_CONFIG");
        std::string config_path = config_env ? config_env : "config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        robot_id_ = config["central_server"]["robot_id"].as<int>(0);
        RCLCPP_INFO(this->get_logger(), "로봇 ID 설정: %d", robot_id_);
    } catch (const std::exception& e) {
        robot_id_ = 0;
        RCLCPP_WARN(this->get_logger(), "로봇 ID 설정 로드 실패: %s (기본 0)", e.what());
    }
    
    // HttpServer 생성 (DatabaseManager를 shared_ptr로 전달)
    auto shared_db_manager = std::shared_ptr<DatabaseManager>(db_manager_.get(), [](DatabaseManager*){});
    http_server_ = std::make_unique<HttpServer>(shared_db_manager, http_port_);
    
    // HttpServer와 RobotNavigationManager 연결
    auto shared_nav_manager = std::shared_ptr<RobotNavigationManager>(nav_manager_.get(), [](RobotNavigationManager*){});
    http_server_->setRobotNavigationManager(shared_nav_manager);
    
    // WebSocket 서버 생성 (GUI 알림용) 및 HttpServer에 직접 주입
    websocket_server_ = std::make_unique<WebSocketServer>(websocket_port_);
    auto shared_ws = std::shared_ptr<WebSocketServer>(websocket_server_.get(), [](WebSocketServer*){});
    http_server_->setWebSocketServer(shared_ws);

    // 로봇 이벤트 → WebSocket(GUI) 전달 + DB 로그 저장 콜백 설정
    nav_manager_->setRobotEventCallback([this](const std::string& event_type) {
        try {
            if (event_type == "arrived_to_call") {
                Json::Value message;
                message["type"] = "arrived_to_call";
                message["timestamp"] = std::to_string(time(nullptr));

                Json::StreamWriterBuilder builder;
                std::string json_message = Json::writeString(builder, message);

                if (websocket_server_) {
                    websocket_server_->broadcastMessageToType("gui", json_message);
                }
            }
            else if (event_type == "navigating_complete") {
                Json::Value message;
                message["type"] = "navigating_complete";
                message["timestamp"] = std::to_string(time(nullptr));

                Json::StreamWriterBuilder builder;
                std::string json_message = Json::writeString(builder, message);

                if (websocket_server_) {
                    websocket_server_->broadcastMessageToType("gui", json_message);
                }
            }
            else if (event_type == "arrived_to_station") {
                Json::Value message;
                message["type"] = "arrived_to_station";
                message["timestamp"] = std::to_string(time(nullptr));

                Json::StreamWriterBuilder builder;
                std::string json_message = Json::writeString(builder, message);

                if (websocket_server_) {
                    websocket_server_->sendAlertIdle(robot_id_);
                }
            }
            else if (event_type == "return_command") {
                Json::Value message;
                message["type"] = "return_command";
                message["timestamp"] = std::to_string(time(nullptr));

                Json::StreamWriterBuilder builder;
                std::string json_message = Json::writeString(builder, message);

                if (websocket_server_) {
                    websocket_server_->broadcastMessageToType("gui", json_message);
                }
            }
            else{
                RCLCPP_WARN(this->get_logger(), "알 수 없는 이벤트 타입: %s", event_type.c_str());
                return;
            }
            
            // 공통 DB 로그 저장 (모든 이벤트 공통)
            if (db_manager_) {
                std::string current_datetime = db_manager_->getCurrentDateTime();
                if (!current_datetime.empty() && robot_id_ != 0) {
                    bool log_ok = db_manager_->insertRobotLogWithType(
                        robot_id_, nullptr, current_datetime, 0, 0, event_type, "");
                    if (!log_ok) {
                        RCLCPP_WARN(this->get_logger(), "robot_log 저장 실패: %s", event_type.c_str());
                    }
                } else if (robot_id_ == 0) {
                    RCLCPP_WARN(this->get_logger(), "robot_id가 0입니다. config.yaml의 central_server.robot_id를 설정하세요.");
                } else {
                    RCLCPP_WARN(this->get_logger(), "현재 시간 조회 실패로 robot_log 미저장 (event: %s)", event_type.c_str());
                }
            }
            // 다른 이벤트는 필요 시 별도 처리 분기 추가
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "로봇 이벤트 웹소켓 전송 실패: %s", e.what());
        }
    });
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

void CentralServer::setupWebSocketServer()
{
    try {
        // 설정 파일 로드
        const char* config_env = std::getenv("CENTRAL_SERVER_CONFIG");
        std::string config_path = config_env ? config_env : "config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        
        // WebSocket 서버 설정
        websocket_port_ = config["central_server"]["websocket_port"].as<int>();
        
        RCLCPP_INFO(this->get_logger(), "WebSocket 서버 설정 로드 완료:");
        RCLCPP_INFO(this->get_logger(), "  - WebSocket 서버 포트: %d", websocket_port_);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "WebSocket 설정 파일 로드 실패: %s", e.what());
        // 기본값 사용
        websocket_port_ = 3000;
        
        RCLCPP_WARN(this->get_logger(), "기본값 사용: WebSocket 포트=%d", websocket_port_);
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
    
    RCLCPP_INFO(this->get_logger(), "3단계: WebSocket 스레드 시작중...");
    websocket_thread_ = std::thread(&CentralServer::runWebSocketThread, this);
    
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
    
    if (websocket_thread_.joinable()) {
        websocket_thread_.join();
        RCLCPP_INFO(this->get_logger(), "WebSocket 스레드 종료됨");
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





void CentralServer::init() {
    RCLCPP_INFO(this->get_logger(), "[init] this ptr: %p", (void*)this);
    
    event_service_ = this->create_service<control_interfaces::srv::EventHandle>(
        "event_handle",
        std::bind(&CentralServer::eventHandleCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    control_event_client_ = this->create_client<control_interfaces::srv::EventHandle>("/control_event");
    navigate_client_ = this->create_client<control_interfaces::srv::NavigateHandle>("/navigate_event");
    teleop_publisher_ = this->create_publisher<std_msgs::msg::String>("/teleop_event", 10);
    tracking_event_client_ = this->create_client<control_interfaces::srv::TrackHandle>("/tracking_event");
    RCLCPP_INFO(this->get_logger(), "ROS2 서비스 설정 완료");
}

void CentralServer::runWebSocketThread() {
    RCLCPP_INFO(this->get_logger(), "WebSocket 스레드 시작됨");
    
    // WebSocket 서버 시작
    if (websocket_server_->start()) {
        RCLCPP_INFO(this->get_logger(), "WebSocket 서버 시작 완료 (포트: %d)", websocket_port_);
    } else {
        RCLCPP_ERROR(this->get_logger(), "WebSocket 서버 시작 실패!");
        return;
    }
    
    while (running_.load() && rclcpp::ok()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // WebSocket 서버 종료
    websocket_server_->stop();
    RCLCPP_INFO(this->get_logger(), "WebSocket 스레드 종료중...");
}
#include "central_server/robot_navigation_manager.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <curl/curl.h>
#include <json/json.h>
#include <ctime>
#include <chrono>
#include <future>

using namespace std::chrono_literals;

RobotNavigationManager::RobotNavigationManager() 
    : Node("robot_navigation_manager") {
    
    RCLCPP_INFO(this->get_logger(), "RobotNavigationManager 초기화 시작");
    
    // 현재 Domain ID를 환경 변수에서 읽기 (기본값 0)
    const char* domain_id_env = std::getenv("ROS_DOMAIN_ID");
    int current_domain_id = domain_id_env ? std::atoi(domain_id_env) : 0;
    
    RCLCPP_INFO(this->get_logger(), "현재 Domain ID: %d", current_domain_id);
    
    // IF-01: 로봇 목적지 전송 퍼블리셔
    navigation_command_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/navigation_command", 10);
    
    // IF-08: 실시간 추적 목표 전송 퍼블리셔
    tracking_goal_pub_ = this->create_publisher<geometry_msgs::msg::Point>(
        "/trackingGoal", 10);
    
    // Teleop 명령 퍼블리셔 (새로운 인터페이스)
    teleop_publisher_ = this->create_publisher<std_msgs::msg::String>(
        "/teleop_event", 10);
    
    // IF-02: 로봇의 현재 위치 서브스크라이버
    pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
        "/pose", 10,
        std::bind(&RobotNavigationManager::poseCallback, this, std::placeholders::_1));
    
    // IF-03: 로봇의 주행 시작점 서브스크라이버
    start_point_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/start_point", 10,
        std::bind(&RobotNavigationManager::startPointCallback, this, std::placeholders::_1));
    
    // IF-04: 로봇의 주행 목적지 서브스크라이버
    target_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/target", 10,
        std::bind(&RobotNavigationManager::targetCallback, this, std::placeholders::_1));
    
    // IF-05: 로봇의 네트워크 상태 서브스크라이버
    net_level_sub_ = this->create_subscription<std_msgs::msg::Int32>(
        "/net_level", 10,
        std::bind(&RobotNavigationManager::networkLevelCallback, this, std::placeholders::_1));
    
    // IF-06: 로봇의 배터리 잔량 서브스크라이버
    battery_sub_ = this->create_subscription<std_msgs::msg::Int32>(
        "/battery", 10,
        std::bind(&RobotNavigationManager::batteryCallback, this, std::placeholders::_1));
    
    // IF-07: 로봇 주행 상태 서브스크라이버
    nav_status_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/nav_status", 10,
        std::bind(&RobotNavigationManager::navStatusCallback, this, std::placeholders::_1));
    
    // IF-09: 장애물 정보 서브스크라이버
    obstacle_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
        "/detected_obstacle", 10,
        std::bind(&RobotNavigationManager::obstacleCallback, this, std::placeholders::_1));
    
    // 서비스 서버 (Robot → Central)
    robot_event_service_ = this->create_service<control_interfaces::srv::EventHandle>(
        "/robot_event",
        std::bind(&RobotNavigationManager::robotEventCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    
    // 장애물 감지 서비스 서버 (Robot → Central)
    detect_obstacle_service_ = this->create_service<control_interfaces::srv::DetectHandle>(
        "/detect_obstacle",
        std::bind(&RobotNavigationManager::detectObstacleCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    
    // 초기값 설정
    current_network_level_ = 0;
    current_battery_ = 0;
    current_robot_x_ = 0.0;
    current_robot_y_ = 0.0;
    current_robot_yaw_ = 0.0;
    
    RCLCPP_INFO(this->get_logger(), "RobotNavigationManager 초기화 완료");
    RCLCPP_INFO(this->get_logger(), "토픽 설정:");
    RCLCPP_INFO(this->get_logger(), "  - 발행: /navigation_command");
    RCLCPP_INFO(this->get_logger(), "  - 발행: /trackingGoal");
    RCLCPP_INFO(this->get_logger(), "  - 발행: /cmd_vel");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /pose");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /start_point");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /target");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /net_level");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /battery");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /nav_status");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /detected_obstacle");
    RCLCPP_INFO(this->get_logger(), "  - 서비스 서버: /robot_event");
    RCLCPP_INFO(this->get_logger(), "  - 서비스 서버: /detect_obstacle");
}

RobotNavigationManager::~RobotNavigationManager() {
    RCLCPP_INFO(this->get_logger(), "RobotNavigationManager 소멸자 호출");
}

// IF-01: 로봇 목적지 전송 (Central → Robot)
bool RobotNavigationManager::sendNavigationCommand(const std::string& command) {
    try {
        auto message = std_msgs::msg::String();
        message.data = "data: '" + command + "'";
        
        navigation_command_pub_->publish(message);
        logNavigationCommand(command);
        
        RCLCPP_INFO(this->get_logger(), "네비게이션 명령 전송: %s", command.c_str());
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "네비게이션 명령 전송 실패: %s", e.what());
        return false;
    }
}

bool RobotNavigationManager::sendWaypointCommand(const std::string& waypoint_name) {
    return sendNavigationCommand(waypoint_name);
}

bool RobotNavigationManager::sendGoStartCommand() {
    return sendNavigationCommand("go_start");
}

bool RobotNavigationManager::sendStopCommand() {
    return sendNavigationCommand("stop/cancel");
}

// IF-08: 실시간 추적 목표 전송 (Central → Robot)
bool RobotNavigationManager::sendTrackingGoal(double x, double y, double z) {
    try {
        auto message = geometry_msgs::msg::Point();
        message.x = x;
        message.y = y;
        message.z = z;
        
        tracking_goal_pub_->publish(message);
        logNavigationCommand("tracking_goal: (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")");
        
        RCLCPP_INFO(this->get_logger(), "추적 목표 전송: (%.2f, %.2f, %.2f)", x, y, z);
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "추적 목표 전송 실패: %s", e.what());
        return false;
    }
}

// Teleop 명령 (새로운 인터페이스)
bool RobotNavigationManager::sendTeleopCommand(const std::string& teleop_key) {
    try {
        auto message = std_msgs::msg::String();
        message.data = teleop_key;
        
        teleop_publisher_->publish(message);
        logNavigationCommand("teleop: " + teleop_key);
        
        RCLCPP_INFO(this->get_logger(), "원격 제어 명령 전송: %s", teleop_key.c_str());
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "원격 제어 명령 전송 실패: %s", e.what());
        return false;
    }
}

// 콜백 설정 함수들
void RobotNavigationManager::setNavStatusCallback(std::function<void(const std::string&)> callback) {
    nav_status_callback_ = callback;
}

void RobotNavigationManager::setRobotPoseCallback(std::function<void(double x, double y, double yaw)> callback) {
    robot_pose_callback_ = callback;
}

void RobotNavigationManager::setStartPointCallback(std::function<void(const std::string&)> callback) {
    start_point_callback_ = callback;
}

void RobotNavigationManager::setTargetCallback(std::function<void(const std::string&)> callback) {
    target_callback_ = callback;
}

void RobotNavigationManager::setNetworkLevelCallback(std::function<void(int)> callback) {
    network_level_callback_ = callback;
}

void RobotNavigationManager::setBatteryCallback(std::function<void(int)> callback) {
    battery_callback_ = callback;
}

void RobotNavigationManager::setObstacleCallback(std::function<void(double x, double y, double yaw)> callback) {
    obstacle_callback_ = callback;
}

void RobotNavigationManager::setRobotEventCallback(std::function<void(const std::string&)> callback) {
    robot_event_callback_ = callback;
}

// 서비스 클라이언트 설정

void RobotNavigationManager::setControlEventClient(std::shared_ptr<rclcpp::Client<control_interfaces::srv::EventHandle>> client) {
    control_event_client_ = client;
}

void RobotNavigationManager::setNavigateClient(std::shared_ptr<rclcpp::Client<control_interfaces::srv::NavigateHandle>> client) {
    navigate_client_ = client;
}

void RobotNavigationManager::setTrackingEventClient(std::shared_ptr<rclcpp::Client<control_interfaces::srv::TrackHandle>> client) {
    tracking_event_client_ = client;
}

// 서비스 통신 함수들

bool RobotNavigationManager::sendControlEvent(const std::string& event_type) {
    if (!control_event_client_) {
        RCLCPP_ERROR(this->get_logger(), "Control Event 클라이언트가 설정되지 않았습니다");
        return false;
    }
    
    // 서비스가 사용 가능한지 확인 (1초 대기)
    while (!control_event_client_->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "ROS 2 시스템이 중단되었습니다. Control Event 전송을 중단합니다.");
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Control Event 서비스가 사용 불가능합니다. 다시 시도합니다...");
    }
    
    try {
        auto request = std::make_shared<control_interfaces::srv::EventHandle::Request>();
        request->event_type = event_type;
        
        auto future = control_event_client_->async_send_request(request);
        
        // 공식 문서 방식으로 응답 대기
        if (rclcpp::spin_until_future_complete(this->shared_from_this(), future) == 
            rclcpp::FutureReturnCode::SUCCESS) {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "Control Event 전송 성공: %s, 응답: %s", 
                       event_type.c_str(), response->status.c_str());
            return true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Control Event 전송 실패: %s", event_type.c_str());
            return false;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Control Event 전송 중 예외 발생: %s, 오류: %s", 
                    event_type.c_str(), e.what());
        return false;
    }
}

bool RobotNavigationManager::sendNavigateEvent(const std::string& event_type, const std::string& command) {
    if (!navigate_client_) {
        RCLCPP_ERROR(this->get_logger(), "Navigate 클라이언트가 설정되지 않았습니다");
        return false;
    }
    
    // 서비스가 사용 가능한지 확인 (1초 대기)
    while (!navigate_client_->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "ROS 2 시스템이 중단되었습니다. Navigate Event 전송을 중단합니다.");
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Navigate Event 서비스가 사용 불가능합니다. 다시 시도합니다...");
    }
    
    try {
        auto request = std::make_shared<control_interfaces::srv::NavigateHandle::Request>();
        request->event_type = event_type;
        request->command = command;
        
        auto future = navigate_client_->async_send_request(request);
        
        // 공식 문서 방식으로 응답 대기
        if (rclcpp::spin_until_future_complete(this->shared_from_this(), future) == 
            rclcpp::FutureReturnCode::SUCCESS) {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "Navigate Event 전송 성공: %s, 명령: %s, 응답: %s", 
                       event_type.c_str(), command.c_str(), response->status.c_str());
            return true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Navigate Event 전송 실패: %s", event_type.c_str());
            return false;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Navigate Event 전송 중 예외 발생: %s, 오류: %s", 
                    event_type.c_str(), e.what());
        return false;
    }
}

bool RobotNavigationManager::sendTrackingEvent(const std::string& event_type, double left_angle, double right_angle) {
    if (!tracking_event_client_) {
        RCLCPP_ERROR(this->get_logger(), "Tracking Event 클라이언트가 설정되지 않았습니다");
        return false;
    }
    
    // 서비스가 사용 가능한지 확인 (1초 대기)
    while (!tracking_event_client_->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "ROS 2 시스템이 중단되었습니다. Tracking Event 전송을 중단합니다.");
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Tracking Event 서비스가 사용 불가능합니다. 다시 시도합니다...");
    }
    
    try {
        auto request = std::make_shared<control_interfaces::srv::TrackHandle::Request>();
        request->event_type = event_type;
        request->left_angle = left_angle;
        request->right_angle = right_angle;
        
        auto future = tracking_event_client_->async_send_request(request);
        
        // 공식 문서 방식으로 응답 대기
        if (rclcpp::spin_until_future_complete(this->shared_from_this(), future) == 
            rclcpp::FutureReturnCode::SUCCESS) {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "Tracking Event 전송 성공: %s, 응답: %s, 거리: %.2f", 
                       event_type.c_str(), response->status.c_str(), response->distance);
            return true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Tracking Event 전송 실패: %s", event_type.c_str());
            return false;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Tracking Event 전송 중 예외 발생: %s, 오류: %s", 
                    event_type.c_str(), e.what());
        return false;
    }
}

// 현재 상태 조회 함수들
std::string RobotNavigationManager::getCurrentNavStatus() {
    return current_nav_status_;
}

std::string RobotNavigationManager::getCurrentStartPoint() {
    return current_start_point_;
}

std::string RobotNavigationManager::getCurrentTarget() {
    return current_target_;
}

int RobotNavigationManager::getCurrentNetworkLevel() {
    return current_network_level_;
}

int RobotNavigationManager::getCurrentBattery() {
    return current_battery_;
}



// 콜백 함수들
void RobotNavigationManager::poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    double x = msg->position.x;
    double y = msg->position.y;
    double yaw = quaternionToYaw(msg->orientation);
    
    // 현재 로봇 위치 저장
    current_robot_x_ = x;
    current_robot_y_ = y;
    current_robot_yaw_ = yaw;
    
    if (robot_pose_callback_) {
        robot_pose_callback_(x, y, yaw);
    }
}

void RobotNavigationManager::startPointCallback(const std_msgs::msg::String::SharedPtr msg) {
    current_start_point_ = msg->data;
    
    if (start_point_callback_) {
        start_point_callback_(msg->data);
    }
}

void RobotNavigationManager::targetCallback(const std_msgs::msg::String::SharedPtr msg) {
    current_target_ = msg->data;
    
    if (target_callback_) {
        target_callback_(msg->data);
    }
}

void RobotNavigationManager::networkLevelCallback(const std_msgs::msg::Int32::SharedPtr msg) {
    current_network_level_ = msg->data;
    
    if (network_level_callback_) {
        network_level_callback_(msg->data);
    }
}

void RobotNavigationManager::batteryCallback(const std_msgs::msg::Int32::SharedPtr msg) {
    current_battery_ = msg->data;
    
    if (battery_callback_) {
        battery_callback_(msg->data);
    }
}

void RobotNavigationManager::navStatusCallback(const std_msgs::msg::String::SharedPtr msg) {
    current_nav_status_ = msg->data;
    
    if (nav_status_callback_) {
        nav_status_callback_(msg->data);
    }
}

void RobotNavigationManager::obstacleCallback(const geometry_msgs::msg::Point::SharedPtr msg) {
    double x = msg->x;
    double y = msg->y;
    double yaw = msg->z; // z 필드를 yaw로 사용
    
    if (obstacle_callback_) {
        obstacle_callback_(x, y, yaw);
    }
}

// 서비스 콜백 함수들
void RobotNavigationManager::robotEventCallback(
    const std::shared_ptr<control_interfaces::srv::EventHandle::Request> request,
    std::shared_ptr<control_interfaces::srv::EventHandle::Response> response) {
    
    RCLCPP_INFO(this->get_logger(), "로봇 이벤트 수신: %s", request->event_type.c_str());
    
    // 이벤트 타입에 따른 처리
    if (request->event_type == "arrived_to_call") {
        response->status = "success";
        RCLCPP_INFO(this->get_logger(), "호출 위치 도착 이벤트 처리");
    } else if (request->event_type == "navigating_complete") {
        response->status = "success";
        RCLCPP_INFO(this->get_logger(), "로비 도착 이벤트 처리");
    } else if (request->event_type == "arrived_to_station") {
        response->status = "success";
        RCLCPP_INFO(this->get_logger(), "대기장소 도착 이벤트 처리");
    } else if (request->event_type == "return_command") {
        response->status = "success";
        RCLCPP_INFO(this->get_logger(), "반환 요청 이벤트 처리");
    } else {
        response->status = "unknown_event";
        RCLCPP_WARN(this->get_logger(), "알 수 없는 이벤트 타입: %s", request->event_type.c_str());
    }
    
    // 콜백 함수 호출
    if (robot_event_callback_) {
        robot_event_callback_(request->event_type);
    }
}

// 유틸리티 함수들
double RobotNavigationManager::quaternionToYaw(const geometry_msgs::msg::Quaternion& quat) {
    // Quaternion을 Euler angles로 변환 (yaw만 추출)
    double siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y);
    double cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z);
    return std::atan2(siny_cosp, cosy_cosp);
}

void RobotNavigationManager::logNavigationCommand(const std::string& command) {
    std::cout << "[RobotNavigation] 명령 전송: " << command << std::endl;
}

// 장애물 감지 서비스 콜백
void RobotNavigationManager::detectObstacleCallback(
    const std::shared_ptr<control_interfaces::srv::DetectHandle::Request> request,
    std::shared_ptr<control_interfaces::srv::DetectHandle::Response> response) {
    
    RCLCPP_INFO(this->get_logger(), "장애물 감지 서비스 수신: left_angle=%.2f, right_angle=%.2f", 
               request->left_angle, request->right_angle);
    
    // AI 서버에 장애물 감지 정보 전송
    int robot_id = 3; // TODO: config에서 가져오기
    bool sent = sendObstacleDetectedToAI(robot_id, request->left_angle, request->right_angle);
    
    if (sent) {
        response->flag = "success";
        RCLCPP_INFO(this->get_logger(), "AI 서버에 장애물 감지 정보 전송 성공");
    } else {
        response->flag = "failed";
        RCLCPP_ERROR(this->get_logger(), "AI 서버에 장애물 감지 정보 전송 실패");
    }
}

// AI 서버 HTTP 통신
bool RobotNavigationManager::sendObstacleDetectedToAI(int robot_id, float left_angle, float right_angle) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        RCLCPP_ERROR(this->get_logger(), "CURL 초기화 실패");
        return false;
    }
    
    // AI 서버 URL
    std::string ai_server_url = "http://192.168.0.27:8000/obstacle/detected";
    
    // JSON 요청 데이터
    Json::Value request_data;
    request_data["robot_id"] = robot_id;
    request_data["left_angle"] = std::to_string(left_angle);
    request_data["rignt_angle"] = std::to_string(right_angle);
    request_data["timestamp"] = std::to_string(time(nullptr));
    
    Json::StreamWriterBuilder builder;
    std::string json_data = Json::writeString(builder, request_data);
    
    // HTTP 헤더 설정
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    // CURL 옵션 설정
    curl_easy_setopt(curl, CURLOPT_URL, ai_server_url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);  // 5초 타임아웃
    
    // 응답 처리
    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](void* contents, size_t size, size_t nmemb, std::string* userp) -> size_t {
        userp->append((char*)contents, size * nmemb);
        return size * nmemb;
    });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    // 요청 실행
    CURLcode res = curl_easy_perform(curl);
    
    // 정리
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        RCLCPP_ERROR(this->get_logger(), "AI 서버 obstacle/detected 요청 실패: %s", curl_easy_strerror(res));
        return false;
    }
    
    // 응답 확인
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    
    if (http_code == 200) {
        RCLCPP_INFO(this->get_logger(), "AI 서버 obstacle/detected 요청 성공: Robot %d", robot_id);
        return true;
    } else {
        RCLCPP_ERROR(this->get_logger(), "AI 서버 obstacle/detected 요청 실패: HTTP %ld", http_code);
        return false;
    }
} 
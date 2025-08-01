#include "central_server/robot_navigation_manager.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

RobotNavigationManager::RobotNavigationManager() 
    : Node("robot_navigation_manager") {
    
    RCLCPP_INFO(this->get_logger(), "RobotNavigationManager 초기화 시작");
    
    // 현재 Domain ID를 환경 변수에서 읽기 (기본값 0)
    const char* domain_id_env = std::getenv("ROS_DOMAIN_ID");
    int current_domain_id = domain_id_env ? std::atoi(domain_id_env) : 0;
    
    RCLCPP_INFO(this->get_logger(), "현재 Domain ID: %d", current_domain_id);
    
    // 퍼블리셔 초기화
    navigation_command_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/navigation_command", 10);
    
    teleop_command_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10);
    
    // 서브스크라이버 초기화
    nav_status_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/fleet/robot1/nav_status", 10,
        std::bind(&RobotNavigationManager::navStatusCallback, this, std::placeholders::_1));
    
    amcl_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
        "/fleet/robot1/pose", 10,
        std::bind(&RobotNavigationManager::amclPoseCallback, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "RobotNavigationManager 초기화 완료");
    RCLCPP_INFO(this->get_logger(), "토픽 설정:");
    RCLCPP_INFO(this->get_logger(), "  - 발행: /navigation_command");
    RCLCPP_INFO(this->get_logger(), "  - 발행: /cmd_vel");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /fleet/robot1/nav_status");
    RCLCPP_INFO(this->get_logger(), "  - 구독: /fleet/robot1/pose");
}

RobotNavigationManager::~RobotNavigationManager() {
    RCLCPP_INFO(this->get_logger(), "RobotNavigationManager 소멸자 호출");
}

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

bool RobotNavigationManager::sendTeleopCommand(const std::string& teleop_key) {
    try {
        auto message = geometry_msgs::msg::Twist();
        
        // teleop_twist_keyboard의 moveBindings에 따라 Twist 메시지 설정
        if (teleop_key == "1") {
            // u: 앞으로 가면서 좌회전 (1,0,0,1)
            message.linear.x = 1.0;
            message.linear.y = 0.0;
            message.angular.z = 1.0;
        } else if (teleop_key == "2") {
            // i: 전진 (1,0,0,0)
            message.linear.x = 1.0;
            message.linear.y = 0.0;
            message.angular.z = 0.0;
        } else if (teleop_key == "3") {
            // o: 앞으로 가면서 우회전 (1,0,0,-1)
            message.linear.x = 1.0;
            message.linear.y = 0.0;
            message.angular.z = -1.0;
        } else if (teleop_key == "4") {
            // j: 제자리에서 왼쪽으로 회전 (0,0,0,1)
            message.linear.x = 0.0;
            message.linear.y = 0.0;
            message.angular.z = 1.0;
        } else if (teleop_key == "5") {
            // k: 정지 (기본값)
            message.linear.x = 0.0;
            message.linear.y = 0.0;
            message.angular.z = 0.0;
        } else if (teleop_key == "6") {
            // l: 제자리에서 오른쪽으로 회전 (0,0,0,-1)
            message.linear.x = 0.0;
            message.linear.y = 0.0;
            message.angular.z = -1.0;
        } else if (teleop_key == "7") {
            // m: 뒤로 가면서 좌회전 (-1,0,0,-1)
            message.linear.x = -1.0;
            message.linear.y = 0.0;
            message.angular.z = -1.0;
        } else if (teleop_key == "8") {
            // ,: 후진 (-1,0,0,0)
            message.linear.x = -1.0;
            message.linear.y = 0.0;
            message.angular.z = 0.0;
        } else if (teleop_key == "9") {
            // .: 뒤로 가면서 우회전 (-1,0,0,1)
            message.linear.x = -1.0;
            message.linear.y = 0.0;
            message.angular.z = 1.0;
        } else {
            // 정지
            message.linear.x = 0.0;
            message.linear.y = 0.0;
            message.angular.z = 0.0;
        }
        
        teleop_command_pub_->publish(message);
        logNavigationCommand("teleop: " + teleop_key);
        
        RCLCPP_INFO(this->get_logger(), "원격 제어 명령 전송: %s", teleop_key.c_str());
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "원격 제어 명령 전송 실패: %s", e.what());
        return false;
    }
}

void RobotNavigationManager::setNavStatusCallback(
    std::function<void(const std::string&)> callback) {
    nav_status_callback_ = callback;
}

void RobotNavigationManager::setRobotPoseCallback(
    std::function<void(double x, double y, double yaw)> callback) {
    robot_pose_callback_ = callback;
}



void RobotNavigationManager::navStatusCallback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "로봇 주행 상태 수신: %s", msg->data.c_str());
    
    // 현재 상태 저장
    {
        std::lock_guard<std::mutex> lock(nav_status_mutex_);
        current_nav_status_ = msg->data;
    }
    
    if (nav_status_callback_) {
        nav_status_callback_(msg->data);
    }
}

std::string RobotNavigationManager::getCurrentNavStatus() {
    std::lock_guard<std::mutex> lock(nav_status_mutex_);
    return current_nav_status_;
}

void RobotNavigationManager::amclPoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    double x = msg->position.x;
    double y = msg->position.y;
    double yaw = quaternionToYaw(msg->orientation);
    
    RCLCPP_INFO(this->get_logger(), 
               "로봇 현재 위치 수신 - 위치: (%.2f, %.2f), 방향: %.2f도", 
               x, y, yaw * 180.0 / M_PI);
    
    if (robot_pose_callback_) {
        robot_pose_callback_(x, y, yaw);
    }
}

double RobotNavigationManager::quaternionToYaw(const geometry_msgs::msg::Quaternion& quat) {
    // Quaternion을 Euler angles로 변환 (yaw만 추출)
    double siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y);
    double cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z);
    return std::atan2(siny_cosp, cosy_cosp);
}

void RobotNavigationManager::logNavigationCommand(const std::string& command) {
    std::cout << "[RobotNavigation] 명령 전송: " << command << std::endl;
} 
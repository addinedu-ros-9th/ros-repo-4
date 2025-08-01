#ifndef ROBOT_NAVIGATION_MANAGER_H
#define ROBOT_NAVIGATION_MANAGER_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <memory>
#include <string>
#include <functional>
#include <map>

class RobotNavigationManager : public rclcpp::Node {
public:
    RobotNavigationManager();
    ~RobotNavigationManager();

    // 로봇 네비게이션 명령 전송
    bool sendNavigationCommand(const std::string& command);
    
    // 로봇 네비게이션 명령 전송 (특정 웨이포인트)
    bool sendWaypointCommand(const std::string& waypoint_name);
    
    // 로봇 시작점으로 복귀 명령
    bool sendGoStartCommand();
    
    // 로봇 주행 정지 명령
    bool sendStopCommand();
    
    // 로봇 원격 제어 명령 (teleop)
    bool sendTeleopCommand(const std::string& teleop_key);
    
    // 로봇 상태 조회
    std::string getCurrentNavStatus();
    
    // 콜백 함수 설정
    void setNavStatusCallback(std::function<void(const std::string&)> callback);
    void setRobotPoseCallback(std::function<void(double x, double y, double yaw)> callback);

private:
    // ROS 2 퍼블리셔
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr navigation_command_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr teleop_command_pub_;
    
    // ROS 2 서브스크라이버
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr start_point_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr target_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr nav_status_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub_;
    
    // 로봇 상태 정보
    std::string current_nav_status_;
    std::mutex nav_status_mutex_;
    
    // 콜백 함수들
    std::function<void(const std::string&)> start_point_callback_;
    std::function<void(const std::string&)> target_callback_;
    std::function<void(const std::string&)> nav_status_callback_;
    std::function<void(double x, double y, double yaw)> robot_pose_callback_;
    
    // 토픽 콜백 함수들
    void startPointCallback(const std_msgs::msg::String::SharedPtr msg);
    void targetCallback(const std_msgs::msg::String::SharedPtr msg);
    void navStatusCallback(const std_msgs::msg::String::SharedPtr msg);
    void amclPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    
    // 유틸리티 함수
    double quaternionToYaw(const geometry_msgs::msg::Quaternion& quat);
    
    // 로깅 함수
    void logNavigationCommand(const std::string& command);
};

#endif // ROBOT_NAVIGATION_MANAGER_H 
#ifndef ROBOT_NAVIGATION_MANAGER_H
#define ROBOT_NAVIGATION_MANAGER_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <control_interfaces/srv/event_handle.hpp>
#include <control_interfaces/srv/track_handle.hpp>
#include <control_interfaces/srv/navigate_handle.hpp>
#include <std_msgs/msg/string.hpp>

#include <functional>
#include <string>
#include <mutex>
#include <memory>

class RobotNavigationManager : public rclcpp::Node {
public:
    RobotNavigationManager();
    ~RobotNavigationManager();
    
    // IF-01: 로봇 목적지 전송 (Central → Robot)
    bool sendNavigationCommand(const std::string& command);
    bool sendWaypointCommand(const std::string& waypoint_name);
    bool sendGoStartCommand();
    bool sendStopCommand();
    
    // IF-08: 실시간 추적 목표 전송 (Central → Robot)
    bool sendTrackingGoal(double x, double y, double z);
    
    // IF-09: 장애물 정보 연동 (Robot → Central) - 콜백으로 처리
    
    // Teleop 명령
    bool sendTeleopCommand(const std::string& teleop_key);
    
    // 콜백 설정
    void setNavStatusCallback(std::function<void(const std::string&)> callback);
    void setRobotPoseCallback(std::function<void(double x, double y, double yaw)> callback);
    void setStartPointCallback(std::function<void(const std::string&)> callback);
    void setTargetCallback(std::function<void(const std::string&)> callback);
    void setNetworkLevelCallback(std::function<void(int)> callback);
    void setBatteryCallback(std::function<void(int)> callback);
    void setObstacleCallback(std::function<void(double x, double y, double yaw)> callback);
    
    // 현재 상태 조회
    std::string getCurrentNavStatus();
    std::string getCurrentStartPoint();
    std::string getCurrentTarget();
    int getCurrentNetworkLevel();
    int getCurrentBattery();
    
    // 현재 로봇 위치 조회 (읽기 전용)
    double getCurrentRobotX() const { return current_robot_x_; }
    double getCurrentRobotY() const { return current_robot_y_; }
    double getCurrentRobotYaw() const { return current_robot_yaw_; }
    
    // 서비스 클라이언트 설정
    void setControlEventClient(std::shared_ptr<rclcpp::Client<control_interfaces::srv::EventHandle>> client);
    void setNavigateClient(std::shared_ptr<rclcpp::Client<control_interfaces::srv::NavigateHandle>> client);
    void setTrackingEventClient(std::shared_ptr<rclcpp::Client<control_interfaces::srv::TrackHandle>> client);
    
    // 서비스 통신 함수들
    bool sendControlEvent(const std::string& event_type);
    bool sendNavigateEvent(const std::string& event_type, const std::string& command);
    bool sendTrackingEvent(const std::string& event_type, double left_angle, double right_angle);
    
    // 로봇 이벤트 콜백 설정
    void setRobotEventCallback(std::function<void(const std::string&)> callback);

private:
    // IF-01: 로봇 목적지 전송 퍼블리셔
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr navigation_command_pub_;
    
    // IF-08: 실시간 추적 목표 전송 퍼블리셔
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr tracking_goal_pub_;
    
    // Teleop 명령 퍼블리셔 (새로운 인터페이스)
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr teleop_publisher_;
    
    // IF-02: 로봇의 현재 위치 서브스크라이버
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
    
    // IF-03: 로봇의 주행 시작점 서브스크라이버
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr start_point_sub_;
    
    // IF-04: 로봇의 주행 목적지 서브스크라이버
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr target_sub_;
    
    // IF-05: 로봇의 네트워크 상태 서브스크라이버
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr net_level_sub_;
    
    // IF-06: 로봇의 배터리 잔량 서브스크라이버
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr battery_sub_;
    
    // IF-07: 로봇 주행 상태 서브스크라이버
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr nav_status_sub_;
    
    // IF-09: 장애물 정보 서브스크라이버 (커스텀 메시지 대신 Point 사용)
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr obstacle_sub_;
    
    // 서비스 서버 (Robot → Central)
    rclcpp::Service<control_interfaces::srv::EventHandle>::SharedPtr robot_event_service_;
    
    // 서비스 클라이언트들 (Central → Robot)
    std::shared_ptr<rclcpp::Client<control_interfaces::srv::EventHandle>> control_event_client_;
    std::shared_ptr<rclcpp::Client<control_interfaces::srv::NavigateHandle>> navigate_client_;
    std::shared_ptr<rclcpp::Client<control_interfaces::srv::TrackHandle>> tracking_event_client_;
    
    // 콜백 함수들
    std::function<void(const std::string&)> nav_status_callback_;
    std::function<void(double x, double y, double yaw)> robot_pose_callback_;
    std::function<void(const std::string&)> start_point_callback_;
    std::function<void(const std::string&)> target_callback_;
    std::function<void(int)> network_level_callback_;
    std::function<void(int)> battery_callback_;
    std::function<void(double x, double y, double yaw)> obstacle_callback_;
    std::function<void(const std::string&)> robot_event_callback_;
    
    // 현재 상태 저장
    std::string current_nav_status_;
    std::string current_start_point_;
    std::string current_target_;
    int current_network_level_;
    int current_battery_;
    
    // 현재 로봇 위치 저장
    double current_robot_x_;
    double current_robot_y_;
    double current_robot_yaw_;
    

    
    // 콜백 함수들
    void poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg);
    void startPointCallback(const std_msgs::msg::String::SharedPtr msg);
    void targetCallback(const std_msgs::msg::String::SharedPtr msg);
    void networkLevelCallback(const std_msgs::msg::Int32::SharedPtr msg);
    void batteryCallback(const std_msgs::msg::Int32::SharedPtr msg);
    void navStatusCallback(const std_msgs::msg::String::SharedPtr msg);
    void obstacleCallback(const geometry_msgs::msg::Point::SharedPtr msg);
    
    // 서비스 콜백 함수들
    void robotEventCallback(
        const std::shared_ptr<control_interfaces::srv::EventHandle::Request> request,
        std::shared_ptr<control_interfaces::srv::EventHandle::Response> response);
    
    // 유틸리티 함수들
    double quaternionToYaw(const geometry_msgs::msg::Quaternion& quat);
    void logNavigationCommand(const std::string& command);
};

#endif // ROBOT_NAVIGATION_MANAGER_H 
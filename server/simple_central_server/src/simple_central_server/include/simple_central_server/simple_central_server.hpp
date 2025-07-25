#ifndef SIMPLE_CENTRAL_SERVER_HPP
#define SIMPLE_CENTRAL_SERVER_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <map>
#include <string>
#include <mutex>
#include <vector>

struct WaypointInfo {
    std::string name;
    double x;
    double y;
    double yaw;  // degrees
    std::string description;
};

struct RobotMonitorInfo {
    std::string robot_id;
    geometry_msgs::msg::Pose current_pose;
    geometry_msgs::msg::Twist current_velocity;
    std::string navigation_status;  // "idle", "navigating", "reached", "failed"
    std::string current_target;
    bool is_online;
    rclcpp::Time last_update;
};

class SimpleCentralServer : public rclcpp::Node
{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using GoalHandleNavigate = rclcpp_action::ClientGoalHandle<NavigateToPose>;

    SimpleCentralServer();

private:
    // 데이터 저장
    std::map<std::string, std::shared_ptr<RobotMonitorInfo>> robots_;
    std::map<std::string, WaypointInfo> waypoints_;
    std::mutex robots_mutex_;
    
    // Action clients
    std::map<std::string, rclcpp_action::Client<NavigateToPose>::SharedPtr> nav_clients_;
    
    // 구독자들
    std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr> amcl_subscribers_;
    std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr> cmd_vel_subscribers_;
    
    // 네비게이션 명령 구독자 (새로 추가)
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr nav_command_subscriber_;
    
    // 개별 토픽 퍼블리셔들
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr> pose_publishers_;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr> velocity_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr> nav_status_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr> target_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr> online_status_publishers_;
    
    // 명령 로그 퍼블리셔
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_log_publisher_;
    
    // 타이머 (자동 명령 타이머 제거)
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    // 초기화 함수들
    void initializeWaypoints();
    void initializeRobotSubscribers();
    void setupActionClients();
    void setupIndividualPublishers();
    void setupNavigationCommandSubscriber();
    
    // 콜백 함수들
    void amclCallback(const std::string& robot_id, 
                     const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void cmdVelCallback(const std::string& robot_id,
                       const geometry_msgs::msg::Twist::SharedPtr msg);
    void navigationCommandCallback(const std_msgs::msg::String::SharedPtr msg);
    
    void statusTimerCallback();
    
    // Navigation 관련 함수들
    bool sendNavigationGoal(const std::string& robot_id, const std::string& waypoint_name);
    geometry_msgs::msg::PoseStamped createPoseStamped(double x, double y, double yaw);
    tf2::Quaternion getQuaternionFromYaw(double yaw_degrees);
    
    // Action 콜백들
    void goalResponseCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr& goal_handle);
    void feedbackCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr,
                         const std::shared_ptr<const NavigateToPose::Feedback> feedback);
    void resultCallback(const std::string& robot_id, const GoalHandleNavigate::WrappedResult& result);
    
    // 유틸리티 함수들
    void publishIndividualRobotData();
    void publishCommandLog(const std::string& message);
    void publishAvailableWaypoints();
};

#endif
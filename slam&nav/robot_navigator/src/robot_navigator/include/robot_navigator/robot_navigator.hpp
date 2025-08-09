#ifndef ROBOT_NAVIGATOR_HPP
#define ROBOT_NAVIGATOR_HPP

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
#include "control_interfaces/srv/event_handle.hpp"
#include "control_interfaces/srv/track_handle.hpp"
#include "control_interfaces/srv/navigate_handle.hpp"
#include <map>
#include <string>
#include <mutex>
#include <vector>
#include <limits>
#include <algorithm>
#include <cctype>
#include <unordered_map>

struct WaypointInfo {
    std::string name;
    double x;
    double y;
    double yaw;  // degrees
    std::string description;
};

struct RobotInfo {
    geometry_msgs::msg::Pose current_pose;
    geometry_msgs::msg::Twist current_velocity;
    std::string navigation_status;  // "idle", "navigating", "reached", "failed"
    std::string current_target;
    std::string start_point_name;   // 시작점 waypoint 이름
    bool is_online;
    bool start_point_set;
    rclcpp::Time last_update;
    rclcpp::Time canceled_time; // navigation canceled 시각
    bool teleop_active;
};

class RobotNavigator : public rclcpp::Node
{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using GoalHandleNavigate = rclcpp_action::ClientGoalHandle<NavigateToPose>;

    RobotNavigator();

private:
    // 데이터 저장
    std::shared_ptr<RobotInfo> robot_info_;
    std::map<std::string, WaypointInfo> waypoints_;
    std::mutex robot_mutex_;
    
    // Action client
    rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
    
    // 구독자들
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr teleop_event_subscriber_;
    
    // 네비게이션 명령 구독자
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr nav_command_subscriber_;
    
    // 개별 토픽 퍼블리셔들
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr start_point_publisher_;  // 시작점 이름 퍼블리셔
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr nav_status_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr target_publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr online_status_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr teleop_command_publisher_;
    
    // 명령 로그 퍼블리셔
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_log_publisher_;

    //service server & client
    rclcpp::Service<control_interfaces::srv::EventHandle>::SharedPtr control_event_server_;
    rclcpp::Service<control_interfaces::srv::TrackHandle>::SharedPtr tracking_event_server_;
    rclcpp::Service<control_interfaces::srv::NavigateHandle>::SharedPtr navigate_event_server_;
    rclcpp::Client<control_interfaces::srv::EventHandle>::SharedPtr robot_event_client_;
    
    // 타이머
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    // 초기화 함수들
    void initializeWaypoints();
    void initializeRobotSubscribers();
    void setupActionClient();
    void setupPublishers();
    void setupNavigationCommandSubscriber();
    void setupServices();
    
    // 콜백 함수들
    void amclCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void teleopEventCallback(const std_msgs::msg::String::SharedPtr teleop_key);
    void navigationCommandCallback(const std_msgs::msg::String::SharedPtr msg);
    
    void statusTimerCallback();

    // Event service handle 함수
    void controlEventHandle(
        const std::shared_ptr<control_interfaces::srv::EventHandle::Request> control_req,
        std::shared_ptr<control_interfaces::srv::EventHandle::Response> control_res
    );
    void trackEventHandle(
    const std::shared_ptr<control_interfaces::srv::TrackHandle::Request> track_req,
    std::shared_ptr<control_interfaces::srv::TrackHandle::Response> track_res
    );

    void navigateEventHandle(
    const std::shared_ptr<control_interfaces::srv::NavigateHandle::Request> nav_req,
    std::shared_ptr<control_interfaces::srv::NavigateHandle::Response> nav_res
    );

    
    // Navigation 관련 함수들
    bool sendNavigationGoal(const std::string& waypoint_name);
    geometry_msgs::msg::PoseStamped createPoseStamped(double x, double y, double yaw);
    tf2::Quaternion getQuaternionFromYaw(double yaw_degrees);
    
    // 시작점 관련 함수들
    std::string findNearestWaypoint(double x, double y) const;
    void setStartPoint(const std::string& waypoint_name);
    bool sendRobotToStartPoint();
    bool sendRobotToLobby();
    
    // Action 콜백들
    void goalResponseCallback(const GoalHandleNavigate::SharedPtr& goal_handle);
    void feedbackCallback(const GoalHandleNavigate::SharedPtr,
                         const std::shared_ptr<const NavigateToPose::Feedback> feedback);
    void resultCallback(const GoalHandleNavigate::WrappedResult& result);
    
    // 유틸리티 함수들
    void publishRobotData();
    void publishCommandLog(const std::string& message);
    void publishAvailableWaypoints();

    void callEventService(const std::string& event_type);
};

#endif
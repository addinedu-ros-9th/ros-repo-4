#ifndef ROBOT_CENTRAL_SERVER_HPP
#define ROBOT_CENTRAL_SERVER_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav2_msgs/action/follow_waypoints.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <memory>

// 커스텀 서비스 메시지
#include "robot_central_interfaces/srv/navigate_robot.hpp"
#include "robot_central_interfaces/srv/get_robot_status.hpp"
#include "robot_central_interfaces/msg/robot_status.hpp"

struct FacilityInfo {
    std::string name;
    double x;
    double y;
    double yaw; // degrees
    std::string description;
};

struct RobotInfo {
    std::string robot_id;
    std::string status; // "idle", "navigating", "error", "offline"
    geometry_msgs::msg::Pose current_pose;
    std::string current_task;
    rclcpp::Time last_update;
    std::shared_ptr<rclcpp_action::Client<nav2_msgs::action::NavigateToPose>> nav_client;
    std::shared_ptr<rclcpp_action::Client<nav2_msgs::action::FollowWaypoints>> waypoint_client;
};

class RobotCentralServer : public rclcpp::Node
{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using FollowWaypoints = nav2_msgs::action::FollowWaypoints;
    using NavigateRobot = robot_central_interfaces::srv::NavigateRobot;
    using GetRobotStatus = robot_central_interfaces::srv::GetRobotStatus;
    using RobotStatus = robot_central_interfaces::msg::RobotStatus;

    RobotCentralServer();
    
private:
    // 시설 및 로봇 관리
    std::map<std::string, FacilityInfo> facilities_;
    std::map<std::string, std::shared_ptr<RobotInfo>> robots_;
    std::mutex robots_mutex_;
    
    // 서비스 서버들
    rclcpp::Service<NavigateRobot>::SharedPtr navigate_service_;
    rclcpp::Service<GetRobotStatus>::SharedPtr status_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr emergency_stop_service_;
    
    // 퍼블리셔들
    rclcpp::Publisher<RobotStatus>::SharedPtr robot_status_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr system_log_publisher_;
    
    // 타이머
    rclcpp::TimerBase::SharedPtr status_update_timer_;
    rclcpp::TimerBase::SharedPtr heartbeat_timer_;
    
    // 초기화 함수들
    void initializeFacilities();
    void initializeRobots();
    void setupServices();
    void setupPublishers();
    void setupTimers();
    
    // 서비스 콜백들
    void navigateRobotCallback(const std::shared_ptr<NavigateRobot::Request> request,
                              std::shared_ptr<NavigateRobot::Response> response);
    void getRobotStatusCallback(const std::shared_ptr<GetRobotStatus::Request> request,
                               std::shared_ptr<GetRobotStatus::Response> response);
    void emergencyStopCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                              std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    
    // 타이머 콜백들
    void statusUpdateCallback();
    void heartbeatCallback();
    
    // 로봇 관리 함수들
    bool registerRobot(const std::string& robot_id);
    bool unregisterRobot(const std::string& robot_id);
    void updateRobotPose(const std::string& robot_id, const geometry_msgs::msg::Pose& pose);
    void updateRobotStatus(const std::string& robot_id, const std::string& status);
    
    // 네비게이션 함수들
    bool sendNavigationGoal(const std::string& robot_id, const std::string& facility_name);
    bool sendWaypointGoal(const std::string& robot_id, const std::vector<std::string>& facility_names);
    geometry_msgs::msg::PoseStamped createPoseStamped(double x, double y, double yaw);
    tf2::Quaternion getQuaternionFromYaw(double yaw_degrees);
    
    // 액션 콜백들
    void navigationResultCallback(const std::string& robot_id,
                                 const rclcpp_action::ClientGoalHandle<NavigateToPose>::WrappedResult& result);
    void waypointsFeedbackCallback(const std::string& robot_id, 
                                const std::shared_ptr<const FollowWaypoints::Feedback> feedback);
    void waypointsResultCallback(const std::string& robot_id,
                               const rclcpp_action::ClientGoalHandle<FollowWaypoints>::WrappedResult& result);
    
    // 유틸리티 함수들
    void publishSystemLog(const std::string& message, const std::string& level = "INFO");
    void publishRobotStatus();
    bool isRobotOnline(const std::shared_ptr<RobotInfo>& robot);
    void checkRobotHealth();
};

#endif // ROBOT_CENTRAL_SERVER_HPP
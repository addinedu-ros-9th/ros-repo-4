#ifndef CENTRAL_SERVER_H
#define CENTRAL_SERVER_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <robot_interfaces/msg/robot_status.hpp>
#include <robot_interfaces/srv/change_robot_status.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <map>
#include <string>
#include <mutex>

#include "database_manager.h"
#include "http_server.h"

#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <mutex>

// simple_central_server에서 가져온 구조체
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

class CentralServer : public rclcpp::Node
{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using GoalHandleNavigate = rclcpp_action::ClientGoalHandle<NavigateToPose>;
    CentralServer();
    ~CentralServer();
    
    void init();
    void start();
    void stop();

private:
    // 기존 함수들
    void runDatabaseThread();
    void runHttpThread();
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void statusCallback(const robot_interfaces::msg::RobotStatus::SharedPtr msg);
    void changeStatusCallback(
        const std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Request> request,
        std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Response> response);
    
    // HTTP 기반 실시간 통신 함수들
    void setupHttpServer();
    void sendRobotLocationToGui(int robot_id, float location_x, float location_y);
    void sendRobotStatusToGui(int robot_id, const std::string& status, const std::string& source);
    void sendArrivalNotificationToGui(int robot_id);
    void broadcastToGuiClients(const std::string& message);
    
    // 기존 멤버 변수들
    std::atomic<bool> running_;
    
    std::unique_ptr<DatabaseManager> db_manager_;
    std::unique_ptr<HttpServer> http_server_;
    
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Subscriber image_subscriber_;
    rclcpp::Subscription<robot_interfaces::msg::RobotStatus>::SharedPtr status_subscriber_;
    rclcpp::Service<robot_interfaces::srv::ChangeRobotStatus>::SharedPtr status_service_;
    
    std::thread db_thread_;
    std::thread http_thread_;
    
    // HTTP 서버 설정
    int http_port_;
    std::string http_host_;

    // simple_central_server에서 가져온 멤버
    std::map<std::string, std::shared_ptr<RobotMonitorInfo>> robots_;
    std::map<std::string, WaypointInfo> waypoints_;
    std::mutex robots_mutex_;
    std::map<std::string, rclcpp_action::Client<NavigateToPose>::SharedPtr> nav_clients_;
    std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr> amcl_subscribers_;
    std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr> cmd_vel_subscribers_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr nav_command_subscriber_;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr> pose_publishers_;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr> velocity_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr> nav_status_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr> target_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr> online_status_publishers_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_log_publisher_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    // simple_central_server 함수 선언
    void initializeWaypoints();
    void initializeRobotSubscribers();
    void setupActionClients();
    void setupIndividualPublishers();
    void setupNavigationCommandSubscriber();
    void amclCallback(const std::string& robot_id, const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void cmdVelCallback(const std::string& robot_id, const geometry_msgs::msg::Twist::SharedPtr msg);
    void navigationCommandCallback(const std_msgs::msg::String::SharedPtr msg);
    void statusTimerCallback();
    bool sendNavigationGoal(const std::string& robot_id, const std::string& waypoint_name);
    geometry_msgs::msg::PoseStamped createPoseStamped(double x, double y, double yaw);
    tf2::Quaternion getQuaternionFromYaw(double yaw_degrees);
    void goalResponseCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr& goal_handle);
    void feedbackCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr, const std::shared_ptr<const NavigateToPose::Feedback> feedback);
    void resultCallback(const std::string& robot_id, const GoalHandleNavigate::WrappedResult& result);
    void publishIndividualRobotData();
    void publishCommandLog(const std::string& message);
    void publishAvailableWaypoints();
};

#endif // CENTRAL_SERVER_H
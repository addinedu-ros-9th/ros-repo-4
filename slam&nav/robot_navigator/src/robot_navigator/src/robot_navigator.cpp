#include "robot_navigator/robot_navigator.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;

RobotNavigator::RobotNavigator() : Node("robot_navigator")
{
    RCLCPP_INFO(this->get_logger(), "Starting Robot Navigator with Nearest Waypoint Start Point...");
    
    // 웨이포인트 초기화
    initializeWaypoints();
    
    // 명령 로그 퍼블리셔 초기화
    command_log_publisher_ = this->create_publisher<std_msgs::msg::String>("navigation_commands", 10);
    
    // 로봇별 구독자 및 Action 클라이언트 초기화
    initializeRobotSubscribers();
    setupActionClients();
    setupIndividualPublishers();
    setupNavigationCommandSubscriber();
    
    // 상태 업데이트 타이머
    status_timer_ = this->create_wall_timer(
        1s, std::bind(&RobotNavigator::statusTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "Robot Navigator with Nearest Waypoint Start Point initialized");
    publishCommandLog("Robot Navigator started - Start point will be set to nearest waypoint");
    
    // 사용 가능한 waypoint 목록 출력
    publishAvailableWaypoints();
}

void RobotNavigator::initializeWaypoints()
{
    // 미리 정의된 웨이포인트들
    waypoints_["lobby_station"] = {"Main Lobby", 9.53, -1.76, 90.0, "병원 로비 스테이션"};
    waypoints_["breast_cancer"] = {"Breast Cancer Center", 7.67, 1.12, 180.0, "유방암 센터"};
    waypoints_["brain_tumor"] = {"Brain Tumor Center", 6.1, 1.12, 180.0, "뇌종양 센터"};
    waypoints_["lung_cancer"] = {"Lung Cancer Center", 5.07, -2.17, 0.0, "폐암 센터"};
    waypoints_["stomach_cancer"] = {"Stomach Cancer Center", 3.65, -2.17, 0.0, "위암 센터"};
    waypoints_["colon_cancer"] = {"Colon Cancer Center", 0.79, -2.17, 0.0, "대장암 센터"};
    //waypoints_["gateway_a"] = {"Gateway A", 0.0, 4.03, 180.0, "통로 A"};
    //waypoints_["gateway_b"] = {"Gateway B", -5.58, 4.03, 0.0, "통로 B"};
    waypoints_["x_ray"] = {"X-ray", -6, 4.03, 180.0, "X-ray 검사실"};
    waypoints_["ct"] = {"CT", -5.58, -1.88, 90.0, "CT 검사실"};
    waypoints_["echography"] = {"Echography", -5.58, -1.88, 90.0, "초음파 검사실"};
    
    RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_.size());
    
    // 웨이포인트 목록 출력
    for (const auto& [name, waypoint] : waypoints_) {
        RCLCPP_INFO(this->get_logger(), "Waypoint '%s': (%.2f, %.2f, %.1f°) - %s", 
                   name.c_str(), waypoint.x, waypoint.y, waypoint.yaw, waypoint.description.c_str());
    }
}

void RobotNavigator::initializeRobotSubscribers()
{
    std::vector<std::string> robot_ids = {"robot1"};
    
    for (const auto& robot_id : robot_ids) {
        // 로봇 정보 초기화
        auto robot_info = std::make_shared<RobotMonitorInfo>();
        robot_info->robot_id = robot_id;
        robot_info->navigation_status = "idle";
        robot_info->current_target = "none";
        robot_info->start_point_name = "none";  // 시작점 이름 초기화
        robot_info->is_online = false;
        robot_info->start_point_set = false;
        robot_info->last_update = this->get_clock()->now();
        
        {
            std::lock_guard<std::mutex> lock(robots_mutex_);
            robots_[robot_id] = robot_info;
        }
        
        // AMCL Pose 구독자
        std::string amcl_topic = "/amcl_pose";
        amcl_subscribers_[robot_id] = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            amcl_topic, 10, 
            [this, robot_id](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                this->amclCallback(robot_id, msg);
            });
        
        // CMD_VEL 구독자
        std::string cmd_vel_topic = "/cmd_vel";
        cmd_vel_subscribers_[robot_id] = this->create_subscription<geometry_msgs::msg::Twist>(
            cmd_vel_topic, 10,
            [this, robot_id](const geometry_msgs::msg::Twist::SharedPtr msg) {
                this->cmdVelCallback(robot_id, msg);
            });
        
        RCLCPP_INFO(this->get_logger(), "Initialized subscribers for %s", robot_id.c_str());
    }
}

void RobotNavigator::setupActionClients()
{
    for (const auto& [robot_id, robot_info] : robots_) {
        try {
            std::string nav_topic = "/navigate_to_pose";
            nav_clients_[robot_id] = rclcpp_action::create_client<NavigateToPose>(this, nav_topic);
            RCLCPP_INFO(this->get_logger(), "Created navigation action client for %s", robot_id.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create action client for %s: %s", 
                        robot_id.c_str(), e.what());
        }
    }
}

void RobotNavigator::setupIndividualPublishers()
{
    for (const auto& [robot_id, robot_info] : robots_) {
        // 개별 토픽 퍼블리셔들 생성
        std::string pose_topic = "/fleet/" + robot_id + "/pose";
        std::string start_point_topic = "/fleet/" + robot_id + "/start_point";  // 시작점 이름 토픽
        std::string velocity_topic = "/fleet/" + robot_id + "/velocity";
        std::string nav_status_topic = "/fleet/" + robot_id + "/nav_status";
        std::string target_topic = "/fleet/" + robot_id + "/target";
        std::string online_topic = "/fleet/" + robot_id + "/online";
        
        pose_publishers_[robot_id] = this->create_publisher<geometry_msgs::msg::Pose>(pose_topic, 10);
        start_point_publishers_[robot_id] = this->create_publisher<std_msgs::msg::String>(start_point_topic, 10);  // String 타입
        velocity_publishers_[robot_id] = this->create_publisher<geometry_msgs::msg::Twist>(velocity_topic, 10);
        nav_status_publishers_[robot_id] = this->create_publisher<std_msgs::msg::String>(nav_status_topic, 10);
        target_publishers_[robot_id] = this->create_publisher<std_msgs::msg::String>(target_topic, 10);
        online_status_publishers_[robot_id] = this->create_publisher<std_msgs::msg::Bool>(online_topic, 10);
        
        RCLCPP_INFO(this->get_logger(), "Created individual publishers for %s:", robot_id.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", pose_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s (waypoint name)", start_point_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", velocity_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", nav_status_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", target_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", online_topic.c_str());
    }
}

void RobotNavigator::setupNavigationCommandSubscriber()
{
    nav_command_subscriber_ = this->create_subscription<std_msgs::msg::String>(
        "navigation_command", 10,
        std::bind(&RobotNavigator::navigationCommandCallback, this, _1));
    
    RCLCPP_INFO(this->get_logger(), "Navigation command subscriber created: /navigation_command");
    RCLCPP_INFO(this->get_logger(), "Available commands:");
    RCLCPP_INFO(this->get_logger(), "  - waypoint_name: Navigate to specific waypoint");
    RCLCPP_INFO(this->get_logger(), "  - go_start: Return to start point waypoint");
    RCLCPP_INFO(this->get_logger(), "  - stop/cancel: Cancel navigation");
    RCLCPP_INFO(this->get_logger(), "  - status: Check current status");
    RCLCPP_INFO(this->get_logger(), "  - list: Show available waypoints");
    RCLCPP_INFO(this->get_logger(), "Note: Start point is automatically updated upon reaching destinations");
}

std::string RobotNavigator::findNearestWaypoint(double x, double y) const
{
    double best_distance = std::numeric_limits<double>::max();
    std::string nearest_waypoint = "lobby_station";  // 기본값
    
    for (const auto& [name, waypoint] : waypoints_) {
        double dx = x - waypoint.x;
        double dy = y - waypoint.y;
        double distance = std::sqrt(dx * dx + dy * dy);
        
        if (distance < best_distance) {
            best_distance = distance;
            nearest_waypoint = name;
        }
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Nearest waypoint to (%.2f, %.2f) is '%s' (distance: %.2f)", 
                x, y, nearest_waypoint.c_str(), best_distance);
    
    return nearest_waypoint;
}

void RobotNavigator::setStartPoint(const std::string& robot_id, const std::string& waypoint_name)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) != robots_.end() && waypoints_.find(waypoint_name) != waypoints_.end()) {
        robots_[robot_id]->start_point_name = waypoint_name;
        robots_[robot_id]->start_point_set = true;
        
        const auto& wp = waypoints_[waypoint_name];
        RCLCPP_INFO(this->get_logger(), "Start point set for %s: '%s' (%.2f, %.2f, %.1f°)", 
                   robot_id.c_str(), waypoint_name.c_str(), wp.x, wp.y, wp.yaw);
        
        publishCommandLog("SET_START: " + robot_id + " -> " + waypoint_name + 
                         " (" + std::to_string(wp.x) + ", " + std::to_string(wp.y) + ")");
    }
}

bool RobotNavigator::sendRobotToStartPoint(const std::string& robot_id)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) == robots_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Robot %s not found", robot_id.c_str());
        return false;
    }
    
    if (!robots_[robot_id]->start_point_set) {
        RCLCPP_ERROR(this->get_logger(), "Start point not set for robot %s", robot_id.c_str());
        publishCommandLog("ERROR: Start point not set for " + robot_id);
        return false;
    }
    
    std::string start_waypoint = robots_[robot_id]->start_point_name;
    robots_mutex_.unlock();  // unlock before calling sendNavigationGoal
    
    RCLCPP_INFO(this->get_logger(), "Sending %s to start point waypoint: %s", 
               robot_id.c_str(), start_waypoint.c_str());
    
    return sendNavigationGoal(robot_id, start_waypoint);
}

bool RobotNavigator::sendRobotToLobby(const std::string& robot_id)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    robots_[robot_id]->navigation_status = "moving_to_station";
    
    if (robots_.find(robot_id) == robots_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Robot %s not found", robot_id.c_str());
        return false;
    }

    robots_mutex_.unlock();  // unlock before calling sendNavigationGoal
    
    RCLCPP_INFO(this->get_logger(), "Sending %s to lobby station", 
               robot_id.c_str());
    
    return sendNavigationGoal(robot_id, "lobby_station");
}

void RobotNavigator::amclCallback(const std::string& robot_id, 
                                      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) != robots_.end()) {
        robots_[robot_id]->current_pose = msg->pose.pose;
        robots_[robot_id]->is_online = true;
        robots_[robot_id]->last_update = this->get_clock()->now();
        
        // 첫 번째 위치 수신 시 가장 가까운 waypoint를 시작점으로 설정
        if (!robots_[robot_id]->start_point_set) {
            std::string nearest_wp = findNearestWaypoint(
                msg->pose.pose.position.x, 
                msg->pose.pose.position.y);
            
            robots_[robot_id]->start_point_name = nearest_wp;
            robots_[robot_id]->start_point_set = true;
            
            const auto& wp = waypoints_[nearest_wp];
            RCLCPP_INFO(this->get_logger(), "Auto-set start point for %s: '%s' (%.2f, %.2f) - nearest to current position (%.2f, %.2f)", 
                       robot_id.c_str(), nearest_wp.c_str(), wp.x, wp.y,
                       msg->pose.pose.position.x, msg->pose.pose.position.y);
            
            publishCommandLog("AUTO: Start point set for " + robot_id + " -> " + nearest_wp + 
                            " (nearest to current position)");
        }
    }
}

void RobotNavigator::cmdVelCallback(const std::string& robot_id,
                                        const geometry_msgs::msg::Twist::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) != robots_.end()) {
        robots_[robot_id]->current_velocity = *msg;
        robots_[robot_id]->last_update = this->get_clock()->now();
    }
}

void RobotNavigator::navigationCommandCallback(const std_msgs::msg::String::SharedPtr msg)
{
    std::string command = msg->data;
    RCLCPP_INFO(this->get_logger(), "Received navigation command: '%s'", command.c_str());
    
    // 명령 파싱
    std::string robot_id = "robot1";
    std::string waypoint_name = command;
    
    size_t colon_pos = command.find(':');
    if (colon_pos != std::string::npos) {
        robot_id = command.substr(0, colon_pos);
        waypoint_name = command.substr(colon_pos + 1);
    }
    
    // set_start 명령 제거 (자동으로 설정되므로)
    /*
    if (waypoint_name == "set_start") {
        publishCommandLog("INFO: Start point is now automatically updated upon reaching destinations");
        RCLCPP_INFO(this->get_logger(), "Start point is automatically updated when reaching destinations. Manual setting disabled.");
        return;
    }
    */
    
    if (waypoint_name == "go_start" || waypoint_name == "return_start") {
        if (sendRobotToStartPoint(robot_id)) {
            publishCommandLog("GO_START: " + robot_id + " returning to start point");
            RCLCPP_INFO(this->get_logger(), "Sending %s to start point", robot_id.c_str());
        } else {
            publishCommandLog("ERROR: Failed to send " + robot_id + " to start point");
            RCLCPP_ERROR(this->get_logger(), "Failed to send %s to start point", robot_id.c_str());
        }
        return;
    }

    if (waypoint_name == "return_lobby")
    {
        if (sendRobotToLobby(robot_id)) {
            publishCommandLog("RETURN_LOBBY: " + robot_id + " returning to lobby station");
            RCLCPP_INFO(this->get_logger(), "Sending %s to lobby station", robot_id.c_str());
            robots_[robot_id]->navigation_status = "moving_to_station";
        } else {
            publishCommandLog("ERROR: Failed to send " + robot_id + " lobby station");
            RCLCPP_ERROR(this->get_logger(), "Failed to send %s to lobby station", robot_id.c_str());
            robots_[robot_id]->navigation_status = "failed";
        }
        return;
    }
    
    // 기존 특수 명령들
    if (waypoint_name == "stop" || waypoint_name == "cancel") {
        std::lock_guard<std::mutex> lock(robots_mutex_);
        if (robots_.find(robot_id) != robots_.end()) {
            robots_[robot_id]->navigation_status = "idle";
            robots_[robot_id]->current_target = "canceled";
        }
        publishCommandLog("CANCEL: Navigation canceled for " + robot_id);
        return;
    }
    
    if (waypoint_name == "status") {
        std::lock_guard<std::mutex> lock(robots_mutex_);
        if (robots_.find(robot_id) != robots_.end()) {
            auto robot = robots_[robot_id];
            publishCommandLog("STATUS: " + robot_id + " - " + robot->navigation_status + 
                            " (target: " + robot->current_target + 
                            ", start_point: " + robot->start_point_name + 
                            ", auto_start: enabled)");
        }
        return;
    }
    
    if (waypoint_name == "list") {
        publishAvailableWaypoints();
        return;
    }
    
    // 실제 네비게이션 명령 처리
    if (waypoints_.find(waypoint_name) != waypoints_.end()) {
        if (sendNavigationGoal(robot_id, waypoint_name)) {
            publishCommandLog("COMMAND: " + robot_id + " -> " + waypoint_name);
            RCLCPP_INFO(this->get_logger(), "Manual navigation command executed: %s -> %s", 
                       robot_id.c_str(), waypoint_name.c_str());
        } else {
            publishCommandLog("ERROR: Failed to send command " + robot_id + " -> " + waypoint_name);
        }
    } else {
        publishCommandLog("ERROR: Unknown waypoint '" + waypoint_name + "'");
        publishAvailableWaypoints();
    }
}

bool RobotNavigator::sendNavigationGoal(const std::string& robot_id, const std::string& waypoint_name)
{
    if (waypoints_.find(waypoint_name) == waypoints_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Waypoint '%s' not found", waypoint_name.c_str());
        return false;
    }
    
    if (nav_clients_.find(robot_id) == nav_clients_.end()) {
        RCLCPP_ERROR(this->get_logger(), "No navigation client found for robot %s", robot_id.c_str());
        return false;
    }
    
    auto nav_client = nav_clients_[robot_id];
    
    if (!nav_client->wait_for_action_server(std::chrono::seconds(5))) {
        RCLCPP_ERROR(this->get_logger(), "Navigation action server not available for %s", robot_id.c_str());
        return false;
    }
    
    const WaypointInfo& waypoint = waypoints_[waypoint_name];
    
    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose = createPoseStamped(waypoint.x, waypoint.y, waypoint.yaw);
    
    {
        std::lock_guard<std::mutex> lock(robots_mutex_);
        if (robots_.find(robot_id) != robots_.end()) {
            robots_[robot_id]->navigation_status = "navigating";
            robots_[robot_id]->current_target = waypoint_name;
        }
    }
    
    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    
    send_goal_options.goal_response_callback =
        [this, robot_id](const GoalHandleNavigate::SharedPtr& goal_handle) {
            this->goalResponseCallback(robot_id, goal_handle);
        };
    
    send_goal_options.feedback_callback =
        [this, robot_id](const GoalHandleNavigate::SharedPtr goal_handle,
                        const std::shared_ptr<const NavigateToPose::Feedback> feedback) {
            this->feedbackCallback(robot_id, goal_handle, feedback);
        };
    
    send_goal_options.result_callback =
        [this, robot_id](const GoalHandleNavigate::WrappedResult& result) {
            this->resultCallback(robot_id, result);
        };
    
    nav_client->async_send_goal(goal_msg, send_goal_options);
    
    RCLCPP_INFO(this->get_logger(), "Sent navigation goal: %s -> %s (%.2f, %.2f, %.1f°)", 
               robot_id.c_str(), waypoint.name.c_str(), waypoint.x, waypoint.y, waypoint.yaw);
    
    return true;
}

geometry_msgs::msg::PoseStamped RobotNavigator::createPoseStamped(double x, double y, double yaw)
{
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.header.stamp = this->get_clock()->now();
    
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = 0.0;
    
    tf2::Quaternion q = getQuaternionFromYaw(yaw);
    pose.pose.orientation = tf2::toMsg(q);
    
    return pose;
}

tf2::Quaternion RobotNavigator::getQuaternionFromYaw(double yaw_degrees)
{
    tf2::Quaternion quaternion;
    double yaw_radians = yaw_degrees * M_PI / 180.0;
    quaternion.setRPY(0, 0, yaw_radians);
    return quaternion;
}

void RobotNavigator::goalResponseCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr& goal_handle)
{
    if (!goal_handle) {
        RCLCPP_ERROR(this->get_logger(), "Goal rejected for robot %s", robot_id.c_str());
        std::lock_guard<std::mutex> lock(robots_mutex_);
        if (robots_.find(robot_id) != robots_.end()) {
            robots_[robot_id]->navigation_status = "failed";
        }
        publishCommandLog("REJECTED: Goal rejected for " + robot_id);
    } else {
        RCLCPP_INFO(this->get_logger(), "Goal accepted for robot %s", robot_id.c_str());
        publishCommandLog("ACCEPTED: Goal accepted for " + robot_id);
    }
}

void RobotNavigator::feedbackCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr,
                                          const std::shared_ptr<const NavigateToPose::Feedback> feedback)
{
    RCLCPP_INFO(this->get_logger(), "Robot %s navigation feedback: distance remaining %.2fm", 
               robot_id.c_str(), feedback->distance_remaining);
}

void RobotNavigator::resultCallback(const std::string& robot_id, const GoalHandleNavigate::WrappedResult& result)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) == robots_.end()) return;
    
    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            robots_[robot_id]->navigation_status = "waiting_for_return";
            
            // 도착 시 자동으로 현재 목표를 새로운 시작점으로 설정
            if (robots_[robot_id]->current_target != "start_point") {  // 시작점 복귀가 아닌 경우만
                std::string old_start = robots_[robot_id]->start_point_name;
                robots_[robot_id]->start_point_name = robots_[robot_id]->current_target;
                robots_[robot_id]->start_point_set = true;
                robots_[robot_id]->canceled_time = this->get_clock()->now(); // CANCELED 시각 기록
                RCLCPP_INFO(this->get_logger(), "Robot %s successfully reached target: %s", 
                           robot_id.c_str(), robots_[robot_id]->current_target.c_str());
                RCLCPP_INFO(this->get_logger(), "Auto-updated start point for %s: %s -> %s", 
                           robot_id.c_str(), old_start.c_str(), robots_[robot_id]->start_point_name.c_str());
                
                publishCommandLog("SUCCESS: " + robot_id + " reached " + robots_[robot_id]->current_target);
                publishCommandLog("AUTO_START: Start point updated to " + robots_[robot_id]->start_point_name);              
            } else {
                RCLCPP_INFO(this->get_logger(), "Robot %s returned to start point: %s", 
                           robot_id.c_str(), robots_[robot_id]->start_point_name.c_str());
                publishCommandLog("SUCCESS: " + robot_id + " returned to start point " + robots_[robot_id]->start_point_name);
            }
            break;
            
        case rclcpp_action::ResultCode::ABORTED:
            robots_[robot_id]->navigation_status = "failed";
            RCLCPP_ERROR(this->get_logger(), "Robot %s navigation aborted", robot_id.c_str());
            publishCommandLog("FAILED: " + robot_id + " navigation aborted");
            break;
            
        case rclcpp_action::ResultCode::CANCELED:
            robots_[robot_id]->navigation_status = "idle";
            robots_[robot_id]->canceled_time = this->get_clock()->now(); // CANCELED 시각 기록
            RCLCPP_WARN(this->get_logger(), "Robot %s navigation canceled", robot_id.c_str());
            publishCommandLog("CANCELED: " + robot_id + " navigation canceled");
            break;
            
        default:
            robots_[robot_id]->navigation_status = "failed";
            RCLCPP_ERROR(this->get_logger(), "Robot %s navigation unknown result", robot_id.c_str());
            publishCommandLog("ERROR: " + robot_id + " navigation unknown result");
            break;
    }
}

void RobotNavigator::statusTimerCallback()
{
    // 10초 timeout 후 lobby_station 복귀 로직
    {
        std::lock_guard<std::mutex> lock(robots_mutex_);
        for (auto& [robot_id, robot] : robots_) {
            if (robot->navigation_status == "idle" &&
                robot->canceled_time.nanoseconds() != 0) {
                rclcpp::Duration elapsed = this->get_clock()->now() - robot->canceled_time;
                if (elapsed.seconds() >= 10.0) {
                    publishCommandLog("TIMEOUT: " + robot_id + " canceled -> returning to lobby_station");
                    sendRobotToLobby(robot_id);
                    robot->canceled_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
                }
            }
        }
    }
    publishIndividualRobotData();
}

void RobotNavigator::publishIndividualRobotData()
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    for (const auto& [robot_id, robot] : robots_) {
        auto time_diff = this->get_clock()->now() - robot->last_update;
        bool is_online = time_diff.seconds() < 10.0;
        
        // 위치 정보 발행
        if (pose_publishers_.find(robot_id) != pose_publishers_.end()) {
            pose_publishers_[robot_id]->publish(robot->current_pose);
        }
        
        // 시작점 이름 발행
        if (start_point_publishers_.find(robot_id) != start_point_publishers_.end() && robot->start_point_set) {
            std_msgs::msg::String start_point_msg;
            start_point_msg.data = robot->start_point_name;
            start_point_publishers_[robot_id]->publish(start_point_msg);
        }
        
        // 속도 정보 발행
        if (velocity_publishers_.find(robot_id) != velocity_publishers_.end()) {
            velocity_publishers_[robot_id]->publish(robot->current_velocity);
        }
        
        // 네비게이션 상태 발행
        if (nav_status_publishers_.find(robot_id) != nav_status_publishers_.end()) {
            std_msgs::msg::String nav_status_msg;
            nav_status_msg.data = robot->navigation_status;
            nav_status_publishers_[robot_id]->publish(nav_status_msg);
        }
        
        // 현재 목표 발행
        if (target_publishers_.find(robot_id) != target_publishers_.end()) {
            std_msgs::msg::String target_msg;
            target_msg.data = robot->current_target;
            target_publishers_[robot_id]->publish(target_msg);
        }
        
        // 온라인 상태 발행
        if (online_status_publishers_.find(robot_id) != online_status_publishers_.end()) {
            std_msgs::msg::Bool online_msg;
            online_msg.data = is_online;
            online_status_publishers_[robot_id]->publish(online_msg);
        }
    }
}

void RobotNavigator::publishCommandLog(const std::string& message)
{
    std_msgs::msg::String log_msg;
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] " << message;
    
    log_msg.data = ss.str();
    command_log_publisher_->publish(log_msg);
    
    RCLCPP_INFO(this->get_logger(), "%s", message.c_str());
}

void RobotNavigator::publishAvailableWaypoints()
{
    std::stringstream ss;
    ss << "Available commands: ";
    for (const auto& [name, waypoint] : waypoints_) {
        ss << name << " ";
    }
    ss << "go_start stop status list (auto start point update enabled)";
    publishCommandLog(ss.str());
}
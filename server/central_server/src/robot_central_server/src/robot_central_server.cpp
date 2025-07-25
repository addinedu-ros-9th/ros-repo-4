#include "robot_central_server/robot_central_server.hpp"
#include <chrono>

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;

RobotCentralServer::RobotCentralServer() : Node("robot_central_server")
{
    RCLCPP_INFO(this->get_logger(), "Starting Robot Central Server...");
    
    initializeFacilities();
    initializeRobots();
    setupServices();
    setupPublishers();
    setupTimers();
    
    RCLCPP_INFO(this->get_logger(), "Robot Central Server initialized successfully");
    publishSystemLog("Central Server started", "INFO");
}

void RobotCentralServer::initializeFacilities()
{
    // 시설 정보 초기화
    facilities_["lobby_station"] = {"Main Lobby", 9.53, -1.76, 90.0, "병원 로비 스테이션"};
    facilities_["breast cancer"] = {"Breast Cancer Center", 7.64, 1.63, 180.0, "유방암 센터"};
    facilities_["brain_tumor"] = {"Brain Tumor Center", 5.97, 1.46, 180.0, "뇌종양 센터"};
    facilities_["lung_cancer"] = {"Lung Cancer Center", 5.32, -2.27, 0.0, "폐암 센터"};
    facilities_["stomach_cancer"] = {"Stomach Cancer Center", 3.84, -2.3, 0.0, "위암 센터"};
    facilities_["colon_cancer"] = {"Colon Cancer Center", 0.93, -2.3, 0.0, "대장암 센터"};
    facilities_["gateway_a"] = {"Gateway A", 0.09, 4.0, 180.0, "통로 A"};
    facilities_["gateway_b"] = {"Gateway B", -2.6, 4.18, 0.0, "통로 B"};
    facilities_["x_ray"] = {"X-ray", -5.69, 4.34, 180.0, "X-ray 검사실"};
    facilities_["ct"] = {"CT", -5.79, -1.88, 90.0, "CT 검사실"};
    facilities_["echography"] = {"Echography", -4.9, -1.96, 90.0, "초음파 검사실"};
    
    RCLCPP_INFO(this->get_logger(), "Loaded %zu facilities", facilities_.size());
}

void RobotCentralServer::initializeRobots()
{
    // 기본 로봇들 등록 (실제 환경에서는 동적으로 등록됨)
    std::vector<std::string> default_robots = {"robot1"};
    
    for (const auto& robot_id : default_robots) {
        registerRobot(robot_id);
    }
    
    RCLCPP_INFO(this->get_logger(), "Initialized %zu robots", robots_.size());
}

void RobotCentralServer::setupServices()
{
    // 로봇 네비게이션 서비스
    navigate_service_ = this->create_service<NavigateRobot>(
        "navigate_robot",
        std::bind(&RobotCentralServer::navigateRobotCallback, this, _1, _2));
    
    // 로봇 상태 조회 서비스
    status_service_ = this->create_service<GetRobotStatus>(
        "get_robot_status",
        std::bind(&RobotCentralServer::getRobotStatusCallback, this, _1, _2));
    
    // 긴급 정지 서비스
    emergency_stop_service_ = this->create_service<std_srvs::srv::Trigger>(
        "emergency_stop_all",
        std::bind(&RobotCentralServer::emergencyStopCallback, this, _1, _2));
    
    RCLCPP_INFO(this->get_logger(), "Services initialized");
}

void RobotCentralServer::setupPublishers()
{
    // 로봇 상태 퍼블리셔
    robot_status_publisher_ = this->create_publisher<RobotStatus>("robot_status_updates", 10);
    
    // 시스템 로그 퍼블리셔
    system_log_publisher_ = this->create_publisher<std_msgs::msg::String>("system_logs", 50);
    
    RCLCPP_INFO(this->get_logger(), "Publishers initialized");
}

void RobotCentralServer::setupTimers()
{
    // 상태 업데이트 타이머 (1초마다)
    status_update_timer_ = this->create_wall_timer(
        1s, std::bind(&RobotCentralServer::statusUpdateCallback, this));
    
    // 하트비트 타이머 (5초마다)
    heartbeat_timer_ = this->create_wall_timer(
        5s, std::bind(&RobotCentralServer::heartbeatCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "Timers initialized");
}

bool RobotCentralServer::registerRobot(const std::string& robot_id)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) != robots_.end()) {
        RCLCPP_WARN(this->get_logger(), "Robot %s already registered", robot_id.c_str());
        return false;
    }
    
    auto robot = std::make_shared<RobotInfo>();
    robot->robot_id = robot_id;
    robot->status = "idle";
    robot->current_task = "none";
    robot->last_update = this->get_clock()->now();
    
    // Action clients 생성
    std::string nav_topic = "/" + robot_id + "/navigate_to_pose";
    std::string waypoint_topic = "/" + robot_id + "/follow_waypoints";
    
    robot->nav_client = rclcpp_action::create_client<NavigateToPose>(this, nav_topic);
    robot->waypoint_client = rclcpp_action::create_client<FollowWaypoints>(this, waypoint_topic);
    
    robots_[robot_id] = robot;
    
    RCLCPP_INFO(this->get_logger(), "Robot %s registered successfully", robot_id.c_str());
    publishSystemLog("Robot " + robot_id + " registered", "INFO");
    
    return true;
}

void RobotCentralServer::navigateRobotCallback(
    const std::shared_ptr<NavigateRobot::Request> request,
    std::shared_ptr<NavigateRobot::Response> response)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    // 로봇 존재 확인
    if (robots_.find(request->robot_id) == robots_.end()) {
        response->success = false;
        response->message = "Robot " + request->robot_id + " not found";
        RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
        return;
    }
    
    auto robot = robots_[request->robot_id];
    
    // 로봇 상태 확인
    if (robot->status == "navigating") {
        response->success = false;
        response->message = "Robot " + request->robot_id + " is already navigating";
        return;
    }
    
    // 단일 목적지인지 다중 목적지인지 확인
    if (request->waypoints.size() == 1) {
        // 단일 목적지 네비게이션
        if (sendNavigationGoal(request->robot_id, request->waypoints[0])) {
            response->success = true;
            response->message = "Navigation started to " + request->waypoints[0];
            robot->status = "navigating";
            robot->current_task = "navigate_to_" + request->waypoints[0];
        } else {
            response->success = false;
            response->message = "Failed to start navigation";
        }
    } else {
        // 다중 목적지 네비게이션
        if (sendWaypointGoal(request->robot_id, request->waypoints)) {
            response->success = true;
            response->message = "Waypoint navigation started";
            robot->status = "navigating";
            robot->current_task = "follow_waypoints";
        } else {
            response->success = false;
            response->message = "Failed to start waypoint navigation";
        }
    }
    
    publishSystemLog("Navigation request for " + request->robot_id + ": " + response->message);
}

void RobotCentralServer::getRobotStatusCallback(
    const std::shared_ptr<GetRobotStatus::Request> request,
    std::shared_ptr<GetRobotStatus::Response> response)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (request->robot_id.empty()) {
        // 모든 로봇 상태 반환
        for (const auto& [robot_id, robot] : robots_) {
            RobotStatus status_msg;
            status_msg.robot_id = robot_id;
            status_msg.status = robot->status;
            status_msg.current_task = robot->current_task;
            status_msg.current_pose = robot->current_pose;
            status_msg.last_update = robot->last_update;
            
            response->robot_statuses.push_back(status_msg);
        }
        response->success = true;
        response->message = "Retrieved status for all robots";
    } else {
        // 특정 로봇 상태 반환
        if (robots_.find(request->robot_id) != robots_.end()) {
            auto robot = robots_[request->robot_id];
            
            RobotStatus status_msg;
            status_msg.robot_id = robot->robot_id;
            status_msg.status = robot->status;
            status_msg.current_task = robot->current_task;
            status_msg.current_pose = robot->current_pose;
            status_msg.last_update = robot->last_update;
            
            response->robot_statuses.push_back(status_msg);
            response->success = true;
            response->message = "Retrieved status for " + request->robot_id;
        } else {
            response->success = false;
            response->message = "Robot " + request->robot_id + " not found";
        }
    }
}

void RobotCentralServer::emergencyStopCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    (void)request; // 미사용 변수 경고 제거
    
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    int stopped_robots = 0;
    
    for (auto& [robot_id, robot] : robots_) {
        if (robot->status == "navigating") {
            // 네비게이션 취소 (실제 구현에서는 action cancel 호출)
            robot->status = "idle";
            robot->current_task = "emergency_stopped";
            stopped_robots++;
            
            RCLCPP_WARN(this->get_logger(), "Emergency stop issued for robot %s", robot_id.c_str());
        }
    }
    
    response->success = true;
    response->message = "Emergency stop issued for " + std::to_string(stopped_robots) + " robots";
    
    publishSystemLog("EMERGENCY STOP - " + std::to_string(stopped_robots) + " robots stopped", "WARN");
}

bool RobotCentralServer::sendNavigationGoal(const std::string& robot_id, const std::string& facility_name)
{
    if (facilities_.find(facility_name) == facilities_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Facility %s not found", facility_name.c_str());
        return false;
    }
    
    auto robot = robots_[robot_id];
    
    if (!robot->nav_client->wait_for_action_server(5s)) {
        RCLCPP_ERROR(this->get_logger(), "Navigation action server not available for %s", robot_id.c_str());
        return false;
    }
    
    const FacilityInfo& facility = facilities_[facility_name];
    
    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose = createPoseStamped(facility.x, facility.y, facility.yaw);
    
    // Result callback with robot_id binding
    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_goal_options.result_callback = 
        [this, robot_id](const rclcpp_action::ClientGoalHandle<NavigateToPose>::WrappedResult& result) {
            this->navigationResultCallback(robot_id, result);
        };
    
    robot->nav_client->async_send_goal(goal_msg, send_goal_options);
    
    RCLCPP_INFO(this->get_logger(), "Sent navigation goal to %s for facility %s", 
               robot_id.c_str(), facility_name.c_str());
    
    return true;
}

bool RobotCentralServer::sendWaypointGoal(const std::string& robot_id, const std::vector<std::string>& facility_names)
{
    auto robot = robots_[robot_id];
    
    if (!robot->waypoint_client->wait_for_action_server(5s)) {
        RCLCPP_ERROR(this->get_logger(), "Waypoint action server not available for %s", robot_id.c_str());
        return false;
    }
    
    std::vector<geometry_msgs::msg::PoseStamped> waypoints;
    
    // 각 시설의 좌표를 waypoint로 변환
    for (const std::string& facility_name : facility_names) {
        if (facilities_.find(facility_name) == facilities_.end()) {
            RCLCPP_ERROR(this->get_logger(), "Facility '%s' not found for robot %s", 
                        facility_name.c_str(), robot_id.c_str());
            continue;
        }
        
        const FacilityInfo& facility = facilities_[facility_name];
        waypoints.push_back(createPoseStamped(facility.x, facility.y, facility.yaw));
        
        RCLCPP_INFO(this->get_logger(), "Added waypoint for %s: %s (%.2f, %.2f, %.1f°)", 
                   robot_id.c_str(), facility.name.c_str(), facility.x, facility.y, facility.yaw);
    }
    
    if (waypoints.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No valid waypoints found for robot %s", robot_id.c_str());
        return false;
    }
    
    // Waypoints goal 생성
    auto goal_msg = FollowWaypoints::Goal();
    goal_msg.poses = waypoints;
    
    // Result callback with robot_id binding
    auto send_goal_options = rclcpp_action::Client<FollowWaypoints>::SendGoalOptions();
    send_goal_options.feedback_callback = 
        [this, robot_id](const rclcpp_action::ClientGoalHandle<FollowWaypoints>::SharedPtr,
                        const std::shared_ptr<const FollowWaypoints::Feedback> feedback) {
            this->waypointsFeedbackCallback(robot_id, feedback);
        };
    send_goal_options.result_callback = 
        [this, robot_id](const rclcpp_action::ClientGoalHandle<FollowWaypoints>::WrappedResult& result) {
            this->waypointsResultCallback(robot_id, result);
        };
    
    robot->waypoint_client->async_send_goal(goal_msg, send_goal_options);
    
    RCLCPP_INFO(this->get_logger(), "Sent waypoint goal to %s with %zu waypoints", 
               robot_id.c_str(), waypoints.size());
    
    return true;
}

void RobotCentralServer::navigationResultCallback(
    const std::string& robot_id,
    const rclcpp_action::ClientGoalHandle<NavigateToPose>::WrappedResult& result)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) == robots_.end()) return;
    
    auto robot = robots_[robot_id];
    
    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            robot->status = "idle";
            robot->current_task = "completed";
            RCLCPP_INFO(this->get_logger(), "Robot %s navigation succeeded", robot_id.c_str());
            publishSystemLog("Robot " + robot_id + " navigation completed successfully", "INFO");
            break;
        case rclcpp_action::ResultCode::ABORTED:
            robot->status = "error";
            robot->current_task = "navigation_failed";
            RCLCPP_ERROR(this->get_logger(), "Robot %s navigation aborted", robot_id.c_str());
            publishSystemLog("Robot " + robot_id + " navigation aborted", "ERROR");
            break;
        case rclcpp_action::ResultCode::CANCELED:
            robot->status = "idle";
            robot->current_task = "navigation_canceled";
            RCLCPP_WARN(this->get_logger(), "Robot %s navigation canceled", robot_id.c_str());
            publishSystemLog("Robot " + robot_id + " navigation canceled", "WARN");
            break;
        default:
            robot->status = "error";
            robot->current_task = "unknown_error";
            break;
    }
    
    robot->last_update = this->get_clock()->now();
}

void RobotCentralServer::waypointsFeedbackCallback(
    const std::string& robot_id,
    const std::shared_ptr<const FollowWaypoints::Feedback> feedback)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) == robots_.end()) return;
    
    auto robot = robots_[robot_id];
    
    RCLCPP_INFO(this->get_logger(), "Robot %s executing waypoint: %u of %u", 
               robot_id.c_str(), 
               feedback->current_waypoint + 1,
               feedback->current_waypoint < 100 ? feedback->current_waypoint + 1 : 0); // 안전 체크
    
    // 현재 태스크 업데이트
    robot->current_task = "waypoint_" + std::to_string(feedback->current_waypoint + 1);
    robot->last_update = this->get_clock()->now();
    
    // 시스템 로그 발행
    publishSystemLog("Robot " + robot_id + " reached waypoint " + 
                    std::to_string(feedback->current_waypoint + 1), "INFO");
}

void RobotCentralServer::waypointsResultCallback(
    const std::string& robot_id,
    const rclcpp_action::ClientGoalHandle<FollowWaypoints>::WrappedResult& result)
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    if (robots_.find(robot_id) == robots_.end()) return;
    
    auto robot = robots_[robot_id];
    
    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            robot->status = "idle";
            robot->current_task = "waypoints_completed";
            RCLCPP_INFO(this->get_logger(), "Robot %s completed all waypoints successfully", robot_id.c_str());
            publishSystemLog("Robot " + robot_id + " completed waypoint mission successfully", "INFO");
            break;
            
        case rclcpp_action::ResultCode::ABORTED:
            robot->status = "error";
            robot->current_task = "waypoints_failed";
            RCLCPP_ERROR(this->get_logger(), "Robot %s waypoint mission aborted", robot_id.c_str());
            publishSystemLog("Robot " + robot_id + " waypoint mission aborted", "ERROR");
            break;
            
        case rclcpp_action::ResultCode::CANCELED:
            robot->status = "idle";
            robot->current_task = "waypoints_canceled";
            RCLCPP_WARN(this->get_logger(), "Robot %s waypoint mission canceled", robot_id.c_str());
            publishSystemLog("Robot " + robot_id + " waypoint mission canceled", "WARN");
            break;
            
        default:
            robot->status = "error";
            robot->current_task = "waypoints_unknown_error";
            RCLCPP_ERROR(this->get_logger(), "Robot %s waypoint mission ended with unknown result", robot_id.c_str());
            publishSystemLog("Robot " + robot_id + " waypoint mission unknown error", "ERROR");
            break;
    }
    
    robot->last_update = this->get_clock()->now();
}

void RobotCentralServer::statusUpdateCallback()
{
    publishRobotStatus();
    checkRobotHealth();
}

void RobotCentralServer::heartbeatCallback()
{
    publishSystemLog("Central Server heartbeat - " + std::to_string(robots_.size()) + " robots connected", "DEBUG");
}

void RobotCentralServer::publishRobotStatus()
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    for (const auto& [robot_id, robot] : robots_) {
        RobotStatus status_msg;
        status_msg.robot_id = robot_id;
        status_msg.status = robot->status;
        status_msg.current_task = robot->current_task;
        status_msg.current_pose = robot->current_pose;
        status_msg.last_update = robot->last_update;
        
        robot_status_publisher_->publish(status_msg);
    }
}

void RobotCentralServer::publishSystemLog(const std::string& message, const std::string& level)
{
    std_msgs::msg::String log_msg;
    auto now = this->get_clock()->now();
    log_msg.data = "[" + std::to_string(now.seconds()) + "] [" + level + "] " + message;
    
    system_log_publisher_->publish(log_msg);
    
    // ROS 로거에도 출력
    if (level == "ERROR") {
        RCLCPP_ERROR(this->get_logger(), "%s", message.c_str());
    } else if (level == "WARN") {
        RCLCPP_WARN(this->get_logger(), "%s", message.c_str());
    } else if (level == "DEBUG") {
        RCLCPP_DEBUG(this->get_logger(), "%s", message.c_str());
    } else {
        RCLCPP_INFO(this->get_logger(), "%s", message.c_str());
    }
}

void RobotCentralServer::checkRobotHealth()
{
    std::lock_guard<std::mutex> lock(robots_mutex_);
    
    auto current_time = this->get_clock()->now();
    
    for (auto& [robot_id, robot] : robots_) {
        auto time_diff = current_time - robot->last_update;
        
        if (time_diff.seconds() > 30) { // 30초 이상 업데이트 없으면 오프라인으로 간주
            if (robot->status != "offline") {
                robot->status = "offline";
                robot->current_task = "connection_lost";
                RCLCPP_WARN(this->get_logger(), "Robot %s went offline", robot_id.c_str());
                publishSystemLog("Robot " + robot_id + " connection lost", "WARN");
            }
        }
    }
}

geometry_msgs::msg::PoseStamped RobotCentralServer::createPoseStamped(double x, double y, double yaw)
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

tf2::Quaternion RobotCentralServer::getQuaternionFromYaw(double yaw_degrees)
{
    tf2::Quaternion quaternion;
    double yaw_radians = yaw_degrees * M_PI / 180.0;
    quaternion.setRPY(0, 0, yaw_radians);
    return quaternion;
}
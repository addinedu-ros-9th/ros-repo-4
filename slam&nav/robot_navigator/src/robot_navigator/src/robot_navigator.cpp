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
    
    // 로봇 정보 초기화
    robot_info_ = std::make_shared<RobotInfo>();
    robot_info_->navigation_status = "idle";
    robot_info_->current_target = "none";
    robot_info_->start_point_name = "none";
    robot_info_->is_online = false;
    robot_info_->start_point_set = false;
    robot_info_->last_update = this->get_clock()->now();
    robot_info_->teleop_active = false;
    
    // 명령 로그 퍼블리셔 초기화
    command_log_publisher_ = this->create_publisher<std_msgs::msg::String>("navigation_commands", 10);
    
    // 로봇 구독자 및 Action 클라이언트 초기화
    initializeRobotSubscribers();
    setupActionClient();
    setupPublishers();
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
    waypoints_["lobby_station"] = {"Main Lobby", 9, -1.76, 90.0, "병원 로비 스테이션"};
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
    // AMCL Pose 구독자
    amcl_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/amcl_pose", 10, 
        std::bind(&RobotNavigator::amclCallback, this, _1));
    
    // CMD_VEL 구독자
    cmd_vel_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        std::bind(&RobotNavigator::cmdVelCallback, this, _1));

    // teleop_event 구독자
    teleop_event_subscriber_ = this->create_subscription<std_msgs::msg::String>(
        "/teleop_event", 10,
        std::bind(&RobotNavigator::teleopEventCallback, this, _1));
    
    RCLCPP_INFO(this->get_logger(), "Initialized robot subscribers");
}

void RobotNavigator::setupActionClient()
{
    try {
        nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "/navigate_to_pose");
        RCLCPP_INFO(this->get_logger(), "Created navigation action client");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create action client: %s", e.what());
    }
}

void RobotNavigator::setupPublishers()
{
    // 개별 토픽 퍼블리셔들 생성
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("/pose", 10);
    start_point_publisher_ = this->create_publisher<std_msgs::msg::String>("/start_point", 10);
    velocity_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/velocity", 10);
    nav_status_publisher_ = this->create_publisher<std_msgs::msg::String>("/nav_status", 10);
    target_publisher_ = this->create_publisher<std_msgs::msg::String>("/target", 10);
    online_status_publisher_ = this->create_publisher<std_msgs::msg::Bool>("/online", 10);
    teleop_command_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    
    RCLCPP_INFO(this->get_logger(), "Created robot publishers");
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

void RobotNavigator::setupServices()
{
    control_event_server_ = this->create_service<control_interfaces::srv::EventHandle>(
        "control_service", &controlEventHandle);
    tracking_event_server_ = this->create_service<control_interfaces::srv::TrackHandle>(
        "control_service", &trackEventHandle);
    navigate_event_server_ = this->create_service<control_interfaces::srv::NavigateHandle>(
        "control_service", &navigateEventHandle);
    //robot_event_client_ = this->create_client<control_interfaces::srv::EventHandle>(
    //    "robot_service");

    RCLCPP_INFO(this->get_logger(), "Service Server & Client created!");
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

void RobotNavigator::setStartPoint(const std::string& waypoint_name)
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    
    if (waypoints_.find(waypoint_name) != waypoints_.end()) {
        robot_info_->start_point_name = waypoint_name;
        robot_info_->start_point_set = true;
        
        const auto& wp = waypoints_[waypoint_name];
        RCLCPP_INFO(this->get_logger(), "Start point set: '%s' (%.2f, %.2f, %.1f°)", 
                   waypoint_name.c_str(), wp.x, wp.y, wp.yaw);
        
        publishCommandLog("SET_START: " + waypoint_name + 
                         " (" + std::to_string(wp.x) + ", " + std::to_string(wp.y) + ")");
    }
}

bool RobotNavigator::sendRobotToStartPoint()
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    
    if (!robot_info_->start_point_set) {
        RCLCPP_ERROR(this->get_logger(), "Start point not set");
        publishCommandLog("ERROR: Start point not set");
        return false;
    }
    
    std::string start_waypoint = robot_info_->start_point_name;
    robot_mutex_.unlock();  // unlock before calling sendNavigationGoal
    
    RCLCPP_INFO(this->get_logger(), "Sending robot to start point waypoint: %s", start_waypoint.c_str());
    
    return sendNavigationGoal(start_waypoint);
}

bool RobotNavigator::sendRobotToLobby()
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    robot_info_->navigation_status = "moving_to_station";
    robot_mutex_.unlock();  // unlock before calling sendNavigationGoal
    
    RCLCPP_INFO(this->get_logger(), "Sending robot to lobby station");
    
    return sendNavigationGoal("lobby_station");
}

void RobotNavigator::amclCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    
    robot_info_->current_pose = msg->pose.pose;
    robot_info_->is_online = true;
    robot_info_->last_update = this->get_clock()->now();
    
    // 첫 번째 위치 수신 시 가장 가까운 waypoint를 시작점으로 설정
    if (!robot_info_->start_point_set) {
        std::string nearest_wp = findNearestWaypoint(
            msg->pose.pose.position.x, 
            msg->pose.pose.position.y);
        
        robot_info_->start_point_name = nearest_wp;
        robot_info_->start_point_set = true;
        
        const auto& wp = waypoints_[nearest_wp];
        RCLCPP_INFO(this->get_logger(), "Auto-set start point: '%s' (%.2f, %.2f) - nearest to current position (%.2f, %.2f)", 
                   nearest_wp.c_str(), wp.x, wp.y,
                   msg->pose.pose.position.x, msg->pose.pose.position.y);
        
        publishCommandLog("AUTO: Start point set -> " + nearest_wp + 
                        " (nearest to current position)");
    }
}

void RobotNavigator::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    
    robot_info_->current_velocity = *msg;
    robot_info_->last_update = this->get_clock()->now();
}

void RobotNavigator::teleopEventCallback(const std_msgs::msg::String::SharedPtr teleop_key)
{
    std::lock_guard<std::mutex> lock(robot_mutex_);

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
        
        teleop_command_publisher_->publish(message);
        logNavigationCommand("teleop: " + teleop_key);
        
        RCLCPP_INFO(this->get_logger(), "원격 제어 명령 전송: %s", teleop_key.c_str());
        //return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "원격 제어 명령 전송 실패: %s", e.what());
        //return false;
    }
}

void controlEventHandle(
    const std::shared_ptr<control_interfaces::srv::EventHandle::Request> control_req,
    std::shared_ptr<control_interfaces::srv::EventHandle::Response> control_res
)
{
    std::lock_guard<std::mutex> lock(robot_mutex_);

    std::string event_type = control_req->event_type;
    //call_with_voice, call_with_screen, control_by_admin 처리
    if (event_type == "call_with_voice" || 
        event_type == "call_with_screen" || event_type == "control_by_admin")
    {
        robot_info_->navigation_status = "waiting_for_navigation";
        control_res->status = "waiting_for_navigation";
        RCLCPP_INFO(this->get_logger(), "Status changed into " + robot_info_->navigation_status);
    }
    //teleop_request 처리
    else if (robot_info_->navigation_status = "waiting_for_navigation" && event_type == "teleop_request")
    {
        robot_info_->teleop_active = true;
        robot_info_->navigation_status = "moving_manual";
        control_res->status = "moving_manual";
        RCLCPP_INFO(this->get_logger(), "Status changed into " + robot_info_->navigation_status);
    }
    //teleop_complete 처리
    else if (robot_info_->navigation_status = "moving_manual" && event_type == "teleop_complete")
    {
        robot_info_->teleop_active = false;
        robot_info_->navigation_status = "waiting_for_navigation";
        control_res->status = "waiting_for_navigation";
        RCLCPP_INFO(this->get_logger(), "Status changed into " + robot_info_->navigation_status);
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "State 변경 실패: %s", e.what());
    }
}

void trackEventHandle(
    const std::shared_ptr<control_interfaces::srv::trackHandle::Request> track_req,
    std::shared_ptr<control_interfaces::srv::EventHandle::Response> track_res
)
{
    //call_with_gesture
}

void navigateEventHandle(
    const std::shared_ptr<control_interfaces::srv::navigateHandle::Request> nav_req,
    std::shared_ptr<control_interfaces::srv::navigateHandle::Response> nav_res
)
{
    if (event_type != "patient_navigating" && 
        event_type != "unknown_navigating" &&
        event_type != "unknown_navigating")
    {
        
    }
    std::string event_type = nav_req->event_type;
    std::string command = nav_req->command;
    RCLCPP_INFO(this->get_logger(), "Received navigation command: '%s'", command.c_str());
    
    if (command == "go_start" || command == "return_start") {
        if (sendRobotToStartPoint()) {
            publishCommandLog("GO_START: returning to start point");
            RCLCPP_INFO(this->get_logger(), "Sending robot to start point");
        } else {
            publishCommandLog("ERROR: Failed to send robot to start point");
            RCLCPP_ERROR(this->get_logger(), "Failed to send robot to start point");
        }
        return;
    }

    if (command == "return_lobby")
    {
        if (sendRobotToLobby()) {
            publishCommandLog("RETURN_LOBBY: returning to lobby station");
            RCLCPP_INFO(this->get_logger(), "Sending robot to lobby station");
        } else {
            publishCommandLog("ERROR: Failed to send robot to lobby station");
            RCLCPP_ERROR(this->get_logger(), "Failed to send robot to lobby station");
        }
        return;
    }
    
    // 기존 특수 명령들
    if (command == "stop" || command == "cancel") {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        robot_info_->navigation_status = "idle";
        robot_info_->current_target = "canceled";
        publishCommandLog("CANCEL: Navigation canceled");
        return;
    }
    
    if (command == "status") {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        publishCommandLog("STATUS: " + robot_info_->navigation_status + 
                        " (target: " + robot_info_->current_target + 
                        ", start_point: " + robot_info_->start_point_name + 
                        ", auto_start: enabled)");
        return;
    }
    
    if (command == "list") {
        publishAvailableWaypoints();
        return;
    }
    
    // 실제 네비게이션 명령 처리
    if (waypoints_.find(command) != waypoints_.end()) {
        if (sendNavigationGoal(command)) {
            publishCommandLog("COMMAND: -> " + command);
            RCLCPP_INFO(this->get_logger(), "Manual navigation command executed: -> %s", command.c_str());
        } else {
            publishCommandLog("ERROR: Failed to send command -> " + command);
        }
    } else {
        publishCommandLog("ERROR: Unknown waypoint '" + command + "'");
        publishAvailableWaypoints();
    }
}

void RobotNavigator::navigationCommandCallback(const std_msgs::msg::String::SharedPtr msg)
{
    std::string command = msg->data;
    RCLCPP_INFO(this->get_logger(), "Received navigation command: '%s'", command.c_str());
    
    if (command == "go_start" || command == "return_start") {
        if (sendRobotToStartPoint()) {
            publishCommandLog("GO_START: returning to start point");
            RCLCPP_INFO(this->get_logger(), "Sending robot to start point");
        } else {
            publishCommandLog("ERROR: Failed to send robot to start point");
            RCLCPP_ERROR(this->get_logger(), "Failed to send robot to start point");
        }
        return;
    }

    if (command == "return_lobby")
    {
        if (sendRobotToLobby()) {
            publishCommandLog("RETURN_LOBBY: returning to lobby station");
            RCLCPP_INFO(this->get_logger(), "Sending robot to lobby station");
        } else {
            publishCommandLog("ERROR: Failed to send robot to lobby station");
            RCLCPP_ERROR(this->get_logger(), "Failed to send robot to lobby station");
        }
        return;
    }
    
    // 기존 특수 명령들
    if (command == "stop" || command == "cancel") {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        robot_info_->navigation_status = "idle";
        robot_info_->current_target = "canceled";
        publishCommandLog("CANCEL: Navigation canceled");
        return;
    }
    
    if (command == "status") {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        publishCommandLog("STATUS: " + robot_info_->navigation_status + 
                        " (target: " + robot_info_->current_target + 
                        ", start_point: " + robot_info_->start_point_name + 
                        ", auto_start: enabled)");
        return;
    }
    
    if (command == "list") {
        publishAvailableWaypoints();
        return;
    }
    
    // 실제 네비게이션 명령 처리
    if (waypoints_.find(command) != waypoints_.end()) {
        if (sendNavigationGoal(command)) {
            publishCommandLog("COMMAND: -> " + command);
            RCLCPP_INFO(this->get_logger(), "Manual navigation command executed: -> %s", command.c_str());
        } else {
            publishCommandLog("ERROR: Failed to send command -> " + command);
        }
    } else {
        publishCommandLog("ERROR: Unknown waypoint '" + command + "'");
        publishAvailableWaypoints();
    }
}

bool RobotNavigator::sendNavigationGoal(const std::string& waypoint_name)
{
    if (robot_info_->teleop_active == true)
    {
        RCLCPP_ERROR(this->get_logger(), "Navigation action server not available due to manual control");
        return false;
    }

    if (waypoints_.find(waypoint_name) == waypoints_.end()) {
        RCLCPP_ERROR(this->get_logger(), "Waypoint '%s' not found", waypoint_name.c_str());
        return false;
    }
    
    if (!nav_client_) {
        RCLCPP_ERROR(this->get_logger(), "No navigation client found");
        return false;
    }
    
    if (!nav_client_->wait_for_action_server(std::chrono::seconds(5))) {
        RCLCPP_ERROR(this->get_logger(), "Navigation action server not available");
        return false;
    }
    
    const WaypointInfo& waypoint = waypoints_[waypoint_name];
    
    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose = createPoseStamped(waypoint.x, waypoint.y, waypoint.yaw);
    
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        robot_info_->navigation_status = "navigating";
        robot_info_->current_target = waypoint_name;
    }
    
    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    
    send_goal_options.goal_response_callback =
        [this](const GoalHandleNavigate::SharedPtr& goal_handle) {
            this->goalResponseCallback(goal_handle);
        };
    
    send_goal_options.feedback_callback =
        [this](const GoalHandleNavigate::SharedPtr goal_handle,
               const std::shared_ptr<const NavigateToPose::Feedback> feedback) {
            this->feedbackCallback(goal_handle, feedback);
        };
    
    send_goal_options.result_callback =
        [this](const GoalHandleNavigate::WrappedResult& result) {
            this->resultCallback(result);
        };
    
    nav_client_->async_send_goal(goal_msg, send_goal_options);
    
    RCLCPP_INFO(this->get_logger(), "Sent navigation goal: -> %s (%.2f, %.2f, %.1f°)", 
               waypoint.name.c_str(), waypoint.x, waypoint.y, waypoint.yaw);
    
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

void RobotNavigator::goalResponseCallback(const GoalHandleNavigate::SharedPtr& goal_handle)
{
    if (!goal_handle) {
        RCLCPP_ERROR(this->get_logger(), "Goal rejected");
        std::lock_guard<std::mutex> lock(robot_mutex_);
        robot_info_->navigation_status = "failed";
        publishCommandLog("REJECTED: Goal rejected");
    } else {
        RCLCPP_INFO(this->get_logger(), "Goal accepted");
        publishCommandLog("ACCEPTED: Goal accepted");
    }
}

void RobotNavigator::feedbackCallback(const GoalHandleNavigate::SharedPtr,
                                     const std::shared_ptr<const NavigateToPose::Feedback> feedback)
{
    RCLCPP_INFO(this->get_logger(), "Navigation feedback: distance remaining %.2fm", 
               feedback->distance_remaining);
}

void RobotNavigator::resultCallback(const GoalHandleNavigate::WrappedResult& result)
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    
    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            robot_event_client_ = this->create_client
            robot_info_->navigation_status = "waiting_for_return";
            
            // 도착 시 자동으로 현재 목표를 새로운 시작점으로 설정
            if (robot_info_->current_target != "start_point") {  // 시작점 복귀가 아닌 경우만
                std::string old_start = robot_info_->start_point_name;
                robot_info_->start_point_name = robot_info_->current_target;
                robot_info_->start_point_set = true;
                robot_info_->canceled_time = this->get_clock()->now(); // CANCELED 시각 기록
                RCLCPP_INFO(this->get_logger(), "Robot successfully reached target: %s", 
                           robot_info_->current_target.c_str());
                RCLCPP_INFO(this->get_logger(), "Auto-updated start point: %s -> %s", 
                           old_start.c_str(), robot_info_->start_point_name.c_str());
                
                publishCommandLog("SUCCESS: reached " + robot_info_->current_target);
                publishCommandLog("AUTO_START: Start point updated to " + robot_info_->start_point_name);              
            } else {
                RCLCPP_INFO(this->get_logger(), "Robot returned to start point: %s", 
                           robot_info_->start_point_name.c_str());
                publishCommandLog("SUCCESS: returned to start point " + robot_info_->start_point_name);
            }
            break;
            
        case rclcpp_action::ResultCode::ABORTED:
            robot_info_->navigation_status = "failed";
            RCLCPP_ERROR(this->get_logger(), "Navigation aborted");
            publishCommandLog("FAILED: navigation aborted");
            break;
            
        case rclcpp_action::ResultCode::CANCELED:
            robot_info_->navigation_status = "idle";
            robot_info_->canceled_time = this->get_clock()->now(); // CANCELED 시각 기록
            RCLCPP_WARN(this->get_logger(), "Navigation canceled");
            publishCommandLog("CANCELED: navigation canceled");
            break;
            
        default:
            robot_info_->navigation_status = "failed";
            RCLCPP_ERROR(this->get_logger(), "Navigation unknown result");
            publishCommandLog("ERROR: navigation unknown result");
            break;
    }
}

void RobotNavigator::statusTimerCallback()
{
    // 10초 timeout 후 lobby_station 복귀 로직
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->navigation_status == "idle" &&
            robot_info_->canceled_time.nanoseconds() != 0) {
            rclcpp::Duration elapsed = this->get_clock()->now() - robot_info_->canceled_time;
            if (elapsed.seconds() >= 10.0) {
                publishCommandLog("TIMEOUT: canceled -> returning to lobby_station");
                sendRobotToLobby();
                robot_info_->canceled_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
            }
        }
    }
    publishRobotData();
}

void RobotNavigator::publishRobotData()
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    
    auto time_diff = this->get_clock()->now() - robot_info_->last_update;
    bool is_online = time_diff.seconds() < 10.0;
    
    // 위치 정보 발행
    pose_publisher_->publish(robot_info_->current_pose);
    
    // 시작점 이름 발행
    if (robot_info_->start_point_set) {
        std_msgs::msg::String start_point_msg;
        start_point_msg.data = robot_info_->start_point_name;
        start_point_publisher_->publish(start_point_msg);
    }
    
    // 속도 정보 발행
    velocity_publisher_->publish(robot_info_->current_velocity);
    
    // 네비게이션 상태 발행
    std_msgs::msg::String nav_status_msg;
    nav_status_msg.data = robot_info_->navigation_status;
    nav_status_publisher_->publish(nav_status_msg);
    
    // 현재 목표 발행
    std_msgs::msg::String target_msg;
    target_msg.data = robot_info_->current_target;
    target_publisher_->publish(target_msg);
    
    // 온라인 상태 발행
    std_msgs::msg::Bool online_msg;
    online_msg.data = is_online;
    online_status_publisher_->publish(online_msg);
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
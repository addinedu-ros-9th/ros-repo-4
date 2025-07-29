#include "central_server/central_server.h"
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <json/json.h>
#include <iomanip>
#include <sstream>

CentralServer::CentralServer() : Node("central_server") {
    RCLCPP_INFO(this->get_logger(), "CentralServer 생성자 호출됨");
    
    http_port_ = 8080;
    http_host_ = "0.0.0.0";

    setupHttpServer();

    // DatabaseManager 생성
    db_manager_ = std::make_unique<DatabaseManager>();
    
    // HttpServer 생성 (DatabaseManager를 shared_ptr로 전달)
    auto shared_db_manager = std::shared_ptr<DatabaseManager>(db_manager_.get(), [](DatabaseManager*){});
    http_server_ = std::make_unique<HttpServer>(shared_db_manager, http_port_);
    
    // simple_central_server 기능 통합: 웨이포인트/로봇/네비게이션 초기화
    initializeWaypoints();
    initializeRobotSubscribers();
    setupActionClients();
    setupIndividualPublishers();
    setupNavigationCommandSubscriber();
    status_timer_ = this->create_wall_timer(
        std::chrono::seconds(1), std::bind(&CentralServer::statusTimerCallback, this));
    command_log_publisher_ = this->create_publisher<std_msgs::msg::String>("navigation_commands", 10);
    publishCommandLog("Central Server started - Ready for manual navigation commands");
    publishAvailableWaypoints();

    RCLCPP_INFO(this->get_logger(), "ROS2 토픽 및 서비스 설정 완료");
}

CentralServer::~CentralServer() {
    stop();
}

void CentralServer::setupHttpServer()
{
    try {
        // 설정 파일 로드
        const char* config_env = std::getenv("CENTRAL_SERVER_CONFIG");
        std::string config_path = config_env ? config_env : "/home/wonho/ros-repo-4/config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        
        // HTTP 서버 설정
        http_port_ = config["central_server"]["http_port"].as<int>();
        http_host_ = config["central_server"]["http_host"].as<std::string>();
        
        RCLCPP_INFO(this->get_logger(), "HTTP 서버 설정 로드 완료:");
        RCLCPP_INFO(this->get_logger(), "  - HTTP 서버 포트: %d", http_port_);
        RCLCPP_INFO(this->get_logger(), "  - HTTP 서버 호스트: %s", http_host_.c_str());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "설정 파일 로드 실패: %s", e.what());
        // 기본값 사용
        http_port_ = 8080;
        http_host_ = "0.0.0.0";
        
        RCLCPP_WARN(this->get_logger(), "기본값 사용: HTTP 포트=%d, 호스트=%s",
                   http_port_, http_host_.c_str());
    }
}

void CentralServer::start() {
    if (running_.load()) {
        RCLCPP_WARN(this->get_logger(), "서버가 이미 실행중입니다");
        return;
    }
    
    running_.store(true);
    
    RCLCPP_INFO(this->get_logger(), "1단계: DB 스레드 시작중...");
    db_thread_ = std::thread(&CentralServer::runDatabaseThread, this);
    
    RCLCPP_INFO(this->get_logger(), "2단계: HTTP 스레드 시작중...");
    http_thread_ = std::thread(&CentralServer::runHttpThread, this);
    
    // 잠시 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    RCLCPP_INFO(this->get_logger(), "Central Server 시작 완료!");
}

void CentralServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "서버 종료중...");
    running_.store(false);
    
    // 기존 스레드들 종료 대기
    if (db_thread_.joinable()) {
        db_thread_.join();
        RCLCPP_INFO(this->get_logger(), "DB 스레드 종료됨");
    }
    
    if (http_thread_.joinable()) {
        http_thread_.join();
        RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료됨");
    }
    
    RCLCPP_INFO(this->get_logger(), "서버 종료 완료");
}

void CentralServer::runDatabaseThread() {
    RCLCPP_INFO(this->get_logger(), "DB 스레드 시작됨");
    
    // DB 연결 시도
    if (db_manager_->connect()) {
        RCLCPP_INFO(this->get_logger(), "MySQL 데이터베이스 연결 성공!");
    } else {
        RCLCPP_ERROR(this->get_logger(), "MySQL 데이터베이스 연결 실패!");
        RCLCPP_WARN(this->get_logger(), "DB 없이 계속 실행합니다...");
    }
    
    while (running_.load() && rclcpp::ok()) {
        // DB 연결 상태 확인 (5초마다)
        if (!db_manager_->isConnected()) {
            RCLCPP_WARN(this->get_logger(), "DB 연결 끊어짐. 재연결 시도...");
            db_manager_->connect();
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    RCLCPP_INFO(this->get_logger(), "DB 스레드 종료중...");
}

void CentralServer::runHttpThread() {
    RCLCPP_INFO(this->get_logger(), "HTTP 스레드 시작됨");
    
    // HTTP 서버 시작
    http_server_->start();
    RCLCPP_INFO(this->get_logger(), "HTTP 서버 시작 완료 (포트: %d)", http_port_);
    
    while (running_.load() && rclcpp::ok()) {
        // HTTP 서버 상태 확인
        if (!http_server_->isRunning()) {
            RCLCPP_ERROR(this->get_logger(), "HTTP 서버가 중지됨!");
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // HTTP 서버 정지
    http_server_->stop();
    RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료중...");
}

void CentralServer::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    try {
        // ROS2 Image 메시지를 OpenCV Mat으로 변환
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        
        // 이미지 정보 로그 (너무 자주 출력하지 않도록 제한)
        static int frame_count = 0;
        frame_count++;
        
        if (frame_count % 60 == 0) { // 60프레임마다 로그
            RCLCPP_INFO(this->get_logger(), 
                       "AI 서버로부터 이미지 수신 - 크기: %dx%d, 프레임 #%d", 
                       cv_ptr->image.cols, cv_ptr->image.rows, frame_count);
        }
        
        // 여기에 이미지 처리 로직을 추가할 수 있습니다
        // 예: 이미지 저장, 분석, 웹으로 스트리밍 등
        
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "이미지 변환 실패: %s", e.what());
    }
}

void CentralServer::statusCallback(const robot_interfaces::msg::RobotStatus::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), 
               "AI 서버 상태 수신 - Robot ID: %d, Status: %s", 
               msg->robot_id, msg->status.c_str());
    
    // HTTP를 통해 GUI 클라이언트들에게 로봇 상태 전송
    sendRobotStatusToGui(msg->robot_id, msg->status, "robot_controller");
    
    if (db_manager_->isConnected()) {
        RCLCPP_DEBUG(this->get_logger(), "상태를 데이터베이스에 저장 중...");
    }
}

void CentralServer::changeStatusCallback(
    const std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Request> request,
    std::shared_ptr<robot_interfaces::srv::ChangeRobotStatus::Response> response)
{
    RCLCPP_INFO(this->get_logger(), 
               "상태 변경 요청 - Robot ID: %d, New Status: %s", 
               request->robot_id, request->new_status.c_str());
    
    // HTTP를 통해 GUI 클라이언트들에게 상태 변경 알림
    sendRobotStatusToGui(request->robot_id, request->new_status, "user_gui");
    
    response->success = true;
    response->message = "상태 변경 완료";
    
    RCLCPP_INFO(this->get_logger(), "상태 변경 서비스 응답 완료");
}

void CentralServer::sendRobotLocationToGui(int robot_id, float location_x, float location_y)
{
    Json::Value message;
    message["robot_id"] = robot_id;
    message["location_x"] = location_x;
    message["location_y"] = location_y;
    
    std::string json_message = message.toStyledString();
    broadcastToGuiClients(json_message);
    
    RCLCPP_INFO(this->get_logger(), "로봇 위치 전송: Robot ID %d at (%.2f, %.2f)", 
               robot_id, location_x, location_y);
}

void CentralServer::sendRobotStatusToGui(int robot_id, const std::string& status, const std::string& source)
{
    Json::Value message;
    message["robot_id"] = robot_id;
    message["status"] = status;
    message["source"] = source;
    
    std::string json_message = message.toStyledString();
    broadcastToGuiClients(json_message);
    
    RCLCPP_INFO(this->get_logger(), "로봇 상태 전송: Robot ID %d, Status %s, Source %s", 
               robot_id, status.c_str(), source.c_str());
}

void CentralServer::sendArrivalNotificationToGui(int robot_id)
{
    Json::Value message;
    message["robot_id"] = robot_id;
    message["status"] = "arrived";
    
    std::string json_message = message.toStyledString();
    broadcastToGuiClients(json_message);
    
    RCLCPP_INFO(this->get_logger(), "도착 알림 전송: Robot ID %d", robot_id);
}

void CentralServer::broadcastToGuiClients(const std::string& message)
{
    // HttpServer를 통해 GUI 클라이언트들에게 메시지 브로드캐스트
    if (http_server_) {
        http_server_->broadcastToClients(message);
        RCLCPP_DEBUG(this->get_logger(), "GUI 클라이언트들에게 메시지 브로드캐스트: %s", message.c_str());
    }
}

void CentralServer::init() {
    RCLCPP_INFO(this->get_logger(), "[init] this ptr: %p", (void*)this);
    try {
        auto self = rclcpp::Node::shared_from_this();
        RCLCPP_INFO(this->get_logger(), "[init] shared_from_this() ptr: %p", (void*)self.get());
        image_transport_ = std::make_shared<image_transport::ImageTransport>(self);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[init] Exception during shared_from_this(): %s", e.what());
        throw;
    }
    image_subscriber_ = image_transport_->subscribe(
        "webcam/image_raw", 1, 
        std::bind(&CentralServer::imageCallback, this, std::placeholders::_1));
    status_subscriber_ = this->create_subscription<robot_interfaces::msg::RobotStatus>(
        "robot_status", 10,
        std::bind(&CentralServer::statusCallback, this, std::placeholders::_1));
    status_service_ = this->create_service<robot_interfaces::srv::ChangeRobotStatus>(
        "change_robot_status",
        std::bind(&CentralServer::changeStatusCallback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(this->get_logger(), "ROS2 토픽 및 서비스 설정 완료");
}

// simple_central_server 함수들 구현
void CentralServer::initializeWaypoints() {
    waypoints_["lobby_station"] = {"Main Lobby", 9.53, -1.76, 90.0, "병원 로비 스테이션"};
    waypoints_["breast_cancer"] = {"Breast Cancer Center", 7.64, 1.63, 180.0, "유방암 센터"};
    waypoints_["brain_tumor"] = {"Brain Tumor Center", 5.97, 1.46, 180.0, "뇌종양 센터"};
    waypoints_["lung_cancer"] = {"Lung Cancer Center", 5.32, -2.27, 0.0, "폐암 센터"};
    waypoints_["stomach_cancer"] = {"Stomach Cancer Center", 3.84, -2.3, 0.0, "위암 센터"};
    waypoints_["colon_cancer"] = {"Colon Cancer Center", 0.93, -2.3, 0.0, "대장암 센터"};
    waypoints_["gateway_a"] = {"Gateway A", 0.09, 4.0, 180.0, "통로 A"};
    waypoints_["gateway_b"] = {"Gateway B", -2.6, 4.18, 0.0, "통로 B"};
    waypoints_["x_ray"] = {"X-ray", -5.69, 4.34, 180.0, "X-ray 검사실"};
    waypoints_["ct"] = {"CT", -5.79, -1.88, 90.0, "CT 검사실"};
    waypoints_["echography"] = {"Echography", -4.9, -1.96, 90.0, "초음파 검사실"};
    RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_.size());
}

void CentralServer::initializeRobotSubscribers() {
    std::vector<std::string> robot_ids = {"robot1"};
    for (const auto& robot_id : robot_ids) {
        auto robot_info = std::make_shared<RobotMonitorInfo>();
        robot_info->robot_id = robot_id;
        robot_info->navigation_status = "idle";
        robot_info->current_target = "none";
        robot_info->is_online = false;
        robot_info->last_update = this->get_clock()->now();
        {
            std::lock_guard<std::mutex> lock(robots_mutex_);
            robots_[robot_id] = robot_info;
        }
        std::string amcl_topic = "/amcl_pose";
        amcl_subscribers_[robot_id] = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            amcl_topic, 10,
            [this, robot_id](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                this->amclCallback(robot_id, msg);
            });
        std::string cmd_vel_topic = "/cmd_vel";
        cmd_vel_subscribers_[robot_id] = this->create_subscription<geometry_msgs::msg::Twist>(
            cmd_vel_topic, 10,
            [this, robot_id](const geometry_msgs::msg::Twist::SharedPtr msg) {
                this->cmdVelCallback(robot_id, msg);
            });
        RCLCPP_INFO(this->get_logger(), "Initialized subscribers for %s", robot_id.c_str());
    }
}

void CentralServer::setupActionClients() {
    for (const auto& [robot_id, robot_info] : robots_) {
        try {
            std::string nav_topic = "/navigate_to_pose";
            nav_clients_[robot_id] = rclcpp_action::create_client<NavigateToPose>(this, nav_topic);
            RCLCPP_INFO(this->get_logger(), "Created navigation action client for %s", robot_id.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create action client for %s: %s", robot_id.c_str(), e.what());
        }
    }
}

void CentralServer::setupIndividualPublishers() {
    for (const auto& [robot_id, robot_info] : robots_) {
        std::string pose_topic = "/fleet/" + robot_id + "/pose";
        std::string velocity_topic = "/fleet/" + robot_id + "/velocity";
        std::string nav_status_topic = "/fleet/" + robot_id + "/nav_status";
        std::string target_topic = "/fleet/" + robot_id + "/target";
        std::string online_topic = "/fleet/" + robot_id + "/online";
        pose_publishers_[robot_id] = this->create_publisher<geometry_msgs::msg::Pose>(pose_topic, 10);
        velocity_publishers_[robot_id] = this->create_publisher<geometry_msgs::msg::Twist>(velocity_topic, 10);
        nav_status_publishers_[robot_id] = this->create_publisher<std_msgs::msg::String>(nav_status_topic, 10);
        target_publishers_[robot_id] = this->create_publisher<std_msgs::msg::String>(target_topic, 10);
        online_status_publishers_[robot_id] = this->create_publisher<std_msgs::msg::Bool>(online_topic, 10);
        RCLCPP_INFO(this->get_logger(), "Created individual publishers for %s", robot_id.c_str());
    }
}

void CentralServer::setupNavigationCommandSubscriber() {
    nav_command_subscriber_ = this->create_subscription<std_msgs::msg::String>(
        "navigation_command", 10,
        std::bind(&CentralServer::navigationCommandCallback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Navigation command subscriber created: /navigation_command");
}

void CentralServer::amclCallback(const std::string& robot_id, const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(robots_mutex_);
    if (robots_.find(robot_id) != robots_.end()) {
        robots_[robot_id]->current_pose = msg->pose.pose;
        robots_[robot_id]->is_online = true;
        robots_[robot_id]->last_update = this->get_clock()->now();
    }
}

void CentralServer::cmdVelCallback(const std::string& robot_id, const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(robots_mutex_);
    if (robots_.find(robot_id) != robots_.end()) {
        robots_[robot_id]->current_velocity = *msg;
        robots_[robot_id]->last_update = this->get_clock()->now();
    }
}

void CentralServer::navigationCommandCallback(const std_msgs::msg::String::SharedPtr msg) {
    std::string command = msg->data;
    RCLCPP_INFO(this->get_logger(), "Received navigation command: '%s'", command.c_str());
    std::string robot_id = "robot1";
    std::string waypoint_name = command;
    size_t colon_pos = command.find(':');
    if (colon_pos != std::string::npos) {
        robot_id = command.substr(0, colon_pos);
        waypoint_name = command.substr(colon_pos + 1);
    }
    if (waypoint_name == "stop" || waypoint_name == "cancel") {
        std::lock_guard<std::mutex> lock(robots_mutex_);
        if (robots_.find(robot_id) != robots_.end()) {
            robots_[robot_id]->navigation_status = "idle";
            robots_[robot_id]->current_target = "canceled";
        }
        publishCommandLog("CANCEL: Navigation canceled for " + robot_id);
        RCLCPP_INFO(this->get_logger(), "Navigation canceled for %s", robot_id.c_str());
        return;
    }
    if (waypoint_name == "status") {
        std::lock_guard<std::mutex> lock(robots_mutex_);
        if (robots_.find(robot_id) != robots_.end()) {
            auto robot = robots_[robot_id];
            publishCommandLog("STATUS: " + robot_id + " - " + robot->navigation_status + " (target: " + robot->current_target + ")");
        }
        return;
    }
    if (waypoint_name == "list") {
        publishAvailableWaypoints();
        return;
    }
    if (waypoints_.find(waypoint_name) != waypoints_.end()) {
        if (sendNavigationGoal(robot_id, waypoint_name)) {
            publishCommandLog("COMMAND: " + robot_id + " -> " + waypoint_name);
            RCLCPP_INFO(this->get_logger(), "Manual navigation command executed: %s -> %s", robot_id.c_str(), waypoint_name.c_str());
        } else {
            publishCommandLog("ERROR: Failed to send command " + robot_id + " -> " + waypoint_name);
            RCLCPP_ERROR(this->get_logger(), "Failed to execute navigation command: %s -> %s", robot_id.c_str(), waypoint_name.c_str());
        }
    } else {
        publishCommandLog("ERROR: Unknown waypoint '" + waypoint_name + "'");
        RCLCPP_WARN(this->get_logger(), "Unknown waypoint: '%s'", waypoint_name.c_str());
        std::stringstream ss;
        ss << "Available waypoints: ";
        for (const auto& [name, waypoint] : waypoints_) {
            ss << name << " ";
        }
        publishCommandLog(ss.str());
    }
}

bool CentralServer::sendNavigationGoal(const std::string& robot_id, const std::string& waypoint_name) {
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

geometry_msgs::msg::PoseStamped CentralServer::createPoseStamped(double x, double y, double yaw) {
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

tf2::Quaternion CentralServer::getQuaternionFromYaw(double yaw_degrees) {
    tf2::Quaternion quaternion;
    double yaw_radians = yaw_degrees * M_PI / 180.0;
    quaternion.setRPY(0, 0, yaw_radians);
    return quaternion;
}

void CentralServer::goalResponseCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr& goal_handle) {
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

void CentralServer::feedbackCallback(const std::string& robot_id, const GoalHandleNavigate::SharedPtr,
                                     const std::shared_ptr<const NavigateToPose::Feedback> feedback) {
    RCLCPP_INFO(this->get_logger(), "Robot %s navigation feedback: distance remaining %.2fm",
               robot_id.c_str(), feedback->distance_remaining);
}

void CentralServer::resultCallback(const std::string& robot_id, const GoalHandleNavigate::WrappedResult& result) {
    std::lock_guard<std::mutex> lock(robots_mutex_);
    if (robots_.find(robot_id) == robots_.end()) return;
    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            robots_[robot_id]->navigation_status = "reached";
            RCLCPP_INFO(this->get_logger(), "Robot %s successfully reached target: %s",
                       robot_id.c_str(), robots_[robot_id]->current_target.c_str());
            publishCommandLog("SUCCESS: " + robot_id + " reached " + robots_[robot_id]->current_target);
            break;
        case rclcpp_action::ResultCode::ABORTED:
            robots_[robot_id]->navigation_status = "failed";
            RCLCPP_ERROR(this->get_logger(), "Robot %s navigation aborted", robot_id.c_str());
            publishCommandLog("FAILED: " + robot_id + " navigation aborted");
            break;
        case rclcpp_action::ResultCode::CANCELED:
            robots_[robot_id]->navigation_status = "idle";
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

void CentralServer::statusTimerCallback() {
    publishIndividualRobotData();
}

void CentralServer::publishIndividualRobotData() {
    std::lock_guard<std::mutex> lock(robots_mutex_);
    for (const auto& [robot_id, robot] : robots_) {
        auto time_diff = this->get_clock()->now() - robot->last_update;
        bool is_online = time_diff.seconds() < 10.0;
        if (pose_publishers_.find(robot_id) != pose_publishers_.end()) {
            pose_publishers_[robot_id]->publish(robot->current_pose);
        }
        if (velocity_publishers_.find(robot_id) != velocity_publishers_.end()) {
            velocity_publishers_[robot_id]->publish(robot->current_velocity);
        }
        if (nav_status_publishers_.find(robot_id) != nav_status_publishers_.end()) {
            std_msgs::msg::String nav_status_msg;
            nav_status_msg.data = robot->navigation_status;
            nav_status_publishers_[robot_id]->publish(nav_status_msg);
        }
        if (target_publishers_.find(robot_id) != target_publishers_.end()) {
            std_msgs::msg::String target_msg;
            target_msg.data = robot->current_target;
            target_publishers_[robot_id]->publish(target_msg);
        }
        if (online_status_publishers_.find(robot_id) != online_status_publishers_.end()) {
            std_msgs::msg::Bool online_msg;
            online_msg.data = is_online;
            online_status_publishers_[robot_id]->publish(online_msg);
        }
    }
}

void CentralServer::publishCommandLog(const std::string& message) {
    std_msgs::msg::String log_msg;
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] " << message;
    log_msg.data = ss.str();
    if (command_log_publisher_) command_log_publisher_->publish(log_msg);
    RCLCPP_INFO(this->get_logger(), "%s", message.c_str());
}

void CentralServer::publishAvailableWaypoints() {
    std::stringstream ss;
    ss << "Available waypoints: ";
    for (const auto& [name, waypoint] : waypoints_) {
        ss << name << " (" << waypoint.x << "," << waypoint.y << "," << waypoint.yaw << "°) ";
    }
    publishCommandLog(ss.str());
    RCLCPP_INFO(this->get_logger(), "Published available waypoints list");
}
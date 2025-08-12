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
    robot_info_->net_signal_level = 0;
    
    // 명령 로그 퍼블리셔 초기화
    command_log_publisher_ = this->create_publisher<std_msgs::msg::String>("navigation_commands", 10);
    
    // 로봇 구독자 및 Action 클라이언트 초기화
    initializeRobotSubscribers();
    setupActionClient();
    setupPublishers();
    setupNavigationCommandSubscriber();
    setupServices();
    
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

    // 파라미터 선언 및 스캔 토픽 채택
    this->declare_parameter<std::string>("scan_topic", scan_topic_);
    this->get_parameter("scan_topic", scan_topic_);

    // LaserScan 구독자 (전방 장애물 감지 및 DetectHandle 서비스 트리거)
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_, 10,
        std::bind(&RobotNavigator::scanCallback, this, _1));
    
    RCLCPP_INFO(this->get_logger(), "Initialized robot subscribers (scan_topic=%s)", scan_topic_.c_str());
}

void RobotNavigator::setupActionClient()
{
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "/navigate_to_pose");
    RCLCPP_INFO(this->get_logger(), "✅ Navigation action client created.");
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
    net_level_publisher_ = this->create_publisher<std_msgs::msg::Int32>("/net_level", 10);
    
    RCLCPP_INFO(this->get_logger(), "Created robot publishers");
}

void RobotNavigator::setupNavigationCommandSubscriber()
{
    nav_command_subscriber_ = this->create_subscription<std_msgs::msg::String>(
        "/navigation_command", 10,
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
        "control_event",
        std::bind(&RobotNavigator::controlEventHandle, this, _1, _2));

    tracking_event_server_ = this->create_service<control_interfaces::srv::TrackHandle>(
        "tracking_event",
        std::bind(&RobotNavigator::trackEventHandle, this, std::placeholders::_1, std::placeholders::_2));

    navigate_event_server_ = this->create_service<control_interfaces::srv::NavigateHandle>(
        "navigate_event",
        std::bind(&RobotNavigator::navigateEventHandle, this, std::placeholders::_1, std::placeholders::_2));

    robot_event_client_ = this->create_client<control_interfaces::srv::EventHandle>("robot_event");
    detect_event_client_ = this->create_client<control_interfaces::srv::DetectHandle>("detect_obstacle");

    RCLCPP_INFO(this->get_logger(), "✅ Service Servers created: control_event_service, tracking_event_service, navigate_event_service");
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
    std::string start_waypoint;
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (!robot_info_->start_point_set) {
            RCLCPP_ERROR(this->get_logger(), "Start point not set");
            publishCommandLog("ERROR: Start point not set");
            return false;
        }
        start_waypoint = robot_info_->start_point_name;
    } // unlock
    RCLCPP_INFO(this->get_logger(), "Sending robot to start point waypoint: %s", start_waypoint.c_str());
    return sendNavigationGoal(start_waypoint);
}

bool RobotNavigator::sendRobotToLobby()
{
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        robot_info_->navigation_status = "moving_to_station"; // 복귀 중 표시 유지
    }
    RCLCPP_INFO(this->get_logger(), "Sending robot to lobby station");
    return sendNavigationGoal("lobby_station", /*keep_status=*/true);
}


void RobotNavigator::amclCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    robot_info_->current_pose = msg->pose.pose;
    robot_info_->is_online = true;
    robot_info_->last_update = this->get_clock()->now();
    
    // 첫 위치 수신 시 현재 위치에서 가장 가까운 waypoint를 시작점으로 설정
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
        publishCommandLog("AUTO: Start point set -> " + nearest_wp + " (nearest to current position)");
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
        if (teleop_key->data == "1") {
            message.linear.x = 1.0; message.linear.y = 0.0; message.angular.z = 1.0;
        } else if (teleop_key->data == "2") {
            message.linear.x = 1.0; message.linear.y = 0.0; message.angular.z = 0.0;
        } else if (teleop_key->data == "3") {
            message.linear.x = 1.0; message.linear.y = 0.0; message.angular.z = -1.0;
        } else if (teleop_key->data == "4") {
            message.linear.x = 0.0; message.linear.y = 0.0; message.angular.z = 1.0;
        } else if (teleop_key->data == "5") {
            message.linear.x = 0.0; message.linear.y = 0.0; message.angular.z = 0.0;
        } else if (teleop_key->data == "6") {
            message.linear.x = 0.0; message.linear.y = 0.0; message.angular.z = -1.0;
        } else if (teleop_key->data == "7") {
            message.linear.x = -1.0; message.linear.y = 0.0; message.angular.z = -1.0;
        } else if (teleop_key->data == "8") {
            message.linear.x = -1.0; message.linear.y = 0.0; message.angular.z = 0.0;
        } else if (teleop_key->data == "9") {
            message.linear.x = -1.0; message.linear.y = 0.0; message.angular.z = 1.0;
        } else {
            message.linear.x = 0.0; message.linear.y = 0.0; message.angular.z = 0.0;
        }
        teleop_command_publisher_->publish(message);
        RCLCPP_INFO(this->get_logger(), "원격 제어 명령 전송: %s", teleop_key->data.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "원격 제어 명령 전송 실패: %s", e.what());
    }
}

void RobotNavigator::netLevelCallback() {
    // popen/pclose 경고 제거용 커스텀 deleter
    struct PcloseDeleter {
        void operator()(FILE* f) const noexcept { if (f) pclose(f); }
    };

    auto exec = [](const char* cmd) -> std::string {
        char buf[256];
        std::string out;
        std::unique_ptr<FILE, PcloseDeleter> pipe(popen(cmd, "r"));
        if (!pipe) return out;  // 실패 시 빈 문자열
        while (fgets(buf, sizeof(buf), pipe.get())) out += buf;
        return out;
    };

    std::string iw = exec("iwconfig 2>/dev/null");
    std::smatch m;
    static const std::regex sigRe(R"(Signal level=(-?\d+))");

    int level = 0; // 기본값: 알 수 없음
    if (std::regex_search(iw, m, sigRe) && m.size() >= 2) {
        int rssi = std::stoi(m[1].str());
        if (rssi >= -50)      level = 4;
        else if (rssi >= -65) level = 3;
        else if (rssi >= -80) level = 2;
        else                  level = 1;
    } else {
        RCLCPP_INFO(rclcpp::get_logger("RobotNavigator"), "Wi-Fi 신호를 찾을 수 없음");
    }

    // 공유 상태 업데이트는 락으로 보호
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        robot_info_->net_signal_level = level;
    }
}

void RobotNavigator::controlEventHandle(
    const std::shared_ptr<control_interfaces::srv::EventHandle::Request> control_req,
    std::shared_ptr<control_interfaces::srv::EventHandle::Response> control_res
)
{
    const std::string event_type = control_req->event_type;

    if (event_type == "call_with_voice" || 
        event_type == "call_with_screen" || 
        event_type == "control_by_admin")
    {
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            robot_info_->navigation_status = "waiting_for_navigating";
        }
        control_res->status = "waiting_for_navigating";
        RCLCPP_INFO(rclcpp::get_logger("RobotNavigator"), "Status changed into waiting_for_navigating");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->navigation_status == "waiting_for_navigating" && event_type == "teleop_request") {
            robot_info_->teleop_active = true;
            robot_info_->navigation_status = "moving_manual";
            control_res->status = "moving_manual";
            RCLCPP_INFO(rclcpp::get_logger("RobotNavigator"), "Status changed into moving_manual");
            return;
        }
        if (robot_info_->navigation_status == "moving_manual" && event_type == "teleop_complete") {
            robot_info_->teleop_active = false;
            robot_info_->navigation_status = "waiting_for_navigating";
            control_res->status = "waiting_for_navigating";
            RCLCPP_INFO(rclcpp::get_logger("RobotNavigator"), "Status changed into waiting_for_navigating");
            return;
        }
    }

    if (event_type == "patient_return" || event_type == "unknown_return" || event_type == "admin_return") {
        bool should_return = false;
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            if (robot_info_->navigation_status == "waiting_for_return") {
                robot_info_->navigation_status = "moving_to_station";
                control_res->status = "moving_to_station";
                should_return = true;
                RCLCPP_INFO(rclcpp::get_logger("RobotNavigator"), "Status changed into moving_to_station");
            } else {
                control_res->status = "invalid_state";
            }
        } // unlock
        if (should_return) {
            if (sendRobotToLobby()) publishCommandLog("RETURN_LOBBY: returning to lobby station");
            else publishCommandLog("ERROR: Failed to send robot to lobby station");
        }
        return;
    }

    RCLCPP_ERROR(rclcpp::get_logger("RobotNavigator"), "❌ Unknown or invalid event_type: %s", event_type.c_str());
}

void RobotNavigator::trackEventHandle(
    const std::shared_ptr<control_interfaces::srv::TrackHandle::Request> track_req,
    std::shared_ptr<control_interfaces::srv::TrackHandle::Response> track_res
)
{
    (void)track_req;
    (void)track_res;
    // TODO
}

void RobotNavigator::navigateEventHandle(
    const std::shared_ptr<control_interfaces::srv::NavigateHandle::Request> nav_req,
    std::shared_ptr<control_interfaces::srv::NavigateHandle::Response> nav_res
)
{
    const std::string event_type = nav_req->event_type;
    std::string command = nav_req->command;

    // 1) 이벤트 분류
    const bool is_navigating_event =
        (event_type == "patient_navigating" ||
         event_type == "unknown_navigating" ||
         event_type == "admin_navigating");

    const bool is_control_event =
        (event_type == "pause_request" ||
         event_type == "restart_navigating" ||
         event_type == "stop_navigating" ||
         event_type == "cancel_navigating");

    RCLCPP_INFO(this->get_logger(), "Received navigation command: '%s' (type=%s)",
                command.c_str(), event_type.c_str());

    // 2) 특수 명령(상태 무관)
    if (command == "go_start" || command == "return_start") {
        if (sendRobotToStartPoint()) { publishCommandLog("GO_START: returning to start point"); nav_res->status = "success"; }
        else { publishCommandLog("ERROR: Failed to send robot to start point"); nav_res->status = "failed"; }
        return;
    }
    if (command == "return_lobby") {
        if (sendRobotToLobby()) { publishCommandLog("RETURN_LOBBY: returning to lobby station"); nav_res->status = "success"; }
        else { publishCommandLog("ERROR: Failed to send robot to lobby station"); nav_res->status = "failed"; }
        return;
    }
    if (command == "status") {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        publishCommandLog("STATUS: " + robot_info_->navigation_status +
                          " (target: " + robot_info_->current_target +
                          ", start_point: " + robot_info_->start_point_name + ")");
        nav_res->status = "status_reported";
        return;
    }
    if (command == "list") {
        publishAvailableWaypoints();
        nav_res->status = "list_sent";
        return;
    }

    // 3) 제어 이벤트
    // pause: navigating에서만
    if (event_type == "pause_request" && command == "pause") {
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            if (robot_info_->navigation_status != "navigating") {
                RCLCPP_WARN(this->get_logger(), "pause only in 'navigating' (curr=%s)", robot_info_->navigation_status.c_str());
                nav_res->status = "invalid_state";
                return;
            }
        }
        if (pauseNavigation()) {
            publishCommandLog("PAUSE: Navigation paused");
            nav_res->status = "success";
        } else {
            publishCommandLog("PAUSE: Navigation pause failed");
            nav_res->status = "failed";
        }
        return;
    }

    // resume: pause에서만
    if (event_type == "restart_navigating") {
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            if (robot_info_->navigation_status != "pause") {
                RCLCPP_WARN(this->get_logger(), "resume only in 'pause' (curr=%s)", robot_info_->navigation_status.c_str());
                nav_res->status = "invalid_state";
                return;
            }
        }
        if (resumeNavigation()) { publishCommandLog("RESUME: Navigation resumed"); nav_res->status = "success"; }
        else { publishCommandLog("RESUME: Navigation resume failed"); nav_res->status = "failed"; }
        return;
    }

    // cancel_navigating: navigating에서만 → (요구) 대기 없이 바로 복귀 = moving_to_station
    if (event_type == "cancel_navigating" && command == "cancel") {
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            if (robot_info_->navigation_status != "navigating") {
                RCLCPP_WARN(this->get_logger(), "cancel_navigating only in 'navigating' (curr=%s)", robot_info_->navigation_status.c_str());
                nav_res->status = "invalid_state";
                return;
            }
        }
        if (cancelNavigation()) {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            robot_info_->navigation_status = "moving_to_station";
            publishCommandLog("CANCEL: cancel_navigating → moving_to_station");
            nav_res->status = "success";
        } else {
            publishCommandLog("CANCEL: cancel_navigating failed");
            nav_res->status = "failed";
        }
        return;
    }

    // stop_navigating: pause에서만 → (요구) 바로 복귀 = moving_to_station
    if (event_type == "stop_navigating" && command == "cancel") {
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            if (robot_info_->navigation_status != "pause") {
                RCLCPP_WARN(this->get_logger(), "stop_navigating only in 'pause' (curr=%s)", robot_info_->navigation_status.c_str());
                nav_res->status = "invalid_state";
                return;
            }
        }
        if (cancelNavigation()) {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            robot_info_->navigation_status = "moving_to_station";
            publishCommandLog("CANCEL: stop_navigating → moving_to_station");
            nav_res->status = "success";
        } else {
            publishCommandLog("CANCEL: stop_navigating failed");
            nav_res->status = "failed";
        }
        return;
    }

    // 4) 네비게이션 이벤트 유효성
    if (!is_navigating_event && !is_control_event) {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->navigation_status != "waiting_for_navigating") {
            RCLCPP_WARN(this->get_logger(), "Invalid event_type: %s", event_type.c_str());
            nav_res->status = "invalid_event";
            return;
        }
    }
    if (is_navigating_event) {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->teleop_active) {
            RCLCPP_WARN(this->get_logger(), "Reject navigating: teleop_active");
            nav_res->status = "teleop_active";
            return;
        }
        if (robot_info_->navigation_status != "waiting_for_navigating") {
            RCLCPP_WARN(this->get_logger(),
                        "Reject navigating in state '%s' (need waiting_for_navigating)",
                        robot_info_->navigation_status.c_str());
            nav_res->status = "invalid_state";
            return;
        }
    }

    // 5) 한글/자연어 커맨드 → waypoint 키 매핑
    auto normalize = [](std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    std::string cmd_key = command;
    if (waypoints_.find(cmd_key) == waypoints_.end()) {
        static const std::unordered_map<std::string, std::string> name2key = {
            {"ct검사실","ct"}, {"ct","ct"},
            {"초음파검사실","echography"}, {"초음파","echography"},
            {"x-ray검사실","x_ray"}, {"xray","x_ray"},
            {"대장암센터","colon_cancer"},
            {"위암센터","stomach_cancer"},
            {"폐암센터","lung_cancer"},
            {"뇌종양센터","brain_tumor"},
            {"유방암센터","breast_cancer"},
            {"병원로비","lobby_station"}, {"로비","lobby_station"}
        };
        const std::string n = normalize(command);
        auto it = name2key.find(n);
        if (it != name2key.end()) cmd_key = it->second;
    }

    // 6) 최종 웨이포인트 전송
    if (waypoints_.find(cmd_key) != waypoints_.end()) {
        if (sendNavigationGoal(cmd_key)) {
            publishCommandLog("COMMAND(" + event_type + "): -> " + cmd_key);
            nav_res->status = "success";
        } else {
            publishCommandLog("ERROR: Failed to send command -> " + cmd_key);
            nav_res->status = "failed";
        }
        return;
    }

    // 7) 알 수 없는 명령 처리
    publishCommandLog("ERROR: Unknown waypoint '" + command + "'");
    publishAvailableWaypoints();
    nav_res->status = "unknown_command";
}

void RobotNavigator::navigationCommandCallback(const std_msgs::msg::String::SharedPtr msg)
{
    std::string command = msg->data;
    RCLCPP_INFO(this->get_logger(), "Received navigation command: '%s'", command.c_str());

    // 1) call_with_* : (로컬 디버그용) 상태 전이만 수행. 센터로는 절대 보내지 않음.
    if (command == "call_with_screen" || command == "call_with_voice" || command == "control_by_admin") {
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            if (robot_info_->navigation_status != "idle") {
                RCLCPP_WARN(this->get_logger(), "⚠️ 명령 [%s] 무시됨 - 현재 상태: %s", command.c_str(), robot_info_->navigation_status.c_str());
                return;
            }
            robot_info_->navigation_status = "waiting_for_navigating";
            robot_info_->current_target = "waiting_for_user";
        }
        // 로컬에서만 상태 전이했음을 명확히 표기
        publishCommandLog("CALL(local): " + command + " → 상태 전이 [waiting_for_navigating]");

        // UI/센터 구독 토픽과 동기화 위해 nav_status 즉시 퍼블리시
        std_msgs::msg::String s;
        s.data = "waiting_for_navigating";
        nav_status_publisher_->publish(s);

        return; // 여기서 종료. 중앙서버로 /robot_event 송신하지 않음
    }
    
    // 2) 복귀/특수
    if (command == "go_start" || command == "return_start") {
        if (sendRobotToStartPoint()) publishCommandLog("GO_START: returning to start point");
        else publishCommandLog("ERROR: Failed to send robot to start point");
        return;
    }
    if (command == "return_lobby") {
        if (sendRobotToLobby()) publishCommandLog("RETURN_LOBBY: returning to lobby station");
        else publishCommandLog("ERROR: Failed to send robot to lobby station");
        return;
    }

    // 3) 취소/정지 (요구: 대기 없이 바로 복귀 → cancel/stop 모두 cancelNavigation 사용)
    if (command == "cancel" || command == "stop") {
        std::string curr;
        { std::lock_guard<std::mutex> lock(robot_mutex_); curr = robot_info_->navigation_status; }

        if (command == "cancel" && curr != "navigating") {
            publishCommandLog("WARN: 'cancel' ignored in state " + curr);
            return;
        }
        if (command == "stop" && curr != "pause") {
            publishCommandLog("WARN: 'stop' ignored in state " + curr);
            return;
        }
        if (cancelNavigation()) {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            robot_info_->navigation_status = "moving_to_station";
            robot_info_->current_target = "canceled";
            publishCommandLog(std::string("CANCEL: ") + command + " → moving_to_station");

            // 상태 브로드캐스트(일관성)
            std_msgs::msg::String s;
            s.data = "moving_to_station";
            nav_status_publisher_->publish(s);
        } else {
            publishCommandLog(std::string("ERROR: ") + command + " failed");
        }
        return;
    }
    
    // 4) 조회
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
    
    // 5) 실제 네비게이션 명령 처리
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


bool RobotNavigator::sendNavigationGoal(const std::string& waypoint_name, bool keep_status)
{
    if (robot_info_->teleop_active) {
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

    const WaypointInfo& waypoint = waypoints_.at(waypoint_name);
    NavigateToPose::Goal goal_msg;
    goal_msg.pose = createPoseStamped(waypoint.x, waypoint.y, waypoint.yaw);

    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        // keep_status=true면 (예: 복귀 중) 기존 상태를 유지
        if (!keep_status) {
            robot_info_->navigation_status = "navigating";
        }
        robot_info_->current_target = waypoint_name;
    }

    // 상태/로그 퍼블리시
    if (!keep_status) {
        RCLCPP_INFO(rclcpp::get_logger("RobotNavigator"),
                    "Status changed into navigating (target=%s)", waypoint_name.c_str());
        publishCommandLog("STATE: waiting_for_navigating → navigating (target: " + waypoint_name + ")");

        std_msgs::msg::String nav_status_msg;
        nav_status_msg.data = "navigating";
        nav_status_publisher_->publish(nav_status_msg);
    } else {
        // 상태 유지 모드에서도 현재 상태를 즉시 퍼블리시해 UI가 바로 반응하도록
        std_msgs::msg::String nav_status_msg;
        {
            std::lock_guard<std::mutex> lock(robot_mutex_);
            nav_status_msg.data = robot_info_->navigation_status; // e.g., "moving_to_station"
        }
        nav_status_publisher_->publish(nav_status_msg);
    }

    // target은 항상 퍼블리시
    {
        std_msgs::msg::String target_msg;
        target_msg.data = waypoint_name;
        target_publisher_->publish(target_msg);
    }

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_goal_options.goal_response_callback =
        [this](const GoalHandleNavigate::SharedPtr& goal_handle) {
            current_goal_handle_ = goal_handle;
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


// 🔧 개선: 취소 응답 대기 + 핸들 reset + 로비 복귀까지
bool RobotNavigator::cancelNavigation()
{
    if (!current_goal_handle_) {
        RCLCPP_WARN(this->get_logger(), "No active goal to cancel");
        return false;
    }

    auto fut = nav_client_->async_cancel_goal(current_goal_handle_);
    if (fut.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
        publishCommandLog("CANCEL: cancel request timed out");
        return false;
    }
    (void)fut.get(); // 필요시 응답 코드 검증
    current_goal_handle_.reset();

    RCLCPP_WARN(this->get_logger(), "Navigation goal canceled by user");
    publishCommandLog("CANCEL: Navigation canceled by user");

    if (sendRobotToLobby()) {
        publishCommandLog("RETURN_LOBBY: returning to lobby station");
        return true;
    }
    publishCommandLog("ERROR: Failed to send robot to lobby station");
    return false;
}

bool RobotNavigator::pauseNavigation()
{
    if (is_paused_ || !current_goal_handle_) return false;

    auto fut = nav_client_->async_cancel_goal(current_goal_handle_);
    (void)fut; // 필요시 대기/검증 추가 가능

    paused_waypoint_ = robot_info_->current_target;
    current_goal_handle_.reset();
    is_paused_ = true;

    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        robot_info_->navigation_status = "pause";
    }

    publishCommandLog("PAUSE: Navigation paused at user request");
    return true;
}

bool RobotNavigator::resumeNavigation()
{
    if (!is_paused_ || paused_waypoint_.empty()) return false;
    is_paused_ = false;
    publishCommandLog("RESUME: Navigation resumed");
    return sendNavigationGoal(paused_waypoint_);
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
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Navigation feedback: distance remaining %.2fm", feedback->distance_remaining);
}

void RobotNavigator::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->navigation_status != "navigating") return;
    }

    const float front_half_angle_rad = 0.6f; // 약 ±34도
    const float block_distance_m = 0.9f;

    const float a_min = msg->angle_min;
    const float a_inc = msg->angle_increment;
    const int n = static_cast<int>(msg->ranges.size());

    int i_start = static_cast<int>(std::ceil(( -front_half_angle_rad - a_min) / a_inc));
    int i_end   = static_cast<int>(std::floor(( +front_half_angle_rad - a_min) / a_inc));
    i_start = std::max(0, std::min(n - 1, i_start));
    i_end   = std::max(0, std::min(n - 1, i_end));
    if (i_start > i_end) std::swap(i_start, i_end);

    int best_cluster_start = -1, best_cluster_end = -1;
    int cur_start = -1;

    auto is_block = [&](int idx) -> bool {
        float r = msg->ranges[idx];
        return std::isfinite(r) && r > msg->range_min && r < std::min(block_distance_m, msg->range_max);
    };

    for (int i = i_start; i <= i_end; ++i) {
        if (is_block(i)) {
            if (cur_start == -1) cur_start = i;
        } else {
            if (cur_start != -1) {
                int cur_end = i - 1;
                if (best_cluster_start == -1 || (cur_end - cur_start) > (best_cluster_end - best_cluster_start)) {
                    best_cluster_start = cur_start;
                    best_cluster_end = cur_end;
                }
                cur_start = -1;
            }
        }
    }
    if (cur_start != -1) {
        int cur_end = i_end;
        if (best_cluster_start == -1 || (cur_end - cur_start) > (best_cluster_end - best_cluster_start)) {
            best_cluster_start = cur_start;
            best_cluster_end = cur_end;
        }
    }

    if (best_cluster_start == -1) return;

    const float left_angle_rad = a_min + a_inc * static_cast<float>(best_cluster_end);
    const float right_angle_rad = a_min + a_inc * static_cast<float>(best_cluster_start);
    const float left_deg = left_angle_rad * 180.0f / static_cast<float>(M_PI);
    const float right_deg = right_angle_rad * 180.0f / static_cast<float>(M_PI);

    const rclcpp::Time now = this->get_clock()->now();
    if (obstacle_angles_available_) {
        const bool small_change = (std::fabs(left_deg - last_obstacle_left_angle_deg_) < 3.0f) &&
                                  (std::fabs(right_deg - last_obstacle_right_angle_deg_) < 3.0f);
        if (small_change && (now - last_obstacle_time_).seconds() < 1.5) return;
    }

    last_obstacle_left_angle_deg_ = left_deg;
    last_obstacle_right_angle_deg_ = right_deg;
    last_obstacle_time_ = now;
    obstacle_angles_available_ = true;

    publishCommandLog("OBSTACLE: left=" + std::to_string(left_deg) + "°, right=" + std::to_string(right_deg) + "°");
    callDetectObstacle(left_deg, right_deg);
}

void RobotNavigator::resultCallback(const GoalHandleNavigate::WrappedResult& result)
{
    std::string event_to_call;         // 락 밖에서 보낼 /robot_event
    bool should_report_obstacle = false;
    float last_left = 0.f, last_right = 0.f;

    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        switch (result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                if (robot_info_->current_target == "lobby_station") {
                    // 로비 도착 = 업무 종료
                    robot_info_->navigation_status = "idle";
                    RCLCPP_INFO(this->get_logger(), "Arrived to lobby_station → idle");
                    publishCommandLog("SUCCESS: arrived to lobby_station → idle");
                    event_to_call = "arrived_to_station";
                } else {
                    // 목적지 도착 = 환자 업무 진행, 복귀 대기
                    robot_info_->navigation_status = "waiting_for_return";
                    RCLCPP_INFO(this->get_logger(), "Robot successfully reached target: %s",
                                robot_info_->current_target.c_str());
                    publishCommandLog("SUCCESS: reached " + robot_info_->current_target);

                    // 시작점 자동 업데이트
                    robot_info_->start_point_name = robot_info_->current_target;
                    robot_info_->start_point_set = true;
                    publishCommandLog("AUTO_START: Start point updated to " + robot_info_->start_point_name);

                    event_to_call = "navigating_complete";
                }
                break;

            case rclcpp_action::ResultCode::ABORTED:
                robot_info_->navigation_status = "failed";
                RCLCPP_ERROR(this->get_logger(), "Navigation aborted");
                publishCommandLog("FAILED: navigation aborted");

                // 최근 장애물 정보가 있으면 락 밖에서 전송
                if (obstacle_angles_available_ &&
                    (this->get_clock()->now() - last_obstacle_time_).seconds() < 5.0) {
                    should_report_obstacle = true;
                    last_left = static_cast<float>(last_obstacle_left_angle_deg_);
                    last_right = static_cast<float>(last_obstacle_right_angle_deg_);
                }
                break;

            case rclcpp_action::ResultCode::CANCELED:
                // pause / moving_to_station 중에 발생한 취소는 상태 유지
                if (robot_info_->navigation_status == "pause" ||
                    robot_info_->navigation_status == "moving_to_station") {
                    RCLCPP_WARN(this->get_logger(), "Previous goal canceled while state=%s",
                                robot_info_->navigation_status.c_str());
                    publishCommandLog("CANCELED: previous goal canceled (" + robot_info_->navigation_status + ")");
                } else {
                    // 그 외 상황의 취소만 idle로 정리
                    robot_info_->navigation_status = "idle";
                    robot_info_->canceled_time = this->get_clock()->now();
                    RCLCPP_WARN(this->get_logger(), "Navigation canceled → idle");
                    publishCommandLog("CANCELED: navigation canceled → idle");
                }
                break;

            default:
                robot_info_->navigation_status = "failed";
                RCLCPP_ERROR(this->get_logger(), "Navigation unknown result");
                publishCommandLog("ERROR: navigation unknown result");
                break;
        }
    } // unlock

    // 락 밖에서 서비스 호출
    if (!event_to_call.empty()) {
        this->callEventService(event_to_call);
    }
    if (should_report_obstacle) {
        callDetectObstacle(last_left, last_right);
    }
}


void RobotNavigator::callEventService(const std::string& event_type)
{
    if (!robot_event_client_ || !robot_event_client_->wait_for_service(1s)) {
        RCLCPP_WARN(this->get_logger(), "❗ /robot_event 서비스 연결 실패");
        return;
    }

    auto request = std::make_shared<control_interfaces::srv::EventHandle::Request>();
    request->event_type = event_type;
    auto future = robot_event_client_->async_send_request(request);

    RCLCPP_INFO(this->get_logger(), "📡 /robot_event 전송: [%s]", event_type.c_str());
    try {
        auto response = future.get();
        RCLCPP_INFO(this->get_logger(), "✅ /robot_event 응답: [%s]", response->status.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "❌ 응답 수신 실패: %s", e.what());
    }
}

void RobotNavigator::callDetectObstacle(float left_angle_deg, float right_angle_deg)
{
    if (!detect_event_client_) {
        RCLCPP_WARN(this->get_logger(), "Detect service client not initialized");
        return;
    }
    if (!detect_event_client_->wait_for_service(500ms)) {
        RCLCPP_WARN(this->get_logger(), "❗ detect_obstacle 서비스가 준비되지 않음");
        return;
    }

    auto request = std::make_shared<control_interfaces::srv::DetectHandle::Request>();
    request->left_angle = left_angle_deg;
    request->right_angle = right_angle_deg;

    auto future = detect_event_client_->async_send_request(request);
    try {
        auto response = future.get();
        RCLCPP_INFO(this->get_logger(), "✅ detect_obstacle 응답 flag: [%s]", response->flag.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "❌ detect_obstacle 응답 실패: %s", e.what());
    }
}

void RobotNavigator::statusTimerCallback()
{
    // 30초 timeout 후 lobby_station 복귀 로직
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->navigation_status == "idle" &&
            robot_info_->canceled_time.nanoseconds() != 0) {
            rclcpp::Duration elapsed = this->get_clock()->now() - robot_info_->canceled_time;
            if (elapsed.seconds() >= 30.0) {
                publishCommandLog("TIMEOUT: canceled -> returning to lobby_station");
                this->callEventService("return_command");
                // 🔽 락 해제 후 호출
                // (락은 위 블록 끝에서 풀림)
            }
        }
    }
    // timeout 로직 실행은 락 밖에서
    // (현재 구현은 바로 sendRobotToLobby() 호출을 안 했는데,
    //  필요하면 위 조건에서 flag를 잡아 여기서 호출하는 형태로 변경)
    publishRobotData();
}

void RobotNavigator::publishRobotData()
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    auto time_diff = this->get_clock()->now() - robot_info_->last_update;
    bool is_online = time_diff.seconds() < 10.0;
    
    pose_publisher_->publish(robot_info_->current_pose);
    if (robot_info_->start_point_set) {
        std_msgs::msg::String start_point_msg;
        start_point_msg.data = robot_info_->start_point_name;
        start_point_publisher_->publish(start_point_msg);
    }
    velocity_publisher_->publish(robot_info_->current_velocity);
    std_msgs::msg::String nav_status_msg;
    nav_status_msg.data = robot_info_->navigation_status;
    nav_status_publisher_->publish(nav_status_msg);
    std_msgs::msg::String target_msg;
    target_msg.data = robot_info_->current_target;
    target_publisher_->publish(target_msg);
    std_msgs::msg::Bool online_msg;
    online_msg.data = is_online;
    online_status_publisher_->publish(online_msg);
    std_msgs::msg::Int32 net_level_msg;
    net_level_msg.data = robot_info_->net_signal_level;
    net_level_publisher_->publish(net_level_msg);
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
    ss << "go_start return_lobby cancel stop status list (auto start point update enabled)";
    publishCommandLog(ss.str());
}

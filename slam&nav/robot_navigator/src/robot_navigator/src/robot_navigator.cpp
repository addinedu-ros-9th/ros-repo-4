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

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // 상태 업데이트 타이머
    status_timer_ = this->create_wall_timer(
        1s, std::bind(&RobotNavigator::statusTimerCallback, this));
    
    // 서비스 클라이언트 초기화 확인
    checkServiceClients();
    
    RCLCPP_INFO(this->get_logger(), "Robot Navigator with Nearest Waypoint Start Point initialized");
    publishCommandLog("Robot Navigator started - Start point will be set to nearest waypoint");
    
    // 사용 가능한 waypoint 목록 출력
    publishAvailableWaypoints();
    
    // 디버깅: 서비스 클라이언트 상태 출력
    RCLCPP_INFO(this->get_logger(), "🔧 Debug: detect_event_client_ pointer = %p", 
               static_cast<void*>(detect_event_client_.get()));
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

    global_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "/plan", 10, 
        std::bind(&RobotNavigator::globalPathCallback, this, _1));
    
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

    robot_event_client_ = this->create_client<control_interfaces::srv::EventHandle>("/robot_event");
    detect_event_client_ = this->create_client<control_interfaces::srv::DetectHandle>("/detect_obstacle");

    // 서비스 클라이언트 생성 확인
    if (!robot_event_client_) {
        RCLCPP_ERROR(this->get_logger(), "❌ Failed to create /robot_event service client");
    } else {
        RCLCPP_INFO(this->get_logger(), "✅ /robot_event service client created");
    }
    
    if (!detect_event_client_) {
        RCLCPP_ERROR(this->get_logger(), "❌ Failed to create /detect_obstacle service client");
    } else {
        RCLCPP_INFO(this->get_logger(), "✅ /detect_obstacle service client created");
    }

    RCLCPP_INFO(this->get_logger(), "✅ Service Servers created: control_event_service, tracking_event_service, navigate_event_service");
    RCLCPP_INFO(this->get_logger(), "✅ Service Clients created: /robot_event, /detect_obstacle");
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
        if (rssi >= -60)      level = 4;
        else if (rssi >= -70) level = 3;
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
    RCLCPP_INFO(this->get_logger(),
                "[TrackEvent] type=%s, left=%.2f°, right=%.2f°",
                track_req->event_type.c_str(),
                track_req->left_angle, track_req->right_angle);

    if (track_req->event_type != "call_with_gesture") {
        RCLCPP_ERROR(this->get_logger(),
                "[TrackEvent] tracking failed. Invalid event type");
        return;
    }

    // 라디안 변환
    double left_rad = track_req->left_angle * M_PI / 180.0;
    double right_rad = track_req->right_angle * M_PI / 180.0;

    if (!latest_scan_) {
        RCLCPP_ERROR(this->get_logger(),
                "[TrackEvent] tracking failed. No scan");
        return;
    }

    // ---- TF로 로봇의 heading과 라이다 오프셋 구하기 ----
    double robot_heading = 0.0;
    double yaw_scan_in_base = 0.0;
    geometry_msgs::msg::Point robot_position;
    try {
        geometry_msgs::msg::TransformStamped map_from_base =
            tf_buffer_->lookupTransform("map", "base_link", latest_scan_->header.stamp);

        tf2::Quaternion q_base;
        tf2::fromMsg(map_from_base.transform.rotation, q_base);
        double roll_bim, pitch_bim;
        tf2::Matrix3x3(q_base).getRPY(roll_bim, pitch_bim, robot_heading);

        robot_position.x = map_from_base.transform.translation.x;
        robot_position.y = map_from_base.transform.translation.y;

        geometry_msgs::msg::TransformStamped base_from_scan =
            tf_buffer_->lookupTransform("base_link", latest_scan_->header.frame_id, latest_scan_->header.stamp);

        tf2::Quaternion q_scan;
        tf2::fromMsg(base_from_scan.transform.rotation, q_scan);
        double roll_sib, pitch_sib;
        tf2::Matrix3x3(q_scan).getRPY(roll_sib, pitch_sib, yaw_scan_in_base);
    }
    catch (tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
        return;
    }

    // 라이다 스캔 파라미터
    const float a_min = latest_scan_->angle_min;
    const float a_inc = latest_scan_->angle_increment;
    const int n = (int)latest_scan_->ranges.size();

    // 가장 가까운 장애물 찾기
    double min_distance = std::numeric_limits<double>::infinity();
    double target_angle_rad = 0.0;
    double obstacle_x = 0.0, obstacle_y = 0.0;

    for (int i = 0; i < n; ++i) {
        float r = latest_scan_->ranges[i];
        if (!std::isfinite(r) || r < latest_scan_->range_min || r > latest_scan_->range_max)
            continue;

        double laser_angle = a_min + i * a_inc; // 라이다 프레임
        double global_angle = robot_heading + yaw_scan_in_base + laser_angle;
        double relative_angle = std::atan2(std::sin(global_angle - robot_heading),
                                           std::cos(global_angle - robot_heading));

        // 서버에서 받은 각도 범위에 있는지 체크
        if (relative_angle >= right_rad && relative_angle <= left_rad) {
            if (r < min_distance) {
                min_distance = r;
                target_angle_rad = relative_angle;
                obstacle_x = robot_position.x + r * std::cos(global_angle);
                obstacle_y = robot_position.y + r * std::sin(global_angle);
            }
        }
    }

    if (!std::isfinite(min_distance)) {
        RCLCPP_ERROR(this->get_logger(), "No obstacle found in given angle range");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Closest obstacle: dist=%.2f m, rel_angle=%.2f°, world=(%.2f, %.2f)",
                min_distance, target_angle_rad * 180.0 / M_PI, obstacle_x, obstacle_y);

    // ---- cmd_vel로 이동 ----
    auto cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    const double goal_tolerance = 0.2; // m
    const double linear_speed = 0.2;   // m/s
    const double angular_speed = 0.3;  // rad/s
    const double timeout_sec = 30.0;   // 최대 이동 시간

    rclcpp::Rate rate(10);
    rclcpp::Time start_time = this->now();

    while (rclcpp::ok()) {
        // 현재 로봇 위치 및 heading 갱신
        try {
            geometry_msgs::msg::TransformStamped map_from_base =
                tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
            tf2::Quaternion q;
            tf2::fromMsg(map_from_base.transform.rotation, q);
            double pitch_h, roll_h;
            tf2::Matrix3x3(q).getRPY(pitch_h, roll_h, robot_heading);
            robot_position.x = map_from_base.transform.translation.x;
            robot_position.y = map_from_base.transform.translation.y;
        }
        catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF update failed: %s", ex.what());
            break;
        }

        // 목표까지의 거리와 방향
        double dx = obstacle_x - robot_position.x;
        double dy = obstacle_y - robot_position.y;
        double dist_to_goal = std::sqrt(dx*dx + dy*dy);
        double heading_to_goal = std::atan2(dy, dx);
        double angle_error = heading_to_goal - robot_heading;
        // -pi~pi 정규화
        while (angle_error > M_PI)  angle_error -= 2.0*M_PI;
        while (angle_error < -M_PI) angle_error += 2.0*M_PI;

        if (dist_to_goal <= goal_tolerance) {
            RCLCPP_INFO(this->get_logger(), "Reached target obstacle position");
            break;
        }

        if ((this->now() - start_time).seconds() > timeout_sec) {
            RCLCPP_WARN(this->get_logger(), "Timeout reached before goal");
            break;
        }

        geometry_msgs::msg::Twist twist;
        if (std::fabs(angle_error) > 0.1) {
            twist.angular.z = (angle_error > 0) ? angular_speed : -angular_speed;
        } else {
            twist.linear.x = linear_speed;
        }
        cmd_vel_pub->publish(twist);

        rate.sleep();
    }

    // 정지
    geometry_msgs::msg::Twist stop;
    cmd_vel_pub->publish(stop);

    robot_info_->navigation_status = "waiting_for_navigation";
    track_res->status = "waiting_for_navigation";
    track_res->distance = min_distance;
    callEventService("arrived_to_call");
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
    if ((event_type == "pause_request" || event_type == "user_disappear") && command == "pause") {
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
    if (event_type == "restart_navigating" || event_type == "user_appear") {
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

    // 1) call_with_* : idle일 때만
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
        publishCommandLog("CALL: " + command + " → 상태 전이 [waiting_for_navigating]");
        callEventService(command);
        return;
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
    if (robot_info_->start_point_set) {
        RCLCPP_ERROR(this->get_logger(), "Already in start point");
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

void RobotNavigator::globalPathCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
    current_global_path_ = *msg;
}

geometry_msgs::msg::PoseStamped RobotNavigator::getNextWaypointFromPath(const geometry_msgs::msg::Point& robot_pos)
{
    geometry_msgs::msg::PoseStamped next_waypoint;
    
    if (current_global_path_.poses.empty()) {
        return next_waypoint; // 빈 waypoint 반환
    }
    
    // 로봇과 가장 가까운 path point 찾기
    double min_dist = 0.20;
    size_t closest_idx = 0;
    
    for (size_t i = 0; i < current_global_path_.poses.size(); ++i) {
        double dx = current_global_path_.poses[i].pose.position.x - robot_pos.x;
        double dy = current_global_path_.poses[i].pose.position.y - robot_pos.y;
        double dist = std::sqrt(dx * dx + dy * dy);
        
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    // 앞쪽 몇 개 점을 다음 waypoint로 사용 (예: 10개 점 앞)
    size_t next_idx = std::min(closest_idx + 1, current_global_path_.poses.size() - 1);
    next_waypoint = current_global_path_.poses[next_idx];
    
    return next_waypoint;
}

void RobotNavigator::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    latest_scan_ = msg;
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->navigation_status != "navigating") return;
    }

    const float front_half_angle_rad = 0.428f; // 49도/2 = 24.5도 = 0.428 라디안
    const float block_distance_m = 0.9f;

    // TF2 변환을 통한 로봇 방향 및 라이다 오프셋 계산
    double robot_heading = 0.0;
    double yaw_scan_in_base = 0.0;
    double waypoint_direction = 0.0;
    
    try {
        // 스캔 시점에서의 로봇 방향 계산 (map 기준)
        geometry_msgs::msg::TransformStamped map_from_base = tf_buffer_->lookupTransform(
            "map", "base_link", msg->header.stamp);
        
        tf2::Quaternion q_base_in_map;
        tf2::fromMsg(map_from_base.transform.rotation, q_base_in_map);
        tf2::Matrix3x3 m_base_in_map(q_base_in_map);
        double roll_bim, pitch_bim;
        m_base_in_map.getRPY(roll_bim, pitch_bim, robot_heading);
        
        // 라이다 프레임의 yaw 오프셋 (base_link 기준)
        geometry_msgs::msg::TransformStamped base_from_scan = tf_buffer_->lookupTransform(
            "base_link", msg->header.frame_id, msg->header.stamp);
        
        tf2::Quaternion q_scan_in_base;
        tf2::fromMsg(base_from_scan.transform.rotation, q_scan_in_base);
        tf2::Matrix3x3 m_scan_in_base(q_scan_in_base);
        double roll_sib, pitch_sib;
        m_scan_in_base.getRPY(roll_sib, pitch_sib, yaw_scan_in_base);
        
        // Global planner가 바라보는 방향 계산 (next waypoint 방향)
        geometry_msgs::msg::Point robot_position;
        robot_position.x = map_from_base.transform.translation.x;
        robot_position.y = map_from_base.transform.translation.y;

        geometry_msgs::msg::PoseStamped next_waypoint = getNextWaypointFromPath(robot_position);
        if (next_waypoint.pose.position.x == 0.0 && next_waypoint.pose.position.y == 0.0) {
            // 다음 웨이포인트가 없으면 로봇의 현재 방향 사용
            waypoint_direction = robot_heading;
        } else {
            // 현재 로봇 위치에서 다음 웨이포인트로의 방향 계산
            geometry_msgs::msg::PoseStamped current_pose;
            current_pose.pose.position.x = map_from_base.transform.translation.x;
            current_pose.pose.position.y = map_from_base.transform.translation.y;
            
            double dx = next_waypoint.pose.position.x - current_pose.pose.position.x;
            double dy = next_waypoint.pose.position.y - current_pose.pose.position.y;
            waypoint_direction = std::atan2(dy, dx);
        }
        
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "Failed to get transform: %s", ex.what());
        return;
    }

    const float a_min = msg->angle_min;
    const float a_inc = msg->angle_increment;
    const int n = static_cast<int>(msg->ranges.size());

    // Global planner 방향 기준으로 스캔 영역 계산
    double scan_center_angle = waypoint_direction - robot_heading - yaw_scan_in_base;
    
    // 각도 정규화
    auto normalizeAngle = [](double angle) -> double {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    };
    scan_center_angle = normalizeAngle(scan_center_angle);

    // 스캔 범위 계산 (waypoint 방향 중심으로 ±24.5도)
    double scan_start_angle = scan_center_angle - front_half_angle_rad;
    double scan_end_angle = scan_center_angle + front_half_angle_rad;

    int i_start = static_cast<int>(std::ceil((scan_start_angle - a_min) / a_inc));
    int i_end   = static_cast<int>(std::floor((scan_end_angle - a_min) / a_inc));
    i_start = std::max(0, std::min(n - 1, i_start));
    i_end   = std::max(0, std::min(n - 1, i_end));
    
    // 범위가 뒤바뀐 경우 (각도가 ±π 경계를 넘나드는 경우) 처리
    if (i_start > i_end) {
        // 두 개의 구간으로 나누어 처리: [0, i_end] + [i_start, n-1]
        // 간단히 전체 범위로 확장 (더 정교한 처리가 필요하다면 별도 구현)
        i_start = 0;
        i_end = n - 1;
    }

    int best_cluster_start = -1, best_cluster_end = -1;
    int cur_start = -1;

    auto is_block = [&](int idx) -> bool {
        float r = msg->ranges[idx];
        return std::isfinite(r) && r > msg->range_min && r < std::min(block_distance_m, msg->range_max);
    };

    // 장애물 클러스터 찾기
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

    // 글로벌 좌표계 기준 각도 계산
    const float left_laser_angle_local = a_min + a_inc * static_cast<float>(best_cluster_end);
    const float right_laser_angle_local = a_min + a_inc * static_cast<float>(best_cluster_start);
    
    // 글로벌 각도 계산
    const float left_global_angle = static_cast<float>(robot_heading + yaw_scan_in_base) + left_laser_angle_local;
    const float right_global_angle = static_cast<float>(robot_heading + yaw_scan_in_base) + right_laser_angle_local;
    
    // Waypoint 방향 기준으로 상대 각도 계산 (Global planner와 동일한 기준)
    float left_angle_rad = left_global_angle - static_cast<float>(waypoint_direction);
    float right_angle_rad = right_global_angle - static_cast<float>(waypoint_direction);
    
    // 각도 정규화 (-π ~ π)
    left_angle_rad = static_cast<float>(normalizeAngle(left_angle_rad));
    right_angle_rad = static_cast<float>(normalizeAngle(right_angle_rad));
    
    const float left_deg = left_angle_rad * 180.0f / static_cast<float>(M_PI);
    const float right_deg = right_angle_rad * 180.0f / static_cast<float>(M_PI);

    // 중복 감지 방지 로직
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

    publishCommandLog("OBSTACLE: left=" + std::to_string(left_deg) + "°, right=" + std::to_string(right_deg) + "° (waypoint-relative)");
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
                    robot_info_->canceled_time = this->get_clock()->now();
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
    static bool service_checked = false;
    static bool service_available = false;
    
    // 서비스 가용성 한 번만 확인
    if (!service_checked) {
        if (!robot_event_client_) {
            RCLCPP_WARN(this->get_logger(), "detect_obstacle client not initialized");
            return;
        }
        service_available = robot_event_client_->wait_for_service(std::chrono::seconds(1));
        service_checked = true;
        if (!service_available) {
            RCLCPP_WARN(this->get_logger(), "detect_obstacle service not available");
            return;
        }
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
    static bool service_checked = false;
    static bool service_available = false;
    
    // 서비스 가용성 한 번만 확인
    if (!service_checked) {
        if (!detect_event_client_) {
            RCLCPP_WARN(this->get_logger(), "detect_obstacle client not initialized");
            return;
        }
        service_available = detect_event_client_->wait_for_service(std::chrono::seconds(1));
        service_checked = true;
        if (!service_available) {
            RCLCPP_WARN(this->get_logger(), "detect_obstacle service not available");
            return;
        }
    }
    
    if (!service_available) {
        return;
    }

    try {
        auto request = std::make_shared<control_interfaces::srv::DetectHandle::Request>();
        request->left_angle = left_angle_deg;
        request->right_angle = right_angle_deg;

        auto future = detect_event_client_->async_send_request(request);
        auto status = future.wait_for(std::chrono::seconds(3));
        if (status == std::future_status::ready) {
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "detect_obstacle response: %s", response->flag.c_str());
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "detect_obstacle failed: %s", e.what());
    }
}

void RobotNavigator::statusTimerCallback()
{
    // 30초 timeout 후 lobby_station 복귀 로직
    {
        std::lock_guard<std::mutex> lock(robot_mutex_);
        if (robot_info_->canceled_time.nanoseconds() != 0) {
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
    //publishRobotData();
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

void RobotNavigator::checkServiceClients()
{
    RCLCPP_INFO(this->get_logger(), "🔍 Checking service client connections...");
    
    // 서비스 클라이언트 가용성 확인
    if (!robot_event_client_) {
        RCLCPP_ERROR(this->get_logger(), "❌ /robot_event service client is null");
    } else if (robot_event_client_->wait_for_service(std::chrono::seconds(3))) {
        RCLCPP_INFO(this->get_logger(), "✅ /robot_event 서비스 클라이언트 연결 성공");
    } else {
        RCLCPP_WARN(this->get_logger(), "⚠️ /robot_event 서비스 클라이언트 연결 실패 (서비스 서버가 실행 중인지 확인)");
    }
    
    if (!detect_event_client_) {
        RCLCPP_ERROR(this->get_logger(), "❌ /detect_obstacle service client is null");
    } else if (detect_event_client_->wait_for_service(std::chrono::seconds(3))) {
        RCLCPP_INFO(this->get_logger(), "✅ /detect_obstacle 서비스 클라이언트 연결 성공");
    } else {
        RCLCPP_WARN(this->get_logger(), "⚠️ /detect_obstacle 서비스 클라이언트 연결 실패 (서비스 서버가 실행 중인지 확인)");
    }
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

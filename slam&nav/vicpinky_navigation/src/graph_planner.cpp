#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <functional>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "nav2_core/global_planner.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "std_msgs/msg/string.hpp"

// Forward declaration to avoid including heavy costmap header
namespace nav2_costmap_2d { class Costmap2DROS; }

namespace graph_planner
{
struct Node
{
    double x;
    double y;
    std::string name;
    std::vector<std::string> neighbors;
    bool is_blocked;  // 장애물로 인한 차단 여부
};

class GraphPlanner : public nav2_core::GlobalPlanner
{
public:
    GraphPlanner() = default;
    ~GraphPlanner() override = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override
    {
        node_ = parent.lock();
        name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;
        
        // Debug 토픽 퍼블리셔
        debug_pub_ = node_->create_publisher<std_msgs::msg::String>("/graph_planner/debug", 10);
        
        // LaserScan 구독자 설정
        scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan_filtered", 10,
            std::bind(&GraphPlanner::scanCallback, this, std::placeholders::_1));
        
        // 파라미터 설정
        // 제거된 파라미터(lookahead_distance, obstacle_threshold)는 선언 생략
        node_->declare_parameter("waypoint_radius", 0.12);
        node_->declare_parameter("forward_detection_angle", 30.0);  // 전방 감지 각도 (도)
        node_->declare_parameter("path_tolerance", 0.15);           // 경로 허용 반경
        node_->declare_parameter("node_block_radius", 0.25);        // 노드 차단 판정 반경
        node_->declare_parameter("edge_block_tolerance", 0.20);     // 엣지(노드-노드 선분) 차단 허용 반경
        node_->declare_parameter("replan_trigger_distance", 1.5);   // 재계획 트리거 최대 거리
        node_->declare_parameter("node_clear_margin", 0.15);        // 노드 클리어 여유
        node_->declare_parameter("node_clear_streak", 3);           // 연속 클리어 임계
        node_->declare_parameter("node_arrival_radius", 0.20);      // 노드 도착 판정 반경
        
        // 제거된 파라미터는 사용하지 않음
        waypoint_radius_ = node_->get_parameter("waypoint_radius").as_double();
        forward_detection_angle_ = node_->get_parameter("forward_detection_angle").as_double() * M_PI / 180.0;
        path_tolerance_ = node_->get_parameter("path_tolerance").as_double();
        node_block_radius_ = node_->get_parameter("node_block_radius").as_double();
        edge_block_tolerance_ = node_->get_parameter("edge_block_tolerance").as_double();
        replan_trigger_distance_ = node_->get_parameter("replan_trigger_distance").as_double();
        node_clear_margin_ = node_->get_parameter("node_clear_margin").as_double();
        node_clear_streak_ = node_->get_parameter("node_clear_streak").as_int();
        node_arrival_radius_ = node_->get_parameter("node_arrival_radius").as_double();
        
        RCLCPP_INFO(node_->get_logger(), "DynamicGraphPlanner configured");
        RCLCPP_INFO(node_->get_logger(), 
                "Waypoint radius: %.3f units (~%.1fcm)", 
                waypoint_radius_, waypoint_radius_ * 100.0);
        RCLCPP_INFO(node_->get_logger(), 
                "Forward detection angle: %.1f degrees, Path tolerance: %.2f units",
                forward_detection_angle_ * 180.0 / M_PI, path_tolerance_);
        initGraph();
    }

    void cleanup() override
    {
    RCLCPP_INFO(node_->get_logger(), "GraphPlanner cleanup");
    }


    void activate() override
    {
    RCLCPP_INFO(node_->get_logger(), "GraphPlanner activate");
    }


    void deactivate() override
    {
    RCLCPP_INFO(node_->get_logger(), "GraphPlanner deactivate");
    }

    nav_msgs::msg::Path createPlan(
        const geometry_msgs::msg::PoseStamped & start,
        const geometry_msgs::msg::PoseStamped & goal,
        std::function<bool()> cancel_checker) override
    {
        // 현재 로봇 위치를 TF로 최신 갱신 (실패 시 입력 start 사용)
        if (!updateRobotPoseLatest()) {
            current_robot_pose_ = start;
        }
        
        // 재계획이 필요한지 판단
        bool need_replan = shouldReplan(start, goal);
        
        if (!need_replan && !current_path_.poses.empty()) {
            // 기존 경로 유지, 인덱스만 업데이트
            size_t closest_idx = findClosestPathIndex();
            if (closest_idx != current_node_index_) {
                current_node_index_ = closest_idx;
                last_path_index_ = current_node_index_;
                
                // 디버그 출력
                std_msgs::msg::String msg;
                std::ostringstream oss;
                oss << "KEEP_PATH: nodes=" << current_path_.poses.size()
                    << ", updated_idx=" << current_node_index_;
                msg.data = oss.str();
                debug_pub_->publish(msg);
            }
            return current_path_;
        }
        
        // 재계획 필요시에만 새 경로 생성
        RCLCPP_INFO(node_->get_logger(), "Planning new path...");
        
        // 장애물 상태 업데이트
        updateObstacleStatus();
        
        // 경로 계획
        nav_msgs::msg::Path path = planPath(start, goal, cancel_checker);
        
        if (!path.poses.empty()) {
            // 현재 경로와 목표 저장
            current_path_ = path;
            current_goal_ = goal;
            current_node_index_ = findClosestPathIndex();
            last_path_index_ = current_node_index_;
            
            // 디버그 출력
            std_msgs::msg::String msg;
            std::ostringstream oss;
            oss << "NEW_PATH: nodes=" << current_path_.poses.size()
                << ", current_idx=" << current_node_index_
                << ", start=(" << start.pose.position.x << "," << start.pose.position.y << ")"
                << ", goal=(" << goal.pose.position.x << "," << goal.pose.position.y << ")";
            msg.data = oss.str();
            debug_pub_->publish(msg);
        }
        
        return path;
    }

private:
    // 재계획이 필요한지 판단하는 함수
    bool shouldReplan(const geometry_msgs::msg::PoseStamped & /* start */, 
                      const geometry_msgs::msg::PoseStamped & goal)
    {
        // 1. 첫 번째 계획이거나 기존 경로가 없는 경우
        if (current_path_.poses.empty() || current_goal_.header.frame_id.empty()) {
            return true;
        }
        
        // 2. 목표가 크게 변경된 경우 (0.3m 이상)
        double goal_distance = std::sqrt(
            std::pow(current_goal_.pose.position.x - goal.pose.position.x, 2) +
            std::pow(current_goal_.pose.position.y - goal.pose.position.y, 2)
        );
        if (goal_distance > 0.3) {
            RCLCPP_INFO(node_->get_logger(), "Goal changed significantly: %.2fm", goal_distance);
            return true;
        }
        
        // 3. 로봇이 경로에서 많이 벗어난 경우 (1.0m 이상)
        size_t closest_idx = findClosestPathIndex();
        if (closest_idx < current_path_.poses.size()) {
            double path_deviation = std::sqrt(
                std::pow(current_path_.poses[closest_idx].pose.position.x - current_robot_pose_.pose.position.x, 2) +
                std::pow(current_path_.poses[closest_idx].pose.position.y - current_robot_pose_.pose.position.y, 2)
            );
            if (path_deviation > 1.0) {
                RCLCPP_INFO(node_->get_logger(), "Robot deviated from path: %.2fm", path_deviation);
                return true;
            }
        }
        
        // 4. 경로상에 실제 장애물이 감지된 경우
        if (isPathBlocked()) {
            RCLCPP_INFO(node_->get_logger(), "Path blocked by obstacle");
            return true;
        }
        
        // 5. 강제 재계획 플래그가 설정된 경우
        if (force_replan_) {
            force_replan_ = false;  // 플래그 리셋
            RCLCPP_INFO(node_->get_logger(), "Forced replanning triggered");
            return true;
        }
        
        return false;  // 재계획 불필요
    }

    // 엣지 키(이름 순서 무관 동일 키 생성)
    std::string makeEdgeKey(const std::string& a, const std::string& b) const {
        if (a < b) return a + "|" + b;
        return b + "|" + a;
    }

    // 엣지 차단 여부(키 기반)
    bool isEdgeBlockedByKey(const std::string& a, const std::string& b) const {
        return blocked_edges_.count(makeEdgeKey(a, b)) > 0;
    }

    // 엣지(선분) 차단 판정: 라이다 포인트가 선분에서 edge_block_tolerance_ 이내인지
    bool isEdgeBlocked(const Node& n1, const Node& n2)
    {
        if (!latest_scan_) return false;
        try {
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", latest_scan_->header.frame_id, latest_scan_->header.stamp);

            // 선분 벡터
            const double x1 = n1.x, y1 = n1.y;
            const double x2 = n2.x, y2 = n2.y;
            const double vx = x2 - x1;
            const double vy = y2 - y1;
            const double c2 = vx * vx + vy * vy;
            if (c2 <= 1e-9) return false; // 동일점 방지

            for (size_t i = 0; i < latest_scan_->ranges.size(); ++i) {
                if (latest_scan_->ranges[i] < latest_scan_->range_min ||
                    latest_scan_->ranges[i] > latest_scan_->range_max) {
                    continue;
                }

                double angle = latest_scan_->angle_min + i * latest_scan_->angle_increment;
                double x_laser = latest_scan_->ranges[i] * std::cos(angle);
                double y_laser = latest_scan_->ranges[i] * std::sin(angle);

                geometry_msgs::msg::PointStamped p_laser, p_map;
                p_laser.header = latest_scan_->header;
                p_laser.point.x = x_laser;
                p_laser.point.y = y_laser;
                p_laser.point.z = 0.0;
                tf2::doTransform(p_laser, p_map, transform);

                // 선분 최근접 거리
                const double wx = p_map.point.x - x1;
                const double wy = p_map.point.y - y1;
                double t = (vx * wx + vy * wy) / c2;
                t = std::max(0.0, std::min(1.0, t));
                const double proj_x = x1 + t * vx;
                const double proj_y = y1 + t * vy;
                const double dx = p_map.point.x - proj_x;
                const double dy = p_map.point.y - proj_y;
                const double dist_seg = std::sqrt(dx * dx + dy * dy);
                if (dist_seg <= edge_block_tolerance_) {
                    return true;
                }
            }
        }
        catch (tf2::TransformException&) {
            return false;
        }
        return false;
    }
    // LaserScan 콜백 함수
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        latest_scan_ = msg;
        
        // 실시간으로 경로상 장애물 체크
        if (!current_path_.poses.empty()) {
            checkPathObstacles();
        }
    }

    // 다음 노드(웨이포인트) 선택: 계획된 노드 시퀀스 기준으로 현재 노드 도착 판정 후 다음 노드 반환
    geometry_msgs::msg::PoseStamped getNextWaypoint()
    {
    geometry_msgs::msg::PoseStamped next_waypoint;
        const auto &poses = current_path_.poses;
        if (poses.empty()) return next_waypoint;

        // 현재 노드 도착 판정
        if (current_node_index_ < poses.size()) {
            const double rx = current_robot_pose_.pose.position.x;
            const double ry = current_robot_pose_.pose.position.y;
            const double nx = poses[current_node_index_].pose.position.x;
            const double ny = poses[current_node_index_].pose.position.y;
            const double dx = nx - rx;
            const double dy = ny - ry;
            const double dist = std::sqrt(dx*dx + dy*dy);
            if (dist <= node_arrival_radius_) {
                current_node_index_ = std::min(current_node_index_ + 1, poses.size() - 1);
            }
        }

        // 다음 노드 반환
        if (current_node_index_ + 1 < poses.size()) {
            next_waypoint = poses[current_node_index_ + 1];
        }
        return next_waypoint;
    }

    // (제거됨) getCurrentHeading(): 미사용

    // 특정 타임스탬프에서 로봇의 진행 방향 계산 (시점 일치)
    double getCurrentHeadingAt(const builtin_interfaces::msg::Time & stamp)
    {
        try {
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", "base_link", stamp);
            
            tf2::Quaternion q;
            tf2::fromMsg(transform.transform.rotation, q);
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            
            return yaw;
        }
        catch (tf2::TransformException& ex) {
            RCLCPP_WARN(node_->get_logger(), "Failed to get robot heading at stamp: %s", ex.what());
            return 0.0;
        }
    }

    // 특정 타임스탬프에서 로봇 포즈를 갱신 (시점 일치)
    bool updateRobotPoseAt(const builtin_interfaces::msg::Time & stamp)
    {
        try {
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", "base_link", stamp);

            current_robot_pose_.header.frame_id = "map";
            current_robot_pose_.header.stamp = stamp;
            current_robot_pose_.pose.position.x = transform.transform.translation.x;
            current_robot_pose_.pose.position.y = transform.transform.translation.y;
            current_robot_pose_.pose.position.z = transform.transform.translation.z;
            current_robot_pose_.pose.orientation = transform.transform.rotation;
            return true;
        }
        catch (tf2::TransformException & ex) {
            RCLCPP_WARN(node_->get_logger(), "Failed to update robot pose at stamp: %s", ex.what());
            return false;
        }
    }

    // 최신 시점으로 로봇 포즈를 갱신 (TimePointZero)
    bool updateRobotPoseLatest()
    {
        try {
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", "base_link", tf2::TimePointZero);

            current_robot_pose_.header.frame_id = "map";
            current_robot_pose_.header.stamp = node_->now();
            current_robot_pose_.pose.position.x = transform.transform.translation.x;
            current_robot_pose_.pose.position.y = transform.transform.translation.y;
            current_robot_pose_.pose.position.z = transform.transform.translation.z;
            current_robot_pose_.pose.orientation = transform.transform.rotation;
            return true;
        }
        catch (tf2::TransformException & ex) {
            RCLCPP_WARN(node_->get_logger(), "Failed to update robot pose (latest): %s", ex.what());
            return false;
        }
    }

    // 다음 waypoint로의 방향 계산
    double getDirectionToWaypoint(const geometry_msgs::msg::PoseStamped& waypoint)
    {
        double dx = waypoint.pose.position.x - current_robot_pose_.pose.position.x;
        double dy = waypoint.pose.position.y - current_robot_pose_.pose.position.y;
        return std::atan2(dy, dx);
    }
    
    // 각도 차이 계산 (-π ~ π 범위로 정규화)
    double normalizeAngle(double angle)
    {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
    
        // 전방 경로상의 장애물만 체크
    bool isObstacleInForwardPath(const geometry_msgs::msg::PoseStamped& next_waypoint)
    {
        if (!latest_scan_) return false;
        
        // 스캔 시점으로 로봇 포즈 및 진행 방향 계산 (시점 일치)
        if (!updateRobotPoseAt(latest_scan_->header.stamp)) {
            return false;
        }
        double robot_heading = getCurrentHeadingAt(latest_scan_->header.stamp);
        double waypoint_direction = getDirectionToWaypoint(next_waypoint);
        
        try {
            geometry_msgs::msg::TransformStamped map_from_scan = tf_->lookupTransform(
                "map", latest_scan_->header.frame_id, latest_scan_->header.stamp);
            geometry_msgs::msg::TransformStamped base_from_scan = tf_->lookupTransform(
                "base_link", latest_scan_->header.frame_id, latest_scan_->header.stamp);

            // 라이다 프레임의 yaw 오프셋 (base_link 기준)
            tf2::Quaternion q_scan_in_base;
            tf2::fromMsg(base_from_scan.transform.rotation, q_scan_in_base);
            tf2::Matrix3x3 m_scan_in_base(q_scan_in_base);
            double roll_sib, pitch_sib, yaw_scan_in_base;
            m_scan_in_base.getRPY(roll_sib, pitch_sib, yaw_scan_in_base);
            
            for (size_t i = 0; i < latest_scan_->ranges.size(); ++i) {
                if (latest_scan_->ranges[i] < latest_scan_->range_min || 
                    latest_scan_->ranges[i] > latest_scan_->range_max) {
                    continue;
                }
                
                // 레이저 각도 계산
                double laser_angle = latest_scan_->angle_min + i * latest_scan_->angle_increment;
                double global_laser_angle = robot_heading + yaw_scan_in_base + laser_angle;
                
                // waypoint 방향 기준 설정된 각도 범위 내에 있는지 체크
                double angle_diff = normalizeAngle(global_laser_angle - waypoint_direction);
                if (std::abs(angle_diff) > forward_detection_angle_) {
                    continue; // 각도 범위 밖의 장애물은 무시
                }
                
                // 레이저 포인트를 맵 좌표계로 변환
                double x_laser = latest_scan_->ranges[i] * std::cos(laser_angle);
                double y_laser = latest_scan_->ranges[i] * std::sin(laser_angle);
                
                geometry_msgs::msg::PointStamped point_laser, point_map;
                point_laser.header = latest_scan_->header;
                point_laser.point.x = x_laser;
                point_laser.point.y = y_laser;
                point_laser.point.z = 0.0;
                
                tf2::doTransform(point_laser, point_map, map_from_scan);
                
                // 로봇-waypoint 직선 경로상에 있는지 체크
                if (isPointOnPath(point_map.point, current_robot_pose_, next_waypoint)) {
                    double distance_to_obstacle = std::sqrt(
                        std::pow(point_map.point.x - current_robot_pose_.pose.position.x, 2) +
                        std::pow(point_map.point.y - current_robot_pose_.pose.position.y, 2)
                    );
                    
                    // waypoint까지의 거리보다 가까우면 장애물로 판단
                    double distance_to_waypoint = std::sqrt(
                        std::pow(next_waypoint.pose.position.x - current_robot_pose_.pose.position.x, 2) +
                        std::pow(next_waypoint.pose.position.y - current_robot_pose_.pose.position.y, 2)
                    );
                    
                    if (distance_to_obstacle < distance_to_waypoint && 
                        distance_to_obstacle > 0.1) { // 최소 거리 필터
                        return true;
                    }
                }
            }
        }
        catch (tf2::TransformException& ex) {
            RCLCPP_WARN(node_->get_logger(), "Transform failed: %s", ex.what());
            return false;
        }
        
        return false;
    }
    
    // 점이 로봇-waypoint 선분 경로상에 있는지 체크
    bool isPointOnPath(const geometry_msgs::msg::Point& point,
                       const geometry_msgs::msg::PoseStamped& start,
                       const geometry_msgs::msg::PoseStamped& end)
    {
        // 선분에 대한 최근접 점을 사용한 거리 계산
        const double x1 = start.pose.position.x;
        const double y1 = start.pose.position.y;
        const double x2 = end.pose.position.x;
        const double y2 = end.pose.position.y;

        const double vx = x2 - x1;
        const double vy = y2 - y1;
        const double wx = point.x - x1;
        const double wy = point.y - y1;

        const double c1 = vx * wx + vy * wy;
        const double c2 = vx * vx + vy * vy;

        double t = 0.0;
        if (c2 > 0.0) {
            t = c1 / c2;
        }
        t = std::max(0.0, std::min(1.0, t));

        const double proj_x = x1 + t * vx;
        const double proj_y = y1 + t * vy;

        const double dx = point.x - proj_x;
        const double dy = point.y - proj_y;
        const double distance_to_segment = std::sqrt(dx * dx + dy * dy);

        // 선분 내부(t in [0,1])이면서 폭 내일 때만 경로상으로 인정
        return (t >= 0.0 && t <= 1.0) && (distance_to_segment <= path_tolerance_);
    }
  
    // 경로가 장애물에 의해 차단되었는지 체크 (재계획 트리거용)
    bool isPathBlocked()
    {
        if (!latest_scan_ || current_path_.poses.empty()) return false;
        
        // 시점 동기화: 스캔 타임스탬프 기준으로 로봇 포즈 갱신
        if (!updateRobotPoseAt(latest_scan_->header.stamp)) {
            return false;
        }

        // 다음 waypoint 가져오기
        geometry_msgs::msg::PoseStamped next_waypoint = getNextWaypoint();
        if (next_waypoint.header.frame_id.empty()) return false;
        
        // 전방 경로상에 장애물이 있는지 체크
        bool obstacle_detected = isObstacleInForwardPath(next_waypoint);
        
        if (obstacle_detected) {
            // 다음 웨이포인트까지 충분히 가까운 경우에만 차단으로 판단
            const double dxw = next_waypoint.pose.position.x - current_robot_pose_.pose.position.x;
            const double dyw = next_waypoint.pose.position.y - current_robot_pose_.pose.position.y;
            const double dist_to_waypoint = std::sqrt(dxw * dxw + dyw * dyw);
            
            return (dist_to_waypoint <= replan_trigger_distance_);
        }
        
        return false;
    }

    // 경로상 장애물 체크 및 재계획 트리거
    void checkPathObstacles()
    {
        if (!latest_scan_ || current_path_.poses.empty()) return;
        
        // 현재 로봇 위치 기반으로 경로상 인덱스 업데이트
        size_t closest_idx = findClosestPathIndex();
        if (closest_idx != current_node_index_) {
            current_node_index_ = closest_idx;
            last_path_index_ = current_node_index_;
        }

        // 경로 차단 여부 확인
        bool path_blocked = isPathBlocked();
        
        // 디버그 정보 출력
        std_msgs::msg::String msg;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "path_blocked=" << (path_blocked ? 1 : 0) 
            << ", node_idx=" << current_node_index_ << "/" << current_path_.poses.size()-1;
        msg.data = std::string("MONITOR: ") + oss.str();
        debug_pub_->publish(msg);
        
        // 경로가 차단되면 즉시 재계획 트리거
        if (path_blocked) {
            RCLCPP_WARN(node_->get_logger(), "Path blocked detected in scan callback - triggering replan");
            triggerReplan();
        }
    }
  
    // (제거됨) getLookaheadWaypoints(): 미사용

    // 현재 위치에서 가장 가까운 경로상 포인트 인덱스 찾기
    size_t findClosestPathIndex()
    {
        size_t closest_idx = 0;
        double min_distance = std::numeric_limits<double>::max();
        
        for (size_t i = 0; i < current_path_.poses.size(); ++i) {
            double dx = current_path_.poses[i].pose.position.x - 
                        current_robot_pose_.pose.position.x;
            double dy = current_path_.poses[i].pose.position.y - 
                        current_robot_pose_.pose.position.y;
            double distance = std::sqrt(dx*dx + dy*dy);
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_idx = i;
            }
        }
        
        return closest_idx;
    }

    // 현재 세그먼트 인덱스(진행 방향 보존용)
    size_t last_path_index_ {0};
    // 경로 노드 인덱스(다익스트라 결과 시퀀스 기준의 현재 위치)
    size_t current_node_index_ {0};

    // (제거됨) isObstacleNearWaypoint(): 미사용

    // 장애물 상태 업데이트
    void updateObstacleStatus()
    {
        // 모든 노드의 차단 상태 초기화
        for (auto& [name, node] : navigation_nodes_) {
            node.is_blocked = false;
        }

        // 현재 스캔 데이터로 노드/엣지 차단 상태 업데이트
        if (latest_scan_) {
            // 노드 차단
            for (auto& [name, node] : navigation_nodes_) {
                if (isObstacleNearPosition(node.x, node.y)) {
                    node.is_blocked = true;
                    RCLCPP_DEBUG(node_->get_logger(), 
                        "Node %s is blocked by obstacle", name.c_str());
                }
            }
            // 엣지 차단 (양방향 중복을 피하기 위해 name < neighbor인 경우만 검사)
            blocked_edges_.clear();
            for (auto& [name, node] : navigation_nodes_) {
                for (const auto& neighbor : node.neighbors) {
                    if (name < neighbor) { // 간단한 정렬 기준
                        if (isEdgeBlocked(node, navigation_nodes_[neighbor])) {
                            blocked_edges_.insert(makeEdgeKey(name, neighbor));
                            RCLCPP_DEBUG(node_->get_logger(),
                                "Edge %s <-> %s blocked by obstacle", name.c_str(), neighbor.c_str());
                        }
                    }
                }
            }
        }
    }

    // 특정 위치 근처에 장애물이 있는지 체크
    bool isObstacleNearPosition(double x, double y)
    {
        if (!latest_scan_) return false;

        try {
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", latest_scan_->header.frame_id, latest_scan_->header.stamp);
            
            for (size_t i = 0; i < latest_scan_->ranges.size(); ++i) {
                if (latest_scan_->ranges[i] < latest_scan_->range_min || 
                    latest_scan_->ranges[i] > latest_scan_->range_max) {
                    continue;
                }
                
                double angle = latest_scan_->angle_min + i * latest_scan_->angle_increment;
                double x_laser = latest_scan_->ranges[i] * std::cos(angle);
                double y_laser = latest_scan_->ranges[i] * std::sin(angle);
                
                geometry_msgs::msg::PointStamped point_laser, point_map;
                point_laser.header = latest_scan_->header;
                point_laser.point.x = x_laser;
                point_laser.point.y = y_laser;
                point_laser.point.z = 0.0;
                
                tf2::doTransform(point_laser, point_map, transform);
                
                double dx = point_map.point.x - x;
                double dy = point_map.point.y - y;
                double distance = std::sqrt(dx*dx + dy*dy);
                if (distance < node_block_radius_) return true;
            }
        }
        catch (tf2::TransformException& ex) {
            return false;
        }

        return false;
    }

    // 강제 재계획 트리거 (외부에서 호출 가능)
    void triggerReplan()
    {
        force_replan_ = true;
        RCLCPP_INFO(node_->get_logger(), "Replan triggered by external request");
    }

        // 실제 경로 계획 (차단된 노드 고려)
        nav_msgs::msg::Path planPath(
            const geometry_msgs::msg::PoseStamped & start,
            const geometry_msgs::msg::PoseStamped & goal,
            std::function<bool()> cancel_checker)
        {
        nav_msgs::msg::Path path;
        path.header.frame_id = "map";
        path.header.stamp = node_->now();

        // Dijkstra 알고리즘 (차단된 노드 제외)
        std::unordered_map<std::string, double> dist;
        std::unordered_map<std::string, std::string> prev;
        auto cmp = [&](const std::pair<double, std::string> &a, 
                        const std::pair<double, std::string> &b) {
            return a.first > b.first;
        };
        std::priority_queue<std::pair<double, std::string>,
                            std::vector<std::pair<double, std::string>>,
                            decltype(cmp)> pq(cmp);

        std::string start_node = findNearestNode(start.pose.position.x, start.pose.position.y);
        std::string goal_node = findNearestNode(goal.pose.position.x, goal.pose.position.y);

        for (auto & kv : navigation_nodes_) {
            dist[kv.first] = std::numeric_limits<double>::infinity();
        }
        dist[start_node] = 0.0;
        pq.push({0.0, start_node});

        while (!pq.empty()) {
            auto [current_dist, current] = pq.top();
            pq.pop();

            if (cancel_checker()) {
                RCLCPP_WARN(node_->get_logger(), "Path planning cancelled!");
                return path;
            }

            if (current == goal_node) break;

            // 차단된 노드는 건너뛰기
            if (navigation_nodes_[current].is_blocked) continue;

            for (auto & neighbor : navigation_nodes_[current].neighbors) {
                // 차단된 이웃 노드는 건너뛰기
                if (navigation_nodes_[neighbor].is_blocked) continue;
                // 엣지 차단이면 건너뛰기
                if (isEdgeBlockedByKey(current, neighbor)) continue;
                
                // 엣지 비용: 실제 유클리드 거리 사용
                const double ex = navigation_nodes_[neighbor].x - navigation_nodes_[current].x;
                const double ey = navigation_nodes_[neighbor].y - navigation_nodes_[current].y;
                double new_dist = current_dist + std::sqrt(ex * ex + ey * ey);
                if (new_dist < dist[neighbor]) {
                    dist[neighbor] = new_dist;
                    prev[neighbor] = current;
                    pq.push({new_dist, neighbor});
                }
            }
        }

        // 도달 불가능한 경우 빈 경로 반환
        if (!std::isfinite(dist[goal_node])) {
            RCLCPP_WARN(node_->get_logger(), 
                "No path to goal: %s (start: %s at %.2f,%.2f, goal: %s at %.2f,%.2f)", 
                goal_node.c_str(), start_node.c_str(), 
                start.pose.position.x, start.pose.position.y,
                goal_node.c_str(), goal.pose.position.x, goal.pose.position.y);
            return path;
        }

        // 역추적으로 경로 구성
        std::vector<std::string> nodes;
        for (std::string at = goal_node; !at.empty(); at = prev.count(at) ? prev[at] : "")
            nodes.push_back(at);
        std::reverse(nodes.begin(), nodes.end());

        for (auto & n : nodes) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path.header;
            pose.pose.position.x = navigation_nodes_[n].x;
            pose.pose.position.y = navigation_nodes_[n].y;
            pose.pose.orientation.w = 1.0;
            path.poses.push_back(pose);
        }

        return path;
    }

    void initGraph()
    {
        // 여기에 네가 준 노드와 양방향 엣지 등록 로직
        auto add_node = [&](std::string key, double x, double y, std::vector<std::string> neighbors, bool is_blocked=false) {
            navigation_nodes_[key] = {x, y, key, neighbors, is_blocked};
        };


        // 예: gateway_a_entrance
        add_node("gateway_a_entrance", 0, 4.03, {"gateway_a_corridor_1", "gateway_bridge_1"});
        add_node("gateway_b_entrance", -5.58, 4.03, {"gateway_bridge_6"});
        add_node("colon_cancer_entrance", 0.79, -2.17, {"colon_approach"});
        add_node("stomach_cancer_entrance", 3.65, -2.17, {"stomach_approach"});
        add_node("lung_cancer_entrance", 5.07, -2.17,{"lung_approach"});
        add_node("breast_cancer_entrance", 7.67, 1.12, {"breast_approach"});
        add_node("brain_tumor_entrance", 6.1, 1.12, {"brain_approach"});
        add_node("lobby_entrance", 9, -2.17, {"main_horizontal_corridor_9", "lobby_corridor_1"});
        add_node("gateway_a_corridor_1", 0, 3, {"gateway_a_corridor_2"});
        add_node("gateway_a_corridor_2", 0, 2, {"main_junction_north"});
        add_node("main_junction_north", 0, 1.12, {"upper_horizontal_corridor_6", "gateway_a_corridor_4"});
        add_node("gateway_a_corridor_4", 0, 0, {"gateway_a_corridor_5"});
        add_node("gateway_a_corridor_5", 0, -1, {"colon_approach"});
        add_node("colon_approach", 0, -2.17, {"main_horizontal_corridor_1"});
        add_node("main_horizontal_corridor_1", 1, -2.17,{"main_horizontal_corridor_2"});
        add_node("main_horizontal_corridor_2", 2, -2.17, {"main_horizontal_corridor_3"});
        add_node("main_horizontal_corridor_3", 3.11, -2.17, {"stomach_approach", "vertical_connector_1_mid", "main_horizontal_corridor_4"});
        add_node("main_horizontal_corridor_4", 4, -2.17, {"main_horizontal_corridor_5"});
        add_node("main_horizontal_corridor_5", 5, -2.17, {"main_horizontal_corridor_6"});
        add_node("main_horizontal_corridor_6", 5.87, -2.17, {"lung_approach", "vertical_connector_2_mid", "main_horizontal_corridor_7"});
        add_node("main_horizontal_corridor_7", 7, -2.17, {"main_horizontal_corridor_8"});
        add_node("main_horizontal_corridor_8", 8, -2.17, {"main_horizontal_corridor_9"});
        add_node("main_horizontal_corridor_9", 9, -2.17, {"lobby_entrance"});
        add_node("lobby_corridor_1", 9.26, -2, {"lobby_corridor_2"});
        add_node("lobby_corridor_2", 9.26, -1, {"lobby_corridor_3"});
        add_node("lobby_corridor_3", 9.26, 0, {"lobby_corridor_4"});
        add_node("lobby_corridor_4", 9.26, 1, {"breast_approach"});
        add_node("breast_approach", 7.67, 1, {"upper_horizontal_corridor_1"});
        add_node("upper_horizontal_corridor_1", 5.87, 1.12, {"brain_approach", "vertical_connector_2_top", "upper_horizontal_corridor_2"});
        add_node("upper_horizontal_corridor_2", 5, 1.12, {"upper_horizontal_corridor_3"});
        add_node("upper_horizontal_corridor_3", 4, 1.12, {"upper_horizontal_corridor_4"});
        add_node("upper_horizontal_corridor_4", 3.11, 1.12, {"vertical_connector_1_top", "upper_horizontal_corridor_5"});
        add_node("upper_horizontal_corridor_5", 2, 1.12, {"upper_horizontal_corridor_6"});
        add_node("upper_horizontal_corridor_6", 1, 1.12, {"upper_horizontal_corridor_5", "main_junction_north"});
        add_node("vertical_connector_1_top", 3.11, 0, {"vertical_connector_1_mid"});
        add_node("vertical_connector_1_mid", 3.11, -1, {"main_horizontal_corridor_3"});
        add_node("vertical_connector_2_top", 5.87, 0, {"vertical_connector_2_mid"});
        add_node("vertical_connector_2_mid", 5.87, -1, {"main_horizontal_corridor_6"});
        add_node("stomach_approach", 3.65, -2.17, {"main_horizontal_corridor_3"});
        add_node("lung_approach", 5.07, -2.17, {"main_horizontal_corridor_6"});
        add_node("brain_approach", 6.1, 1.12, {"upper_horizontal_corridor_1"});
        add_node("xray_entrance", -6, 4.03, {"gateway_bridge_6"});
        add_node("ct_echo_entrance", -5.58, -1.88, {"gateway_bridge_11"});
        add_node("gateway_bridge_1", -1, 4.03, {"gateway_bridge_2"});
        add_node("gateway_bridge_2", -2, 4.03, {"gateway_bridge_3"});
        add_node("gateway_bridge_3", -3, 4.03, {"gateway_bridge_4"});
        add_node("gateway_bridge_4", -4, 4.03, {"gateway_bridge_5"});
        add_node("gateway_bridge_5", -5, 4.03, {"gateway_bridge_6"});
        add_node("gateway_bridge_6", -5.58, 4.03, {"xray_entrance", "gateway_bridge_7"});
        add_node("gateway_bridge_7", -5.58, 3, {"gateway_bridge_8"});
        add_node("gateway_bridge_8", -5.58, 2, {"gateway_bridge_9"});
        add_node("gateway_bridge_9", -5.58, 1, {"gateway_bridge_10"});
        add_node("gateway_bridge_10", -5.58, 0, {"gateway_bridge_11"});
        add_node("gateway_bridge_11", -5.58, -1, {"ct_echo_entrance"});
        // (각 노드 추가 시 양방향 edge는 아래 loop로 보장)
        for (auto &[name, node] : navigation_nodes_) {
            for (auto &neighbor : node.neighbors) {
            if (std::find(navigation_nodes_[neighbor].neighbors.begin(),
                navigation_nodes_[neighbor].neighbors.end(), name)
                == navigation_nodes_[neighbor].neighbors.end()) {
                navigation_nodes_[neighbor].neighbors.push_back(name);
            }
            }
        }
    }

    std::string findNearestNode(double x, double y)
    {
        std::string nearest;
        double min_dist = std::numeric_limits<double>::max();
        for (auto &[name, node] : navigation_nodes_) {
            double dx = node.x - x;
            double dy = node.y - y;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < min_dist) {
            min_dist = dist;
            nearest = name;
            }
        }
        return nearest;
    }

    // 멤버 변수들
    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    std::unordered_map<std::string, Node> navigation_nodes_;
    std::unordered_set<std::string> blocked_edges_;

    // 새로 추가된 멤버 변수들
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;

    nav_msgs::msg::Path current_path_;
    geometry_msgs::msg::PoseStamped current_robot_pose_;
    geometry_msgs::msg::PoseStamped current_goal_;

    // 제거: lookahead_distance_, obstacle_threshold_
    double waypoint_radius_;
    double forward_detection_angle_;
    double path_tolerance_;
    double node_block_radius_;
    double edge_block_tolerance_;
    // 추가: 재계획 트리거/노드 클리어 파라미터
    double replan_trigger_distance_;
    double node_clear_margin_;
    int node_clear_streak_;
    double node_arrival_radius_;
    
    // 강제 재계획 플래그
    bool force_replan_ {false};
};
}

PLUGINLIB_EXPORT_CLASS(graph_planner::GraphPlanner, nav2_core::GlobalPlanner)


#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
#include <functional>
#include <cmath>

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
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_core/global_planner.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace graph_planner
{
struct Node
{
    double x;
    double y;
    std::string name;
    std::vector<std::string> neighbors;
    double cost;
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
        
        // LaserScan 구독자 설정
        scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan_filtered", 10,
            std::bind(&GraphPlanner::scanCallback, this, std::placeholders::_1));
        
        // 파라미터 설정
        node_->declare_parameter("lookahead_distance", 2.0);
        node_->declare_parameter("obstacle_threshold", 0.5);
        node_->declare_parameter("waypoint_radius", 0.12);
        node_->declare_parameter("forward_detection_angle", 30.0);  // 전방 감지 각도 (도)
        node_->declare_parameter("path_tolerance", 0.15);           // 경로 허용 반경
        
        lookahead_distance_ = node_->get_parameter("lookahead_distance").as_double();
        obstacle_threshold_ = node_->get_parameter("obstacle_threshold").as_double();
        waypoint_radius_ = node_->get_parameter("waypoint_radius").as_double();
        forward_detection_angle_ = node_->get_parameter("forward_detection_angle").as_double() * M_PI / 180.0;
        path_tolerance_ = node_->get_parameter("path_tolerance").as_double();
        
        RCLCPP_INFO(node_->get_logger(), "DynamicGraphPlanner configured");
        RCLCPP_INFO(node_->get_logger(), 
                "Waypoint radius: %.3f units (~%.1fcm)", 
                waypoint_radius_, waypoint_radius_ * 98.0);
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
        // 현재 로봇 위치 가져오기
        current_robot_pose_ = start;
        
        // 장애물 상태 업데이트
        //updateObstacleStatus();
        checkPathObstacles();
        
        // 경로 계획
        nav_msgs::msg::Path path = planPath(start, goal, cancel_checker);
        
        // 현재 경로와 목표 저장 (재계획을 위해)
        current_path_ = path;
        current_goal_ = goal;
        
        return path;
    }

private:
    // LaserScan 콜백 함수
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        latest_scan_ = msg;
        
        // 실시간으로 경로상 장애물 체크
        if (!current_path_.poses.empty()) {
            checkPathObstacles();
        }
    }

    // 다음 waypoint만 반환
    geometry_msgs::msg::PoseStamped getNextWaypoint()
    {
    geometry_msgs::msg::PoseStamped next_waypoint;

    if (current_path_.poses.empty()) return next_waypoint;

    // 현재 로봇 위치에서 가장 가까운 경로상 포인트 찾기
    size_t closest_idx = findClosestPathIndex();

    // 다음 waypoint 반환 (현재 위치보다 앞에 있는 첫 번째 point)
    if (closest_idx + 1 < current_path_.poses.size()) {
        next_waypoint = current_path_.poses[closest_idx + 1];
    }

    return next_waypoint;   
    }

    // 로봇의 현재 진행 방향 계산
    double getCurrentHeading()
    {
        try {
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", "base_link", tf2::TimePointZero);
            
            tf2::Quaternion q;
            tf2::fromMsg(transform.transform.rotation, q);
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            
            return yaw;
        }
        catch (tf2::TransformException& ex) {
            RCLCPP_WARN(node_->get_logger(), "Failed to get robot heading: %s", ex.what());
            return 0.0;
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
        
        // 현재 로봇 진행 방향과 다음 waypoint 방향 계산
        double robot_heading = getCurrentHeading();
        double waypoint_direction = getDirectionToWaypoint(next_waypoint);
        
        try {
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", latest_scan_->header.frame_id, latest_scan_->header.stamp);
            
            for (size_t i = 0; i < latest_scan_->ranges.size(); ++i) {
                if (latest_scan_->ranges[i] < latest_scan_->range_min || 
                    latest_scan_->ranges[i] > latest_scan_->range_max) {
                    continue;
                }
                
                // 레이저 각도 계산
                double laser_angle = latest_scan_->angle_min + i * latest_scan_->angle_increment;
                double global_laser_angle = robot_heading + laser_angle;
                
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
                
                tf2::doTransform(point_laser, point_map, transform);
                
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
    
    // 점이 로봇-waypoint 직선 경로상에 있는지 체크
    bool isPointOnPath(const geometry_msgs::msg::Point& point,
                       const geometry_msgs::msg::PoseStamped& start,
                       const geometry_msgs::msg::PoseStamped& end)
    {
        // 직선 경로에서 점까지의 거리 계산 (점-직선 거리 공식)
        double A = end.pose.position.y - start.pose.position.y;
        double B = start.pose.position.x - end.pose.position.x;
        double C = end.pose.position.x * start.pose.position.y - 
                   start.pose.position.x * end.pose.position.y;
        
        double distance_to_line = std::abs(A * point.x + B * point.y + C) / 
                                  std::sqrt(A * A + B * B);
        
        return distance_to_line <= path_tolerance_;
    }
  
    // 경로상 장애물 체크 (다음 waypoint만)
    void checkPathObstacles()
    {
        if (!latest_scan_ || current_path_.poses.empty()) return;
        
        // 다음 waypoint만 가져오기
        geometry_msgs::msg::PoseStamped next_waypoint = getNextWaypoint();
        
        if (next_waypoint.header.frame_id.empty()) return;
        
        // 전방 경로상에 장애물이 있는지만 체크
        bool obstacle_detected = isObstacleInForwardPath(next_waypoint);
        
        if (obstacle_detected) {
            RCLCPP_WARN(node_->get_logger(), 
                "Forward path obstacle detected near next waypoint (%.2f, %.2f)", 
                next_waypoint.pose.position.x, next_waypoint.pose.position.y);
            replanPath();
        }
    }
  
    // Lookahead 거리 내의 waypoint들 반환
    std::vector<geometry_msgs::msg::PoseStamped> getLookaheadWaypoints()
    {
        std::vector<geometry_msgs::msg::PoseStamped> lookahead_waypoints;
        
        if (current_path_.poses.empty()) return lookahead_waypoints;
        
        // 현재 로봇 위치에서 가장 가까운 경로상 포인트 찾기
        size_t closest_idx = findClosestPathIndex();
        
        // lookahead 거리 내의 waypoint들 수집
        double accumulated_distance = 0.0;
        for (size_t i = closest_idx; i < current_path_.poses.size() - 1; ++i) {
            double dx = current_path_.poses[i+1].pose.position.x - 
                        current_path_.poses[i].pose.position.x;
            double dy = current_path_.poses[i+1].pose.position.y - 
                        current_path_.poses[i].pose.position.y;
            accumulated_distance += std::sqrt(dx*dx + dy*dy);
            
            if (accumulated_distance > lookahead_distance_) break;
            
            lookahead_waypoints.push_back(current_path_.poses[i+1]);
        }
        
        return lookahead_waypoints;
    }

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

    // 특정 waypoint 근처에 장애물이 있는지 체크
    bool isObstacleNearWaypoint(const geometry_msgs::msg::PoseStamped& waypoint)
    {
        if (!latest_scan_) return false;
        
        // LaserScan 데이터를 waypoint 좌표계로 변환하여 체크
        try {
            // 로봇 프레임에서 맵 프레임으로 변환
            geometry_msgs::msg::TransformStamped transform = tf_->lookupTransform(
                "map", latest_scan_->header.frame_id, latest_scan_->header.stamp);
            
            for (size_t i = 0; i < latest_scan_->ranges.size(); ++i) {
                if (latest_scan_->ranges[i] < latest_scan_->range_min || 
                    latest_scan_->ranges[i] > latest_scan_->range_max) {
                    continue;
                }
                
                // 레이저 포인트를 맵 좌표계로 변환
                double angle = latest_scan_->angle_min + i * latest_scan_->angle_increment;
                double x_laser = latest_scan_->ranges[i] * std::cos(angle);
                double y_laser = latest_scan_->ranges[i] * std::sin(angle);
                
                geometry_msgs::msg::PointStamped point_laser, point_map;
                point_laser.header = latest_scan_->header;
                point_laser.point.x = x_laser;
                point_laser.point.y = y_laser;
                point_laser.point.z = 0.0;
                
                tf2::doTransform(point_laser, point_map, transform);
                
                // waypoint와의 거리 계산
                double dx = point_map.point.x - waypoint.pose.position.x;
                double dy = point_map.point.y - waypoint.pose.position.y;
                double distance = std::sqrt(dx*dx + dy*dy);
                
                // waypoint 반경 내에 장애물이 있으면 true 반환
                if (distance < waypoint_radius_) {
                    return true;
                }
            }
        }
        catch (tf2::TransformException& ex) {
            RCLCPP_WARN(node_->get_logger(), "Transform failed: %s", ex.what());
            return false;
        }
        
        return false;
    }

    // 장애물 상태 업데이트
    void updateObstacleStatus()
    {
        // 모든 노드의 차단 상태 초기화
        for (auto& [name, node] : navigation_nodes_) {
            node.is_blocked = false;
        }

        // 현재 스캔 데이터로 노드 차단 상태 업데이트
        if (latest_scan_) {
            for (auto& [name, node] : navigation_nodes_) {
                if (isObstacleNearPosition(node.x, node.y)) {
                    node.is_blocked = true;
                    RCLCPP_DEBUG(node_->get_logger(), 
                        "Node %s is blocked by obstacle", name.c_str());
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
                
                if (distance < waypoint_radius_) {
                    return true;
                }
            }
        }
        catch (tf2::TransformException& ex) {
            return false;
        }

        return false;
    }

    // 경로 재계획
    void replanPath()
    {
        if (current_goal_.header.frame_id.empty()) return;

        RCLCPP_INFO(node_->get_logger(), "Replanning path due to obstacle detection");

        // 장애물 상태 업데이트
        updateObstacleStatus();

        // 새로운 경로 계산
        auto dummy_cancel_checker = []() { return false; };
        nav_msgs::msg::Path new_path = planPath(
            current_robot_pose_, current_goal_, dummy_cancel_checker);

        if (!new_path.poses.empty()) {
            current_path_ = new_path;
            RCLCPP_INFO(node_->get_logger(), "Path replanned successfully");
        }
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
                
                double new_dist = current_dist + navigation_nodes_[neighbor].cost;
                if (new_dist < dist[neighbor]) {
                    dist[neighbor] = new_dist;
                    prev[neighbor] = current;
                    pq.push({new_dist, neighbor});
                }
            }
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
        auto add_node = [&](std::string key, double x, double y, std::vector<std::string> neighbors, double cost, bool is_blocked=false) {
            navigation_nodes_[key] = {x, y, key, neighbors, cost, is_blocked};
        };


        // 예: gateway_a_entrance
        add_node("gateway_a_entrance", 0, 4.03, {"gateway_a_corridor_1", "gateway_bridge_1"}, 1.0);
        add_node("gateway_b_entrance", -5.58, 4.03, {"gateway_bridge_6"}, 1.0);
        add_node("colon_cancer_entrance", 0.79, -2.17, {"colon_approach"}, 1.0);
        add_node("stomach_cancer_entrance", 3.65, -2.17, {"stomach_approach"}, 1.0);
        add_node("lung_cancer_entrance", 5.07, -2.17,{"lung_approach"}, 1.0);
        add_node("breast_cancer_entrance", 7.67, 1.12, {"breast_approach"}, 1.0);
        add_node("brain_tumor_entrance", 6.1, 1.12, {"brain_approach"}, 1.0);
        add_node("lobby_entrance", 9, -2.17, {"main_horizontal_corridor_9", "lobby_corridor_1"}, 1.0);
        add_node("gateway_a_corridor_1", 0, 3, {"gateway_a_corridor_2"}, 1.0);
        add_node("gateway_a_corridor_2", 0, 2, {"main_junction_north"}, 1.0);
        add_node("main_junction_north", 0, 1.12, {"upper_horizontal_corridor_6", "gateway_a_corridor_4"}, 1.0);
        add_node("gateway_a_corridor_4", 0, 0, {"gateway_a_corridor_5"}, 1.0);
        add_node("gateway_a_corridor_5", 0, -1, {"colon_approach"}, 1.0);
        add_node("colon_approach", 0, -2.17, {"main_horizontal_corridor_1"}, 1.0);
        add_node("main_horizontal_corridor_1", 1, -2.17,{"main_horizontal_corridor_2"}, 1.0);
        add_node("main_horizontal_corridor_2", 2, -2.17, {"main_horizontal_corridor_3"}, 1.0);
        add_node("main_horizontal_corridor_3", 3.11, -2.17, {"stomach_approach", "vertical_connector_1_mid", "main_horizontal_corridor_4"}, 1.0);
        add_node("main_horizontal_corridor_4", 4, -2.17, {"main_horizontal_corridor_5"}, 1.0);
        add_node("main_horizontal_corridor_5", 5, -2.17, {"main_horizontal_corridor_6"}, 1.0);
        add_node("main_horizontal_corridor_6", 5.87, -2.17, {"lung_approach", "vertical_connector_2_mid", "main_horizontal_corridor_7"}, 1.0);
        add_node("main_horizontal_corridor_7", 7, -2.17, {"main_horizontal_corridor_8"}, 1.0);
        add_node("main_horizontal_corridor_8", 8, -2.17, {"main_horizontal_corridor_9"}, 1.0);
        add_node("main_horizontal_corridor_9", 9, -2.17, {"lobby_entrance"}, 1.0);
        add_node("lobby_corridor_1", 9.26, -2, {"lobby_corridor_2"}, 1.0);
        add_node("lobby_corridor_2", 9.26, -1, {"lobby_corridor_3"}, 1.0);
        add_node("lobby_corridor_3", 9.26, 0, {"lobby_corridor_4"}, 1.0);
        add_node("lobby_corridor_4", 9.26, 1, {"breast_approach"}, 1.0);
        add_node("breast_approach", 7.67, 1, {"upper_horizontal_corridor_1"}, 1.0);
        add_node("upper_horizontal_corridor_1", 5.87, 1.12, {"brain_approach", "vertical_connector_2_top", "upper_horizontal_corridor_2"}, 1.0);
        add_node("upper_horizontal_corridor_2", 5, 1.12, {"upper_horizontal_corridor_3"}, 1.0);
        add_node("upper_horizontal_corridor_3", 4, 1.12, {"upper_horizontal_corridor_4"}, 1.0);
        add_node("upper_horizontal_corridor_4", 3.11, 1.12, {"vertical_connector_1_top", "upper_horizontal_corridor_5"}, 1.0);
        add_node("upper_horizontal_corridor_5", 2, 1.12, {"upper_horizontal_corridor_6"}, 1.0);
        add_node("upper_horizontal_corridor_6", 1, 1.12, {"upper_horizontal_corridor_5", "main_junction_north"}, 1.0);
        add_node("vertical_connector_1_top", 3.11, 0, {"vertical_connector_1_mid"}, 1.0);
        add_node("vertical_connector_1_mid", 3.11, -1, {"main_horizontal_corridor_3"}, 1.0);
        add_node("vertical_connector_2_top", 5.87, 0, {"vertical_connector_2_mid"}, 1.0);
        add_node("vertical_connector_2_mid", 5.87, -1, {"main_horizontal_corridor_6"}, 1.0);
        add_node("stomach_approach", 3.65, -2.17, {"main_horizontal_corridor_3"}, 1.0);
        add_node("lung_approach", 5.07, -2.17, {"main_horizontal_corridor_6"}, 1.0);
        add_node("brain_approach", 6.1, 1.12, {"upper_horizontal_corridor_1"}, 1.0);
        add_node("xray_entrance", -6, 4.03, {"gateway_bridge_6"}, 1.0);
        add_node("ct_echo_entrance", -5.58, -1.88, {"gateway_bridge_11"}, 1.0);
        add_node("gateway_bridge_1", -1, 4.03, {"gateway_bridge_2"}, 1.0);
        add_node("gateway_bridge_2", -2, 4.03, {"gateway_bridge_3"}, 1.0);
        add_node("gateway_bridge_3", -3, 4.03, {"gateway_bridge_4"}, 1.0);
        add_node("gateway_bridge_4", -4, 4.03, {"gateway_bridge_5"}, 1.0);
        add_node("gateway_bridge_5", -5, 4.03, {"gateway_bridge_6"}, 1.0);
        add_node("gateway_bridge_6", -5.58, 4.03, {"xray_entrance", "gateway_bridge_7"}, 1.0);
        add_node("gateway_bridge_7", -5.58, 3, {"gateway_bridge_8"}, 1.0);
        add_node("gateway_bridge_8", -5.58, 2, {"gateway_bridge_9"}, 1.0);
        add_node("gateway_bridge_9", -5.58, 1, {"gateway_bridge_10"}, 1.0);
        add_node("gateway_bridge_10", -5.58, 0, {"gateway_bridge_11"}, 1.0);
        add_node("gateway_bridge_11", -5.58, -1, {"ct_echo_entrance"}, 1.0);
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

    // 새로 추가된 멤버 변수들
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;

    nav_msgs::msg::Path current_path_;
    geometry_msgs::msg::PoseStamped current_robot_pose_;
    geometry_msgs::msg::PoseStamped current_goal_;

    double lookahead_distance_;
    double obstacle_threshold_;
    double waypoint_radius_;
    double forward_detection_angle_;
    double path_tolerance_;
};
}

PLUGINLIB_EXPORT_CLASS(graph_planner::GraphPlanner, nav2_core::GlobalPlanner)


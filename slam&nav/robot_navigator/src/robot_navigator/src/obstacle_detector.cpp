#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <control_interfaces/msg/detected_obstacle.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

class ObstacleDetector : public rclcpp::Node
{
public:
    ObstacleDetector() : Node("obstacle_detector")
    {
        // 파라미터 설정
        this->declare_parameter("min_obstacle_distance", 0.5);  // 최소 장애물 감지 거리 (미터)
        this->declare_parameter("max_allowed_time_to_collision", 1.0);  // 최대 허용 충돌 시간 (초)
        this->declare_parameter("robot_speed_threshold", 0.1);  // 로봇 속도 임계값 (m/s)
        this->declare_parameter("obstacle_detection_angle_range", 60.0);  // 장애물 감지 각도 범위 (도)
        
        min_obstacle_distance_ = this->get_parameter("min_obstacle_distance").as_double();
        max_allowed_time_to_collision_ = this->get_parameter("max_allowed_time_to_collision").as_double();
        robot_speed_threshold_ = this->get_parameter("robot_speed_threshold").as_double();
        obstacle_detection_angle_range_ = this->get_parameter("obstacle_detection_angle_range").as_double();
        
        // TF 리스너 초기화
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
        
        // 구독자 설정
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10, std::bind(&ObstacleDetector::scanCallback, this, std::placeholders::_1));
        
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10, std::bind(&ObstacleDetector::cmdVelCallback, this, std::placeholders::_1));
        
        // 발행자 설정
        obstacle_pub_ = this->create_publisher<control_interfaces::msg::DetectedObstacle>(
            "detected_obstacle", 10);
        
        // 로봇 정지 명령 발행자
        emergency_stop_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "cmd_vel", 10);
        
        RCLCPP_INFO(this->get_logger(), "장애물 감지 노드가 시작되었습니다.");
        RCLCPP_INFO(this->get_logger(), "최소 장애물 거리: %.2f m", min_obstacle_distance_);
        RCLCPP_INFO(this->get_logger(), "최대 허용 충돌 시간: %.1f 초", max_allowed_time_to_collision_);
        RCLCPP_INFO(this->get_logger(), "감지 각도 범위: %.1f도", obstacle_detection_angle_range_);
    }

private:
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
    {
        if (!scan_msg) return;
        
        // 현재 로봇 속도 확인
        double current_speed = std::sqrt(
            current_cmd_vel_.linear.x * current_cmd_vel_.linear.x + 
            current_cmd_vel_.linear.y * current_cmd_vel_.linear.y);
        
        // 로봇이 정지 상태면 장애물 감지하지 않음
        if (current_speed < robot_speed_threshold_) {
            return;
        }
        
        // 장애물 감지
        std::vector<ObstacleInfo> detected_obstacles = detectObstacles(scan_msg);
        
        // 가장 위험한 장애물 찾기 (가장 가까운 장애물)
        if (!detected_obstacles.empty()) {
            ObstacleInfo closest_obstacle = detected_obstacles[0];
            
            // 충돌 시간 계산 (속도가 0이 아닌 경우에만)
            double time_to_collision = (current_speed > 0.001) ? 
                closest_obstacle.distance / current_speed : 999.0;
            
            RCLCPP_INFO(this->get_logger(), 
                "🚨 장애물 감지: 거리=%.2fm, 각도=%.1f도, 충돌시간=%.2f초, 로봇속도=%.2fm/s", 
                closest_obstacle.distance, closest_obstacle.angle_degrees, 
                time_to_collision, current_speed);
            
            // 중앙 서버로 장애물 정보 전송
            publishObstacleInfo(closest_obstacle);
            
            // 충돌 위험 시 로봇 정지
            if (time_to_collision <= max_allowed_time_to_collision_) {
                RCLCPP_WARN(this->get_logger(), 
                    "⚠️ 충돌 위험! 로봇을 정지합니다. (충돌까지 %.2f초)", time_to_collision);
                emergencyStop();
            }
        }
    }
    
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr cmd_vel_msg)
    {
        current_cmd_vel_ = *cmd_vel_msg;
    }
    
    struct ObstacleInfo {
        double distance;      // 거리 (미터)
        double angle_radians; // 각도 (라디안)
        double angle_degrees; // 각도 (도)
        double x;            // 로컬 x 좌표
        double y;            // 로컬 y 좌표
    };
    
    std::vector<ObstacleInfo> detectObstacles(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg)
    {
        std::vector<ObstacleInfo> obstacles;
        
        // 라이다 각도 범위 계산 (라디안)
        double angle_range_rad = obstacle_detection_angle_range_ * M_PI / 180.0;
        double start_angle = -angle_range_rad / 2.0;
        double end_angle = angle_range_rad / 2.0;
        
        int detected_count = 0;
        
        for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
            double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
            
            // 감지 각도 범위 내에서만 확인 (로봇 전방)
            if (angle >= start_angle && angle <= end_angle) {
                double range = scan_msg->ranges[i];
                
                // 유효한 거리값이고 최소 거리보다 가까우면 장애물로 판단
                if (range > scan_msg->range_min && 
                    range < scan_msg->range_max && 
                    range < min_obstacle_distance_) {
                    
                    ObstacleInfo obstacle;
                    obstacle.distance = range;
                    obstacle.angle_radians = angle;
                    obstacle.angle_degrees = angle * 180.0 / M_PI;
                    
                    // 로컬 좌표계로 변환 (라이다 기준)
                    obstacle.x = range * cos(angle);
                    obstacle.y = range * sin(angle);
                    
                    obstacles.push_back(obstacle);
                    detected_count++;
                }
            }
        }
        
        // 거리순으로 정렬 (가장 가까운 것이 먼저)
        std::sort(obstacles.begin(), obstacles.end(), 
            [](const ObstacleInfo& a, const ObstacleInfo& b) {
                return a.distance < b.distance;
            });
        
        if (detected_count > 0) {
            RCLCPP_DEBUG(this->get_logger(), 
                "감지된 장애물 개수: %d개", detected_count);
        }
        
        return obstacles;
    }
    
    void publishObstacleInfo(const ObstacleInfo& obstacle)
    {
        auto obstacle_msg = control_interfaces::msg::DetectedObstacle();
        obstacle_msg.x = obstacle.x;
        obstacle_msg.y = obstacle.y;
        obstacle_msg.yaw = obstacle.angle_degrees;  // 도 단위로 전송
        
        obstacle_pub_->publish(obstacle_msg);
        
        RCLCPP_INFO(this->get_logger(), 
            "📡 장애물 정보 전송: x=%.2f, y=%.2f, yaw=%.1f도", 
            obstacle_msg.x, obstacle_msg.y, obstacle_msg.yaw);
    }
    
    void emergencyStop()
    {
        auto stop_msg = geometry_msgs::msg::Twist();
        stop_msg.linear.x = 0.0;
        stop_msg.linear.y = 0.0;
        stop_msg.linear.z = 0.0;
        stop_msg.angular.x = 0.0;
        stop_msg.angular.y = 0.0;
        stop_msg.angular.z = 0.0;
        
        emergency_stop_pub_->publish(stop_msg);
        RCLCPP_WARN(this->get_logger(), "🛑 비상 정지 명령 전송됨");
    }
    
    // 파라미터
    double min_obstacle_distance_;
    double max_allowed_time_to_collision_;
    double robot_speed_threshold_;
    double obstacle_detection_angle_range_;
    
    // 현재 로봇 속도
    geometry_msgs::msg::Twist current_cmd_vel_;
    
    // TF 관련
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // 구독자
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    
    // 발행자
    rclcpp::Publisher<control_interfaces::msg::DetectedObstacle>::SharedPtr obstacle_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr emergency_stop_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObstacleDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 
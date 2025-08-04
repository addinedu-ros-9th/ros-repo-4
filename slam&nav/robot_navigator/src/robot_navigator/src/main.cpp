#include "robot_navigator/robot_navigator.hpp"
#include <signal.h>

std::shared_ptr<RobotNavigator> g_server;

void signalHandler(int signum) {
    if (g_server) {
        RCLCPP_INFO(g_server->get_logger(), "Received signal %d. Shutting down gracefully...", signum);
        rclcpp::shutdown();
    }
    exit(signum);
}

void printUsageInstructions() {
    std::cout << "\n=== Robot Navigator with Nearest Waypoint Start Point ===" << std::endl;
    std::cout << "Navigation Commands:" << std::endl;
    std::cout << "  ros2 topic pub /navigation_command std_msgs/msg/String \"data: 'WAYPOINT_NAME'\"" << std::endl;
    std::cout << "\nAvailable Waypoints:" << std::endl;
    std::cout << "  - lobby_station : 병원 로비 스테이션" << std::endl;
    std::cout << "  - breast_cancer : 유방암 센터" << std::endl;
    std::cout << "  - brain_tumor   : 뇌종양 센터" << std::endl;
    std::cout << "  - lung_cancer   : 폐암 센터" << std::endl;
    std::cout << "  - stomach_cancer: 위암 센터" << std::endl;
    std::cout << "  - colon_cancer  : 대장암 센터" << std::endl;
    std::cout << "  - gateway_a     : 통로 A" << std::endl;
    std::cout << "  - gateway_b     : 통로 B" << std::endl;
    std::cout << "  - x_ray         : X-ray 검사실" << std::endl;
    std::cout << "  - ct            : CT 검사실" << std::endl;
    std::cout << "  - echography    : 초음파 검사실" << std::endl;
    std::cout << "\nStart Point Commands:" << std::endl;
    std::cout << "  - go_start      : Return to start point waypoint" << std::endl;
    std::cout << "\nSpecial Commands:" << std::endl;
    std::cout << "  - stop/cancel   : Cancel current navigation" << std::endl;
    std::cout << "  - status        : Check current robot status" << std::endl;
    std::cout << "  - list          : Show available waypoints" << std::endl;
    std::cout << "\nMonitoring Topics:" << std::endl;
    std::cout << "  - /navigation_commands       : Command logs" << std::endl;
    std::cout << "  - /fleet/robot1/pose         : Robot position" << std::endl;
    std::cout << "  - /fleet/robot1/start_point  : Start point waypoint name" << std::endl;
    std::cout << "  - /fleet/robot1/nav_status   : Navigation status" << std::endl;
    std::cout << "  - /fleet/robot1/target       : Current target" << std::endl;
    std::cout << "  - /fleet/robot1/velocity     : Robot velocity" << std::endl;
    std::cout << "  - /fleet/robot1/online       : Online status" << std::endl;
    std::cout << "\nExample Commands:" << std::endl;
    std::cout << "  ros2 topic pub /navigation_command std_msgs/msg/String \"data: 'lobby_station'\"" << std::endl;
    std::cout << "  ros2 topic pub /navigation_command std_msgs/msg/String \"data: 'go_start'\"" << std::endl;
    std::cout << "  ros2 topic echo /fleet/robot1/start_point" << std::endl;
    std::cout << "======================================================" << std::endl;
}

int main(int argc, char** argv)
{
    // ROS2 초기화
    rclcpp::init(argc, argv);
    
    // 신호 핸들러 등록
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        // 서버 인스턴스 생성
        g_server = std::make_shared<RobotNavigator>();
        
        // 시작 메시지
        RCLCPP_INFO(g_server->get_logger(), "==========================================================");
        RCLCPP_INFO(g_server->get_logger(), "Robot Navigator with Nearest Waypoint Start Point");
        RCLCPP_INFO(g_server->get_logger(), "==========================================================");
        
        // 사용법 출력
        printUsageInstructions();
        
        // 서버 상태 정보
        RCLCPP_INFO(g_server->get_logger(), "Navigator Status:");
        RCLCPP_INFO(g_server->get_logger(), "  - Navigation command topic: /navigation_command");
        RCLCPP_INFO(g_server->get_logger(), "  - Command log topic: /navigation_commands");
        RCLCPP_INFO(g_server->get_logger(), "  - Robot monitoring: /fleet/robot1/*");
        RCLCPP_INFO(g_server->get_logger(), "  - Start point: Nearest waypoint (auto-detected)");
        RCLCPP_INFO(g_server->get_logger(), "  - Auto navigation: DISABLED (manual control only)");
        
        RCLCPP_INFO(g_server->get_logger(), "Waiting for robot position and navigation commands...");
        RCLCPP_INFO(g_server->get_logger(), "Press Ctrl+C to shutdown");
        
        // ROS2 스핀
        rclcpp::spin(g_server);
        
    } catch (const rclcpp::exceptions::RCLError& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "RCL Error: %s", e.what());
        return 1;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception occurred: %s", e.what());
        return 1;
    } catch (...) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Unknown exception occurred");
        return 1;
    }
    
    // 정상 종료
    RCLCPP_INFO(rclcpp::get_logger("main"), "Robot Navigator stopped gracefully");
    return 0;
}
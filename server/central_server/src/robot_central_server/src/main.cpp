#include "robot_central_server/robot_central_server.hpp"
#include <signal.h>

std::shared_ptr<RobotCentralServer> g_server;

void signalHandler(int signum) {
    if (g_server) {
        RCLCPP_INFO(g_server->get_logger(), "Central Server shutting down...");
        rclcpp::shutdown();
    }
    exit(signum);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    // 신호 핸들러 등록
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    g_server = std::make_shared<RobotCentralServer>();
    
    RCLCPP_INFO(g_server->get_logger(), "Robot Central Server is running...");
    RCLCPP_INFO(g_server->get_logger(), "Available services:");
    RCLCPP_INFO(g_server->get_logger(), "  - /navigate_robot");
    RCLCPP_INFO(g_server->get_logger(), "  - /get_robot_status");
    RCLCPP_INFO(g_server->get_logger(), "  - /emergency_stop_all");
    RCLCPP_INFO(g_server->get_logger(), "Available topics:");
    RCLCPP_INFO(g_server->get_logger(), "  - /robot_status_updates");
    RCLCPP_INFO(g_server->get_logger(), "  - /system_logs");
    
    try {
        rclcpp::spin(g_server);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(g_server->get_logger(), "Exception in main loop: %s", e.what());
    }
    
    RCLCPP_INFO(g_server->get_logger(), "Central Server stopped");
    return 0;
}
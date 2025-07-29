#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include "central_server/central_server.h"

std::shared_ptr<CentralServer> server_node;

void signalHandler(int signum) {
    if (server_node) {
        server_node->stop();
    }
    rclcpp::shutdown();
    exit(0);
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    try {
        RCLCPP_INFO(rclcpp::get_logger("central_server"), "[main] Before make_shared, server_node ptr: %p", server_node.get());
        server_node = std::make_shared<CentralServer>();
        RCLCPP_INFO(rclcpp::get_logger("central_server"), "[main] After make_shared, server_node ptr: %p", server_node.get());
        RCLCPP_INFO(rclcpp::get_logger("central_server"), "[main] Before init()");
        server_node->init();
        RCLCPP_INFO(rclcpp::get_logger("central_server"), "[main] After init()");

        RCLCPP_INFO(server_node->get_logger(), "중앙서버 시작중...");

        server_node->start();

        RCLCPP_INFO(server_node->get_logger(), "중앙서버 시작 완료!");

        rclcpp::spin(server_node);

    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("central_server"), "서버 시작 실패: %s", e.what());
        return 1;
    }

    RCLCPP_INFO(rclcpp::get_logger("central_server"), "중앙서버 종료됨");
    return 0;
}
/*
#include "simple_central_server/simple_central_server.hpp"
#include <signal.h>

std::shared_ptr<SimpleCentralServer> g_server;

void signalHandler(int signum) {
    if (g_server) {
        RCLCPP_INFO(g_server->get_logger(), "Received signal %d. Shutting down gracefully...", signum);
        rclcpp::shutdown();
    }
    exit(signum);
}

void printUsageInstructions() {
    std::cout << "\n=== Simple Central Server Usage ===" << std::endl;
    std::cout << "Navigation Commands:" << std::endl;
    std::cout << "  ros2 topic pub /navigation_command std_msgs/msg/String \"data: 'WAYPOINT_NAME'\"" << std::endl;
    std::cout << "\nAvailable Waypoints:" << std::endl;
    std::cout << "  - home          : Home Position (0.0, 0.0, 0.0°)" << std::endl;
    std::cout << "  - reception     : Reception Desk (1.5, 1.0, 0.0°)" << std::endl;
    std::cout << "  - lobby         : Main Lobby (-1.5, 1.0, 180.0°)" << std::endl;
    std::cout << "  - meeting_room  : Meeting Room (2.0, -1.0, 90.0°)" << std::endl;
    std::cout << "  - elevator      : Elevator (-1.0, -2.0, 270.0°)" << std::endl;
    std::cout << "  - cafeteria     : Cafeteria (2.5, -1.8, 90.0°)" << std::endl;
    std::cout << "  - information   : Information Desk (0.0, -1.0, 90.0°)" << std::endl;
    std::cout << "\nSpecial Commands:" << std::endl;
    std::cout << "  - stop/cancel   : Cancel current navigation" << std::endl;
    std::cout << "  - status        : Check current robot status" << std::endl;
    std::cout << "  - list          : Show available waypoints" << std::endl;
    std::cout << "\nMonitoring Topics:" << std::endl;
    std::cout << "  - /navigation_commands       : Command logs" << std::endl;
    std::cout << "  - /fleet/robot1/pose         : Robot position" << std::endl;
    std::cout << "  - /fleet/robot1/nav_status   : Navigation status" << std::endl;
    std::cout << "  - /fleet/robot1/target       : Current target" << std::endl;
    std::cout << "  - /fleet/robot1/velocity     : Robot velocity" << std::endl;
    std::cout << "  - /fleet/robot1/online       : Online status" << std::endl;
    std::cout << "\nExample Commands:" << std::endl;
    std::cout << "  ros2 topic pub /navigation_command std_msgs/msg/String \"data: 'reception'\"" << std::endl;
    std::cout << "  ros2 topic pub /navigation_command std_msgs/msg/String \"data: 'stop'\"" << std::endl;
    std::cout << "  ros2 topic echo /navigation_commands" << std::endl;
    std::cout << "================================" << std::endl;
}

int main(int argc, char** argv)
{
    // ROS2 초기화
    rclcpp::init(argc, argv);
    
    // 신호 핸들러 등록 (Ctrl+C 등 안전한 종료를 위해)
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        // 서버 인스턴스 생성
        g_server = std::make_shared<SimpleCentralServer>();
        
        // 시작 메시지
        RCLCPP_INFO(g_server->get_logger(), "=================================================");
        RCLCPP_INFO(g_server->get_logger(), "Simple Central Server is now running...");
        RCLCPP_INFO(g_server->get_logger(), "=================================================");
        
        // 사용법 출력
        printUsageInstructions();
        
        // 서버 상태 정보
        RCLCPP_INFO(g_server->get_logger(), "Server Status:");
        RCLCPP_INFO(g_server->get_logger(), "  - Navigation command topic: /navigation_command");
        RCLCPP_INFO(g_server->get_logger(), "  - Command log topic: /navigation_commands");
        RCLCPP_INFO(g_server->get_logger(), "  - Robot monitoring: /fleet/robot1/*");
        RCLCPP_INFO(g_server->get_logger(), "  - Auto navigation: DISABLED (manual control only)");
        
        RCLCPP_INFO(g_server->get_logger(), "Waiting for navigation commands...");
        RCLCPP_INFO(g_server->get_logger(), "Press Ctrl+C to shutdown");
        
        // ROS2 스핀 (메인 이벤트 루프)
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
    RCLCPP_INFO(rclcpp::get_logger("main"), "Simple Central Server stopped gracefully");
    return 0;
}
*/
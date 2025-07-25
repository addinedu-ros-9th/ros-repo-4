#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include "central_server/central_server.h"

std::shared_ptr<CentralServer> server_node;

void signalHandler(int signum) {
    RCLCPP_INFO(rclcpp::get_logger("central_server"), "시그널 받음 (%d). 서버 종료중...", signum);
    if (server_node) {
        server_node->stop();
    }
    rclcpp::shutdown();
    exit(0);
}

int main(int argc, char* argv[]) {
    // ROS2 초기화
    rclcpp::init(argc, argv);
    
    // 시그널 핸들러 설정
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        // 중앙서버 노드 생성
        server_node = std::make_shared<CentralServer>();
        
        RCLCPP_INFO(server_node->get_logger(), "중앙서버 시작중...");
        
        // 서버 시작
        server_node->start();
        
        RCLCPP_INFO(server_node->get_logger(), "중앙서버 시작 완료!");
        
        // ROS2 spin
        rclcpp::spin(server_node);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("central_server"), "서버 시작 실패: %s", e.what());
        return 1;
    }
    
    RCLCPP_INFO(rclcpp::get_logger("central_server"), "중앙서버 종료됨");
    return 0;
} 
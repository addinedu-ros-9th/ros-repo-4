#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include "ai_server/ai_server.h"

std::shared_ptr<AIServer> ai_server_node;

void signalHandler(int signum) {
    RCLCPP_INFO(rclcpp::get_logger("ai_server"), "시그널 받음 (%d). AI 서버 종료중...", signum);
    if (ai_server_node) {
        ai_server_node->stop();
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
        // AI 서버 노드 생성
        ai_server_node = std::make_shared<AIServer>();
        
        RCLCPP_INFO(ai_server_node->get_logger(), "AI 서버 초기화 중...");
        
        // 초기화 (shared_from_this 사용하는 부분)
        ai_server_node->initialize();
        
        RCLCPP_INFO(ai_server_node->get_logger(), "AI 서버 시작중...");
        
        // 서버 시작
        ai_server_node->start();
        
        RCLCPP_INFO(ai_server_node->get_logger(), "AI 서버 시작 완료!");
        
        // ROS2 spin
        rclcpp::spin(ai_server_node);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("ai_server"), "AI 서버 시작 실패: %s", e.what());
        return 1;
    }
    
    RCLCPP_INFO(rclcpp::get_logger("ai_server"), "AI 서버 종료됨");
    return 0;
}

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
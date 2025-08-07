#include "central_server/robot_request_handler.h"
#include <iostream>
#include <json/json.h>

RobotRequestHandler::RobotRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                                       std::shared_ptr<RobotNavigationManager> nav_manager)
    : db_manager_(db_manager), nav_manager_(nav_manager) {
}

// 로봇 관련 API 핸들러들 (central → GUI)

std::string RobotRequestHandler::handleAlertOccupied(const Json::Value& request) {
    std::cout << "[ROBOT] 환자 사용중 블락 알림 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: GUI 클라이언트들에게 환자 사용중 상태 브로드캐스트
    // WebSocket을 통해 실시간 알림 전송
    
    std::cout << "[ROBOT] 환자 사용중 블락 알림 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string RobotRequestHandler::handleAlertIdle(const Json::Value& request) {
    std::cout << "[ROBOT] 사용 가능한 상태 알림 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: GUI 클라이언트들에게 사용 가능한 상태 브로드캐스트
    // WebSocket을 통해 실시간 알림 전송
    
    std::cout << "[ROBOT] 사용 가능한 상태 알림 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string RobotRequestHandler::handleNavigatingComplete(const Json::Value& request) {
    std::cout << "[ROBOT] 길안내 완료 알림 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: GUI 클라이언트들에게 길안내 완료 상태 브로드캐스트
    // WebSocket을 통해 실시간 알림 전송
    
    std::cout << "[ROBOT] 길안내 완료 알림 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

// 유틸리티 함수들

std::string RobotRequestHandler::createErrorResponse(const std::string& message) {
    Json::Value response;
    response["error"] = message;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string RobotRequestHandler::createStatusResponse(int status_code) {
    Json::Value response;
    response["status_code"] = status_code;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
} 
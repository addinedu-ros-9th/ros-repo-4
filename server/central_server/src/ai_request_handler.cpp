#include "central_server/ai_request_handler.h"
#include <iostream>
#include <ctime>

AiRequestHandler::AiRequestHandler(std::shared_ptr<DatabaseManager> db_manager,
                                   std::shared_ptr<RobotNavigationManager> nav_manager,
                                   std::shared_ptr<WebSocketServer> websocket_server)
    : db_manager_(std::move(db_manager)), nav_manager_(std::move(nav_manager)), websocket_server_(std::move(websocket_server)) {}

void AiRequestHandler::setRobotNavigationManager(std::shared_ptr<RobotNavigationManager> nav_manager) {
    nav_manager_ = std::move(nav_manager);
}

void AiRequestHandler::setWebSocketServer(std::shared_ptr<WebSocketServer> websocket_server) {
    websocket_server_ = std::move(websocket_server);
}

std::string AiRequestHandler::createStatusResponse(int status_code) {
    Json::Value response;
    response["status_code"] = status_code;
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AiRequestHandler::createErrorResponse(const std::string& message) {
    Json::Value response;
    response["error"] = message;
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

// IF-03: /gesture/come
std::string AiRequestHandler::handleGestureCome(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("left_angle") || !request.isMember("right_angle") || !request.isMember("timestamp")) {
        return createErrorResponse("Missing robot_id, left_angle, right_angle, or timestamp");
    }

    int robot_id = request["robot_id"].asInt();
    float left_angle = std::stof(request["left_angle"].asString());
    float right_angle = std::stof(request["right_angle"].asString());
    long long timestamp = std::stoll(request["timestamp"].asString());

    if (!nav_manager_) {
        return createErrorResponse("Navigation manager not available");
    }

    // 로봇에 call_wtih_gesture 이벤트 전송 (angles 포함)
    bool sent = nav_manager_->sendTrackingEvent("call_wtih_gesture", left_angle, right_angle);
    if (!sent) {
        return createErrorResponse("Failed to send tracking event");
    }

    // 웹소켓으로 관리자에게 alert_occupied 메시지 전송
    if (websocket_server_) {
        websocket_server_->sendAlertOccupied(robot_id, "admin");
    }

    // DB 로깅: patient_id = null, admin_id = ""
    if (db_manager_) {
        std::string current_datetime = db_manager_->getCurrentDateTime();
        if (!current_datetime.empty()) {
            int* null_patient = nullptr;
            bool log_ok = db_manager_->insertRobotLogWithType(
                robot_id, null_patient, current_datetime, 0, 0, "call_wtih_gesture", "");
            if (!log_ok) {
                std::cout << "[AI] call_wtih_gesture 로그 저장 실패" << std::endl;
            }
        }
    }

    return createStatusResponse(200);
}

// IF-04: /user_disappear
std::string AiRequestHandler::handleUserDisappear(const Json::Value& request) {
    if (!request.isMember("robot_id")) {
        return createErrorResponse("Missing robot_id");
    }

    int robot_id = request["robot_id"].asInt();

    if (!nav_manager_) {
        return createErrorResponse("Navigation manager not available");
    }

    // 1. 로봇에 user_disappear 이벤트 전송
    bool sent = nav_manager_->sendControlEvent("user_disappear");
    if (!sent) {
        return createErrorResponse("Failed to send user_disappear event");
    }

    // 2. 웹소켓으로 GUI에게 user_disappear 이벤트 전송
    if (websocket_server_) {
        Json::Value message;
        message["type"] = "user_disappear";
        message["robot_id"] = robot_id;
        message["timestamp"] = std::to_string(time(nullptr));

        Json::StreamWriterBuilder builder;
        std::string json_message = Json::writeString(builder, message);
        websocket_server_->broadcastMessageToType("gui", json_message);
    }

    // 3. DB 로깅: patient_id = null, admin_id = ""
    if (db_manager_) {
        std::string current_datetime = db_manager_->getCurrentDateTime();
        if (!current_datetime.empty()) {
            int* null_patient = nullptr;
            bool log_ok = db_manager_->insertRobotLogWithType(
                robot_id, null_patient, current_datetime, 0, 0, "user_disappear", "");
            if (!log_ok) {
                std::cout << "[AI] user_disappear 로그 저장 실패" << std::endl;
            }
        }
    }



    return createStatusResponse(200);
}

// IF-05: /user_appear
std::string AiRequestHandler::handleUserAppear(const Json::Value& request) {
    if (!request.isMember("robot_id")) {
        return createErrorResponse("Missing robot_id");
    }

    int robot_id = request["robot_id"].asInt();



    if (!nav_manager_) {
        return createErrorResponse("Navigation manager not available");
    }

    // 2. 로봇에 restart_navigating 이벤트 전송
    bool sent = nav_manager_->sendControlEvent("user_appear");
    if (!sent) {
        return createErrorResponse("Failed to send user_appear event");
    }

    // 3. 웹소켓으로 GUI에게 user_appear 이벤트 전송
    if (websocket_server_) {
        Json::Value message;
        message["type"] = "user_appear";
        message["robot_id"] = robot_id;
        message["timestamp"] = std::to_string(time(nullptr));

        Json::StreamWriterBuilder builder;
        std::string json_message = Json::writeString(builder, message);
        websocket_server_->broadcastMessageToType("gui", json_message);
    }

    // 4. DB 로깅: patient_id = null, admin_id = ""
    if (db_manager_) {
        std::string current_datetime = db_manager_->getCurrentDateTime();
        if (!current_datetime.empty()) {
            int* null_patient = nullptr;
            bool log_ok = db_manager_->insertRobotLogWithType(
                robot_id, null_patient, current_datetime, 0, 0, "user_appear", "");
            if (!log_ok) {
                std::cout << "[AI] user_appear 로그 저장 실패" << std::endl;
            }
        }
    }

    return createStatusResponse(200);
}

// IF-06: /stop_tracking
std::string AiRequestHandler::handleStopTracking(const Json::Value& request) {
    if (!request.isMember("robot_id")) {
        return createErrorResponse("Missing robot_id");
    }

    int robot_id = request["robot_id"].asInt();

    if (!nav_manager_) {
        return createErrorResponse("Navigation manager not available");
    }

    // 1. 로봇에 return_command 이벤트 전송
    bool sent = nav_manager_->sendControlEvent("return_command");
    if (!sent) {
        return createErrorResponse("Failed to send return_command event");
    }

    // 2. DB 로깅: stop_tracking 이벤트 기록 (patient_id = NULL, 대기장소 dest=8)
    if (db_manager_) {
        std::string current_datetime = db_manager_->getCurrentDateTime();
        if (!current_datetime.empty()) {
            int* null_patient = nullptr;
            bool log_ok = db_manager_->insertRobotLogWithType(
                robot_id, null_patient, current_datetime, 0, 8, "stop_tracking", "");
            if (!log_ok) {
                std::cout << "[AI] stop_tracking 로그 저장 실패" << std::endl;
            }
        }
    }

    // 3. 성공 응답 반환
    return createStatusResponse(200);
}


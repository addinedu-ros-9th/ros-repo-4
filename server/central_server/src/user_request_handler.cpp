#include "central_server/user_request_handler.h"
#include "central_server/config.h"
#include "central_server/websocket_server.h"
#include <iostream>
#include <sstream>
#include <json/json.h>


UserRequestHandler::UserRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                                     std::shared_ptr<RobotNavigationManager> nav_manager)
    : db_manager_(db_manager), nav_manager_(nav_manager) {
    // WebSocket 서버 초기화 및 시작
    websocket_server_ = std::make_unique<WebSocketServer>(Config::GUI_PORT);
    if (!websocket_server_->start()) {
        std::cerr << "[USER] WebSocket 서버 시작 실패" << std::endl;
    } else {
        std::cout << "[USER] WebSocket 서버 시작 완료 (포트: " << Config::GUI_PORT << ")" << std::endl;
    }
}

// User GUI API 핸들러들

std::string UserRequestHandler::handleAuthSSN(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("ssn")) {
        return createErrorResponse("Missing robot_id or ssn");
    }
    
    std::string ssn = request["ssn"].asString();
    
    PatientInfo patient;
    if (db_manager_->getPatientBySSN(ssn, patient)) {
        return handleCommonAuth(patient);
    } else {
        return createErrorResponse("Patient not found");
    }
}

std::string UserRequestHandler::handleAuthPatientId(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return createErrorResponse("Missing robot_id or patient_id");
    }
    
    int patient_id = request["patient_id"].asInt();
    
    PatientInfo patient;
    if (db_manager_->getPatientById(patient_id, patient)) {
        return handleCommonAuth(patient);
    } else {
        return createErrorResponse("Patient not found");
    }
}

std::string UserRequestHandler::handleAuthRFID(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("rfid")) {
        return createErrorResponse("Missing robot_id or rfid");
    }
    
    std::string rfid = request["rfid"].asString();
    
    PatientInfo patient;
    if (db_manager_->getPatientByRFID(rfid, patient)) {
        return handleCommonAuth(patient);
    } else {
        return createErrorResponse("Patient not found");
    }
}

std::string UserRequestHandler::handleAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("department_id") || !request.isMember("patient_id")) {
        return createErrorResponse("Missing robot_id, department_id, or patient_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int department_id = request["department_id"].asInt();
    int patient_id = request["patient_id"].asInt();
    
    return processDirectionRequest(robot_id, department_id, &patient_id, "moving_by_patient");
}

std::string UserRequestHandler::handleRobotReturn(const Json::Value& request) {
    std::cout << "[USER] 로봇 복귀 요청 처리 (IF-06)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id, patient_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string patient_id_str = request["patient_id"].asString();
    int patient_id = std::stoi(patient_id_str);
    
    return processRobotReturnRequest(robot_id, &patient_id, "return_by_patient");
}

std::string UserRequestHandler::handleWithoutAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("department_id")) {
        return createErrorResponse("Missing robot_id or department_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int department_id = request["department_id"].asInt();
    
    return processDirectionRequest(robot_id, department_id, nullptr, "moving_by_unknown");
}

std::string UserRequestHandler::handleWithoutAuthRobotReturn(const Json::Value& request) {
    std::cout << "[USER] 비인증 로봇 복귀 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    return processRobotReturnRequest(robot_id, nullptr, "return_by_unknown");
}

std::string UserRequestHandler::handleRobotStatus(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("status")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string status = request["status"].asString();
    
    // GUI에서 화면을 터치했을 때 보내는 요청으로 로봇은 즉시 이동을 멈춰야 함
    if (nav_manager_) {
        if (!nav_manager_->sendStopCommand()) {
            return "500"; // Internal Server Error
        }
        std::cout << "[USER] 로봇 정지 명령 전송 완료: Robot " << robot_id << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return "200"; // 성공
}

std::string UserRequestHandler::handleGetLLMConfig(const Json::Value&) {
    Json::Value response;
    response["ip"] = Config::LLM_SERVER_IP;  // LLM 서버 IP
    response["port"] = Config::LLM_SERVER_PORT;
    return response.toStyledString();
}

std::string UserRequestHandler::handleCallWithVoice(const Json::Value& request) {
    std::cout << "[USER] 음성 호출 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 음성 호출 명령 전송
    std::cout << "[USER] 음성 호출 명령 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string UserRequestHandler::handleCallWithScreen(const Json::Value& request) {
    std::cout << "[USER] 화면 호출 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 화면 호출 명령 전송
    std::cout << "[USER] 화면 호출 명령 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string UserRequestHandler::handleAlertTimeout(const Json::Value& request) {
    std::cout << "[USER] 30초 타임아웃 알림 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 타임아웃 처리
    std::cout << "[USER] 타임아웃 처리 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string UserRequestHandler::handlePauseRequest(const Json::Value& request) {
    std::cout << "[USER] 일시정지 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 일시정지 명령 전송
    std::cout << "[USER] 일시정지 명령 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string UserRequestHandler::handleRestartNavigation(const Json::Value& request) {
    std::cout << "[USER] 길안내 재개 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 길안내 재개 명령 전송
    std::cout << "[USER] 길안내 재개 명령 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string UserRequestHandler::handleStopNavigating(const Json::Value& request) {
    std::cout << "[USER] 길안내 중지 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 길안내 중지 명령 전송
    std::cout << "[USER] 길안내 중지 명령 전송 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

// 공통 인증 로직
std::string UserRequestHandler::handleCommonAuth(const PatientInfo& patient) {
    // 오늘 날짜의 "예약" 상태인 건을 조회
    SeriesInfo series;
    std::string department_name;
    if (db_manager_->getTodayReservationWithDepartmentName(patient.patient_id, series, department_name)) {
        // 변경 전 상태 저장
        std::string original_status = series.status;
        
        // status가 '예약' 상태일 때 '접수'로 변경
        if (series.status == "예약") {
            if (db_manager_->updateSeriesStatus(patient.patient_id, series.reservation_date, "접수")) {
                std::cout << "[USER] 환자 상태 변경: 예약 -> 접수" << std::endl;
            }
        }
        
        // dttm 값을 그대로 사용 (YYYY-MM-DD HH:MM:SS 형식)
        std::string datetime = series.dttm;
        
        return createSuccessResponse(patient.name, datetime, department_name, original_status, patient.patient_id);
    } else {
        // Series 정보가 없는 경우
        return createErrorResponse("Reservation not found");
    }
}

// 네비게이션 처리 함수들
std::string UserRequestHandler::processDirectionRequest(int robot_id, int department_id, int* patient_id, const std::string& log_type) {
    // 1. department_id를 department_name으로 변환
    DepartmentInfo department;
    if (!db_manager_->getDepartmentById(department_id, department)) {
        return createErrorResponse("Department not found: " + std::to_string(department_id));
    }
    
    // 2. 로봇에게 navigation 명령 전송
    if (nav_manager_) {
        if (!nav_manager_->sendWaypointCommand(department.department_name)) {
            return createErrorResponse("Failed to send navigation command");
        }
        std::cout << "[USER] 로봇 네비게이션 명령 전송 완료: " << department.department_name << "로 이동" << std::endl;
    } else {
        return createErrorResponse("Navigation manager not available");
    }
    
    // 3. 현재 로봇 위치 확인 (읽기만 하므로 mutex 불필요)
    double current_x, current_y;
    // TODO: nav_manager_에서 현재 로봇 위치 정보를 가져와야 함
    current_x = 0.0;  // 더미 데이터
    current_y = 0.0;  // 더미 데이터
    
    // 4. 가장 가까운 부서 찾기 (출발지)
    int orig_department_id = db_manager_->findNearestDepartment(current_x, current_y);
    
    if (orig_department_id == -1) {
        return createErrorResponse("Failed to find nearest department");
    }
    
    // 5. robot_log에 insert
    std::string current_datetime = db_manager_->getCurrentDate();
    if (current_datetime.empty()) {
        return createErrorResponse("Failed to get current datetime");
    }
    
    // patient_id 저장 (nullptr이면 NULL, 아니면 실제 patient_id)
    bool success = db_manager_->insertRobotLogWithType(robot_id, patient_id, current_datetime, 
                                                     orig_department_id, department_id, log_type);
    
    if (!success) {
        return createErrorResponse("Failed to insert robot log");
    }
    
    std::cout << "[USER] 네비게이션 명령 처리 완료: Robot " << robot_id 
              << " -> Department " << department.department_name << " (ID: " << department_id 
              << "), Patient: " << (patient_id ? std::to_string(*patient_id) : "NULL") 
              << ", Type: " << log_type << std::endl;
    
    return createStatusResponse(200);
}

std::string UserRequestHandler::processRobotReturnRequest(int robot_id, int* patient_id, const std::string& log_type) {
    // 1. 현재 로봇 위치 확인 (amcl_pose)
    double current_x, current_y;
    // TODO: nav_manager_에서 현재 로봇 위치 정보를 가져와야 함
    current_x = 0.0;  // 더미 데이터
    current_y = 0.0;  // 더미 데이터
    
    // 2. 현재 위치에서 가장 가까운 부서 찾기 (orig)
    int orig_department_id = db_manager_->findNearestDepartment(current_x, current_y);
    if (orig_department_id == -1) {
        return createErrorResponse("가장 가까운 부서를 찾을 수 없습니다");
    }
    
    // 3. 복귀 명령 전송 (destination_id = 8, 병원 로비)
    DepartmentInfo lobby_department;
    if (!db_manager_->getDepartmentById(8, lobby_department)) {
        return createErrorResponse("병원 로비 정보를 찾을 수 없습니다");
    }
    
    if (nav_manager_) {
        if (!nav_manager_->sendWaypointCommand(lobby_department.department_name)) {
            return createErrorResponse("복귀 명령 전송 실패");
        }
        std::cout << "[USER] 로봇 복귀 명령 전송 완료: " << lobby_department.department_name << "로 이동" << std::endl;
    } else {
        return createErrorResponse("네비게이션 관리자를 사용할 수 없습니다");
    }
    
    // 4. robot_log에 데이터 저장
    std::string current_datetime = db_manager_->getCurrentDate();
    if (current_datetime.empty()) {
        return createErrorResponse("현재 날짜/시간을 가져올 수 없습니다");
    }
    
    bool log_success = db_manager_->insertRobotLogWithType(robot_id, patient_id, current_datetime, 
                                                         orig_department_id, 8, log_type);
    if (!log_success) {
        return createErrorResponse("로봇 로그 저장 실패");
    }
    
    std::cout << "[USER] 로봇 복귀 명령 처리 완료: Robot " << robot_id 
              << " -> 대기장소 (ID: 8), Patient: " << (patient_id ? std::to_string(*patient_id) : "NULL") 
              << ", Type: " << log_type << std::endl;
    
    // 5. 응답 반환
    Json::Value response;
    response["status_code"] = 200;
    response["dest"] = 8;  // 대기장소의 목적지 ID
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

// 유틸리티 함수들

std::string UserRequestHandler::createErrorResponse(const std::string& message) {
    Json::Value response;
    response["error"] = message;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string UserRequestHandler::createSuccessResponse(const std::string& name, 
                                                    const std::string& datetime, 
                                                    const std::string& department,
                                                    const std::string& status,
                                                    int patient_id) {
    Json::Value response;
    response["name"] = name;
    response["datetime"] = datetime;  // YY:DD:HH:MM 형식
    response["department"] = department;
    response["status"] = status;
    response["patient_id"] = patient_id;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string UserRequestHandler::createStatusResponse(int status_code) {
    Json::Value response;
    response["status_code"] = status_code;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

// GUI로 보내는 통신 핸들러들



// GUI로 직접 HTTP 요청을 보내는 함수들



// 메시지 전송 함수들

void UserRequestHandler::sendAlertOccupied(int robot_id) {
    std::cout << "[USER] 모든 클라이언트에게 관리자 사용중 블락 알림 전송 중..." << std::endl;
    
    // JSON 메시지 생성
    Json::Value message;
    message["type"] = "alert_occupied";
    message["robot_id"] = robot_id;
    message["timestamp"] = std::to_string(time(nullptr));
    
    Json::StreamWriterBuilder builder;
    std::string json_message = Json::writeString(builder, message);
    
    // 모든 클라이언트에게 브로드캐스트
    websocket_server_->broadcastMessage(json_message);
    
    std::cout << "[USER] 모든 클라이언트에게 관리자 사용중 블락 알림 전송 완료: Robot " << robot_id << std::endl;
}

void UserRequestHandler::sendAlertIdle(int robot_id) {
    std::cout << "[USER] 모든 클라이언트에게 사용 가능한 상태 알림 전송 중..." << std::endl;
    
    // JSON 메시지 생성
    Json::Value message;
    message["type"] = "alert_idle";
    message["robot_id"] = robot_id;
    message["timestamp"] = std::to_string(time(nullptr));
    
    Json::StreamWriterBuilder builder;
    std::string json_message = Json::writeString(builder, message);
    
    // 모든 클라이언트에게 브로드캐스트
    websocket_server_->broadcastMessage(json_message);
    
    std::cout << "[USER] 모든 클라이언트에게 사용 가능한 상태 알림 전송 완료: Robot " << robot_id << std::endl;
}

void UserRequestHandler::sendNavigatingComplete(int robot_id) {
    std::cout << "[USER] GUI 클라이언트들에게 길안내 완료 알림 전송 중..." << std::endl;
    
    // JSON 메시지 생성
    Json::Value message;
    message["type"] = "navigating_complete";
    message["robot_id"] = robot_id;
    message["timestamp"] = std::to_string(time(nullptr));
    
    Json::StreamWriterBuilder builder;
    std::string json_message = Json::writeString(builder, message);
    
    // GUI 클라이언트들에게만 브로드캐스트
    websocket_server_->broadcastMessageToType("gui", json_message);
    
    std::cout << "[USER] GUI 클라이언트들에게 길안내 완료 알림 전송 완료: Robot " << robot_id << std::endl;
}

// 연결된 클라이언트 정보 조회

std::vector<std::string> UserRequestHandler::getConnectedClients() const {
    return websocket_server_->getConnectedIPs();
}

bool UserRequestHandler::isClientConnected(const std::string& ip_address) const {
    return websocket_server_->isClientConnected(ip_address);
}

// 클라이언트 타입 관리

bool UserRequestHandler::setClientType(const std::string& ip_address, const std::string& client_type) {
    return websocket_server_->setClientType(ip_address, client_type);
}

 
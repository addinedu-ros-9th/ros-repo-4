#include "central_server/admin_request_handler.h"
#include "central_server/config.h"
#include <iostream>
#include <sstream>
#include <json/json.h>

AdminRequestHandler::AdminRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                                       std::shared_ptr<RobotNavigationManager> nav_manager,
                                       std::shared_ptr<WebSocketServer> websocket_server)
    : db_manager_(db_manager), nav_manager_(nav_manager), websocket_server_(websocket_server) {
}

// Admin GUI API 핸들러들

std::string AdminRequestHandler::handleAuthLogin(const Json::Value& request) {
    std::cout << "[ADMIN] 로그인 요청 처리 (IF-01)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("admin_id") || !request.isMember("password")) {
        std::cout << "[ADMIN] 로그인 요청: admin_id 또는 password 누락" << std::endl;
        return "404"; // Missing fields
    }
    
    std::string admin_id = request["admin_id"].asString();
    std::string password = request["password"].asString();
    
    // 빈 필드 검증
    if (admin_id.empty() || password.empty()) {
        std::cout << "[ADMIN] 로그인 요청: 빈 필드 존재" << std::endl;
        return "404"; // Empty fields
    }
    
    // 먼저 admin_id가 데이터베이스에 존재하는지 확인
    if (!db_manager_->isAdminIdExists(admin_id)) {
        std::cout << "[ADMIN] 로그인 실패: 존재하지 않는 admin_id - " << admin_id << std::endl;
        return "401"; // ID doesn't exist
    }
    
    // Admin 테이블에서 로그인 검증 (ID는 존재하므로 비밀번호만 검증)
    AdminInfo admin;
    if (db_manager_->authenticateAdmin(admin_id, password, admin)) {
        std::cout << "[ADMIN] 로그인 성공: " << admin_id << std::endl;
        return "200"; // 성공
    } else {
        std::cout << "[ADMIN] 로그인 실패: 비밀번호 불일치 - " << admin_id << std::endl;
        return "402"; // Wrong password
    }
}

std::string AdminRequestHandler::handleAuthDetail(const Json::Value& request) {
    std::cout << "[ADMIN] 세부 정보 요청 처리 (IF-02)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("admin_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: admin_id");
    }
    
    std::string admin_id = request["admin_id"].asString();
    
    // Admin 테이블에서 관리자 정보 조회
    AdminInfo admin;
    if (db_manager_->getAdminById(admin_id, admin)) {
        Json::Value response;
        response["name"] = admin.name;
        response["email"] = admin.email;
        response["hospital_name"] = admin.hospital_name;
        
        Json::StreamWriterBuilder builder;
        return Json::writeString(builder, response);
    } else {
        return createErrorResponse("관리자를 찾을 수 없습니다");
    }
}

std::string AdminRequestHandler::handleGetRobotLocation(const Json::Value& request) {
    std::cout << "[ADMIN] 로봇 위치 요청 처리 (IF-03)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id;
    if (request["robot_id"].isString()) {
        try {
            robot_id = std::stoi(request["robot_id"].asString());
        } catch (const std::exception& e) {
            std::cout << "[ADMIN] robot_id 변환 실패: " << request["robot_id"].asString() << " - " << e.what() << std::endl;
            return createErrorResponse("Invalid robot_id format");
        }
    } else if (request["robot_id"].isInt()) {
        robot_id = request["robot_id"].asInt();
    } else {
        return createErrorResponse("robot_id는 정수 또는 문자열이어야 합니다");
    }
    
    // 실제 로봇 위치 정보 조회 (RobotNavigationManager에서 받은 데이터)
    double x = nav_manager_->getCurrentRobotX();
    double y = nav_manager_->getCurrentRobotY();
    double yaw = nav_manager_->getCurrentRobotYaw();
    
    Json::Value response;
    response["x"] = x;
    response["y"] = y;
    response["yaw"] = yaw;
    
    std::cout << "[ADMIN] 로봇 위치 정보 반환: Robot " << robot_id 
              << " - 위치: (" << response["x"].asDouble() << ", " << response["y"].asDouble() 
              << "), 방향: " << response["yaw"].asDouble() << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AdminRequestHandler::handleGetRobotStatus(const Json::Value& request) {
    std::cout << "[ADMIN] 로봇 상태 요청 처리 (IF-04)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // 실제 로봇 시스템에서 상태 정보 조회
    if (!nav_manager_) {
        return createErrorResponse("네비게이션 관리자를 사용할 수 없습니다");
    }
    
    // 실제 로봇 상태 정보 조회
    std::string nav_status = nav_manager_->getCurrentNavStatus();
    std::string start_point = nav_manager_->getCurrentStartPoint();
    std::string target = nav_manager_->getCurrentTarget();
    int battery = nav_manager_->getCurrentBattery();
    int network_level = nav_manager_->getCurrentNetworkLevel();
    
    // IF-04 명세에 따라 응답
    Json::Value response;
    response["status"] = nav_status.empty() ? "unknown" : nav_status;
    
    // 문자열 그대로 사용 (정수 변환하지 않음)
    response["orig"] = start_point.empty() ? "none" : start_point;  // 출발지 정보
    response["dest"] = target.empty() ? "none" : target;  // 목적지 정보
    response["battery"] = battery;  // 실제 배터리 정보
    response["network"] = network_level;   // 실제 네트워크 정보
    
    std::cout << "[ADMIN] 로봇 상태 정보 반환: Robot " << robot_id 
              << " - Status: " << response["status"].asString() << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AdminRequestHandler::handleGetPatientInfo(const Json::Value& request) {
    std::cout << "[ADMIN] 환자 정보 요청 처리 (IF-05)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id;
    if (request["robot_id"].isString()) {
        try {
            robot_id = std::stoi(request["robot_id"].asString());
        } catch (const std::exception& e) {
            std::cout << "[ADMIN] robot_id 변환 실패: " << request["robot_id"].asString() << " - " << e.what() << std::endl;
            return createErrorResponse("Invalid robot_id format");
        }
    } else if (request["robot_id"].isInt()) {
        robot_id = request["robot_id"].asInt();
    } else {
        return createErrorResponse("robot_id는 정수 또는 문자열이어야 합니다");
    }
    
    // 로봇 상태 확인 - navigating일 때만 환자 정보 반환
    if (!nav_manager_) {
        return createErrorResponse("네비게이션 관리자를 사용할 수 없습니다");
    }
    
    std::string nav_status = nav_manager_->getCurrentNavStatus();
    if (nav_status != "navigating") {
        return createErrorResponse("로봇이 네비게이션 중이 아닙니다. 현재 상태: " + nav_status);
    }
    
    // 실제 데이터베이스에서 해당 로봇의 최근 환자 정보 조회
    PatientInfo patient;
    if (!db_manager_->getPatientByRobotId(robot_id, patient)) {
        return createErrorResponse("해당 로봇을 이용중인 환자 정보가 없습니다");
    }
    
    // IF-05 명세에 따라 응답
    Json::Value response;
    response["patient_id"] = std::to_string(patient.patient_id);
    response["phone"] = patient.phone;
    response["rfid"] = patient.rfid;
    response["name"] = patient.name;
    
    std::cout << "[ADMIN] 환자 정보 반환: Robot " << robot_id 
              << " (상태: " << nav_status << ")" << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AdminRequestHandler::handleControlByAdmin(const Json::Value& request) {
    std::cout << "[ADMIN] 원격 제어 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("admin_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string admin_id = request["admin_id"].asString();


    // 웹소켓으로 관리자에게 alert_occupied 메시지 전송
    if (websocket_server_) {
        websocket_server_->sendAlertOccupied(robot_id, "admin");
        websocket_server_->sendAlertOccupied(robot_id, "ai");
        std::cout << "[ADMIN] 관리자에게 alert_occupied 메시지 전송: Robot " << robot_id << std::endl;
    }


    // 공통 함수를 사용하여 원격 제어 명령 처리
    return sendControlCommand(robot_id, nullptr, "control_by_admin", "원격 제어 요청", admin_id);
}

std::string AdminRequestHandler::handleReturnCommand(const Json::Value& request) {
    std::cout << "[ADMIN] 원격 제어 취소 요청 처리 (IF-07)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("admin_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string admin_id = request["admin_id"].asString();
    
    // 공통 함수를 사용하여 원격 제어 명령 처리
    return sendControlCommand(robot_id, nullptr, "return_command", "원격 제어 취소", admin_id);
}

std::string AdminRequestHandler::handleTeleopRequest(const Json::Value& request) {
    std::cout << "[ADMIN] 수동제어 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("admin_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string admin_id = request["admin_id"].asString();
    // TODO: 실제 로봇 시스템에서 수동제어 모드 활성화
    std::cout << "[ADMIN] 수동제어 요청 처리 완료: Robot " << robot_id << std::endl;

    return sendControlCommand(robot_id, nullptr, "teleop_request", "수동제어 요청", admin_id);
}

std::string AdminRequestHandler::handleTeleopComplete(const Json::Value& request) {
    std::cout << "[ADMIN] 수동제어 완료 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("admin_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string admin_id = request["admin_id"].asString();
    // TODO: 실제 로봇 시스템에서 수동제어 모드 비활성화
    std::cout << "[ADMIN] 수동제어 완료 처리: Robot " << robot_id << std::endl;
    
    return sendControlCommand(robot_id, nullptr, "teleop_complete", "수동제어 완료", admin_id);
}

std::string AdminRequestHandler::handleCommandMoveTeleop(const Json::Value& request) {
    std::cout << "[ADMIN] teleop 이동 명령 처리 (IF-08)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("teleop_key")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    int teleop_key = request["teleop_key"].asInt();
    
    // teleop 키에 따른 이동 명령 매핑 (teleop_twist_keyboard moveBindings 기반)
    std::string teleop_command;
    switch (teleop_key) {
        case 1: teleop_command = "1"; break;   // u: 앞으로 가면서 좌회전
        case 2: teleop_command = "2"; break;   // i: 전진
        case 3: teleop_command = "3"; break;   // o: 앞으로 가면서 우회전
        case 4: teleop_command = "4"; break;   // j: 제자리에서 왼쪽으로 회전
        case 5: teleop_command = "5"; break;   // k: 정지
        case 6: teleop_command = "6"; break;   // l: 제자리에서 오른쪽으로 회전
        case 7: teleop_command = "7"; break;   // m: 뒤로 가면서 좌회전
        case 8: teleop_command = "8"; break;   // ,: 후진
        case 9: teleop_command = "9"; break;   // .: 뒤로 가면서 우회전
        default:
            return "400"; // Bad Request - 잘못된 teleop_key
    }
    
    // 실제 로봇 시스템에서 teleop 명령 전송
    if (nav_manager_) {
        if (!nav_manager_->sendTeleopCommand(teleop_command)) {
            return "500"; // Internal Server Error
        }
        std::cout << "[ADMIN] 로봇 teleop 명령 전송 완료: Robot " << robot_id 
                  << ", Key: " << teleop_key << ", Command: " << teleop_command << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return "200"; // 성공
}

std::string AdminRequestHandler::handleCommandMoveDest(const Json::Value& request) {
    std::cout << "[ADMIN] 목적지 이동 명령 처리 (IF-09)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("dest") || !request.isMember("admin_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    int dest = request["dest"].asInt();
    std::string admin_id = request["admin_id"].asString();

    // 목적지 ID를 department_name으로 변환
    DepartmentInfo department;
    if (!db_manager_->getDepartmentById(dest, department)) {
        return "400"; // Bad Request - 잘못된 목적지 ID
    }
    std::string waypoint_name = department.department_name;

    // 네비게이션 이벤트 전송 (유저 흐름과 동일한 방식)
    if (nav_manager_) {
        bool sent = nav_manager_->sendNavigateEvent("admin_navigating", waypoint_name);
        if (!sent) {
            return "500"; // Internal Server Error
        }
        std::cout << "[ADMIN] 로봇 네비게이션 명령 전송 완료: Robot " << robot_id
                  << ", Dest: " << dest << ", Waypoint: " << waypoint_name << std::endl;
    } else {
        return "503"; // Service Unavailable
    }

    // 현재 로봇 위치 기반 출발지 계산
    double current_x = nav_manager_->getCurrentRobotX();
    double current_y = nav_manager_->getCurrentRobotY();
    int orig_department_id = db_manager_->findNearestDepartment(static_cast<float>(current_x), static_cast<float>(current_y));
    if (orig_department_id == -1) {
        std::cout << "[ADMIN][WARNING] 출발지 부서를 찾지 못했습니다" << std::endl;
        return "500";
    }

    // 로깅 (robot_log, navigating_log)
    std::string current_datetime = db_manager_->getCurrentDateTime();
    if (current_datetime.empty()) {
        return "500";
    }

    bool log_success = db_manager_->insertRobotLogWithType(robot_id, nullptr, current_datetime, orig_department_id, dest, "admin_navigating", admin_id);
    if (!log_success) {
        return "500";
    }

    return "200"; // 성공
}

std::string AdminRequestHandler::handleCancelNavigating(const Json::Value& request) {
    std::cout << "[ADMIN] 길안내 취소 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("admin_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string admin_id = request["admin_id"].asString();
    // 실제 로봇 시스템에서 길안내 취소 명령 전송
    if (nav_manager_) {
        if (!nav_manager_->sendStopCommand()) {
            return "500"; // Internal Server Error
        }
        std::cout << "[ADMIN] 길안내 취소 명령 전송 완료: Robot " << robot_id << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return sendControlCommand(robot_id, nullptr, "cancel_navigating", "길안내 취소", admin_id);
}

std::string AdminRequestHandler::handleGetLogData(const Json::Value& request) {
    std::cout << "[ADMIN] 로봇 로그 데이터 요청 처리 (IF-10)" << std::endl;
    
    // 요청 데이터 검증 및 파라미터 처리
    std::string period = "";
    std::string start_date = "";
    std::string end_date = "";
    
    // period 파라미터 처리
    if (request.isMember("period")) {
        if (request["period"].isString()) {
            std::string period_str = request["period"].asString();
            if (period_str != "None" && period_str != "null" && !period_str.empty()) {
                period = period_str;
            }
        }
    }
    
    // start_date 파라미터 처리
    if (request.isMember("start_date")) {
        if (request["start_date"].isString()) {
            std::string start_date_str = request["start_date"].asString();
            if (start_date_str != "None" && start_date_str != "null" && !start_date_str.empty()) {
                start_date = start_date_str;
            }
        }
    }
    
    // end_date 파라미터 처리
    if (request.isMember("end_date")) {
        if (request["end_date"].isString()) {
            std::string end_date_str = request["end_date"].asString();
            if (end_date_str != "None" && end_date_str != "null" && !end_date_str.empty()) {
                end_date = end_date_str;
            }
        }
    }
    
    // 에러 검증: period와 start_date/end_date가 동시에 값이 있으면 에러
    if (!period.empty() && (!start_date.empty() || !end_date.empty())) {
        return createErrorResponse("period와 start_date/end_date는 동시에 사용할 수 없습니다");
    }
    
    // 실제 데이터베이스에서 로그 데이터 조회
    std::vector<std::map<std::string, std::string>> log_data = 
        db_manager_->getRobotLogData(period, start_date, end_date);
    
    Json::Value response = Json::Value(Json::arrayValue);
    
    // DB에서 조회한 로그 데이터를 JSON으로 변환
    for (const auto& log_entry : log_data) {
        Json::Value json_entry;
        json_entry["patient_id"] = log_entry.at("patient_id");  // 문자열로 유지
        
        // 안전한 문자열을 정수로 변환
        try {
            json_entry["orig"] = std::stoi(log_entry.at("orig"));
        } catch (const std::exception& e) {
            std::cout << "[ADMIN] orig 변환 실패: " << log_entry.at("orig") << " - " << e.what() << std::endl;
            json_entry["orig"] = 0;
        }
        
        try {
            json_entry["dest"] = std::stoi(log_entry.at("dest"));
        } catch (const std::exception& e) {
            std::cout << "[ADMIN] dest 변환 실패: " << log_entry.at("dest") << " - " << e.what() << std::endl;
            json_entry["dest"] = 0;
        }
        
        json_entry["datetime"] = log_entry.at("date");  // 올바른 필드명 사용
        
        response.append(json_entry);
    }
    
    std::cout << "[ADMIN] 로그 데이터 응답 생성 완료: " << response.size() << "개 레코드" << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}
// 유틸리티 함수들

std::string AdminRequestHandler::createErrorResponse(const std::string& message) {
    Json::Value response;
    response["error"] = message;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AdminRequestHandler::createSuccessResponse(const Json::Value& data) {
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, data);
}

std::string AdminRequestHandler::createStatusResponse(int status_code) {
    Json::Value response;
    response["status_code"] = status_code;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AdminRequestHandler::sendControlCommand(int robot_id, int* patient_id, const std::string& log_type, 
                                                   const std::string& command_name, const std::string& admin_id) {
    // 1. 제어 명령 전송
    if (nav_manager_) {
        bool success = nav_manager_->sendControlEvent(log_type);
        if (success) {
            std::cout << "[ADMIN] 로봇 " << command_name << " 명령 전송 완료: " << log_type << " 전송 성공" << std::endl;
        } else {
            std::cout << "[ADMIN] 로봇 " << command_name << " 명령 전송 실패: " << log_type << " 전송 실패" << std::endl;
            return createErrorResponse(command_name + " 명령 전송 실패");
        }
    } else {
        return createErrorResponse("네비게이션 관리자를 사용할 수 없습니다");
    }
    
    // 2. robot_log에 데이터 저장
    std::string current_datetime = db_manager_->getCurrentDateTime();
    if (current_datetime.empty()) {
        return createErrorResponse("현재 날짜/시간을 가져올 수 없습니다");
    }
    
    bool log_success = db_manager_->insertRobotLogWithType(robot_id, patient_id, current_datetime, 
                                                         0, 0, log_type, admin_id);  // orig_department_id = 0, dest_department_id = 0 (제어 중)
    if (!log_success) {
        return createErrorResponse("로봇 로그 저장 실패");
    }
    
    std::cout << "[ADMIN] 로봇 " << command_name << " 명령 처리 완료: Robot " << robot_id 
              << ", Patient: " << (patient_id ? std::to_string(*patient_id) : "NULL") 
              << ", Type: " << log_type << std::endl;
    
    // 3. 응답 반환
    Json::Value response;
    response["status_code"] = 200;
    response["message"] = command_name + " 모드 활성화 완료";
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
} 
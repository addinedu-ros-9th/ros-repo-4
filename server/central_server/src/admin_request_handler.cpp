#include "central_server/admin_request_handler.h"
#include "central_server/config.h"
#include <iostream>
#include <sstream>
#include <json/json.h>

AdminRequestHandler::AdminRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                                       std::shared_ptr<RobotNavigationManager> nav_manager)
    : db_manager_(db_manager), nav_manager_(nav_manager) {
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
        robot_id = std::stoi(request["robot_id"].asString());
    } else if (request["robot_id"].isInt()) {
        robot_id = request["robot_id"].asInt();
    } else {
        return createErrorResponse("robot_id는 정수 또는 문자열이어야 합니다");
    }
    
    // 실제 로봇 위치 정보 조회 (amcl_pose에서 받은 데이터)
    // TODO: nav_manager_에서 현재 로봇 위치 정보를 가져와야 함
    // 현재는 더미 데이터로 응답
    
    Json::Value response;
    response["x"] = 5.0;  // TODO: 실제 로봇 위치로 변경
    response["y"] = -1.0; // TODO: 실제 로봇 위치로 변경
    response["yaw"] = -0.532151; // TODO: 실제 로봇 방향으로 변경
    
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
    
    std::string nav_status = nav_manager_->getCurrentNavStatus();
    
    // IF-04 명세에 따라 응답
    Json::Value response;
    response["status"] = nav_status.empty() ? "unknown" : nav_status;
    response["orig"] = 0;  // TODO: 실제 출발지 정보 조회
    response["dest"] = 3;  // TODO: 실제 목적지 정보 조회
    response["battery"] = 70;  // TODO: 실제 배터리 정보 조회
    response["network"] = 4;   // TODO: 실제 네트워크 정보 조회
    
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
    
    // TODO: 실제 데이터베이스에서 해당 로봇을 이용중인 환자 정보 조회
    // IF-05 명세에 따라 응답
    Json::Value response;
    response["patient_id"] = "00000000";
    response["phone"] = "010-1111-1111";
    response["rfid"] = "33F7ADEC";
    response["name"] = "김환자";
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AdminRequestHandler::handleControlByAdmin(const Json::Value& request) {
    std::cout << "[ADMIN] 원격 제어 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 원격 제어 모드 활성화
    std::cout << "[ADMIN] 원격 제어 요청 처리 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string AdminRequestHandler::handleReturnCommand(const Json::Value& request) {
    std::cout << "[ADMIN] 원격 제어 취소 요청 처리 (IF-07)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // 실제 로봇 시스템에서 취소 명령 전송 (정지 후 대기장소로 이동)
    if (nav_manager_) {
        // 1. 먼저 정지
        if (!nav_manager_->sendStopCommand()) {
            return "500"; // Internal Server Error
        }
        
        // 2. 병원 로비로 이동
        DepartmentInfo lobby_department;
        if (!db_manager_->getDepartmentById(8, lobby_department)) {
            return "500"; // Internal Server Error
        }
        
        if (!nav_manager_->sendWaypointCommand(lobby_department.department_name)) {
            return "500"; // Internal Server Error
        }
        
        std::cout << "[ADMIN] 로봇 원격 제어 취소 명령 전송 완료: Robot " << robot_id 
                  << " (정지 후 " << lobby_department.department_name << "로 이동)" << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return "200"; // 성공
}

std::string AdminRequestHandler::handleTeleopRequest(const Json::Value& request) {
    std::cout << "[ADMIN] 수동제어 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 수동제어 모드 활성화
    std::cout << "[ADMIN] 수동제어 요청 처리 완료: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
}

std::string AdminRequestHandler::handleTeleopComplete(const Json::Value& request) {
    std::cout << "[ADMIN] 수동제어 완료 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 수동제어 모드 비활성화
    std::cout << "[ADMIN] 수동제어 완료 처리: Robot " << robot_id << std::endl;
    
    return "200"; // 성공
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
    if (!request.isMember("robot_id") || !request.isMember("dest")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    int dest = request["dest"].asInt();
    
    // 목적지 ID를 department_name으로 변환
    DepartmentInfo department;
    if (!db_manager_->getDepartmentById(dest, department)) {
        return "400"; // Bad Request - 잘못된 목적지 ID
    }
    std::string waypoint_name = department.department_name;
    
    // 실제 로봇 시스템에서 목적지 이동 명령 전송
    if (nav_manager_) {
        if (!nav_manager_->sendWaypointCommand(waypoint_name)) {
            return "500"; // Internal Server Error
        }
        std::cout << "[ADMIN] 로봇 목적지 이동 명령 전송 완료: Robot " << robot_id 
                  << ", Dest: " << dest << ", Waypoint: " << waypoint_name << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return "200"; // 성공
}

std::string AdminRequestHandler::handleCancelNavigating(const Json::Value& request) {
    std::cout << "[ADMIN] 길안내 취소 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // 실제 로봇 시스템에서 길안내 취소 명령 전송
    if (nav_manager_) {
        if (!nav_manager_->sendStopCommand()) {
            return "500"; // Internal Server Error
        }
        std::cout << "[ADMIN] 길안내 취소 명령 전송 완료: Robot " << robot_id << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return "200"; // 성공
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
        json_entry["orig"] = std::stoi(log_entry.at("orig"));
        json_entry["dest"] = std::stoi(log_entry.at("dest"));
        json_entry["datetime"] = log_entry.at("date");  // 올바른 필드명 사용
        
        response.append(json_entry);
    }
    
    std::cout << "[ADMIN] 로그 데이터 응답 생성 완료: " << response.size() << "개 레코드" << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string AdminRequestHandler::handleGetHeatmap(const Json::Value& request) {
    std::cout << "[ADMIN] 히트맵 데이터 요청 처리 (IF-11)" << std::endl;
    
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
    
    // TODO: 실제 데이터베이스에서 히트맵 데이터 조회
    // 현재는 더미 데이터로 응답
    Json::Value response;
    Json::Value matrix = Json::Value(Json::arrayValue);
    
    // 8x8 히트맵 매트릭스 생성 (더미 데이터)
    int heatmap_data[8][8] = {
        {0, 4, 2, 0, 0, 0, 1, 0},
        {2, 0, 3, 0, 1, 0, 0, 0},
        {1, 1, 0, 0, 0, 2, 0, 0},
        {0, 0, 0, 0, 5, 0, 1, 0},
        {0, 0, 0, 3, 0, 2, 0, 1},
        {0, 0, 1, 0, 1, 0, 4, 0},
        {2, 0, 0, 0, 0, 1, 0, 2},
        {0, 0, 0, 0, 0, 0, 1, 0}
    };
    
    for (int i = 0; i < 8; i++) {
        Json::Value row = Json::Value(Json::arrayValue);
        for (int j = 0; j < 8; j++) {
            row.append(heatmap_data[i][j]);
        }
        matrix.append(row);
    }
    
    response["matrix"] = matrix;
    
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
#include "central_server/user_request_handler.h"
#include "central_server/config.h"
#include "central_server/websocket_server.h"
#include <iostream>
#include <sstream>
#include <json/json.h>
#include <curl/curl.h>


UserRequestHandler::UserRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                                     std::shared_ptr<RobotNavigationManager> nav_manager,
                                     std::shared_ptr<WebSocketServer> websocket_server)
    : db_manager_(db_manager), nav_manager_(nav_manager), websocket_server_(websocket_server) {
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
    
    // AI 서버에 start_tracking 명령 전송
    sendStartTrackingToAI(robot_id);
    
    return processDirectionRequest(robot_id, department_id, &patient_id, "patient_navigating");
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
    
    return sendReturnCommand(robot_id, &patient_id, "patient_return");
}

std::string UserRequestHandler::handleWithoutAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("department_id")) {
        return createErrorResponse("Missing robot_id or department_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int department_id = request["department_id"].asInt();
    
    // AI 서버에 start_tracking 명령 전송
    sendStartTrackingToAI(robot_id);
    
    return processDirectionRequest(robot_id, department_id, nullptr, "unknown_navigating");
}

std::string UserRequestHandler::handleWithoutAuthRobotReturn(const Json::Value& request) {
    std::cout << "[USER] 비인증 로봇 복귀 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    return sendReturnCommand(robot_id, nullptr, "unknown_return");
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

    
    
    // 음성 호출 요청 시 call_with_voice 메시지를 서비스를 통해 전송
    if (nav_manager_) {
        bool success = nav_manager_->sendControlEvent("call_with_voice");
        if (success) {
            std::cout << "[USER] 음성 호출 처리 완료: Robot " << robot_id << " - call_with_voice 전송 성공" << std::endl;
        } else {
            std::cout << "[USER] 음성 호출 처리 실패: Robot " << robot_id << " - call_with_voice 전송 실패" << std::endl;
            return "500"; // Internal Server Error
        }
    } else {
        std::cout << "[USER] 음성 호출 처리 실패: Robot " << robot_id << " - nav_manager가 설정되지 않음" << std::endl;
        return "500"; // Internal Server Error
    }

    // 웹소켓으로 관리자에게 alert_occupied 메시지 전송
    if (websocket_server_) {
        websocket_server_->sendAlertOccupied(robot_id, "admin");
        websocket_server_->sendAlertOccupied(robot_id, "ai");
        std::cout << "[USER] 관리자에게 alert_occupied 메시지 전송: Robot " << robot_id << std::endl;
    }

    // robot_log에 이벤트 저장
    std::string current_datetime = db_manager_->getCurrentDateTime();
    if (!current_datetime.empty()) {
        bool log_success = db_manager_->insertRobotLogWithType(robot_id, nullptr, current_datetime, 
                                                             0, 0, "call_with_voice", "");
        if (!log_success) {
            std::cout << "[WARNING] Failed to insert robot log for call_with_voice event" << std::endl;
        }
    }

    
    return "200"; // 성공
}

std::string UserRequestHandler::handleCallWithScreen(const Json::Value& request) {
    std::cout << "[USER] 화면 호출 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // 화면 호출 요청 시 call_with_screen 메시지를 서비스를 통해 전송
    if (nav_manager_) {
        bool success = nav_manager_->sendControlEvent("call_with_screen");
        if (success) {
            std::cout << "[USER] 화면 호출 처리 완료: Robot " << robot_id << " - call_with_screen 전송 성공" << std::endl;
        } else {
            std::cout << "[USER] 화면 호출 처리 실패: Robot " << robot_id << " - call_with_screen 전송 실패" << std::endl;
            return "500"; // Internal Server Error
        }
    } else {
        std::cout << "[USER] 화면 호출 처리 실패: Robot " << robot_id << " - nav_manager가 설정되지 않음" << std::endl;
        return "500"; // Internal Server Error
    }

    // 웹소켓으로 관리자에게 alert_occupied 메시지 전송
    if (websocket_server_) {
        websocket_server_->sendAlertOccupied(robot_id, "admin");
        websocket_server_->sendAlertOccupied(robot_id, "ai");
        std::cout << "[USER] 관리자에게 alert_occupied 메시지 전송: Robot " << robot_id << std::endl;
    }

    // robot_log에 이벤트 저장
    std::string current_datetime = db_manager_->getCurrentDateTime();
    if (!current_datetime.empty()) {
        bool log_success = db_manager_->insertRobotLogWithType(robot_id, nullptr, current_datetime, 
                                                             0, 0, "call_with_screen", "");
        if (!log_success) {
            std::cout << "[WARNING] Failed to insert robot log for call_with_screen event" << std::endl;
        }
    }
    
    return "200"; // 성공
}

std::string UserRequestHandler::handleAlertTimeout(const Json::Value& request) {
    std::cout << "[USER] 30초 타임아웃 알림 처리" << std::endl;
    
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string patient_id_str = request["patient_id"].asString();
    int patient_id = std::stoi(patient_id_str);
    
    // 30초 타임아웃 발생 시 return_command 메시지를 서비스를 통해 전송
    return sendReturnCommand(robot_id, &patient_id, "return_command");
}

std::string UserRequestHandler::handlePauseRequest(const Json::Value& request) {
    std::cout << "[USER] 일시정지 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string patient_id_str = request["patient_id"].asString();
    int patient_id = std::stoi(patient_id_str);
    
    return sendReturnCommand(robot_id, &patient_id, "pause_request");
}

std::string UserRequestHandler::handleRestartNavigation(const Json::Value& request) {
    std::cout << "[USER] 길안내 재개 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string patient_id_str = request["patient_id"].asString();
    int patient_id = std::stoi(patient_id_str);
    
    return sendReturnCommand(robot_id, &patient_id, "restart_navigation");
}

std::string UserRequestHandler::handleStopNavigating(const Json::Value& request) {
    std::cout << "[USER] 길안내 중지 요청 처리" << std::endl;
    
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string patient_id_str = request["patient_id"].asString();
    int patient_id = std::stoi(patient_id_str);
    
    return sendReturnCommand(robot_id, &patient_id, "stop_navigating");
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
        bool success = nav_manager_->sendNavigateEvent(log_type, department.department_name);
        if (success) {
            std::cout << "[USER] 로봇 네비게이션 명령 전송 완료: " << department.department_name << "로 이동" << std::endl;
        } else {
            std::cout << "[USER] 로봇 네비게이션 명령 전송 실패: " << department.department_name << std::endl;
            return createErrorResponse("Failed to send navigation command");
        }
    } else {
        return createErrorResponse("Navigation manager not available");
    }
    
    // 3. 현재 로봇 위치 확인
    double current_x = nav_manager_->getCurrentRobotX();
    double current_y = nav_manager_->getCurrentRobotY();
    
    // 4. 가장 가까운 부서 찾기 (출발지)
    int orig_department_id = db_manager_->findNearestDepartment(current_x, current_y);
    
    if (orig_department_id == -1) {
        return createErrorResponse("Failed to find nearest department");
    }
    
    // 5. robot_log에 insert
    std::string current_datetime = db_manager_->getCurrentDateTime();
    if (current_datetime.empty()) {
        return createErrorResponse("Failed to get current datetime");
    }
    
    // patient_id 저장 (nullptr이면 NULL, 아니면 실제 patient_id)
    bool success = db_manager_->insertRobotLogWithType(robot_id, patient_id, current_datetime, 
                                                     orig_department_id, department_id, log_type, "");
    
    if (!success) {
        return createErrorResponse("Failed to insert robot log");
    }
    
    // 6. navigating_log에 insert (navigation 이벤트의 시작지/목적지 저장)
    bool nav_success = db_manager_->insertNavigatingLog(robot_id, current_datetime, orig_department_id, department_id);
    if (!nav_success) {
        std::cout << "[WARNING] Failed to insert navigating log for Robot " << robot_id << std::endl;
        // navigating_log 실패는 경고만 하고 계속 진행
    }
    
    std::cout << "[USER] 네비게이션 명령 처리 완료: Robot " << robot_id 
              << " -> Department " << department.department_name << " (ID: " << department_id 
              << "), Patient: " << (patient_id ? std::to_string(*patient_id) : "NULL") 
              << ", Type: " << log_type << std::endl;
    
    return createStatusResponse(200);
}

std::string UserRequestHandler::sendReturnCommand(int robot_id, int* patient_id, const std::string& log_type, bool save_log) {
    // 1. 복귀 명령 전송
    if (nav_manager_) {
        bool success = nav_manager_->sendControlEvent(log_type);
        if (success) {
            std::cout << "[USER] 로봇 복귀 명령 전송 완료: return_command 전송 성공" << std::endl;
        } else {
            std::cout << "[USER] 로봇 복귀 명령 전송 실패: return_command 전송 실패" << std::endl;
            return createErrorResponse("복귀 명령 전송 실패");
        }
    } else {
        return createErrorResponse("네비게이션 관리자를 사용할 수 없습니다");
    }
    
    // 2. robot_log에 데이터 저장 (save_log가 true일 때만)
    if (save_log) {
        std::string current_datetime = db_manager_->getCurrentDateTime();
        if (current_datetime.empty()) {
            return createErrorResponse("현재 날짜/시간을 가져올 수 없습니다");
        }
        
        bool log_success = db_manager_->insertRobotLogWithType(robot_id, patient_id, current_datetime, 
                                                             0, 8, log_type, "");  // orig_department_id = 0 (알 수 없음)
        if (!log_success) {
            return createErrorResponse("로봇 로그 저장 실패");
        }
        
        std::cout << "[USER] 로봇 복귀 명령 처리 완료: Robot " << robot_id 
                  << ", Patient: " << (patient_id ? std::to_string(*patient_id) : "NULL") 
                  << ", Type: " << log_type << std::endl;
    }
    
    // 3. 응답 반환
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

// AI 서버 HTTP 통신
bool UserRequestHandler::sendStartTrackingToAI(int robot_id) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cout << "[USER] CURL 초기화 실패" << std::endl;
        return false;
    }
    
    // AI 서버 URL (config에서 가져오거나 기본값 사용)
    std::string ai_server_url = "http://192.168.0.27:8000/start_tracking";  // AI 서버 주소로 변경 필요
    
    // JSON 요청 데이터
    Json::Value request_data;
    request_data["robot_id"] = robot_id;
    
    Json::StreamWriterBuilder builder;
    std::string json_data = Json::writeString(builder, request_data);
    
    // HTTP 헤더 설정
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    // CURL 옵션 설정
    curl_easy_setopt(curl, CURLOPT_URL, ai_server_url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);  // 5초 타임아웃
    
    // 응답 처리
    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](void* contents, size_t size, size_t nmemb, std::string* userp) -> size_t {
        userp->append((char*)contents, size * nmemb);
        return size * nmemb;
    });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    // 요청 실행
    CURLcode res = curl_easy_perform(curl);
    
    // 정리
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        std::cout << "[USER] AI 서버 start_tracking 요청 실패: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    
    // 응답 확인 (간단히 HTTP 상태 코드만 확인)
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    
    if (http_code == 200) {
        std::cout << "[USER] AI 서버 start_tracking 요청 성공: Robot " << robot_id << std::endl;
        return true;
    } else {
        std::cout << "[USER] AI 서버 start_tracking 요청 실패: HTTP " << http_code << std::endl;
        return false;
    }
}

// GUI로 보내는 통신 핸들러들



// GUI로 직접 HTTP 요청을 보내는 함수들







 
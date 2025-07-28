#include "central_server/http_server.h"
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <json/json.h> // Added for JSON parsing/writing

HttpServer::HttpServer(std::shared_ptr<DatabaseManager> db_manager, int port)
    : db_manager_(db_manager), port_(port), running_(false) {
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    server_thread_ = std::thread(&HttpServer::serverLoop, this);
    std::cout << "[HTTP] HTTP 서버 시작됨 (포트: " << port_ << ")" << std::endl;
}

void HttpServer::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    std::cout << "[HTTP] HTTP 서버 중지됨" << std::endl;
}

bool HttpServer::isRunning() const {
    return running_;
}

void HttpServer::serverLoop() {
    // 소켓 생성
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        std::cerr << "[HTTP] 소켓 생성 실패" << std::endl;
        return;
    }
    
    // 소켓 옵션 설정 (재사용 가능)
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // 주소 설정
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);
    
    // 바인딩
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[HTTP] 바인딩 실패" << std::endl;
        close(server_fd);
        return;
    }
    
    // 리스닝 시작
    if (listen(server_fd, 3) < 0) {
        std::cerr << "[HTTP] 리스닝 실패" << std::endl;
        close(server_fd);
        return;
    }
    
    std::cout << "[HTTP] HTTP 서버 리스닝 시작: " << port_ << std::endl;
    
    // 클라이언트 연결 처리
    while (running_) {
        struct sockaddr_in client_address;
        socklen_t client_len = sizeof(client_address);
        
        int client_fd = accept(server_fd, (struct sockaddr*)&client_address, &client_len);
        if (client_fd < 0) {
            if (running_) {
                std::cerr << "[HTTP] 클라이언트 연결 실패" << std::endl;
            }
            continue;
        }
        
        // HTTP 요청 읽기
        char buffer[4096] = {0};
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            std::string request_str(buffer, bytes_read);
            HttpRequest request = parseHttpRequest(request_str);
            
            std::string response = processRequest(request);
            
            // 응답 전송
            send(client_fd, response.c_str(), response.length(), 0);
        }
        
        close(client_fd);
    }
    
    close(server_fd);
}

std::string HttpServer::processRequest(const HttpRequest& request) {
    try {
        std::cout << "[HTTP] " << request.method << " " << request.path << std::endl;
        
        if (request.method != "POST") {
            return createHttpResponse(405, "application/json", 
                createErrorResponse("Method not allowed"));
        }
        
        Json::Value json_request = parseJson(request.body);
        std::string response_body;
        
        if (request.path == "/auth/ssn") {
            response_body = handleAuthSSN(json_request);
        } else if (request.path == "/auth/patient_id") {
            response_body = handleAuthPatientId(json_request);
        } else if (request.path == "/auth/rfid") {
            response_body = handleAuthRFID(json_request);
        } else if (request.path == "/auth/direction") {
            response_body = handleAuthDirection(json_request);
        } else if (request.path == "/status/robot_return") {
            response_body = handleRobotReturn(json_request);
        } else if (request.path == "/without_auth/direction") {
            response_body = handleWithoutAuthDirection(json_request);
        } else if (request.path == "/robot_status") {
            response_body = handleRobotStatus(json_request);
        } else {
            response_body = createErrorResponse("Not found");
            return createHttpResponse(404, "application/json", response_body);
        }
        
        return createHttpResponse(200, "application/json", response_body);
        
    } catch (const std::exception& e) {
        std::cerr << "[HTTP] 요청 처리 오류: " << e.what() << std::endl;
        return createHttpResponse(500, "application/json", 
            createErrorResponse("Internal server error"));
    }
}

// API 핸들러들
std::string HttpServer::handleAuthSSN(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("ssn")) {
        return createErrorResponse("Missing robot_id or ssn");
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string ssn = request["ssn"].asString();
    
    PatientInfo patient;
    if (db_manager_->getPatientBySSN(ssn, patient)) {
        // 환자의 예약 정보 조회
        ReservationInfo reservation;
        if (db_manager_->getReservationByPatientId(patient.patient_id, reservation)) {
            return createSuccessResponse(patient.name, reservation.time_hhmm, reservation.reservation);
        } else {
            // 예약 정보가 없는 경우
            return createSuccessResponse(patient.name, "00:00", "00");
        }
    } else {
        return createErrorResponse("Patient not found");
    }
}

std::string HttpServer::handleAuthPatientId(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return createErrorResponse("Missing robot_id or patient_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int patient_id = request["patient_id"].asInt();
    
    PatientInfo patient;
    if (db_manager_->getPatientById(patient_id, patient)) {
        // 환자의 예약 정보 조회
        ReservationInfo reservation;
        if (db_manager_->getReservationByPatientId(patient.patient_id, reservation)) {
            return createSuccessResponse(patient.name, reservation.time_hhmm, reservation.reservation);
        } else {
            // 예약 정보가 없는 경우
            return createSuccessResponse(patient.name, "00:00", "00");
        }
    } else {
        return createErrorResponse("Patient not found");
    }
}

std::string HttpServer::handleAuthRFID(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("rfid")) {
        return createErrorResponse("Missing robot_id or rfid");
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string rfid = request["rfid"].asString();
    
    PatientInfo patient;
    if (db_manager_->getPatientByRFID(rfid, patient)) {
        // 환자의 예약 정보 조회
        ReservationInfo reservation;
        if (db_manager_->getReservationByPatientId(patient.patient_id, reservation)) {
            return createSuccessResponse(patient.name, reservation.time_hhmm, reservation.reservation);
        } else {
            // 예약 정보가 없는 경우
            return createSuccessResponse(patient.name, "00:00", "00");
        }
    } else {
        return createErrorResponse("Patient not found");
    }
}

std::string HttpServer::handleAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("station_id")) {
        return createErrorResponse("Missing robot_id or station_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int station_id = request["station_id"].asInt();
    
    // TODO: 네비게이션 명령 처리
    std::cout << "[HTTP] 네비게이션 명령: Robot " << robot_id << " -> Station " << station_id << std::endl;
    
    return createStatusResponse(200);
}

std::string HttpServer::handleRobotReturn(const Json::Value& request) {
    if (!request.isMember("robot_id")) {
        return createErrorResponse("Missing robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 로봇 복귀 명령 처리
    std::cout << "[HTTP] 로봇 복귀 명령: Robot " << robot_id << std::endl;
    
    return createStatusResponse(200);
}

std::string HttpServer::handleWithoutAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("station_id")) {
        return createErrorResponse("Missing robot_id or station_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int station_id = request["station_id"].asInt();
    
    // TODO: 인증 없는 네비게이션 명령 처리
    std::cout << "[HTTP] 인증 없는 네비게이션: Robot " << robot_id << " -> Station " << station_id << std::endl;
    
    return createStatusResponse(200);
}

std::string HttpServer::handleRobotStatus(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("status")) {
        return createErrorResponse("Missing robot_id or status");
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string status = request["status"].asString();
    
    // TODO: 로봇 상태 업데이트 처리
    std::cout << "[HTTP] 로봇 상태 업데이트: Robot " << robot_id << " -> " << status << std::endl;
    
    return createStatusResponse(200);
}

// 유틸리티 함수들
Json::Value HttpServer::parseJson(const std::string& jsonStr) {
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    std::istringstream stream(jsonStr);
    if (!Json::parseFromStream(builder, stream, &root, &errors)) {
        throw std::runtime_error("JSON 파싱 실패: " + errors);
    }
    
    return root;
}

std::string HttpServer::createSuccessResponse(const std::string& name, 
                                            const std::string& time_hhmm, 
                                            const std::string& reservation) {
    Json::Value response;
    response["name"] = name;
    response["datetime"] = time_hhmm;  // hh:mm 형식
    response["reservation"] = reservation;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::createErrorResponse(const std::string& message) {
    Json::Value response;
    response["error"] = message;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::createStatusResponse(int status_code) {
    Json::Value response;
    response["status_code"] = status_code;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

HttpServer::HttpRequest HttpServer::parseHttpRequest(const std::string& request) {
    HttpRequest http_request;
    std::istringstream stream(request);
    std::string line;
    
    // 첫 번째 줄: Method Path HTTP/1.1
    if (std::getline(stream, line)) {
        std::istringstream first_line(line);
        first_line >> http_request.method >> http_request.path;
    }
    
    // 헤더들 파싱
    while (std::getline(stream, line) && line != "\r" && !line.empty()) {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string header_name = line.substr(0, colon_pos);
            std::string header_value = line.substr(colon_pos + 2); // ": " 건너뛰기
            
            // \r 제거
            if (!header_value.empty() && header_value.back() == '\r') {
                header_value.pop_back();
            }
            
            http_request.headers[header_name] = header_value;
        }
    }
    
    // 바디 읽기
    std::string body;
    std::string body_line;
    while (std::getline(stream, body_line)) {
        body += body_line;
    }
    http_request.body = body;
    
    return http_request;
}

std::string HttpServer::createHttpResponse(int status_code, 
                                          const std::string& content_type,
                                          const std::string& body) {
    std::ostringstream response;
    
    std::string status_text;
    switch (status_code) {
        case 200: status_text = "OK"; break;
        case 404: status_text = "Not Found"; break;
        case 405: status_text = "Method Not Allowed"; break;
        case 500: status_text = "Internal Server Error"; break;
        default: status_text = "Unknown"; break;
    }
    
    response << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
    response << "Content-Type: " << content_type << "\r\n";
    response << "Content-Length: " << body.length() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n";
    response << "Access-Control-Allow-Headers: Content-Type\r\n";
    response << "\r\n";
    response << body;
    
    return response.str();
} 
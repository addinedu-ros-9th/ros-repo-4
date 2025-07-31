#include "central_server/http_server.h"
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <json/json.h> // Added for JSON parsing/writing
#include <openssl/sha.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>

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
    
    // WebSocket 클라이언트들 정리
    {
        std::lock_guard<std::mutex> lock(websocket_clients_mutex_);
        for (int client_socket : websocket_clients_) {
            close(client_socket);
        }
        websocket_clients_.clear();
    }
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    std::cout << "[HTTP] HTTP 서버 중지됨" << std::endl;
}

bool HttpServer::isRunning() const {
    return running_;
}

void HttpServer::broadcastToClients(const std::string& message) {
    std::lock_guard<std::mutex> lock(websocket_clients_mutex_);
    
    auto it = websocket_clients_.begin();
    while (it != websocket_clients_.end()) {
        int client_socket = *it;
        
        try {
            sendWebSocketFrame(client_socket, message);
            ++it;
        } catch (const std::exception& e) {
            std::cout << "[HTTP] WebSocket 클라이언트 전송 실패: " << e.what() << std::endl;
            close(client_socket);
            it = websocket_clients_.erase(it);
        }
    }
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
            
            // WebSocket 업그레이드 요청인지 확인
            if (isWebSocketRequest(request)) {
                std::string response = handleWebSocketUpgrade(request, client_fd);
                write(client_fd, response.c_str(), response.length());
                
                // WebSocket 클라이언트로 등록하고 별도 스레드에서 처리
                {
                    std::lock_guard<std::mutex> lock(websocket_clients_mutex_);
                    websocket_clients_.push_back(client_fd);
                }
                
                std::thread ws_thread(&HttpServer::handleWebSocketClient, this, client_fd);
                ws_thread.detach();
                
                std::cout << "[HTTP] WebSocket 클라이언트 연결됨: " << client_fd << std::endl;
            } else {
                // 일반 HTTP 요청 처리
                std::string response = processRequest(request);
                write(client_fd, response.c_str(), response.length());
                close(client_fd);
            }
        } else {
            close(client_fd);
        }
    }
    
    close(server_fd);
}

std::string HttpServer::processRequest(const HttpRequest& request) {
    std::cout << "[HTTP] 요청 처리: " << request.method << " " << request.path << std::endl;
    
    // CORS 헤더 추가
    std::string cors_headers = "Access-Control-Allow-Origin: *\r\n"
                              "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                              "Access-Control-Allow-Headers: Content-Type\r\n";
    
    // OPTIONS 요청 처리 (CORS preflight)
    if (request.method == "OPTIONS") {
        return createHttpResponse(200, "text/plain", "", cors_headers);
    }
    
    // API 엔드포인트 라우팅
    if (request.path == "/auth/ssn" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleAuthSSN(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/auth/patient_id" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleAuthPatientId(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/auth/rfid" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleAuthRFID(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/auth/direction" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleAuthDirection(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/status/robot_return" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleRobotReturn(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/robot_status" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleRobotStatus(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/without_auth/direction" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleWithoutAuthDirection(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/api/config/llm" && request.method == "GET") {
        Json::Value json_request;
        std::string response = handleGetLLMConfig(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    // 테이블 API 엔드포인트들 (실제 명세)
    else if (request.path == "/auth/login" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleAuthLogin(json_request);
        return createHttpResponse(200, "text/plain", response, cors_headers);
    }
    else if (request.path == "/auth/detail" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleAuthDetail(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/get/robot_location" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetRobotLocation(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/change/camera" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleChangeCamera(json_request);
        return createHttpResponse(200, "text/plain", response, cors_headers);
    }
    else if (request.path == "/get/robot_status" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetRobotStatus(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/get/patient_info" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetPatientInfo(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/stop/status_moving" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleStopStatusMoving(json_request);
        return createHttpResponse(200, "text/plain", response, cors_headers);
    }
    else if (request.path == "/cancel_command" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleCancelCommand(json_request);
        return createHttpResponse(200, "text/plain", response, cors_headers);
    }
    else if (request.path == "/command/move_teleop" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleCommandMoveTeleop(json_request);
        return createHttpResponse(200, "text/plain", response, cors_headers);
    }
    else if (request.path == "/command/move_dest" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleCommandMoveDest(json_request);
        return createHttpResponse(200, "text/plain", response, cors_headers);
    }
    else if (request.path == "/get/log_data" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetLogData(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/get/heatmap" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetHeatmap(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/ws" && request.method == "GET") {
        // WebSocket 연결 요청은 이미 위에서 처리됨
        return createHttpResponse(400, "text/plain", "WebSocket upgrade failed");
    }
    else {
        return createHttpResponse(404, "text/plain", "Not Found");
    }
}

bool HttpServer::isWebSocketRequest(const HttpRequest& request) {
    return request.method == "GET" && 
           request.path == "/ws" && 
           request.headers.find("Upgrade") != request.headers.end() &&
           request.headers.at("Upgrade") == "websocket";
}

std::string HttpServer::handleWebSocketUpgrade(const HttpRequest& request, int client_socket) {
    std::string client_key = request.headers.at("Sec-WebSocket-Key");
    std::string accept_key = generateWebSocketAcceptKey(client_key);
    
    std::string response = "HTTP/1.1 101 Switching Protocols\r\n"
                          "Upgrade: websocket\r\n"
                          "Connection: Upgrade\r\n"
                          "Sec-WebSocket-Accept: " + accept_key + "\r\n"
                          "\r\n";
    
    return response;
}

std::string HttpServer::generateWebSocketAcceptKey(const std::string& client_key) {
    const std::string magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string concatenated = client_key + magic_string;
    
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(concatenated.c_str()), concatenated.length(), hash);
    
    // Base64 인코딩
    BIO* bio = BIO_new(BIO_s_mem());
    BIO* b64 = BIO_new(BIO_f_base64());
    bio = BIO_push(b64, bio);
    
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, hash, SHA_DIGEST_LENGTH);
    BIO_flush(bio);
    
    BUF_MEM* bufferPtr;
    BIO_get_mem_ptr(bio, &bufferPtr);
    
    std::string result(bufferPtr->data, bufferPtr->length);
    
    BIO_free_all(bio);
    
    return result;
}

void HttpServer::handleWebSocketClient(int client_socket) {
    char buffer[1024];
    
    while (running_) {
        ssize_t bytes_read = recv(client_socket, buffer, sizeof(buffer), 0);
        
        if (bytes_read <= 0) {
            break;
        }
        
        // WebSocket 프레임 파싱 (간단한 구현)
        // 실제로는 더 복잡한 WebSocket 프로토콜 파싱이 필요
        std::cout << "[HTTP] WebSocket 메시지 수신: " << std::string(buffer, bytes_read) << std::endl;
    }
    
    removeWebSocketClient(client_socket);
    std::cout << "[HTTP] WebSocket 클라이언트 연결 종료: " << client_socket << std::endl;
}

void HttpServer::sendWebSocketFrame(int client_socket, const std::string& message) {
    // 간단한 WebSocket 텍스트 프레임 생성
    std::vector<unsigned char> frame;
    
    // FIN + RSV + Opcode (텍스트 프레임)
    frame.push_back(0x81);
    
    // Payload length
    if (message.length() < 126) {
        frame.push_back(message.length());
    } else if (message.length() < 65536) {
        frame.push_back(126);
        frame.push_back((message.length() >> 8) & 0xFF);
        frame.push_back(message.length() & 0xFF);
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; --i) {
            frame.push_back((message.length() >> (i * 8)) & 0xFF);
        }
    }
    
    // Payload
    frame.insert(frame.end(), message.begin(), message.end());
    
    send(client_socket, frame.data(), frame.size(), 0);
}

void HttpServer::removeWebSocketClient(int client_socket) {
    std::lock_guard<std::mutex> lock(websocket_clients_mutex_);
    auto it = std::find(websocket_clients_.begin(), websocket_clients_.end(), client_socket);
    if (it != websocket_clients_.end()) {
        websocket_clients_.erase(it);
    }
}

std::string HttpServer::handleGetLLMConfig(const Json::Value& request) {
    Json::Value response;
    response["ip"] = "192.168.0.31";  // minje_pc
    response["port"] = 5000;
    return response.toStyledString();
}

// 공통 인증 로직
std::string HttpServer::handleCommonAuth(const PatientInfo& patient) {
    // 오늘 날짜 조회
    std::string current_date = db_manager_->getCurrentDate();
    
    // Series 정보와 Department 이름 조회 (series_id = 0)
    SeriesInfo series;
    std::string department_name;
    if (db_manager_->getSeriesWithDepartmentName(patient.patient_id, current_date, series, department_name)) {
        // 변경 전 상태 저장
        std::string original_status = series.status;
        
        // status가 '예약' 상태일 때 '접수'로 변경
        if (series.status == "예약") {
            if (db_manager_->updateSeriesStatus(patient.patient_id, current_date, "접수")) {
                std::cout << "[HTTP] 환자 상태 변경: 예약 -> 접수" << std::endl;
            }
        }
        
        // dttm 값을 그대로 사용 (YYYY-MM-DD HH:MM:SS 형식)
        std::string datetime = series.dttm;
        
        return createSuccessResponse(patient.name, datetime, department_name, original_status);
    } else {
        // Series 정보가 없는 경우
        return createErrorResponse("Reservation not found");
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
        return handleCommonAuth(patient);
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
        return handleCommonAuth(patient);
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
        return handleCommonAuth(patient);
    } else {
        return createErrorResponse("Patient not found");
    }
}

std::string HttpServer::handleAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("department_id")) {
        return createErrorResponse("Missing robot_id or department_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int department_id = request["department_id"].asInt();
    
    // TODO: 네비게이션 명령 처리
    std::cout << "[HTTP] 네비게이션 명령: Robot " << robot_id << " -> Department " << department_id << std::endl;
    
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
    if (!request.isMember("robot_id") || !request.isMember("department_id")) {
        return createErrorResponse("Missing robot_id or department_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int department_id = request["department_id"].asInt();
    
    // TODO: 인증 없는 네비게이션 명령 처리
    std::cout << "[HTTP] 인증 없는 네비게이션: Robot " << robot_id << " -> Department " << department_id << std::endl;
    
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
                                            const std::string& datetime, 
                                            const std::string& department,
                                            const std::string& status) {
    Json::Value response;
    response["name"] = name;
    response["datetime"] = datetime;  // YY:DD:HH:MM 형식
    response["department"] = department;
    response["status"] = status;
    
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
                                          const std::string& body,
                                          const std::string& additional_headers) {
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
    
    // 기본 CORS 헤더
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n";
    response << "Access-Control-Allow-Headers: Content-Type\r\n";
    
    // 추가 헤더가 있으면 추가
    if (!additional_headers.empty()) {
        response << additional_headers;
    }
    
    response << "\r\n";
    response << body;
    
    return response.str();
}

// 테이블 API 핸들러들 구현 (실제 명세)

std::string HttpServer::handleAuthLogin(const Json::Value& request) {
    std::cout << "[HTTP] 로그인 요청 처리 (IF-01)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("admin_id") || !request.isMember("password")) {
        return "400"; // Bad Request
    }
    
    std::string admin_id = request["admin_id"].asString(); // 명세서에서는 patient_id로 오지만 실제로는 admin_id
    std::string password = request["password"].asString();
    
    // Admin 테이블에서 로그인 검증
    AdminInfo admin;
    if (db_manager_->authenticateAdmin(admin_id, password, admin)) {
        return "200"; // 성공
    } else {
        return "401"; // 인증 실패
    }
}

std::string HttpServer::handleAuthDetail(const Json::Value& request) {
    std::cout << "[HTTP] 세부 정보 요청 처리 (IF-02)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("admin_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: admin_id");
    }
    
    std::string admin_id = request["admin_id"].asString(); // 명세서에서는 patient_id로 오지만 실제로는 admin_id
    
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

std::string HttpServer::handleGetRobotLocation(const Json::Value& request) {
    std::cout << "[HTTP] 로봇 위치 요청 처리 (IF-03)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 위치 정보 조회
    // IF-03 명세에 따라 응답
    Json::Value response;
    response["x"] = 5.0;
    response["y"] = -1.0;
    response["yaw"] = -0.532151;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::handleChangeCamera(const Json::Value& request) {
    std::cout << "[HTTP] 카메라 변경 요청 처리 (IF-04)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("camera")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string camera = request["camera"].asString();
    
    // TODO: 실제 로봇 시스템에서 카메라 변경 명령 전송
    // IF-04 명세에 따라 status code 반환
    return "200"; // 성공
}

std::string HttpServer::handleGetRobotStatus(const Json::Value& request) {
    std::cout << "[HTTP] 로봇 상태 요청 처리 (IF-05)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 상태 정보 조회
    // IF-05 명세에 따라 응답
    Json::Value response;
    response["status"] = "moving";
    response["orig"] = 0;
    response["dest"] = 3;
    response["battery"] = 70;
    response["network"] = 4;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::handleGetPatientInfo(const Json::Value& request) {
    std::cout << "[HTTP] 환자 정보 요청 처리 (IF-06)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 데이터베이스에서 해당 로봇을 이용중인 환자 정보 조회
    // IF-06 명세에 따라 응답 (ohone은 phone의 오타로 보임)
    Json::Value response;
    response["patient_id"] = "00000000";
    response["phone"] = "010-1111-1111";
    response["rfid"] = "33F7ADEC";
    response["name"] = "김환자";
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::handleStopStatusMoving(const Json::Value& request) {
    std::cout << "[HTTP] 이동 중 정지 요청 처리 (IF-07)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 moving → assigned 상태 변경
    // IF-07 명세에 따라 status code 반환
    return "200"; // 성공
}

std::string HttpServer::handleCancelCommand(const Json::Value& request) {
    std::cout << "[HTTP] 원격 제어 취소 요청 처리 (IF-08)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    
    // TODO: 실제 로봇 시스템에서 assigned/moving → idle 상태 변경 및 대기장소로 이동
    // IF-08 명세에 따라 status code 반환
    return "200"; // 성공
}

std::string HttpServer::handleCommandMoveTeleop(const Json::Value& request) {
    std::cout << "[HTTP] teleop 이동 명령 처리 (IF-09)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("teleop_key")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    int teleop_key = request["teleop_key"].asInt();
    
    // TODO: 실제 로봇 시스템에서 teleop 키에 따른 이동 명령 전송
    // teleop_key 매핑: 123=uio, 456=jkl, 789=m,.
    // IF-09 명세에 따라 status code 반환
    return "200"; // 성공
}

std::string HttpServer::handleCommandMoveDest(const Json::Value& request) {
    std::cout << "[HTTP] 목적지 이동 명령 처리 (IF-10)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("dest")) {
        return "400"; // Bad Request
    }
    
    int robot_id = request["robot_id"].asInt();
    int dest = request["dest"].asInt();
    
    // TODO: 실제 로봇 시스템에서 목적지로 이동 명령 전송
    // IF-10 명세에 따라 status code 반환
    return "200"; // 성공
}

std::string HttpServer::handleGetLogData(const Json::Value& request) {
    std::cout << "[HTTP] 로봇 로그 데이터 요청 처리 (IF-11)" << std::endl;
    
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
    
    // TODO: 실제 데이터베이스에서 로그 데이터 조회
    // 현재는 더미 데이터로 응답
    Json::Value response = Json::Value(Json::arrayValue);
    
    // 더미 로그 데이터 생성
    for (int i = 0; i < 3; i++) {
        Json::Value log_entry;
        log_entry["patient_id"] = 00000000;
        log_entry["orig"] = 0;
        log_entry["dest"] = 3;
        log_entry["date"] = "2024-01-15 14:30:00";
        log_entry["is_checked"] = 0;
        log_entry["video_url"] = "video_34.mp4";
        log_entry["favorite"] = 0;
        
        response.append(log_entry);
    }
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::handleGetHeatmap(const Json::Value& request) {
    std::cout << "[HTTP] 히트맵 데이터 요청 처리 (IF-12)" << std::endl;
    
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
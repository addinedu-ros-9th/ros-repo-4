#include "central_server/http_server.h"
#include "central_server/config.h"
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
    
    // 로봇 위치 초기화
    current_robot_position_.x = 0.0;
    current_robot_position_.y = 0.0;
    current_robot_position_.yaw = 0.0;
    current_robot_position_.valid = false;
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
    
    try {
    
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
    else if (request.path == "/auth/robot_return" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleRobotReturn(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/without_auth/robot_return" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleWithoutAuthRobotReturn(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/change/robot_status" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleRobotStatus(json_request);
        int status_code = std::stoi(response);
        return createHttpResponse(status_code, "text/plain", response, cors_headers);
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
        int status_code = std::stoi(response);
        return createHttpResponse(status_code, "text/plain", response, cors_headers);
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
        std::string response = handleChangeCamera(json_request);//AI 서버의 카메라 전환 로직 구현이 안 되어 추후 구현
        int status_code = std::stoi(response);
        return createHttpResponse(status_code, "text/plain", response, cors_headers);
    }
    else if (request.path == "/get/robot_status" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetRobotStatus(json_request);//추후 구현
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/get/patient_info" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetPatientInfo(json_request);//추후 구현
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/stop/status_moving" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleStopStatusMoving(json_request);
        int status_code = std::stoi(response);
        return createHttpResponse(status_code, "text/plain", response, cors_headers);
    }
    else if (request.path == "/cancel_command" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleCancelCommand(json_request);
        int status_code = std::stoi(response);
        return createHttpResponse(status_code, "text/plain", response, cors_headers);
    }
    else if (request.path == "/command/move_teleop" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleCommandMoveTeleop(json_request);
        int status_code = std::stoi(response);
        return createHttpResponse(status_code, "text/plain", response, cors_headers);
    }
    else if (request.path == "/command/move_dest" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleCommandMoveDest(json_request);
        int status_code = std::stoi(response);
        return createHttpResponse(status_code, "text/plain", response, cors_headers);
    }
    else if (request.path == "/get/log_data" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetLogData(json_request);
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/get/heatmap" && request.method == "POST") {
        Json::Value json_request = parseJson(request.body);
        std::string response = handleGetHeatmap(json_request);//히트맵에서 다른 방식으로 전환하기로 해서 추후 구현
        return createHttpResponse(200, "application/json", response, cors_headers);
    }
    else if (request.path == "/ws" && request.method == "GET") {
        // WebSocket 연결 요청은 이미 위에서 처리됨
        return createHttpResponse(400, "text/plain", "WebSocket upgrade failed");
    }
    else {
        return createHttpResponse(404, "text/plain", "Not Found");
    }
    
    } catch (const std::exception& e) {
        std::cerr << "[HTTP] 요청 처리 중 예외 발생: " << e.what() << std::endl;
        return createHttpResponse(500, "application/json", createErrorResponse("Internal server error: " + std::string(e.what())), cors_headers);
    } catch (...) {
        std::cerr << "[HTTP] 요청 처리 중 알 수 없는 예외 발생" << std::endl;
        return createHttpResponse(500, "application/json", createErrorResponse("Internal server error"), cors_headers);
    }
}

bool HttpServer::isWebSocketRequest(const HttpRequest& request) {
    return request.method == "GET" && 
           request.path == "/ws" && 
           request.headers.find("Upgrade") != request.headers.end() &&
           request.headers.at("Upgrade") == "websocket";
}

std::string HttpServer::handleWebSocketUpgrade(const HttpRequest& request, int) {
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

std::string HttpServer::handleGetLLMConfig(const Json::Value&) {
    Json::Value response;
    response["ip"] = Config::LLM_SERVER_IP;  // LLM 서버 IP
    response["port"] = Config::LLM_SERVER_PORT;
    return response.toStyledString();
}

// 공통 인증 로직
std::string HttpServer::handleCommonAuth(const PatientInfo& patient) {
    // 오늘 날짜의 "예약" 상태인 건을 조회
    SeriesInfo series;
    std::string department_name;
    if (db_manager_->getTodayReservationWithDepartmentName(patient.patient_id, series, department_name)) {
        // 변경 전 상태 저장
        std::string original_status = series.status;
        
        // status가 '예약' 상태일 때 '접수'로 변경
        if (series.status == "예약") {
            if (db_manager_->updateSeriesStatus(patient.patient_id, series.reservation_date, "접수")) {
                std::cout << "[HTTP] 환자 상태 변경: 예약 -> 접수" << std::endl;
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

// API 핸들러들
std::string HttpServer::handleAuthSSN(const Json::Value& request) {
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

std::string HttpServer::handleAuthPatientId(const Json::Value& request) {
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

std::string HttpServer::handleAuthRFID(const Json::Value& request) {
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

std::string HttpServer::handleAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("department_id") || !request.isMember("patient_id")) {
        return createErrorResponse("Missing robot_id, department_id, or patient_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int department_id = request["department_id"].asInt();
    int patient_id = request["patient_id"].asInt();
    
    return processDirectionRequest(robot_id, department_id, &patient_id, "moving_by_patient");
}

std::string HttpServer::handleRobotReturn(const Json::Value& request) {
    std::cout << "[HTTP] 로봇 복귀 요청 처리 (IF-06)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("patient_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id, patient_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    std::string patient_id_str = request["patient_id"].asString();
    int patient_id = std::stoi(patient_id_str);
    
    return processRobotReturnRequest(robot_id, &patient_id, "return_by_patient");
}

std::string HttpServer::handleWithoutAuthDirection(const Json::Value& request) {
    if (!request.isMember("robot_id") || !request.isMember("department_id")) {
        return createErrorResponse("Missing robot_id or department_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    int department_id = request["department_id"].asInt();
    
    return processDirectionRequest(robot_id, department_id, nullptr, "moving_by_unknown");
}

std::string HttpServer::handleRobotStatus(const Json::Value& request) {
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
        std::cout << "[HTTP] 로봇 정지 명령 전송 완료: Robot " << robot_id << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return "200"; // 성공
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
        case 400: status_text = "Bad Request"; break;
        case 401: status_text = "Unauthorized"; break;
        case 404: status_text = "Not Found"; break;
        case 405: status_text = "Method Not Allowed"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 503: status_text = "Service Unavailable"; break;
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
        std::cout << "[HTTP] 로그인 요청: admin_id 또는 password 누락" << std::endl;
        return "404"; // Missing fields
    }
    
    std::string admin_id = request["admin_id"].asString();
    std::string password = request["password"].asString();
    
    // 빈 필드 검증
    if (admin_id.empty() || password.empty()) {
        std::cout << "[HTTP] 로그인 요청: 빈 필드 존재" << std::endl;
        return "404"; // Empty fields
    }
    
    // 먼저 admin_id가 데이터베이스에 존재하는지 확인
    if (!db_manager_->isAdminIdExists(admin_id)) {
        std::cout << "[HTTP] 로그인 실패: 존재하지 않는 admin_id - " << admin_id << std::endl;
        return "401"; // ID doesn't exist
    }
    
    // Admin 테이블에서 로그인 검증 (ID는 존재하므로 비밀번호만 검증)
    AdminInfo admin;
    if (db_manager_->authenticateAdmin(admin_id, password, admin)) {
        std::cout << "[HTTP] 로그인 성공: " << admin_id << std::endl;
        return "200"; // 성공
    } else {
        std::cout << "[HTTP] 로그인 실패: 비밀번호 불일치 - " << admin_id << std::endl;
        return "402"; // Wrong password
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

    
    int robot_id;
    if (request["robot_id"].isString()) {
        robot_id = std::stoi(request["robot_id"].asString());
    } else if (request["robot_id"].isInt()) {
        robot_id = request["robot_id"].asInt();
    } else {
        return createErrorResponse("robot_id는 정수 또는 문자열이어야 합니다");
    }
    
    // 실제 로봇 위치 정보 조회 (amcl_pose에서 받은 데이터)
    std::lock_guard<std::mutex> lock(robot_position_mutex_);
    if (!current_robot_position_.valid) {
        return createErrorResponse("로봇 위치 정보를 사용할 수 없습니다");
    }
    
    Json::Value response;
    response["x"] = current_robot_position_.x;
    response["y"] = current_robot_position_.y;
    response["yaw"] = current_robot_position_.yaw;
    
    std::cout << "[HTTP] 로봇 위치 정보 반환: Robot " << robot_id 
              << " - 위치: (" << current_robot_position_.x << ", " << current_robot_position_.y 
              << "), 방향: " << current_robot_position_.yaw << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::handleChangeCamera(const Json::Value& request) {//AI 서버의 카메라 전환 로직 구현이 안 되어 추후 구현
    std::cout << "[HTTP] 카메라 변경 요청 처리 (IF-04)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id") || !request.isMember("camera")) {
        return "400"; // Bad Request
    }
    
    std::string camera = request["camera"].asString();
    
    // TODO: 실제 로봇 시스템에서 카메라 변경 명령 전송
    // IF-04 명세에 따라 status code 반환
    return "200"; // 성공
}

std::string HttpServer::handleGetRobotStatus(const Json::Value& request) {//현재 로봇 side 코드 상태가 구현이 안되어 있어서 추후 수정 필요
    std::cout << "[HTTP] 로봇 상태 요청 처리 (IF-05)" << std::endl;
    
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
    
    // IF-05 명세에 따라 응답
    Json::Value response;
    response["status"] = nav_status.empty() ? "unknown" : nav_status;
    response["orig"] = 0;  // TODO: 실제 출발지 정보 조회
    response["dest"] = 3;  // TODO: 실제 목적지 정보 조회
    response["battery"] = 70;  // TODO: 실제 배터리 정보 조회
    response["network"] = 4;   // TODO: 실제 네트워크 정보 조회
    
    std::cout << "[HTTP] 로봇 상태 정보 반환: Robot " << robot_id 
              << " - Status: " << response["status"].asString() << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::handleGetPatientInfo(const Json::Value& request) {//로봇의 상태를 체크하고 로그를 뒤져 확인해야 하는데 상태 미구현으로 추후 구현
    std::cout << "[HTTP] 환자 정보 요청 처리 (IF-06)" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
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
    
    // 실제 로봇 시스템에서 정지 명령 전송
    if (nav_manager_) {
        if (!nav_manager_->sendStopCommand()) {
            return "500"; // Internal Server Error
        }
        std::cout << "[HTTP] 로봇 정지 명령 전송 완료: Robot " << robot_id << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
    return "200"; // 성공
}

std::string HttpServer::handleCancelCommand(const Json::Value& request) {
    std::cout << "[HTTP] 원격 제어 취소 요청 처리 (IF-08)" << std::endl;
    
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
        
        std::cout << "[HTTP] 로봇 원격 제어 취소 명령 전송 완료: Robot " << robot_id << " (정지 후 " << lobby_department.department_name << "로 이동)" << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
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
        std::cout << "[HTTP] 로봇 teleop 명령 전송 완료: Robot " << robot_id 
                  << ", Key: " << teleop_key << ", Command: " << teleop_command << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
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
        std::cout << "[HTTP] 로봇 목적지 이동 명령 전송 완료: Robot " << robot_id 
                  << ", Dest: " << dest << ", Waypoint: " << waypoint_name << std::endl;
    } else {
        return "503"; // Service Unavailable
    }
    
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
    
    std::cout << "[HTTP] 로그 데이터 응답 생성 완료: " << response.size() << "개 레코드" << std::endl;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::handleGetHeatmap(const Json::Value& request) {//히트맵에서 다른 방식으로 전환하기로 해서 추후 구현
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

std::string HttpServer::processRobotReturnRequest(int robot_id, int* patient_id, const std::string& log_type) {
    // 1. 현재 로봇 위치 확인 (amcl_pose)
    double current_x, current_y;
    {
        std::lock_guard<std::mutex> lock(robot_position_mutex_);
        if (!current_robot_position_.valid) {
            return createErrorResponse("로봇 위치 정보를 사용할 수 없습니다");
        }
        current_x = current_robot_position_.x;
        current_y = current_robot_position_.y;
    }
    
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
        std::cout << "[HTTP] 로봇 복귀 명령 전송 완료: " << lobby_department.department_name << "로 이동" << std::endl;
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
    
    std::cout << "[HTTP] 로봇 복귀 명령 처리 완료: Robot " << robot_id 
              << " -> 대기장소 (ID: 8), Patient: " << (patient_id ? std::to_string(*patient_id) : "NULL") 
              << ", Type: " << log_type << std::endl;
    
    // 5. 응답 반환
    Json::Value response;
    response["status_code"] = 200;
    response["dest"] = 8;  // 대기장소의 목적지 ID
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, response);
}

std::string HttpServer::processDirectionRequest(int robot_id, int department_id, int* patient_id, const std::string& log_type) {
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
        std::cout << "[HTTP] 로봇 네비게이션 명령 전송 완료: " << department.department_name << "로 이동" << std::endl;
    } else {
        return createErrorResponse("Navigation manager not available");
    }
    
    // 3. 현재 로봇 위치 확인 (읽기만 하므로 mutex 불필요)
    double current_x, current_y;
    {
        std::lock_guard<std::mutex> lock(robot_position_mutex_);
        if (!current_robot_position_.valid) {
            return createErrorResponse("Robot position not available");
        }
        current_x = current_robot_position_.x;
        current_y = current_robot_position_.y;
    } // mutex 해제
    
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
    
    std::cout << "[HTTP] 네비게이션 명령 처리 완료: Robot " << robot_id 
              << " -> Department " << department.department_name << " (ID: " << department_id 
              << "), Patient: " << (patient_id ? std::to_string(*patient_id) : "NULL") 
              << ", Type: " << log_type << std::endl;
    
    return createStatusResponse(200);
}

void HttpServer::setRobotNavigationManager(std::shared_ptr<RobotNavigationManager> nav_manager) {
    nav_manager_ = nav_manager;
    
    if (nav_manager_) {
        // 로봇 위치 콜백 설정
        nav_manager_->setRobotPoseCallback([this](double x, double y, double yaw) {
            std::lock_guard<std::mutex> lock(robot_position_mutex_);
            current_robot_position_.x = x;
            current_robot_position_.y = y;
            current_robot_position_.yaw = yaw;
            current_robot_position_.valid = true;
        });
        
        std::cout << "[HTTP] 로봇 네비게이션 관리자 설정 완료" << std::endl;
    }
}

std::string HttpServer::handleWithoutAuthRobotReturn(const Json::Value& request) {
    std::cout << "[HTTP] 비인증 로봇 복귀 요청 처리" << std::endl;
    
    // 요청 데이터 검증
    if (!request.isMember("robot_id")) {
        return createErrorResponse("필수 필드가 누락되었습니다: robot_id");
    }
    
    int robot_id = request["robot_id"].asInt();
    
    return processRobotReturnRequest(robot_id, nullptr, "return_by_unknown");
}
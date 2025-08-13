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
    
    // 요청 핸들러들 초기화 (nav_manager_는 나중에 setRobotNavigationManager에서 설정)
    admin_handler_ = std::make_unique<AdminRequestHandler>(db_manager_, nullptr);
    user_handler_ = std::make_unique<UserRequestHandler>(db_manager_, nullptr, nullptr);
    ai_handler_ = std::make_unique<AiRequestHandler>(db_manager_, nullptr);
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
            
            // HTTP 요청 처리
            std::string response = processRequest(request);
            // 응답 전체가 전송될 때까지 반복 전송
            size_t total_sent = 0;
            const char* resp_data = response.c_str();
            const size_t resp_size = response.size();
            while (total_sent < resp_size) {
                ssize_t sent = write(client_fd, resp_data + total_sent, resp_size - total_sent);
                if (sent <= 0) {
                    break;
                }
                total_sent += static_cast<size_t>(sent);
            }
            close(client_fd);
        } else {
            close(client_fd);
        }
    }
    
    close(server_fd);
}

std::string HttpServer::processRequest(const HttpRequest& request) {
    std::cout << "[HTTP] 요청 처리: " << request.method << " " << request.path << std::endl;
    std::cout << "[HTTP] 요청 바디: '" << request.body << "'" << std::endl;
    
    // CORS 헤더 추가
    std::string cors_headers = "Access-Control-Allow-Origin: *\r\n"
                              "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                              "Access-Control-Allow-Headers: Content-Type\r\n";
    
    // OPTIONS 요청 처리 (CORS preflight)
    if (request.method == "OPTIONS") {
        return createHttpResponse(200, "text/plain", "", cors_headers);
    }
    
    try {
        // 공통 응답 처리 함수
        auto processHandlerResponse = [&](const std::string& response) -> std::string {
            int status_code = 200;
            std::string content_type = "application/json";
            
            try {
                Json::Value res_json = parseJson(response);
                
                // JSON 객체인 경우에만 isMember() 사용
                if (res_json.isObject()) {
                    if (res_json.isMember("status_code") && res_json["status_code"].isInt()) {
                        status_code = res_json["status_code"].asInt();
                    } else if (res_json.isMember("error")) {
                        status_code = 400; // 클라이언트 에러
                    }
                }
            } catch (const std::exception& e) {
                // JSON 파싱 실패 시 숫자 문자열인지 확인
                try {
                    status_code = std::stoi(response);
                    content_type = "text/plain";
                } catch (...) {
                    status_code = 500; // 서버 에러
                    content_type = "application/json";
                }
            }
            
            return createHttpResponse(status_code, content_type, response, cors_headers);
        };
    
        // User GUI API 엔드포인트 라우팅
        if (request.path == "/auth/ssn" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleAuthSSN(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/auth/patient_id" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleAuthPatientId(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/auth/rfid" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleAuthRFID(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/auth/direction" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleAuthDirection(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/auth/robot_return" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleRobotReturn(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/without_auth/robot_return" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleWithoutAuthRobotReturn(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/change/robot_status" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleRobotStatus(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/without_auth/direction" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleWithoutAuthDirection(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/api/config/llm" && request.method == "GET") {
            Json::Value json_request;
            std::string response = user_handler_->handleGetLLMConfig(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/call_with_voice" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleCallWithVoice(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/call_with_screen" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleCallWithScreen(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/alert_timeout" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleAlertTimeout(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/pause_request" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handlePauseRequest(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/restart_navigation" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleRestartNavigation(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/stop_navigating" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = user_handler_->handleStopNavigating(json_request);
            return processHandlerResponse(response);
        }

        // Admin GUI API 엔드포인트 라우팅
        else if (request.path == "/auth/login" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleAuthLogin(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/auth/detail" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleAuthDetail(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/get/robot_location" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleGetRobotLocation(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/get/robot_status" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleGetRobotStatus(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/get/patient_info" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleGetPatientInfo(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/control_by_admin" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleControlByAdmin(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/return_command" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleReturnCommand(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/teleop_request" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleTeleopRequest(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/teleop_complete" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleTeleopComplete(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/command/move_teleop" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleCommandMoveTeleop(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/command/move_dest" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleCommandMoveDest(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/cancel_navigating" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleCancelNavigating(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/get/log_data" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = admin_handler_->handleGetLogData(json_request);
            return processHandlerResponse(response);
        }
        // AI 서버 HTTP 엔드포인트 라우팅
        else if (request.path == "/gesture/come" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = ai_handler_->handleGestureCome(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/user_disappear" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = ai_handler_->handleUserDisappear(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/user_appear" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = ai_handler_->handleUserAppear(json_request);
            return processHandlerResponse(response);
        }
        else if (request.path == "/stop_tracking" && request.method == "POST") {
            Json::Value json_request = parseJson(request.body);
            std::string response = ai_handler_->handleStopTracking(json_request);
            return processHandlerResponse(response);
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





// 유틸리티 함수들
Json::Value HttpServer::parseJson(const std::string& jsonStr) {
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    // 빈 문자열이면 빈 오브젝트로 처리 (옵션 파라미터 허용 엔드포인트 호환)
    if (jsonStr.empty()) {
        return Json::Value(Json::objectValue);
    }
    
    std::istringstream stream(jsonStr);
    if (!Json::parseFromStream(builder, stream, &root, &errors)) {
        throw std::runtime_error("JSON 파싱 실패: " + errors + " (입력: '" + jsonStr + "')");
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
    while (std::getline(stream, line)) {
        // \r 제거
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        // 빈 줄이면 헤더 끝, 바디 시작
        if (line.empty()) {
            break;
        }
        
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string header_name = line.substr(0, colon_pos);
            std::string header_value = line.substr(colon_pos + 2); // ": " 건너뛰기
            
            http_request.headers[header_name] = header_value;
        }
    }
    
    // 바디 읽기
    std::string body;
    std::string body_line;
    while (std::getline(stream, body_line)) {
        // \r 제거
        if (!body_line.empty() && body_line.back() == '\r') {
            body_line.pop_back();
        }
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
        
        // 모든 핸들러에 nav_manager 설정
        admin_handler_ = std::make_unique<AdminRequestHandler>(db_manager_, nav_manager);
        user_handler_ = std::make_unique<UserRequestHandler>(db_manager_, nav_manager, nullptr);
        ai_handler_ = std::make_unique<AiRequestHandler>(db_manager_, nav_manager, websocket_server_);
        
        std::cout << "[HTTP] 로봇 네비게이션 관리자 설정 완료" << std::endl;
    }
}

void HttpServer::setWebSocketServer(std::shared_ptr<WebSocketServer> websocket_server) {
    websocket_server_ = websocket_server;
    if (ai_handler_) {
        ai_handler_->setWebSocketServer(websocket_server_);
    }
}
#include "central_server/websocket_server.h"
#include <sstream>

WebSocketServer::WebSocketServer(int port)
    : port_(port), server_socket_(-1), running_(false) {
}

WebSocketServer::~WebSocketServer() {
    stop();
}

bool WebSocketServer::start() {
    if (running_) {
        std::cout << "[WebSocket] 서버가 이미 실행 중입니다" << std::endl;
        return true;
    }
    
    // 소켓 생성
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        std::cerr << "[WebSocket] 소켓 생성 실패" << std::endl;
        return false;
    }
    
    // 소켓 옵션 설정 (재사용 가능)
    int opt = 1;
    setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // 주소 설정
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);
    
    // 바인딩
    if (bind(server_socket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[WebSocket] 바인딩 실패" << std::endl;
        close(server_socket_);
        server_socket_ = -1;
        return false;
    }
    
    // 리스닝 시작
    if (listen(server_socket_, 5) < 0) {
        std::cerr << "[WebSocket] 리스닝 실패" << std::endl;
        close(server_socket_);
        server_socket_ = -1;
        return false;
    }
    
    running_ = true;
    server_thread_ = std::thread(&WebSocketServer::serverLoop, this);
    
    std::cout << "[WebSocket] 서버 시작됨 (포트: " << port_ << ")" << std::endl;
    return true;
}

void WebSocketServer::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // 모든 클라이언트 연결 종료
    for (auto& pair : clients_by_ip_) {
        close(pair.second.socket_fd);
    }
    clients_by_ip_.clear();
    clients_by_socket_.clear();
    
    // 서버 소켓 종료
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
    
    // 스레드 종료 대기
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    std::cout << "[WebSocket] 서버 중지됨" << std::endl;
}

void WebSocketServer::broadcastMessage(const std::string& message) {
    auto it = clients_by_ip_.begin();
    while (it != clients_by_ip_.end()) {
        const std::string& ip = it->first;
        ClientInfo& client = it->second;
        
        if (sendWebSocketFrame(client.socket_fd, message)) {
            ++it;
        } else {
            // 전송 실패 시 클라이언트 제거
            std::cout << "[WebSocket] 클라이언트 " << ip << " 연결 종료 (전송 실패)" << std::endl;
            close(client.socket_fd);
            clients_by_socket_.erase(client.socket_fd);
            it = clients_by_ip_.erase(it);
        }
    }
}

bool WebSocketServer::sendMessageToIP(const std::string& ip_address, const std::string& message) {
    auto it = clients_by_ip_.find(ip_address);
    if (it == clients_by_ip_.end()) {
        std::cerr << "[WebSocket] IP " << ip_address << "에 연결된 클라이언트가 없습니다" << std::endl;
        return false;
    }
    
    ClientInfo& client = it->second;
    if (sendWebSocketFrame(client.socket_fd, message)) {
        std::cout << "[WebSocket] IP " << ip_address << "로 메시지 전송 성공" << std::endl;
        return true;
    } else {
        // 전송 실패 시 클라이언트 제거
        std::cout << "[WebSocket] 클라이언트 " << ip_address << " 연결 종료 (전송 실패)" << std::endl;
        close(client.socket_fd);
        clients_by_socket_.erase(client.socket_fd);
        clients_by_ip_.erase(it);
        return false;
    }
}

bool WebSocketServer::sendMessageToClient(int client_socket, const std::string& message) {
    auto it = clients_by_socket_.find(client_socket);
    if (it == clients_by_socket_.end()) {
        std::cerr << "[WebSocket] 소켓 " << client_socket << "에 해당하는 클라이언트가 없습니다" << std::endl;
        return false;
    }
    
    return sendWebSocketFrame(client_socket, message);
}

size_t WebSocketServer::getClientCount() const {
    return clients_by_ip_.size();
}

std::vector<std::string> WebSocketServer::getConnectedIPs() const {
    std::vector<std::string> ips;
    for (const auto& pair : clients_by_ip_) {
        ips.push_back(pair.first);
    }
    return ips;
}

bool WebSocketServer::isClientConnected(const std::string& ip_address) const {
    return clients_by_ip_.find(ip_address) != clients_by_ip_.end();
}

void WebSocketServer::broadcastMessageToType(const std::string& client_type, const std::string& message) {
    auto it = clients_by_ip_.begin();
    while (it != clients_by_ip_.end()) {
        const std::string& ip = it->first;
        ClientInfo& client = it->second;
        
        // 지정된 타입의 클라이언트에게만 전송
        if (client.client_type == client_type) {
            if (sendWebSocketFrame(client.socket_fd, message)) {
                ++it;
            } else {
                // 전송 실패 시 클라이언트 제거
                std::cout << "[WebSocket] 클라이언트 " << ip << " 연결 종료 (전송 실패)" << std::endl;
                int socket_fd = client.socket_fd;
                it = clients_by_ip_.erase(it);
                clients_by_socket_.erase(socket_fd);
                // removeClient를 호출하지 않고 직접 close만 수행
                close(socket_fd);
            }
        } else {
            ++it;
        }
    }
}

std::vector<std::string> WebSocketServer::getClientsByType(const std::string& client_type) const {
    std::vector<std::string> ips;
    
    for (const auto& pair : clients_by_ip_) {
        if (pair.second.client_type == client_type) {
            ips.push_back(pair.first);
        }
    }
    
    return ips;
}

bool WebSocketServer::setClientType(const std::string& ip_address, const std::string& client_type) {
    auto it = clients_by_ip_.find(ip_address);
    if (it == clients_by_ip_.end()) {
        std::cerr << "[WebSocket] IP " << ip_address << "에 연결된 클라이언트가 없습니다" << std::endl;
        return false;
    }
    
    it->second.client_type = client_type;
    std::cout << "[WebSocket] 클라이언트 " << ip_address << " 타입을 " << client_type << "로 설정" << std::endl;
    return true;
}

// 로봇 알림 메시지 전송 함수들

void WebSocketServer::sendAlertOccupied(int robot_id, std::string type) {
    std::cout << "[WebSocket] 관리자 클라이언트들에게 화면 호출 알림 전송 중..." << std::endl;
    
    // JSON 메시지 생성
    Json::Value message;
    message["type"] = "alert_occupied";
    message["robot_id"] = robot_id;
    message["timestamp"] = std::to_string(time(nullptr));
    
    Json::StreamWriterBuilder builder;
    std::string json_message = Json::writeString(builder, message);
    
    // 관리자 클라이언트들에게만 브로드캐스트
    broadcastMessageToType(type, json_message);
    
    std::cout << "[WebSocket] " << type << "에게 알림 전송 완료: Robot " << robot_id << std::endl;
}

void WebSocketServer::sendAlertIdle(int robot_id) {
    std::cout << "[WebSocket] 모든 클라이언트에게 사용 가능한 상태 알림 전송 중..." << std::endl;
    
    // JSON 메시지 생성
    Json::Value message;
    message["type"] = "alert_idle";
    message["robot_id"] = robot_id;
    message["timestamp"] = std::to_string(time(nullptr));
    
    Json::StreamWriterBuilder builder;
    std::string json_message = Json::writeString(builder, message);
    
    // 모든 클라이언트에게 브로드캐스트
    broadcastMessage(json_message);
    
    std::cout << "[WebSocket] 모든 클라이언트에게 사용 가능한 상태 알림 전송 완료: Robot " << robot_id << std::endl;
}

void WebSocketServer::sendNavigatingComplete(int robot_id) {
    std::cout << "[WebSocket] GUI 클라이언트들에게 길안내 완료 알림 전송 중..." << std::endl;
    
    // JSON 메시지 생성
    Json::Value message;
    message["type"] = "navigating_complete";
    message["robot_id"] = robot_id;
    message["timestamp"] = std::to_string(time(nullptr));
    
    Json::StreamWriterBuilder builder;
    std::string json_message = Json::writeString(builder, message);
    
    // GUI 클라이언트들에게만 브로드캐스트
    broadcastMessageToType("gui", json_message);
    
    std::cout << "[WebSocket] GUI 클라이언트들에게 길안내 완료 알림 전송 완료: Robot " << robot_id << std::endl;
}

void WebSocketServer::serverLoop() {
    std::cout << "[WebSocket] 서버 루프 시작" << std::endl;
    
    while (running_) {
        struct sockaddr_in client_address;
        socklen_t client_len = sizeof(client_address);
        
        int client_socket = accept(server_socket_, (struct sockaddr*)&client_address, &client_len);
        if (client_socket < 0) {
            if (running_) {
                std::cerr << "[WebSocket] 클라이언트 연결 실패" << std::endl;
            }
            continue;
        }
        
        // 클라이언트 IP 주소 추출
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_address.sin_addr, ip_str, INET_ADDRSTRLEN);
        std::string client_ip(ip_str);
        
        std::cout << "[WebSocket] 새 클라이언트 연결: " << client_socket << " (IP: " << client_ip << ")" << std::endl;
        
        // 클라이언트를 별도 스레드에서 처리
        std::thread client_thread(&WebSocketServer::handleClient, this, client_socket, client_ip);
        client_thread.detach();
    }
}

void WebSocketServer::handleClient(int client_socket, const std::string& client_ip) {
    char buffer[4096] = {0};
    ssize_t bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    
    if (bytes_received > 0) {
        std::string request(buffer, bytes_received);
        
        if (isWebSocketRequest(request)) {
            // WebSocket 핸드셰이크 처리
            std::string client_key = extractWebSocketKey(request);
            std::string response = createWebSocketHandshakeResponse(client_key);
            
            if (send(client_socket, response.c_str(), response.length(), 0) > 0) {
                // 클라이언트 목록에 추가
                ClientInfo client_info(client_socket, client_ip);
                
                // 쿼리 파라미터에서 클라이언트 타입 추출
                std::string client_type = extractClientTypeFromRequest(request);
                if (!client_type.empty()) {
                    client_info.client_type = client_type;
                    std::cout << "[WebSocket] 클라이언트 " << client_ip << " 타입을 " << client_type << "로 설정" << std::endl;
                }
                
                clients_by_ip_[client_ip] = client_info;
                clients_by_socket_[client_socket] = client_ip;
                
                std::cout << "[WebSocket] 클라이언트 " << client_ip << " (소켓: " << client_socket << ") WebSocket 연결 완료" << std::endl;
                
                // 클라이언트 메시지 수신 대기
                while (running_) {
                    bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
                    if (bytes_received <= 0) {
                        break;
                    }
                    
                    // WebSocket 프레임 파싱 (간단한 구현)
                    std::cout << "[WebSocket] 클라이언트 " << client_ip << " 메시지 수신" << std::endl;
                }
            }
        } else {
            // 일반 HTTP 요청 처리
            std::string response = "HTTP/1.1 400 Bad Request\r\n\r\n";
            send(client_socket, response.c_str(), response.length(), 0);
        }
    }
    
    // 클라이언트 연결 종료
    std::cout << "[WebSocket] 클라이언트 " << client_ip << " (소켓: " << client_socket << ") 연결 종료" << std::endl;
    removeClient(client_socket);
}

bool WebSocketServer::isWebSocketRequest(const std::string& request) {
    return request.find("GET") == 0 && 
           request.find("Upgrade: websocket") != std::string::npos &&
           request.find("Sec-WebSocket-Key:") != std::string::npos;
}

std::string WebSocketServer::extractWebSocketKey(const std::string& request) {
    size_t key_pos = request.find("Sec-WebSocket-Key:");
    if (key_pos == std::string::npos) {
        return "";
    }
    
    size_t start = key_pos + 19; // "Sec-WebSocket-Key:" 길이
    size_t end = request.find("\r\n", start);
    if (end == std::string::npos) {
        return "";
    }
    
    return request.substr(start, end - start);
}

std::string WebSocketServer::extractClientTypeFromRequest(const std::string& request) {
    // GET 요청의 첫 번째 줄에서 URL 추출
    size_t get_pos = request.find("GET ");
    if (get_pos == std::string::npos) {
        return "";
    }
    
    size_t url_start = get_pos + 4; // "GET " 길이
    size_t url_end = request.find(" HTTP/", url_start);
    if (url_end == std::string::npos) {
        return "";
    }
    
    std::string url = request.substr(url_start, url_end - url_start);
    
    // 쿼리 파라미터에서 client_type 추출
    size_t query_pos = url.find("?client_type=");
    if (query_pos == std::string::npos) {
        return "";
    }
    
    size_t type_start = query_pos + 13; // "?client_type=" 길이
    size_t type_end = url.find("&", type_start);
    if (type_end == std::string::npos) {
        type_end = url.length();
    }
    
    return url.substr(type_start, type_end - type_start);
}

std::string WebSocketServer::generateWebSocketAcceptKey(const std::string& client_key) {
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

std::string WebSocketServer::createWebSocketHandshakeResponse(const std::string& client_key) {
    std::string accept_key = generateWebSocketAcceptKey(client_key);
    
    std::string response = 
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + accept_key + "\r\n"
        "\r\n";
    
    return response;
}

bool WebSocketServer::sendWebSocketFrame(int client_socket, const std::string& message) {
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
    
    return send(client_socket, frame.data(), frame.size(), 0) > 0;
}

void WebSocketServer::removeClient(int client_socket) {
    auto it = clients_by_socket_.find(client_socket);
    if (it != clients_by_socket_.end()) {
        const std::string& ip = it->second;
        clients_by_socket_.erase(it);
        
        auto ip_it = clients_by_ip_.find(ip);
        if (ip_it != clients_by_ip_.end()) {
            clients_by_ip_.erase(ip_it);
        }
        
        std::cout << "[WebSocket] 클라이언트 " << ip << " (소켓: " << client_socket << ") 제거됨" << std::endl;
        
        // 소켓 닫기 (클라이언트가 존재할 때만)
        close(client_socket);
    }
} 
#ifndef CENTRAL_SERVER_CONFIG_H
#define CENTRAL_SERVER_CONFIG_H

#include <string>

namespace Config {
    // 서버 IP 주소들
    const std::string CENTRAL_SERVER_IP = "192.168.0.37";
    const std::string LLM_SERVER_IP = "192.168.0.37";  // LLM 서버 IP (별도 설정 가능)
    
    // 포트 설정
    const int CENTRAL_SERVER_PORT = 8080;
    const int LLM_SERVER_PORT = 5000;
    
    // ESP32 IP 주소
    const std::string ESP32_IP = "192.168.0.34";
}

#endif // CENTRAL_SERVER_CONFIG_H 
#ifndef CENTRAL_SERVER_CONFIG_H
#define CENTRAL_SERVER_CONFIG_H

#include <string>

namespace Config {
    // 서버 IP 주소들
    const std::string CENTRAL_SERVER_IP = "192.168.0.10";
    const std::string LLM_SERVER_IP = "192.168.0.10";  // LLM 서버 IP (별도 설정 가능)
    
    // 포트 설정
    const int CENTRAL_SERVER_PORT = 8080;
    const int LLM_SERVER_PORT = 5000;
    
    // ESP32 IP 주소
    const std::string ESP32_IP = "192.168.0.34";
    
    // GUI 설정
    const std::string GUI_IP = "192.168.1.100";
    const int GUI_PORT = 3000;
    const int GUI_TIMEOUT = 5;  // 초
}

#endif // CENTRAL_SERVER_CONFIG_H 
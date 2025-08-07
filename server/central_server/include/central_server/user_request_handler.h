#ifndef USER_REQUEST_HANDLER_H
#define USER_REQUEST_HANDLER_H

#include <json/json.h>
#include <memory>
#include <string>
#include "central_server/database_manager.h"
#include "central_server/robot_navigation_manager.h"
#include "central_server/websocket_server.h"

class UserRequestHandler {
public:
    UserRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                      std::shared_ptr<RobotNavigationManager> nav_manager);
    ~UserRequestHandler() = default;

    // User GUI API 핸들러들
    std::string handleAuthSSN(const Json::Value& request);
    std::string handleAuthPatientId(const Json::Value& request);
    std::string handleAuthRFID(const Json::Value& request);
    std::string handleAuthDirection(const Json::Value& request);
    std::string handleRobotReturn(const Json::Value& request);
    std::string handleWithoutAuthDirection(const Json::Value& request);
    std::string handleWithoutAuthRobotReturn(const Json::Value& request);
    std::string handleRobotStatus(const Json::Value& request);
    std::string handleGetLLMConfig(const Json::Value& request);
    std::string handleCallWithVoice(const Json::Value& request);
    std::string handleCallWithScreen(const Json::Value& request);
    std::string handleAlertTimeout(const Json::Value& request);
    std::string handlePauseRequest(const Json::Value& request);
    std::string handleRestartNavigation(const Json::Value& request);
    std::string handleStopNavigating(const Json::Value& request);
    

    

    
    // 연결된 클라이언트 정보 조회
    std::vector<std::string> getConnectedClients() const;
    bool isClientConnected(const std::string& ip_address) const;
    
    // 메시지 전송 함수들
    void sendAlertOccupied(int robot_id);                    // 모든 클라이언트에게
    void sendAlertIdle(int robot_id);                        // 모든 클라이언트에게
    void sendNavigatingComplete(int robot_id);               // GUI 클라이언트에게만
    
    // 클라이언트 타입 관리
    bool setClientType(const std::string& ip_address, const std::string& client_type);

private:
    std::shared_ptr<DatabaseManager> db_manager_;
    std::shared_ptr<RobotNavigationManager> nav_manager_;
    std::unique_ptr<WebSocketServer> websocket_server_;
    
    // 공통 인증 로직
    std::string handleCommonAuth(const PatientInfo& patient);
    
    // 네비게이션 처리 함수들
    std::string processDirectionRequest(int robot_id, int department_id, int* patient_id, const std::string& log_type);
    std::string processRobotReturnRequest(int robot_id, int* patient_id, const std::string& log_type);

    // 유틸리티 함수들
    std::string createErrorResponse(const std::string& message);
    std::string createSuccessResponse(const std::string& name, 
                                    const std::string& datetime, 
                                    const std::string& department,
                                    const std::string& status,
                                    int patient_id);
    std::string createStatusResponse(int status_code);
};

#endif // USER_REQUEST_HANDLER_H 
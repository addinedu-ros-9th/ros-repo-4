#ifndef ADMIN_REQUEST_HANDLER_H
#define ADMIN_REQUEST_HANDLER_H

#include <json/json.h>
#include <memory>
#include <string>
#include "central_server/database_manager.h"
#include "central_server/robot_navigation_manager.h"
#include "central_server/websocket_server.h"

class AdminRequestHandler {
public:
    AdminRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                       std::shared_ptr<RobotNavigationManager> nav_manager,
                       std::shared_ptr<WebSocketServer> websocket_server);
    ~AdminRequestHandler() = default;

    // Admin GUI API 핸들러들
    std::string handleAuthLogin(const Json::Value& request);
    std::string handleAuthDetail(const Json::Value& request);
    std::string handleGetRobotLocation(const Json::Value& request);
    std::string handleGetRobotStatus(const Json::Value& request);
    std::string handleGetPatientInfo(const Json::Value& request);
    std::string handleControlByAdmin(const Json::Value& request);
    std::string handleReturnCommand(const Json::Value& request);
    std::string handleTeleopRequest(const Json::Value& request);
    std::string handleTeleopComplete(const Json::Value& request);
    std::string handleCommandMoveTeleop(const Json::Value& request);
    std::string handleCommandMoveDest(const Json::Value& request);
    std::string handleCancelNavigating(const Json::Value& request);
    std::string handleGetLogData(const Json::Value& request);
    std::string handleGetHeatmap(const Json::Value& request);

private:
    std::shared_ptr<DatabaseManager> db_manager_;
    std::shared_ptr<RobotNavigationManager> nav_manager_;
    std::shared_ptr<WebSocketServer> websocket_server_;

    // 유틸리티 함수들
    std::string createErrorResponse(const std::string& message);
    std::string createSuccessResponse(const Json::Value& data);
    std::string createStatusResponse(int status_code);
    
    // 공통 로직 함수들
    std::string sendControlCommand(int robot_id, int* patient_id, const std::string& log_type, 
                                  const std::string& command_name, const std::string& admin_id);
};

#endif // ADMIN_REQUEST_HANDLER_H 
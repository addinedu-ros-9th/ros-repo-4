#ifndef ROBOT_REQUEST_HANDLER_H
#define ROBOT_REQUEST_HANDLER_H

#include <json/json.h>
#include <memory>
#include <string>
#include "central_server/database_manager.h"
#include "central_server/robot_navigation_manager.h"

class RobotRequestHandler {
public:
    RobotRequestHandler(std::shared_ptr<DatabaseManager> db_manager, 
                       std::shared_ptr<RobotNavigationManager> nav_manager);
    ~RobotRequestHandler() = default;

    // 로봇 관련 API 핸들러들 (central → GUI)
    std::string handleAlertOccupied(const Json::Value& request);
    std::string handleAlertIdle(const Json::Value& request);
    std::string handleNavigatingComplete(const Json::Value& request);

private:
    std::shared_ptr<DatabaseManager> db_manager_;
    std::shared_ptr<RobotNavigationManager> nav_manager_;

    // 유틸리티 함수들
    std::string createErrorResponse(const std::string& message);
    std::string createStatusResponse(int status_code);
};

#endif // ROBOT_REQUEST_HANDLER_H 
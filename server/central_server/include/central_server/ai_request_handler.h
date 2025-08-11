#ifndef AI_REQUEST_HANDLER_H
#define AI_REQUEST_HANDLER_H

#include <json/json.h>
#include <memory>
#include <string>


#include "central_server/database_manager.h"
#include "central_server/robot_navigation_manager.h"
#include "central_server/websocket_server.h"

class AiRequestHandler {
public:
    AiRequestHandler(std::shared_ptr<DatabaseManager> db_manager,
                     std::shared_ptr<RobotNavigationManager> nav_manager,
                     std::shared_ptr<WebSocketServer> websocket_server = nullptr);
    ~AiRequestHandler() = default;

    // IF-03: 손동작(come) 인식 이벤트 송신 /트래킹
    std::string handleGestureCome(const Json::Value& request);

    // IF-04: 길안내 중 사람 사라짐
    std::string handleUserDisappear(const Json::Value& request);

    // IF-05: 사라졌던 사람 다시 나타남
    std::string handleUserAppear(const Json::Value& request);

    // IF-06: 트래킹 중지
    std::string handleStopTracking(const Json::Value& request);

    void setRobotNavigationManager(std::shared_ptr<RobotNavigationManager> nav_manager);
    void setWebSocketServer(std::shared_ptr<WebSocketServer> websocket_server);

private:
    std::shared_ptr<DatabaseManager> db_manager_;
    std::shared_ptr<RobotNavigationManager> nav_manager_;
    std::shared_ptr<WebSocketServer> websocket_server_;



    std::string createStatusResponse(int status_code);
    std::string createErrorResponse(const std::string& message);
};

#endif // AI_REQUEST_HANDLER_H


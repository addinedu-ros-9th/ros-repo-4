// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_central_interfaces:srv/NavigateRobot.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/srv/navigate_robot.hpp"


#ifndef ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__BUILDER_HPP_
#define ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_central_interfaces/srv/detail/navigate_robot__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_central_interfaces
{

namespace srv
{

namespace builder
{

class Init_NavigateRobot_Request_waypoints
{
public:
  explicit Init_NavigateRobot_Request_waypoints(::robot_central_interfaces::srv::NavigateRobot_Request & msg)
  : msg_(msg)
  {}
  ::robot_central_interfaces::srv::NavigateRobot_Request waypoints(::robot_central_interfaces::srv::NavigateRobot_Request::_waypoints_type arg)
  {
    msg_.waypoints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_central_interfaces::srv::NavigateRobot_Request msg_;
};

class Init_NavigateRobot_Request_robot_id
{
public:
  Init_NavigateRobot_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateRobot_Request_waypoints robot_id(::robot_central_interfaces::srv::NavigateRobot_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_NavigateRobot_Request_waypoints(msg_);
  }

private:
  ::robot_central_interfaces::srv::NavigateRobot_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_central_interfaces::srv::NavigateRobot_Request>()
{
  return robot_central_interfaces::srv::builder::Init_NavigateRobot_Request_robot_id();
}

}  // namespace robot_central_interfaces


namespace robot_central_interfaces
{

namespace srv
{

namespace builder
{

class Init_NavigateRobot_Response_message
{
public:
  explicit Init_NavigateRobot_Response_message(::robot_central_interfaces::srv::NavigateRobot_Response & msg)
  : msg_(msg)
  {}
  ::robot_central_interfaces::srv::NavigateRobot_Response message(::robot_central_interfaces::srv::NavigateRobot_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_central_interfaces::srv::NavigateRobot_Response msg_;
};

class Init_NavigateRobot_Response_success
{
public:
  Init_NavigateRobot_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateRobot_Response_message success(::robot_central_interfaces::srv::NavigateRobot_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_NavigateRobot_Response_message(msg_);
  }

private:
  ::robot_central_interfaces::srv::NavigateRobot_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_central_interfaces::srv::NavigateRobot_Response>()
{
  return robot_central_interfaces::srv::builder::Init_NavigateRobot_Response_success();
}

}  // namespace robot_central_interfaces


namespace robot_central_interfaces
{

namespace srv
{

namespace builder
{

class Init_NavigateRobot_Event_response
{
public:
  explicit Init_NavigateRobot_Event_response(::robot_central_interfaces::srv::NavigateRobot_Event & msg)
  : msg_(msg)
  {}
  ::robot_central_interfaces::srv::NavigateRobot_Event response(::robot_central_interfaces::srv::NavigateRobot_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_central_interfaces::srv::NavigateRobot_Event msg_;
};

class Init_NavigateRobot_Event_request
{
public:
  explicit Init_NavigateRobot_Event_request(::robot_central_interfaces::srv::NavigateRobot_Event & msg)
  : msg_(msg)
  {}
  Init_NavigateRobot_Event_response request(::robot_central_interfaces::srv::NavigateRobot_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_NavigateRobot_Event_response(msg_);
  }

private:
  ::robot_central_interfaces::srv::NavigateRobot_Event msg_;
};

class Init_NavigateRobot_Event_info
{
public:
  Init_NavigateRobot_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateRobot_Event_request info(::robot_central_interfaces::srv::NavigateRobot_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_NavigateRobot_Event_request(msg_);
  }

private:
  ::robot_central_interfaces::srv::NavigateRobot_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_central_interfaces::srv::NavigateRobot_Event>()
{
  return robot_central_interfaces::srv::builder::Init_NavigateRobot_Event_info();
}

}  // namespace robot_central_interfaces

#endif  // ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__BUILDER_HPP_

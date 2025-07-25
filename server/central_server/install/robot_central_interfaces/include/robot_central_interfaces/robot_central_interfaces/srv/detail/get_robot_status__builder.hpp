// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_central_interfaces:srv/GetRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/srv/get_robot_status.hpp"


#ifndef ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__GET_ROBOT_STATUS__BUILDER_HPP_
#define ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__GET_ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_central_interfaces/srv/detail/get_robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_central_interfaces
{

namespace srv
{

namespace builder
{

class Init_GetRobotStatus_Request_robot_id
{
public:
  Init_GetRobotStatus_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::robot_central_interfaces::srv::GetRobotStatus_Request robot_id(::robot_central_interfaces::srv::GetRobotStatus_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_central_interfaces::srv::GetRobotStatus_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_central_interfaces::srv::GetRobotStatus_Request>()
{
  return robot_central_interfaces::srv::builder::Init_GetRobotStatus_Request_robot_id();
}

}  // namespace robot_central_interfaces


namespace robot_central_interfaces
{

namespace srv
{

namespace builder
{

class Init_GetRobotStatus_Response_robot_statuses
{
public:
  explicit Init_GetRobotStatus_Response_robot_statuses(::robot_central_interfaces::srv::GetRobotStatus_Response & msg)
  : msg_(msg)
  {}
  ::robot_central_interfaces::srv::GetRobotStatus_Response robot_statuses(::robot_central_interfaces::srv::GetRobotStatus_Response::_robot_statuses_type arg)
  {
    msg_.robot_statuses = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_central_interfaces::srv::GetRobotStatus_Response msg_;
};

class Init_GetRobotStatus_Response_message
{
public:
  explicit Init_GetRobotStatus_Response_message(::robot_central_interfaces::srv::GetRobotStatus_Response & msg)
  : msg_(msg)
  {}
  Init_GetRobotStatus_Response_robot_statuses message(::robot_central_interfaces::srv::GetRobotStatus_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return Init_GetRobotStatus_Response_robot_statuses(msg_);
  }

private:
  ::robot_central_interfaces::srv::GetRobotStatus_Response msg_;
};

class Init_GetRobotStatus_Response_success
{
public:
  Init_GetRobotStatus_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetRobotStatus_Response_message success(::robot_central_interfaces::srv::GetRobotStatus_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetRobotStatus_Response_message(msg_);
  }

private:
  ::robot_central_interfaces::srv::GetRobotStatus_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_central_interfaces::srv::GetRobotStatus_Response>()
{
  return robot_central_interfaces::srv::builder::Init_GetRobotStatus_Response_success();
}

}  // namespace robot_central_interfaces


namespace robot_central_interfaces
{

namespace srv
{

namespace builder
{

class Init_GetRobotStatus_Event_response
{
public:
  explicit Init_GetRobotStatus_Event_response(::robot_central_interfaces::srv::GetRobotStatus_Event & msg)
  : msg_(msg)
  {}
  ::robot_central_interfaces::srv::GetRobotStatus_Event response(::robot_central_interfaces::srv::GetRobotStatus_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_central_interfaces::srv::GetRobotStatus_Event msg_;
};

class Init_GetRobotStatus_Event_request
{
public:
  explicit Init_GetRobotStatus_Event_request(::robot_central_interfaces::srv::GetRobotStatus_Event & msg)
  : msg_(msg)
  {}
  Init_GetRobotStatus_Event_response request(::robot_central_interfaces::srv::GetRobotStatus_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_GetRobotStatus_Event_response(msg_);
  }

private:
  ::robot_central_interfaces::srv::GetRobotStatus_Event msg_;
};

class Init_GetRobotStatus_Event_info
{
public:
  Init_GetRobotStatus_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetRobotStatus_Event_request info(::robot_central_interfaces::srv::GetRobotStatus_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_GetRobotStatus_Event_request(msg_);
  }

private:
  ::robot_central_interfaces::srv::GetRobotStatus_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_central_interfaces::srv::GetRobotStatus_Event>()
{
  return robot_central_interfaces::srv::builder::Init_GetRobotStatus_Event_info();
}

}  // namespace robot_central_interfaces

#endif  // ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__GET_ROBOT_STATUS__BUILDER_HPP_

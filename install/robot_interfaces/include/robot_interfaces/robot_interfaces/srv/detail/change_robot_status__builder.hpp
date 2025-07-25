// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_interfaces:srv/ChangeRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/srv/change_robot_status.hpp"


#ifndef ROBOT_INTERFACES__SRV__DETAIL__CHANGE_ROBOT_STATUS__BUILDER_HPP_
#define ROBOT_INTERFACES__SRV__DETAIL__CHANGE_ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_interfaces/srv/detail/change_robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_ChangeRobotStatus_Request_new_status
{
public:
  explicit Init_ChangeRobotStatus_Request_new_status(::robot_interfaces::srv::ChangeRobotStatus_Request & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::srv::ChangeRobotStatus_Request new_status(::robot_interfaces::srv::ChangeRobotStatus_Request::_new_status_type arg)
  {
    msg_.new_status = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::ChangeRobotStatus_Request msg_;
};

class Init_ChangeRobotStatus_Request_robot_id
{
public:
  Init_ChangeRobotStatus_Request_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ChangeRobotStatus_Request_new_status robot_id(::robot_interfaces::srv::ChangeRobotStatus_Request::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_ChangeRobotStatus_Request_new_status(msg_);
  }

private:
  ::robot_interfaces::srv::ChangeRobotStatus_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::ChangeRobotStatus_Request>()
{
  return robot_interfaces::srv::builder::Init_ChangeRobotStatus_Request_robot_id();
}

}  // namespace robot_interfaces


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_ChangeRobotStatus_Response_message
{
public:
  explicit Init_ChangeRobotStatus_Response_message(::robot_interfaces::srv::ChangeRobotStatus_Response & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::srv::ChangeRobotStatus_Response message(::robot_interfaces::srv::ChangeRobotStatus_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::ChangeRobotStatus_Response msg_;
};

class Init_ChangeRobotStatus_Response_success
{
public:
  Init_ChangeRobotStatus_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ChangeRobotStatus_Response_message success(::robot_interfaces::srv::ChangeRobotStatus_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_ChangeRobotStatus_Response_message(msg_);
  }

private:
  ::robot_interfaces::srv::ChangeRobotStatus_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::ChangeRobotStatus_Response>()
{
  return robot_interfaces::srv::builder::Init_ChangeRobotStatus_Response_success();
}

}  // namespace robot_interfaces


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_ChangeRobotStatus_Event_response
{
public:
  explicit Init_ChangeRobotStatus_Event_response(::robot_interfaces::srv::ChangeRobotStatus_Event & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::srv::ChangeRobotStatus_Event response(::robot_interfaces::srv::ChangeRobotStatus_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::ChangeRobotStatus_Event msg_;
};

class Init_ChangeRobotStatus_Event_request
{
public:
  explicit Init_ChangeRobotStatus_Event_request(::robot_interfaces::srv::ChangeRobotStatus_Event & msg)
  : msg_(msg)
  {}
  Init_ChangeRobotStatus_Event_response request(::robot_interfaces::srv::ChangeRobotStatus_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_ChangeRobotStatus_Event_response(msg_);
  }

private:
  ::robot_interfaces::srv::ChangeRobotStatus_Event msg_;
};

class Init_ChangeRobotStatus_Event_info
{
public:
  Init_ChangeRobotStatus_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ChangeRobotStatus_Event_request info(::robot_interfaces::srv::ChangeRobotStatus_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_ChangeRobotStatus_Event_request(msg_);
  }

private:
  ::robot_interfaces::srv::ChangeRobotStatus_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::ChangeRobotStatus_Event>()
{
  return robot_interfaces::srv::builder::Init_ChangeRobotStatus_Event_info();
}

}  // namespace robot_interfaces

#endif  // ROBOT_INTERFACES__SRV__DETAIL__CHANGE_ROBOT_STATUS__BUILDER_HPP_

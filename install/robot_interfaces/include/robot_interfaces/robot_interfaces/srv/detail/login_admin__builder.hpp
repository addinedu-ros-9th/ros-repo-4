// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_interfaces:srv/LoginAdmin.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/srv/login_admin.hpp"


#ifndef ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__BUILDER_HPP_
#define ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_interfaces/srv/detail/login_admin__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_LoginAdmin_Request_password
{
public:
  explicit Init_LoginAdmin_Request_password(::robot_interfaces::srv::LoginAdmin_Request & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::srv::LoginAdmin_Request password(::robot_interfaces::srv::LoginAdmin_Request::_password_type arg)
  {
    msg_.password = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Request msg_;
};

class Init_LoginAdmin_Request_id
{
public:
  Init_LoginAdmin_Request_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LoginAdmin_Request_password id(::robot_interfaces::srv::LoginAdmin_Request::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_LoginAdmin_Request_password(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::LoginAdmin_Request>()
{
  return robot_interfaces::srv::builder::Init_LoginAdmin_Request_id();
}

}  // namespace robot_interfaces


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_LoginAdmin_Response_hospital_name
{
public:
  explicit Init_LoginAdmin_Response_hospital_name(::robot_interfaces::srv::LoginAdmin_Response & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::srv::LoginAdmin_Response hospital_name(::robot_interfaces::srv::LoginAdmin_Response::_hospital_name_type arg)
  {
    msg_.hospital_name = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Response msg_;
};

class Init_LoginAdmin_Response_email
{
public:
  explicit Init_LoginAdmin_Response_email(::robot_interfaces::srv::LoginAdmin_Response & msg)
  : msg_(msg)
  {}
  Init_LoginAdmin_Response_hospital_name email(::robot_interfaces::srv::LoginAdmin_Response::_email_type arg)
  {
    msg_.email = std::move(arg);
    return Init_LoginAdmin_Response_hospital_name(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Response msg_;
};

class Init_LoginAdmin_Response_name
{
public:
  explicit Init_LoginAdmin_Response_name(::robot_interfaces::srv::LoginAdmin_Response & msg)
  : msg_(msg)
  {}
  Init_LoginAdmin_Response_email name(::robot_interfaces::srv::LoginAdmin_Response::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_LoginAdmin_Response_email(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Response msg_;
};

class Init_LoginAdmin_Response_success
{
public:
  Init_LoginAdmin_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LoginAdmin_Response_name success(::robot_interfaces::srv::LoginAdmin_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_LoginAdmin_Response_name(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::LoginAdmin_Response>()
{
  return robot_interfaces::srv::builder::Init_LoginAdmin_Response_success();
}

}  // namespace robot_interfaces


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_LoginAdmin_Event_response
{
public:
  explicit Init_LoginAdmin_Event_response(::robot_interfaces::srv::LoginAdmin_Event & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::srv::LoginAdmin_Event response(::robot_interfaces::srv::LoginAdmin_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Event msg_;
};

class Init_LoginAdmin_Event_request
{
public:
  explicit Init_LoginAdmin_Event_request(::robot_interfaces::srv::LoginAdmin_Event & msg)
  : msg_(msg)
  {}
  Init_LoginAdmin_Event_response request(::robot_interfaces::srv::LoginAdmin_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_LoginAdmin_Event_response(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Event msg_;
};

class Init_LoginAdmin_Event_info
{
public:
  Init_LoginAdmin_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LoginAdmin_Event_request info(::robot_interfaces::srv::LoginAdmin_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_LoginAdmin_Event_request(msg_);
  }

private:
  ::robot_interfaces::srv::LoginAdmin_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::LoginAdmin_Event>()
{
  return robot_interfaces::srv::builder::Init_LoginAdmin_Event_info();
}

}  // namespace robot_interfaces

#endif  // ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__BUILDER_HPP_

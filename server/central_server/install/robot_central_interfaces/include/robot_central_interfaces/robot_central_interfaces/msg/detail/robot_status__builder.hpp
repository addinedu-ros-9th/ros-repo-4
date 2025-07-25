// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_central_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/msg/robot_status.hpp"


#ifndef ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_
#define ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_central_interfaces/msg/detail/robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_central_interfaces
{

namespace msg
{

namespace builder
{

class Init_RobotStatus_last_update
{
public:
  explicit Init_RobotStatus_last_update(::robot_central_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  ::robot_central_interfaces::msg::RobotStatus last_update(::robot_central_interfaces::msg::RobotStatus::_last_update_type arg)
  {
    msg_.last_update = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_central_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_current_pose
{
public:
  explicit Init_RobotStatus_current_pose(::robot_central_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_last_update current_pose(::robot_central_interfaces::msg::RobotStatus::_current_pose_type arg)
  {
    msg_.current_pose = std::move(arg);
    return Init_RobotStatus_last_update(msg_);
  }

private:
  ::robot_central_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_current_task
{
public:
  explicit Init_RobotStatus_current_task(::robot_central_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_current_pose current_task(::robot_central_interfaces::msg::RobotStatus::_current_task_type arg)
  {
    msg_.current_task = std::move(arg);
    return Init_RobotStatus_current_pose(msg_);
  }

private:
  ::robot_central_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_status
{
public:
  explicit Init_RobotStatus_status(::robot_central_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_current_task status(::robot_central_interfaces::msg::RobotStatus::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_RobotStatus_current_task(msg_);
  }

private:
  ::robot_central_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_robot_id
{
public:
  Init_RobotStatus_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotStatus_status robot_id(::robot_central_interfaces::msg::RobotStatus::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_RobotStatus_status(msg_);
  }

private:
  ::robot_central_interfaces::msg::RobotStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_central_interfaces::msg::RobotStatus>()
{
  return robot_central_interfaces::msg::builder::Init_RobotStatus_robot_id();
}

}  // namespace robot_central_interfaces

#endif  // ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_

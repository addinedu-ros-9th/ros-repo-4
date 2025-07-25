// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_interfaces:msg/RobotFeedback.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/msg/robot_feedback.hpp"


#ifndef ROBOT_INTERFACES__MSG__DETAIL__ROBOT_FEEDBACK__BUILDER_HPP_
#define ROBOT_INTERFACES__MSG__DETAIL__ROBOT_FEEDBACK__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_interfaces/msg/detail/robot_feedback__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_interfaces
{

namespace msg
{

namespace builder
{

class Init_RobotFeedback_error_message
{
public:
  explicit Init_RobotFeedback_error_message(::robot_interfaces::msg::RobotFeedback & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::msg::RobotFeedback error_message(::robot_interfaces::msg::RobotFeedback::_error_message_type arg)
  {
    msg_.error_message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

class Init_RobotFeedback_detected_objects
{
public:
  explicit Init_RobotFeedback_detected_objects(::robot_interfaces::msg::RobotFeedback & msg)
  : msg_(msg)
  {}
  Init_RobotFeedback_error_message detected_objects(::robot_interfaces::msg::RobotFeedback::_detected_objects_type arg)
  {
    msg_.detected_objects = std::move(arg);
    return Init_RobotFeedback_error_message(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

class Init_RobotFeedback_current_speed
{
public:
  explicit Init_RobotFeedback_current_speed(::robot_interfaces::msg::RobotFeedback & msg)
  : msg_(msg)
  {}
  Init_RobotFeedback_detected_objects current_speed(::robot_interfaces::msg::RobotFeedback::_current_speed_type arg)
  {
    msg_.current_speed = std::move(arg);
    return Init_RobotFeedback_detected_objects(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

class Init_RobotFeedback_network_status
{
public:
  explicit Init_RobotFeedback_network_status(::robot_interfaces::msg::RobotFeedback & msg)
  : msg_(msg)
  {}
  Init_RobotFeedback_current_speed network_status(::robot_interfaces::msg::RobotFeedback::_network_status_type arg)
  {
    msg_.network_status = std::move(arg);
    return Init_RobotFeedback_current_speed(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

class Init_RobotFeedback_battery_percent
{
public:
  explicit Init_RobotFeedback_battery_percent(::robot_interfaces::msg::RobotFeedback & msg)
  : msg_(msg)
  {}
  Init_RobotFeedback_network_status battery_percent(::robot_interfaces::msg::RobotFeedback::_battery_percent_type arg)
  {
    msg_.battery_percent = std::move(arg);
    return Init_RobotFeedback_network_status(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

class Init_RobotFeedback_current_pose
{
public:
  explicit Init_RobotFeedback_current_pose(::robot_interfaces::msg::RobotFeedback & msg)
  : msg_(msg)
  {}
  Init_RobotFeedback_battery_percent current_pose(::robot_interfaces::msg::RobotFeedback::_current_pose_type arg)
  {
    msg_.current_pose = std::move(arg);
    return Init_RobotFeedback_battery_percent(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

class Init_RobotFeedback_status
{
public:
  explicit Init_RobotFeedback_status(::robot_interfaces::msg::RobotFeedback & msg)
  : msg_(msg)
  {}
  Init_RobotFeedback_current_pose status(::robot_interfaces::msg::RobotFeedback::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_RobotFeedback_current_pose(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

class Init_RobotFeedback_robot_id
{
public:
  Init_RobotFeedback_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotFeedback_status robot_id(::robot_interfaces::msg::RobotFeedback::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_RobotFeedback_status(msg_);
  }

private:
  ::robot_interfaces::msg::RobotFeedback msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::msg::RobotFeedback>()
{
  return robot_interfaces::msg::builder::Init_RobotFeedback_robot_id();
}

}  // namespace robot_interfaces

#endif  // ROBOT_INTERFACES__MSG__DETAIL__ROBOT_FEEDBACK__BUILDER_HPP_

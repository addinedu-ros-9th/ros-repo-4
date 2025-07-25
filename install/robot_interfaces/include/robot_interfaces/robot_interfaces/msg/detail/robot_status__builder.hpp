// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/msg/robot_status.hpp"


#ifndef ROBOT_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_
#define ROBOT_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_interfaces/msg/detail/robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_interfaces
{

namespace msg
{

namespace builder
{

class Init_RobotStatus_source
{
public:
  explicit Init_RobotStatus_source(::robot_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::msg::RobotStatus source(::robot_interfaces::msg::RobotStatus::_source_type arg)
  {
    msg_.source = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_network_status
{
public:
  explicit Init_RobotStatus_network_status(::robot_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_source network_status(::robot_interfaces::msg::RobotStatus::_network_status_type arg)
  {
    msg_.network_status = std::move(arg);
    return Init_RobotStatus_source(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_battery_percent
{
public:
  explicit Init_RobotStatus_battery_percent(::robot_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_network_status battery_percent(::robot_interfaces::msg::RobotStatus::_battery_percent_type arg)
  {
    msg_.battery_percent = std::move(arg);
    return Init_RobotStatus_network_status(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_destination_station
{
public:
  explicit Init_RobotStatus_destination_station(::robot_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_battery_percent destination_station(::robot_interfaces::msg::RobotStatus::_destination_station_type arg)
  {
    msg_.destination_station = std::move(arg);
    return Init_RobotStatus_battery_percent(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_current_station
{
public:
  explicit Init_RobotStatus_current_station(::robot_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_destination_station current_station(::robot_interfaces::msg::RobotStatus::_current_station_type arg)
  {
    msg_.current_station = std::move(arg);
    return Init_RobotStatus_destination_station(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_amcl_pose
{
public:
  explicit Init_RobotStatus_amcl_pose(::robot_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_current_station amcl_pose(::robot_interfaces::msg::RobotStatus::_amcl_pose_type arg)
  {
    msg_.amcl_pose = std::move(arg);
    return Init_RobotStatus_current_station(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_status
{
public:
  explicit Init_RobotStatus_status(::robot_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_amcl_pose status(::robot_interfaces::msg::RobotStatus::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_RobotStatus_amcl_pose(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_robot_id
{
public:
  Init_RobotStatus_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotStatus_status robot_id(::robot_interfaces::msg::RobotStatus::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_RobotStatus_status(msg_);
  }

private:
  ::robot_interfaces::msg::RobotStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::msg::RobotStatus>()
{
  return robot_interfaces::msg::builder::Init_RobotStatus_robot_id();
}

}  // namespace robot_interfaces

#endif  // ROBOT_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_

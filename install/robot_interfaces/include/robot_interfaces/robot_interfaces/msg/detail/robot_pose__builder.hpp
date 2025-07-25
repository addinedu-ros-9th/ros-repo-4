// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_interfaces:msg/RobotPose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/msg/robot_pose.hpp"


#ifndef ROBOT_INTERFACES__MSG__DETAIL__ROBOT_POSE__BUILDER_HPP_
#define ROBOT_INTERFACES__MSG__DETAIL__ROBOT_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_interfaces/msg/detail/robot_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_interfaces
{

namespace msg
{

namespace builder
{

class Init_RobotPose_timestamp
{
public:
  explicit Init_RobotPose_timestamp(::robot_interfaces::msg::RobotPose & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::msg::RobotPose timestamp(::robot_interfaces::msg::RobotPose::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::msg::RobotPose msg_;
};

class Init_RobotPose_pose
{
public:
  explicit Init_RobotPose_pose(::robot_interfaces::msg::RobotPose & msg)
  : msg_(msg)
  {}
  Init_RobotPose_timestamp pose(::robot_interfaces::msg::RobotPose::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_RobotPose_timestamp(msg_);
  }

private:
  ::robot_interfaces::msg::RobotPose msg_;
};

class Init_RobotPose_robot_id
{
public:
  Init_RobotPose_robot_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotPose_pose robot_id(::robot_interfaces::msg::RobotPose::_robot_id_type arg)
  {
    msg_.robot_id = std::move(arg);
    return Init_RobotPose_pose(msg_);
  }

private:
  ::robot_interfaces::msg::RobotPose msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::msg::RobotPose>()
{
  return robot_interfaces::msg::builder::Init_RobotPose_robot_id();
}

}  // namespace robot_interfaces

#endif  // ROBOT_INTERFACES__MSG__DETAIL__ROBOT_POSE__BUILDER_HPP_

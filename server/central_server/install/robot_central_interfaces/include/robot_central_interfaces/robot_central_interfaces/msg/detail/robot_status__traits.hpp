// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from robot_central_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/msg/robot_status.hpp"


#ifndef ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__TRAITS_HPP_
#define ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "robot_central_interfaces/msg/detail/robot_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'current_pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"
// Member 'last_update'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace robot_central_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotStatus & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_id
  {
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << ", ";
  }

  // member: status
  {
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << ", ";
  }

  // member: current_task
  {
    out << "current_task: ";
    rosidl_generator_traits::value_to_yaml(msg.current_task, out);
    out << ", ";
  }

  // member: current_pose
  {
    out << "current_pose: ";
    to_flow_style_yaml(msg.current_pose, out);
    out << ", ";
  }

  // member: last_update
  {
    out << "last_update: ";
    to_flow_style_yaml(msg.last_update, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: robot_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << "\n";
  }

  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << "\n";
  }

  // member: current_task
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_task: ";
    rosidl_generator_traits::value_to_yaml(msg.current_task, out);
    out << "\n";
  }

  // member: current_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_pose:\n";
    to_block_style_yaml(msg.current_pose, out, indentation + 2);
  }

  // member: last_update
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "last_update:\n";
    to_block_style_yaml(msg.last_update, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotStatus & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace robot_central_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_central_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_central_interfaces::msg::RobotStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_central_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_central_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const robot_central_interfaces::msg::RobotStatus & msg)
{
  return robot_central_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<robot_central_interfaces::msg::RobotStatus>()
{
  return "robot_central_interfaces::msg::RobotStatus";
}

template<>
inline const char * name<robot_central_interfaces::msg::RobotStatus>()
{
  return "robot_central_interfaces/msg/RobotStatus";
}

template<>
struct has_fixed_size<robot_central_interfaces::msg::RobotStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_central_interfaces::msg::RobotStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_central_interfaces::msg::RobotStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__TRAITS_HPP_

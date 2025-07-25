// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from robot_interfaces:action/NavigateToGoal.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/action/navigate_to_goal.hpp"


#ifndef ROBOT_INTERFACES__ACTION__DETAIL__NAVIGATE_TO_GOAL__TRAITS_HPP_
#define ROBOT_INTERFACES__ACTION__DETAIL__NAVIGATE_TO_GOAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "robot_interfaces/action/detail/navigate_to_goal__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose_stamped__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_Goal & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_id
  {
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
    out << ", ";
  }

  // member: target_pose
  {
    out << "target_pose: ";
    to_flow_style_yaml(msg.target_pose, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_Goal & msg,
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

  // member: target_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_pose:\n";
    to_block_style_yaml(msg.target_pose, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_Goal & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_Goal & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_Goal & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_Goal>()
{
  return "robot_interfaces::action::NavigateToGoal_Goal";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_Goal>()
{
  return "robot_interfaces/action/NavigateToGoal_Goal";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_Goal>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::PoseStamped>::value> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_Goal>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::PoseStamped>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_Goal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_Result & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_Result & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_Result & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_Result & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_Result & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_Result>()
{
  return "robot_interfaces::action::NavigateToGoal_Result";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_Result>()
{
  return "robot_interfaces/action/NavigateToGoal_Result";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_Result>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_Result>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_Result>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'current_pose'
// already included above
// #include "geometry_msgs/msg/detail/pose_stamped__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_Feedback & msg,
  std::ostream & out)
{
  out << "{";
  // member: current_pose
  {
    out << "current_pose: ";
    to_flow_style_yaml(msg.current_pose, out);
    out << ", ";
  }

  // member: current_status
  {
    out << "current_status: ";
    rosidl_generator_traits::value_to_yaml(msg.current_status, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: current_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_pose:\n";
    to_block_style_yaml(msg.current_pose, out, indentation + 2);
  }

  // member: current_status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_status: ";
    rosidl_generator_traits::value_to_yaml(msg.current_status, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_Feedback & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_Feedback & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_Feedback>()
{
  return "robot_interfaces::action::NavigateToGoal_Feedback";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_Feedback>()
{
  return "robot_interfaces/action/NavigateToGoal_Feedback";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_Feedback>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_Feedback>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_Feedback>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'goal'
#include "robot_interfaces/action/detail/navigate_to_goal__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_SendGoal_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: goal_id
  {
    out << "goal_id: ";
    to_flow_style_yaml(msg.goal_id, out);
    out << ", ";
  }

  // member: goal
  {
    out << "goal: ";
    to_flow_style_yaml(msg.goal, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_SendGoal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_block_style_yaml(msg.goal_id, out, indentation + 2);
  }

  // member: goal
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal:\n";
    to_block_style_yaml(msg.goal, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_SendGoal_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_SendGoal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_SendGoal_Request & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_SendGoal_Request>()
{
  return "robot_interfaces::action::NavigateToGoal_SendGoal_Request";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_SendGoal_Request>()
{
  return "robot_interfaces/action/NavigateToGoal_SendGoal_Request";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_SendGoal_Request>
  : std::integral_constant<bool, has_fixed_size<robot_interfaces::action::NavigateToGoal_Goal>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal_Request>
  : std::integral_constant<bool, has_bounded_size<robot_interfaces::action::NavigateToGoal_Goal>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_SendGoal_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_SendGoal_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: accepted
  {
    out << "accepted: ";
    rosidl_generator_traits::value_to_yaml(msg.accepted, out);
    out << ", ";
  }

  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_SendGoal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: accepted
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "accepted: ";
    rosidl_generator_traits::value_to_yaml(msg.accepted, out);
    out << "\n";
  }

  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_SendGoal_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_SendGoal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_SendGoal_Response & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_SendGoal_Response>()
{
  return "robot_interfaces::action::NavigateToGoal_SendGoal_Response";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_SendGoal_Response>()
{
  return "robot_interfaces/action/NavigateToGoal_SendGoal_Response";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_SendGoal_Response>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal_Response>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_SendGoal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_SendGoal_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_SendGoal_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_SendGoal_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_SendGoal_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_SendGoal_Event & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_SendGoal_Event>()
{
  return "robot_interfaces::action::NavigateToGoal_SendGoal_Event";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_SendGoal_Event>()
{
  return "robot_interfaces/action/NavigateToGoal_SendGoal_Event";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_SendGoal_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal_Event>
  : std::integral_constant<bool, has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal_Request>::value && has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_SendGoal_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_SendGoal>()
{
  return "robot_interfaces::action::NavigateToGoal_SendGoal";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_SendGoal>()
{
  return "robot_interfaces/action/NavigateToGoal_SendGoal";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_SendGoal>
  : std::integral_constant<
    bool,
    has_fixed_size<robot_interfaces::action::NavigateToGoal_SendGoal_Request>::value &&
    has_fixed_size<robot_interfaces::action::NavigateToGoal_SendGoal_Response>::value
  >
{
};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal>
  : std::integral_constant<
    bool,
    has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal_Request>::value &&
    has_bounded_size<robot_interfaces::action::NavigateToGoal_SendGoal_Response>::value
  >
{
};

template<>
struct is_service<robot_interfaces::action::NavigateToGoal_SendGoal>
  : std::true_type
{
};

template<>
struct is_service_request<robot_interfaces::action::NavigateToGoal_SendGoal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<robot_interfaces::action::NavigateToGoal_SendGoal_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_GetResult_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: goal_id
  {
    out << "goal_id: ";
    to_flow_style_yaml(msg.goal_id, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_GetResult_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_block_style_yaml(msg.goal_id, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_GetResult_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_GetResult_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_GetResult_Request & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_GetResult_Request>()
{
  return "robot_interfaces::action::NavigateToGoal_GetResult_Request";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_GetResult_Request>()
{
  return "robot_interfaces/action/NavigateToGoal_GetResult_Request";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_GetResult_Request>
  : std::integral_constant<bool, has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult_Request>
  : std::integral_constant<bool, has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_GetResult_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'result'
// already included above
// #include "robot_interfaces/action/detail/navigate_to_goal__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_GetResult_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: status
  {
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << ", ";
  }

  // member: result
  {
    out << "result: ";
    to_flow_style_yaml(msg.result, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_GetResult_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << "\n";
  }

  // member: result
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "result:\n";
    to_block_style_yaml(msg.result, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_GetResult_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_GetResult_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_GetResult_Response & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_GetResult_Response>()
{
  return "robot_interfaces::action::NavigateToGoal_GetResult_Response";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_GetResult_Response>()
{
  return "robot_interfaces/action/NavigateToGoal_GetResult_Response";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_GetResult_Response>
  : std::integral_constant<bool, has_fixed_size<robot_interfaces::action::NavigateToGoal_Result>::value> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult_Response>
  : std::integral_constant<bool, has_bounded_size<robot_interfaces::action::NavigateToGoal_Result>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_GetResult_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
// already included above
// #include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_GetResult_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_GetResult_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_GetResult_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_GetResult_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_GetResult_Event & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_GetResult_Event>()
{
  return "robot_interfaces::action::NavigateToGoal_GetResult_Event";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_GetResult_Event>()
{
  return "robot_interfaces/action/NavigateToGoal_GetResult_Event";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_GetResult_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult_Event>
  : std::integral_constant<bool, has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult_Request>::value && has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_GetResult_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_GetResult>()
{
  return "robot_interfaces::action::NavigateToGoal_GetResult";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_GetResult>()
{
  return "robot_interfaces/action/NavigateToGoal_GetResult";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_GetResult>
  : std::integral_constant<
    bool,
    has_fixed_size<robot_interfaces::action::NavigateToGoal_GetResult_Request>::value &&
    has_fixed_size<robot_interfaces::action::NavigateToGoal_GetResult_Response>::value
  >
{
};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult>
  : std::integral_constant<
    bool,
    has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult_Request>::value &&
    has_bounded_size<robot_interfaces::action::NavigateToGoal_GetResult_Response>::value
  >
{
};

template<>
struct is_service<robot_interfaces::action::NavigateToGoal_GetResult>
  : std::true_type
{
};

template<>
struct is_service_request<robot_interfaces::action::NavigateToGoal_GetResult_Request>
  : std::true_type
{
};

template<>
struct is_service_response<robot_interfaces::action::NavigateToGoal_GetResult_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'feedback'
// already included above
// #include "robot_interfaces/action/detail/navigate_to_goal__traits.hpp"

namespace robot_interfaces
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToGoal_FeedbackMessage & msg,
  std::ostream & out)
{
  out << "{";
  // member: goal_id
  {
    out << "goal_id: ";
    to_flow_style_yaml(msg.goal_id, out);
    out << ", ";
  }

  // member: feedback
  {
    out << "feedback: ";
    to_flow_style_yaml(msg.feedback, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToGoal_FeedbackMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_block_style_yaml(msg.goal_id, out, indentation + 2);
  }

  // member: feedback
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "feedback:\n";
    to_block_style_yaml(msg.feedback, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToGoal_FeedbackMessage & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::action::NavigateToGoal_FeedbackMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::action::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::action::NavigateToGoal_FeedbackMessage & msg)
{
  return robot_interfaces::action::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::action::NavigateToGoal_FeedbackMessage>()
{
  return "robot_interfaces::action::NavigateToGoal_FeedbackMessage";
}

template<>
inline const char * name<robot_interfaces::action::NavigateToGoal_FeedbackMessage>()
{
  return "robot_interfaces/action/NavigateToGoal_FeedbackMessage";
}

template<>
struct has_fixed_size<robot_interfaces::action::NavigateToGoal_FeedbackMessage>
  : std::integral_constant<bool, has_fixed_size<robot_interfaces::action::NavigateToGoal_Feedback>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<robot_interfaces::action::NavigateToGoal_FeedbackMessage>
  : std::integral_constant<bool, has_bounded_size<robot_interfaces::action::NavigateToGoal_Feedback>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<robot_interfaces::action::NavigateToGoal_FeedbackMessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits


namespace rosidl_generator_traits
{

template<>
struct is_action<robot_interfaces::action::NavigateToGoal>
  : std::true_type
{
};

template<>
struct is_action_goal<robot_interfaces::action::NavigateToGoal_Goal>
  : std::true_type
{
};

template<>
struct is_action_result<robot_interfaces::action::NavigateToGoal_Result>
  : std::true_type
{
};

template<>
struct is_action_feedback<robot_interfaces::action::NavigateToGoal_Feedback>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits


#endif  // ROBOT_INTERFACES__ACTION__DETAIL__NAVIGATE_TO_GOAL__TRAITS_HPP_

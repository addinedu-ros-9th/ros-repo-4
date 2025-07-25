// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from robot_central_interfaces:srv/GetRobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/srv/get_robot_status.hpp"


#ifndef ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__GET_ROBOT_STATUS__TRAITS_HPP_
#define ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__GET_ROBOT_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "robot_central_interfaces/srv/detail/get_robot_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace robot_central_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetRobotStatus_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_id
  {
    out << "robot_id: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_id, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetRobotStatus_Request & msg,
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
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetRobotStatus_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace robot_central_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_central_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_central_interfaces::srv::GetRobotStatus_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_central_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_central_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_central_interfaces::srv::GetRobotStatus_Request & msg)
{
  return robot_central_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_central_interfaces::srv::GetRobotStatus_Request>()
{
  return "robot_central_interfaces::srv::GetRobotStatus_Request";
}

template<>
inline const char * name<robot_central_interfaces::srv::GetRobotStatus_Request>()
{
  return "robot_central_interfaces/srv/GetRobotStatus_Request";
}

template<>
struct has_fixed_size<robot_central_interfaces::srv::GetRobotStatus_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_central_interfaces::srv::GetRobotStatus_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_central_interfaces::srv::GetRobotStatus_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'robot_statuses'
#include "robot_central_interfaces/msg/detail/robot_status__traits.hpp"

namespace robot_central_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetRobotStatus_Response & msg,
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
    out << ", ";
  }

  // member: robot_statuses
  {
    if (msg.robot_statuses.size() == 0) {
      out << "robot_statuses: []";
    } else {
      out << "robot_statuses: [";
      size_t pending_items = msg.robot_statuses.size();
      for (auto item : msg.robot_statuses) {
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
  const GetRobotStatus_Response & msg,
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

  // member: robot_statuses
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.robot_statuses.size() == 0) {
      out << "robot_statuses: []\n";
    } else {
      out << "robot_statuses:\n";
      for (auto item : msg.robot_statuses) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetRobotStatus_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace robot_central_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_central_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_central_interfaces::srv::GetRobotStatus_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_central_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_central_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_central_interfaces::srv::GetRobotStatus_Response & msg)
{
  return robot_central_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_central_interfaces::srv::GetRobotStatus_Response>()
{
  return "robot_central_interfaces::srv::GetRobotStatus_Response";
}

template<>
inline const char * name<robot_central_interfaces::srv::GetRobotStatus_Response>()
{
  return "robot_central_interfaces/srv/GetRobotStatus_Response";
}

template<>
struct has_fixed_size<robot_central_interfaces::srv::GetRobotStatus_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_central_interfaces::srv::GetRobotStatus_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_central_interfaces::srv::GetRobotStatus_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace robot_central_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetRobotStatus_Event & msg,
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
  const GetRobotStatus_Event & msg,
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

inline std::string to_yaml(const GetRobotStatus_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace robot_central_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_central_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_central_interfaces::srv::GetRobotStatus_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_central_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_central_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_central_interfaces::srv::GetRobotStatus_Event & msg)
{
  return robot_central_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_central_interfaces::srv::GetRobotStatus_Event>()
{
  return "robot_central_interfaces::srv::GetRobotStatus_Event";
}

template<>
inline const char * name<robot_central_interfaces::srv::GetRobotStatus_Event>()
{
  return "robot_central_interfaces/srv/GetRobotStatus_Event";
}

template<>
struct has_fixed_size<robot_central_interfaces::srv::GetRobotStatus_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_central_interfaces::srv::GetRobotStatus_Event>
  : std::integral_constant<bool, has_bounded_size<robot_central_interfaces::srv::GetRobotStatus_Request>::value && has_bounded_size<robot_central_interfaces::srv::GetRobotStatus_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<robot_central_interfaces::srv::GetRobotStatus_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<robot_central_interfaces::srv::GetRobotStatus>()
{
  return "robot_central_interfaces::srv::GetRobotStatus";
}

template<>
inline const char * name<robot_central_interfaces::srv::GetRobotStatus>()
{
  return "robot_central_interfaces/srv/GetRobotStatus";
}

template<>
struct has_fixed_size<robot_central_interfaces::srv::GetRobotStatus>
  : std::integral_constant<
    bool,
    has_fixed_size<robot_central_interfaces::srv::GetRobotStatus_Request>::value &&
    has_fixed_size<robot_central_interfaces::srv::GetRobotStatus_Response>::value
  >
{
};

template<>
struct has_bounded_size<robot_central_interfaces::srv::GetRobotStatus>
  : std::integral_constant<
    bool,
    has_bounded_size<robot_central_interfaces::srv::GetRobotStatus_Request>::value &&
    has_bounded_size<robot_central_interfaces::srv::GetRobotStatus_Response>::value
  >
{
};

template<>
struct is_service<robot_central_interfaces::srv::GetRobotStatus>
  : std::true_type
{
};

template<>
struct is_service_request<robot_central_interfaces::srv::GetRobotStatus_Request>
  : std::true_type
{
};

template<>
struct is_service_response<robot_central_interfaces::srv::GetRobotStatus_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__GET_ROBOT_STATUS__TRAITS_HPP_

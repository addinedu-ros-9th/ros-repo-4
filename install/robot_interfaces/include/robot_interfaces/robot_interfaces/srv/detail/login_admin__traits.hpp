// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from robot_interfaces:srv/LoginAdmin.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/srv/login_admin.hpp"


#ifndef ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__TRAITS_HPP_
#define ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "robot_interfaces/srv/detail/login_admin__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace robot_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const LoginAdmin_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << ", ";
  }

  // member: password
  {
    out << "password: ";
    rosidl_generator_traits::value_to_yaml(msg.password, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const LoginAdmin_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: password
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "password: ";
    rosidl_generator_traits::value_to_yaml(msg.password, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const LoginAdmin_Request & msg, bool use_flow_style = false)
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

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::srv::LoginAdmin_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::srv::LoginAdmin_Request & msg)
{
  return robot_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::srv::LoginAdmin_Request>()
{
  return "robot_interfaces::srv::LoginAdmin_Request";
}

template<>
inline const char * name<robot_interfaces::srv::LoginAdmin_Request>()
{
  return "robot_interfaces/srv/LoginAdmin_Request";
}

template<>
struct has_fixed_size<robot_interfaces::srv::LoginAdmin_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_interfaces::srv::LoginAdmin_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_interfaces::srv::LoginAdmin_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace robot_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const LoginAdmin_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: email
  {
    out << "email: ";
    rosidl_generator_traits::value_to_yaml(msg.email, out);
    out << ", ";
  }

  // member: hospital_name
  {
    out << "hospital_name: ";
    rosidl_generator_traits::value_to_yaml(msg.hospital_name, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const LoginAdmin_Response & msg,
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

  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: email
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "email: ";
    rosidl_generator_traits::value_to_yaml(msg.email, out);
    out << "\n";
  }

  // member: hospital_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "hospital_name: ";
    rosidl_generator_traits::value_to_yaml(msg.hospital_name, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const LoginAdmin_Response & msg, bool use_flow_style = false)
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

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::srv::LoginAdmin_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::srv::LoginAdmin_Response & msg)
{
  return robot_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::srv::LoginAdmin_Response>()
{
  return "robot_interfaces::srv::LoginAdmin_Response";
}

template<>
inline const char * name<robot_interfaces::srv::LoginAdmin_Response>()
{
  return "robot_interfaces/srv/LoginAdmin_Response";
}

template<>
struct has_fixed_size<robot_interfaces::srv::LoginAdmin_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_interfaces::srv::LoginAdmin_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_interfaces::srv::LoginAdmin_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace robot_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const LoginAdmin_Event & msg,
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
  const LoginAdmin_Event & msg,
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

inline std::string to_yaml(const LoginAdmin_Event & msg, bool use_flow_style = false)
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

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::srv::LoginAdmin_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::srv::LoginAdmin_Event & msg)
{
  return robot_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::srv::LoginAdmin_Event>()
{
  return "robot_interfaces::srv::LoginAdmin_Event";
}

template<>
inline const char * name<robot_interfaces::srv::LoginAdmin_Event>()
{
  return "robot_interfaces/srv/LoginAdmin_Event";
}

template<>
struct has_fixed_size<robot_interfaces::srv::LoginAdmin_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_interfaces::srv::LoginAdmin_Event>
  : std::integral_constant<bool, has_bounded_size<robot_interfaces::srv::LoginAdmin_Request>::value && has_bounded_size<robot_interfaces::srv::LoginAdmin_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<robot_interfaces::srv::LoginAdmin_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<robot_interfaces::srv::LoginAdmin>()
{
  return "robot_interfaces::srv::LoginAdmin";
}

template<>
inline const char * name<robot_interfaces::srv::LoginAdmin>()
{
  return "robot_interfaces/srv/LoginAdmin";
}

template<>
struct has_fixed_size<robot_interfaces::srv::LoginAdmin>
  : std::integral_constant<
    bool,
    has_fixed_size<robot_interfaces::srv::LoginAdmin_Request>::value &&
    has_fixed_size<robot_interfaces::srv::LoginAdmin_Response>::value
  >
{
};

template<>
struct has_bounded_size<robot_interfaces::srv::LoginAdmin>
  : std::integral_constant<
    bool,
    has_bounded_size<robot_interfaces::srv::LoginAdmin_Request>::value &&
    has_bounded_size<robot_interfaces::srv::LoginAdmin_Response>::value
  >
{
};

template<>
struct is_service<robot_interfaces::srv::LoginAdmin>
  : std::true_type
{
};

template<>
struct is_service_request<robot_interfaces::srv::LoginAdmin_Request>
  : std::true_type
{
};

template<>
struct is_service_response<robot_interfaces::srv::LoginAdmin_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__TRAITS_HPP_

// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_interfaces:srv/LoginAdmin.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/srv/login_admin.h"


#ifndef ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__STRUCT_H_
#define ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'id'
// Member 'password'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/LoginAdmin in the package robot_interfaces.
typedef struct robot_interfaces__srv__LoginAdmin_Request
{
  rosidl_runtime_c__String id;
  rosidl_runtime_c__String password;
} robot_interfaces__srv__LoginAdmin_Request;

// Struct for a sequence of robot_interfaces__srv__LoginAdmin_Request.
typedef struct robot_interfaces__srv__LoginAdmin_Request__Sequence
{
  robot_interfaces__srv__LoginAdmin_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__srv__LoginAdmin_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'name'
// Member 'email'
// Member 'hospital_name'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/LoginAdmin in the package robot_interfaces.
typedef struct robot_interfaces__srv__LoginAdmin_Response
{
  bool success;
  rosidl_runtime_c__String name;
  rosidl_runtime_c__String email;
  rosidl_runtime_c__String hospital_name;
} robot_interfaces__srv__LoginAdmin_Response;

// Struct for a sequence of robot_interfaces__srv__LoginAdmin_Response.
typedef struct robot_interfaces__srv__LoginAdmin_Response__Sequence
{
  robot_interfaces__srv__LoginAdmin_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__srv__LoginAdmin_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  robot_interfaces__srv__LoginAdmin_Event__request__MAX_SIZE = 1
};
// response
enum
{
  robot_interfaces__srv__LoginAdmin_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/LoginAdmin in the package robot_interfaces.
typedef struct robot_interfaces__srv__LoginAdmin_Event
{
  service_msgs__msg__ServiceEventInfo info;
  robot_interfaces__srv__LoginAdmin_Request__Sequence request;
  robot_interfaces__srv__LoginAdmin_Response__Sequence response;
} robot_interfaces__srv__LoginAdmin_Event;

// Struct for a sequence of robot_interfaces__srv__LoginAdmin_Event.
typedef struct robot_interfaces__srv__LoginAdmin_Event__Sequence
{
  robot_interfaces__srv__LoginAdmin_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__srv__LoginAdmin_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_INTERFACES__SRV__DETAIL__LOGIN_ADMIN__STRUCT_H_

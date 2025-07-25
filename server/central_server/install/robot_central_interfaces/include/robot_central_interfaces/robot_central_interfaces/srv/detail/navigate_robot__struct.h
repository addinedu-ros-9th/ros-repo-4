// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_central_interfaces:srv/NavigateRobot.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/srv/navigate_robot.h"


#ifndef ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__STRUCT_H_
#define ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'robot_id'
// Member 'waypoints'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/NavigateRobot in the package robot_central_interfaces.
typedef struct robot_central_interfaces__srv__NavigateRobot_Request
{
  rosidl_runtime_c__String robot_id;
  rosidl_runtime_c__String__Sequence waypoints;
} robot_central_interfaces__srv__NavigateRobot_Request;

// Struct for a sequence of robot_central_interfaces__srv__NavigateRobot_Request.
typedef struct robot_central_interfaces__srv__NavigateRobot_Request__Sequence
{
  robot_central_interfaces__srv__NavigateRobot_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_central_interfaces__srv__NavigateRobot_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/NavigateRobot in the package robot_central_interfaces.
typedef struct robot_central_interfaces__srv__NavigateRobot_Response
{
  bool success;
  rosidl_runtime_c__String message;
} robot_central_interfaces__srv__NavigateRobot_Response;

// Struct for a sequence of robot_central_interfaces__srv__NavigateRobot_Response.
typedef struct robot_central_interfaces__srv__NavigateRobot_Response__Sequence
{
  robot_central_interfaces__srv__NavigateRobot_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_central_interfaces__srv__NavigateRobot_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  robot_central_interfaces__srv__NavigateRobot_Event__request__MAX_SIZE = 1
};
// response
enum
{
  robot_central_interfaces__srv__NavigateRobot_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/NavigateRobot in the package robot_central_interfaces.
typedef struct robot_central_interfaces__srv__NavigateRobot_Event
{
  service_msgs__msg__ServiceEventInfo info;
  robot_central_interfaces__srv__NavigateRobot_Request__Sequence request;
  robot_central_interfaces__srv__NavigateRobot_Response__Sequence response;
} robot_central_interfaces__srv__NavigateRobot_Event;

// Struct for a sequence of robot_central_interfaces__srv__NavigateRobot_Event.
typedef struct robot_central_interfaces__srv__NavigateRobot_Event__Sequence
{
  robot_central_interfaces__srv__NavigateRobot_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_central_interfaces__srv__NavigateRobot_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__STRUCT_H_

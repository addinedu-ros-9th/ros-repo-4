// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_central_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/msg/robot_status.h"


#ifndef ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_
#define ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_

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
// Member 'status'
// Member 'current_task'
#include "rosidl_runtime_c/string.h"
// Member 'current_pose'
#include "geometry_msgs/msg/detail/pose__struct.h"
// Member 'last_update'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/RobotStatus in the package robot_central_interfaces.
typedef struct robot_central_interfaces__msg__RobotStatus
{
  rosidl_runtime_c__String robot_id;
  rosidl_runtime_c__String status;
  rosidl_runtime_c__String current_task;
  geometry_msgs__msg__Pose current_pose;
  builtin_interfaces__msg__Time last_update;
} robot_central_interfaces__msg__RobotStatus;

// Struct for a sequence of robot_central_interfaces__msg__RobotStatus.
typedef struct robot_central_interfaces__msg__RobotStatus__Sequence
{
  robot_central_interfaces__msg__RobotStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_central_interfaces__msg__RobotStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_

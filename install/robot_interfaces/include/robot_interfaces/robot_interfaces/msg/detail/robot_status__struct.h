// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/msg/robot_status.h"


#ifndef ROBOT_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_
#define ROBOT_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'status'
// Member 'current_station'
// Member 'destination_station'
// Member 'network_status'
// Member 'source'
#include "rosidl_runtime_c/string.h"
// Member 'amcl_pose'
#include "geometry_msgs/msg/detail/pose_with_covariance_stamped__struct.h"

/// Struct defined in msg/RobotStatus in the package robot_interfaces.
typedef struct robot_interfaces__msg__RobotStatus
{
  int32_t robot_id;
  rosidl_runtime_c__String status;
  geometry_msgs__msg__PoseWithCovarianceStamped amcl_pose;
  rosidl_runtime_c__String current_station;
  rosidl_runtime_c__String destination_station;
  int32_t battery_percent;
  rosidl_runtime_c__String network_status;
  rosidl_runtime_c__String source;
} robot_interfaces__msg__RobotStatus;

// Struct for a sequence of robot_interfaces__msg__RobotStatus.
typedef struct robot_interfaces__msg__RobotStatus__Sequence
{
  robot_interfaces__msg__RobotStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__msg__RobotStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_

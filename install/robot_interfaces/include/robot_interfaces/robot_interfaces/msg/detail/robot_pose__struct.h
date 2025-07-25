// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_interfaces:msg/RobotPose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/msg/robot_pose.h"


#ifndef ROBOT_INTERFACES__MSG__DETAIL__ROBOT_POSE__STRUCT_H_
#define ROBOT_INTERFACES__MSG__DETAIL__ROBOT_POSE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose2_d__struct.h"
// Member 'timestamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in msg/RobotPose in the package robot_interfaces.
/**
  * 로봇의 위치 정보 (간단한 버전) - WebSocket 실시간 전송용
 */
typedef struct robot_interfaces__msg__RobotPose
{
  int32_t robot_id;
  /// x, y, theta (표준 타입 사용)
  geometry_msgs__msg__Pose2D pose;
  builtin_interfaces__msg__Time timestamp;
} robot_interfaces__msg__RobotPose;

// Struct for a sequence of robot_interfaces__msg__RobotPose.
typedef struct robot_interfaces__msg__RobotPose__Sequence
{
  robot_interfaces__msg__RobotPose * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__msg__RobotPose__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_INTERFACES__MSG__DETAIL__ROBOT_POSE__STRUCT_H_

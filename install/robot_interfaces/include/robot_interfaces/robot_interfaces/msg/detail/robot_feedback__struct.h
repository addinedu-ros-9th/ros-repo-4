// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_interfaces:msg/RobotFeedback.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/msg/robot_feedback.h"


#ifndef ROBOT_INTERFACES__MSG__DETAIL__ROBOT_FEEDBACK__STRUCT_H_
#define ROBOT_INTERFACES__MSG__DETAIL__ROBOT_FEEDBACK__STRUCT_H_

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
// Member 'network_status'
// Member 'detected_objects'
// Member 'error_message'
#include "rosidl_runtime_c/string.h"
// Member 'current_pose'
#include "geometry_msgs/msg/detail/pose_with_covariance_stamped__struct.h"

/// Struct defined in msg/RobotFeedback in the package robot_interfaces.
/**
  * 로봇 컨트롤러에서 중앙서버로 보내는 피드백
 */
typedef struct robot_interfaces__msg__RobotFeedback
{
  int32_t robot_id;
  /// "moving", "standby" (로봇이 스스로 판단하는 상태만)
  rosidl_runtime_c__String status;
  geometry_msgs__msg__PoseWithCovarianceStamped current_pose;
  int32_t battery_percent;
  /// "connected", "disconnected", "weak"
  rosidl_runtime_c__String network_status;
  /// 현재 이동 속도 (m/s)
  float current_speed;
  /// 감지된 객체들 ["person", "obstacle", "chair", ...]
  rosidl_runtime_c__String__Sequence detected_objects;
  /// 오류 발생시 메시지 (정상시 빈 문자열)
  rosidl_runtime_c__String error_message;
} robot_interfaces__msg__RobotFeedback;

// Struct for a sequence of robot_interfaces__msg__RobotFeedback.
typedef struct robot_interfaces__msg__RobotFeedback__Sequence
{
  robot_interfaces__msg__RobotFeedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__msg__RobotFeedback__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_INTERFACES__MSG__DETAIL__ROBOT_FEEDBACK__STRUCT_H_

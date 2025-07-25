// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from robot_interfaces:msg/RobotFeedback.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "robot_interfaces/msg/detail/robot_feedback__rosidl_typesupport_introspection_c.h"
#include "robot_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "robot_interfaces/msg/detail/robot_feedback__functions.h"
#include "robot_interfaces/msg/detail/robot_feedback__struct.h"


// Include directives for member types
// Member `status`
// Member `network_status`
// Member `detected_objects`
// Member `error_message`
#include "rosidl_runtime_c/string_functions.h"
// Member `current_pose`
#include "geometry_msgs/msg/pose_with_covariance_stamped.h"
// Member `current_pose`
#include "geometry_msgs/msg/detail/pose_with_covariance_stamped__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  robot_interfaces__msg__RobotFeedback__init(message_memory);
}

void robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_fini_function(void * message_memory)
{
  robot_interfaces__msg__RobotFeedback__fini(message_memory);
}

size_t robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__size_function__RobotFeedback__detected_objects(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__get_const_function__RobotFeedback__detected_objects(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__get_function__RobotFeedback__detected_objects(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__fetch_function__RobotFeedback__detected_objects(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__get_const_function__RobotFeedback__detected_objects(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__assign_function__RobotFeedback__detected_objects(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__get_function__RobotFeedback__detected_objects(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__resize_function__RobotFeedback__detected_objects(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_member_array[8] = {
  {
    "robot_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, robot_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, current_pose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "battery_percent",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, battery_percent),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "network_status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, network_status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "current_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, current_speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "detected_objects",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, detected_objects),  // bytes offset in struct
    NULL,  // default value
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__size_function__RobotFeedback__detected_objects,  // size() function pointer
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__get_const_function__RobotFeedback__detected_objects,  // get_const(index) function pointer
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__get_function__RobotFeedback__detected_objects,  // get(index) function pointer
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__fetch_function__RobotFeedback__detected_objects,  // fetch(index, &value) function pointer
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__assign_function__RobotFeedback__detected_objects,  // assign(index, value) function pointer
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__resize_function__RobotFeedback__detected_objects  // resize(index) function pointer
  },
  {
    "error_message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(robot_interfaces__msg__RobotFeedback, error_message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_members = {
  "robot_interfaces__msg",  // message namespace
  "RobotFeedback",  // message name
  8,  // number of fields
  sizeof(robot_interfaces__msg__RobotFeedback),
  false,  // has_any_key_member_
  robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_member_array,  // message members
  robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_init_function,  // function to initialize message memory (memory has to be allocated)
  robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_type_support_handle = {
  0,
  &robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_members,
  get_message_typesupport_handle_function,
  &robot_interfaces__msg__RobotFeedback__get_type_hash,
  &robot_interfaces__msg__RobotFeedback__get_type_description,
  &robot_interfaces__msg__RobotFeedback__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_robot_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, robot_interfaces, msg, RobotFeedback)() {
  robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PoseWithCovarianceStamped)();
  if (!robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_type_support_handle.typesupport_identifier) {
    robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &robot_interfaces__msg__RobotFeedback__rosidl_typesupport_introspection_c__RobotFeedback_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

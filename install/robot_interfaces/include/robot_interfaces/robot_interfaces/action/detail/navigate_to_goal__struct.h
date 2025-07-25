// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_interfaces:action/NavigateToGoal.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_interfaces/action/navigate_to_goal.h"


#ifndef ROBOT_INTERFACES__ACTION__DETAIL__NAVIGATE_TO_GOAL__STRUCT_H_
#define ROBOT_INTERFACES__ACTION__DETAIL__NAVIGATE_TO_GOAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose_stamped__struct.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_Goal
{
  int32_t robot_id;
  geometry_msgs__msg__PoseStamped target_pose;
} robot_interfaces__action__NavigateToGoal_Goal;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_Goal.
typedef struct robot_interfaces__action__NavigateToGoal_Goal__Sequence
{
  robot_interfaces__action__NavigateToGoal_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_Goal__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_Result
{
  bool success;
  rosidl_runtime_c__String message;
} robot_interfaces__action__NavigateToGoal_Result;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_Result.
typedef struct robot_interfaces__action__NavigateToGoal_Result__Sequence
{
  robot_interfaces__action__NavigateToGoal_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_Result__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'current_pose'
// already included above
// #include "geometry_msgs/msg/detail/pose_stamped__struct.h"
// Member 'current_status'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_Feedback
{
  geometry_msgs__msg__PoseStamped current_pose;
  /// "planning", "navigating", "arrived"
  rosidl_runtime_c__String current_status;
} robot_interfaces__action__NavigateToGoal_Feedback;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_Feedback.
typedef struct robot_interfaces__action__NavigateToGoal_Feedback__Sequence
{
  robot_interfaces__action__NavigateToGoal_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_Feedback__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "robot_interfaces/action/detail/navigate_to_goal__struct.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  robot_interfaces__action__NavigateToGoal_Goal goal;
} robot_interfaces__action__NavigateToGoal_SendGoal_Request;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_SendGoal_Request.
typedef struct robot_interfaces__action__NavigateToGoal_SendGoal_Request__Sequence
{
  robot_interfaces__action__NavigateToGoal_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_SendGoal_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} robot_interfaces__action__NavigateToGoal_SendGoal_Response;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_SendGoal_Response.
typedef struct robot_interfaces__action__NavigateToGoal_SendGoal_Response__Sequence
{
  robot_interfaces__action__NavigateToGoal_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_SendGoal_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  robot_interfaces__action__NavigateToGoal_SendGoal_Event__request__MAX_SIZE = 1
};
// response
enum
{
  robot_interfaces__action__NavigateToGoal_SendGoal_Event__response__MAX_SIZE = 1
};

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_SendGoal_Event
{
  service_msgs__msg__ServiceEventInfo info;
  robot_interfaces__action__NavigateToGoal_SendGoal_Request__Sequence request;
  robot_interfaces__action__NavigateToGoal_SendGoal_Response__Sequence response;
} robot_interfaces__action__NavigateToGoal_SendGoal_Event;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_SendGoal_Event.
typedef struct robot_interfaces__action__NavigateToGoal_SendGoal_Event__Sequence
{
  robot_interfaces__action__NavigateToGoal_SendGoal_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_SendGoal_Event__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} robot_interfaces__action__NavigateToGoal_GetResult_Request;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_GetResult_Request.
typedef struct robot_interfaces__action__NavigateToGoal_GetResult_Request__Sequence
{
  robot_interfaces__action__NavigateToGoal_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_GetResult_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "robot_interfaces/action/detail/navigate_to_goal__struct.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_GetResult_Response
{
  int8_t status;
  robot_interfaces__action__NavigateToGoal_Result result;
} robot_interfaces__action__NavigateToGoal_GetResult_Response;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_GetResult_Response.
typedef struct robot_interfaces__action__NavigateToGoal_GetResult_Response__Sequence
{
  robot_interfaces__action__NavigateToGoal_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_GetResult_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
// already included above
// #include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  robot_interfaces__action__NavigateToGoal_GetResult_Event__request__MAX_SIZE = 1
};
// response
enum
{
  robot_interfaces__action__NavigateToGoal_GetResult_Event__response__MAX_SIZE = 1
};

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_GetResult_Event
{
  service_msgs__msg__ServiceEventInfo info;
  robot_interfaces__action__NavigateToGoal_GetResult_Request__Sequence request;
  robot_interfaces__action__NavigateToGoal_GetResult_Response__Sequence response;
} robot_interfaces__action__NavigateToGoal_GetResult_Event;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_GetResult_Event.
typedef struct robot_interfaces__action__NavigateToGoal_GetResult_Event__Sequence
{
  robot_interfaces__action__NavigateToGoal_GetResult_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_GetResult_Event__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "robot_interfaces/action/detail/navigate_to_goal__struct.h"

/// Struct defined in action/NavigateToGoal in the package robot_interfaces.
typedef struct robot_interfaces__action__NavigateToGoal_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  robot_interfaces__action__NavigateToGoal_Feedback feedback;
} robot_interfaces__action__NavigateToGoal_FeedbackMessage;

// Struct for a sequence of robot_interfaces__action__NavigateToGoal_FeedbackMessage.
typedef struct robot_interfaces__action__NavigateToGoal_FeedbackMessage__Sequence
{
  robot_interfaces__action__NavigateToGoal_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__action__NavigateToGoal_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_INTERFACES__ACTION__DETAIL__NAVIGATE_TO_GOAL__STRUCT_H_

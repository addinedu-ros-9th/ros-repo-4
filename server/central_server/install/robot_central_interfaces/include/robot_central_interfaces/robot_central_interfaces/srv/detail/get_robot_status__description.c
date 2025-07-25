// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from robot_central_interfaces:srv/GetRobotStatus.idl
// generated code does not contain a copyright notice

#include "robot_central_interfaces/srv/detail/get_robot_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__GetRobotStatus__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x0a, 0x50, 0xe9, 0x8e, 0xc4, 0xfa, 0xbc, 0x7d,
      0xd4, 0x16, 0x82, 0xd3, 0x37, 0x46, 0xac, 0x73,
      0x49, 0xb3, 0x23, 0xb6, 0x2c, 0xb0, 0xb3, 0x7b,
      0xcb, 0xa8, 0xcc, 0x0b, 0xa6, 0xb4, 0xf1, 0xc1,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__GetRobotStatus_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x4a, 0x9f, 0xd8, 0x94, 0x18, 0xa2, 0x89, 0xd7,
      0xb2, 0xf7, 0xf2, 0x46, 0x32, 0x81, 0xb1, 0x4d,
      0x80, 0x3f, 0xf1, 0xf0, 0xb1, 0x18, 0xe3, 0x10,
      0x79, 0x36, 0x24, 0x97, 0x1a, 0x8b, 0x7f, 0x5b,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__GetRobotStatus_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xac, 0xb2, 0x4e, 0x5c, 0x6d, 0x59, 0x9e, 0xae,
      0x07, 0xb7, 0x53, 0x3e, 0x0a, 0xcb, 0x1a, 0x1e,
      0x42, 0x18, 0x96, 0xcc, 0x86, 0x1f, 0xe6, 0xa7,
      0xf3, 0xae, 0x55, 0xd1, 0xee, 0xce, 0x40, 0xb3,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__GetRobotStatus_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xac, 0xa7, 0x37, 0x71, 0x2e, 0x2d, 0x42, 0x81,
      0xd7, 0x37, 0xdb, 0xa0, 0x85, 0xe0, 0x00, 0x4a,
      0x83, 0xdd, 0x8f, 0x37, 0x9a, 0x75, 0x88, 0xed,
      0x44, 0x7b, 0x30, 0xdb, 0xa0, 0xc6, 0x13, 0x87,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "geometry_msgs/msg/detail/point__functions.h"
#include "geometry_msgs/msg/detail/quaternion__functions.h"
#include "geometry_msgs/msg/detail/pose__functions.h"
#include "robot_central_interfaces/msg/detail/robot_status__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t geometry_msgs__msg__Point__EXPECTED_HASH = {1, {
    0x69, 0x63, 0x08, 0x48, 0x42, 0xa9, 0xb0, 0x44,
    0x94, 0xd6, 0xb2, 0x94, 0x1d, 0x11, 0x44, 0x47,
    0x08, 0xd8, 0x92, 0xda, 0x2f, 0x4b, 0x09, 0x84,
    0x3b, 0x9c, 0x43, 0xf4, 0x2a, 0x7f, 0x68, 0x81,
  }};
static const rosidl_type_hash_t geometry_msgs__msg__Pose__EXPECTED_HASH = {1, {
    0xd5, 0x01, 0x95, 0x4e, 0x94, 0x76, 0xce, 0xa2,
    0x99, 0x69, 0x84, 0xe8, 0x12, 0x05, 0x4b, 0x68,
    0x02, 0x6a, 0xe0, 0xbf, 0xae, 0x78, 0x9d, 0x9a,
    0x10, 0xb2, 0x3d, 0xaf, 0x35, 0xcc, 0x90, 0xfa,
  }};
static const rosidl_type_hash_t geometry_msgs__msg__Quaternion__EXPECTED_HASH = {1, {
    0x8a, 0x76, 0x5f, 0x66, 0x77, 0x8c, 0x8f, 0xf7,
    0xc8, 0xab, 0x94, 0xaf, 0xcc, 0x59, 0x0a, 0x2e,
    0xd5, 0x32, 0x5a, 0x1d, 0x9a, 0x07, 0x6f, 0xff,
    0xf3, 0x8f, 0xbc, 0xe3, 0x6f, 0x45, 0x86, 0x84,
  }};
static const rosidl_type_hash_t robot_central_interfaces__msg__RobotStatus__EXPECTED_HASH = {1, {
    0x32, 0x11, 0x4f, 0x35, 0x20, 0x3b, 0xae, 0xa8,
    0x99, 0xeb, 0xed, 0xf9, 0x8e, 0xc3, 0x77, 0xd6,
    0x55, 0xbc, 0x7a, 0xaa, 0x64, 0x64, 0x0d, 0x80,
    0x2a, 0x87, 0xd9, 0x69, 0xb5, 0xf2, 0xbd, 0x29,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char robot_central_interfaces__srv__GetRobotStatus__TYPE_NAME[] = "robot_central_interfaces/srv/GetRobotStatus";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char geometry_msgs__msg__Point__TYPE_NAME[] = "geometry_msgs/msg/Point";
static char geometry_msgs__msg__Pose__TYPE_NAME[] = "geometry_msgs/msg/Pose";
static char geometry_msgs__msg__Quaternion__TYPE_NAME[] = "geometry_msgs/msg/Quaternion";
static char robot_central_interfaces__msg__RobotStatus__TYPE_NAME[] = "robot_central_interfaces/msg/RobotStatus";
static char robot_central_interfaces__srv__GetRobotStatus_Event__TYPE_NAME[] = "robot_central_interfaces/srv/GetRobotStatus_Event";
static char robot_central_interfaces__srv__GetRobotStatus_Request__TYPE_NAME[] = "robot_central_interfaces/srv/GetRobotStatus_Request";
static char robot_central_interfaces__srv__GetRobotStatus_Response__TYPE_NAME[] = "robot_central_interfaces/srv/GetRobotStatus_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char robot_central_interfaces__srv__GetRobotStatus__FIELD_NAME__request_message[] = "request_message";
static char robot_central_interfaces__srv__GetRobotStatus__FIELD_NAME__response_message[] = "response_message";
static char robot_central_interfaces__srv__GetRobotStatus__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__GetRobotStatus__FIELDS[] = {
  {
    {robot_central_interfaces__srv__GetRobotStatus__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_central_interfaces__srv__GetRobotStatus_Request__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_central_interfaces__srv__GetRobotStatus_Response__TYPE_NAME, 52, 52},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_central_interfaces__srv__GetRobotStatus_Event__TYPE_NAME, 49, 49},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_central_interfaces__srv__GetRobotStatus__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Point__TYPE_NAME, 23, 23},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Pose__TYPE_NAME, 22, 22},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Quaternion__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__msg__RobotStatus__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Event__TYPE_NAME, 49, 49},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Request__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Response__TYPE_NAME, 52, 52},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_central_interfaces__srv__GetRobotStatus__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__GetRobotStatus__TYPE_NAME, 43, 43},
      {robot_central_interfaces__srv__GetRobotStatus__FIELDS, 3, 3},
    },
    {robot_central_interfaces__srv__GetRobotStatus__REFERENCED_TYPE_DESCRIPTIONS, 9, 9},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Point__EXPECTED_HASH, geometry_msgs__msg__Point__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = geometry_msgs__msg__Point__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Pose__EXPECTED_HASH, geometry_msgs__msg__Pose__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = geometry_msgs__msg__Pose__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Quaternion__EXPECTED_HASH, geometry_msgs__msg__Quaternion__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = geometry_msgs__msg__Quaternion__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&robot_central_interfaces__msg__RobotStatus__EXPECTED_HASH, robot_central_interfaces__msg__RobotStatus__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = robot_central_interfaces__msg__RobotStatus__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[5].fields = robot_central_interfaces__srv__GetRobotStatus_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[6].fields = robot_central_interfaces__srv__GetRobotStatus_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[7].fields = robot_central_interfaces__srv__GetRobotStatus_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[8].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_central_interfaces__srv__GetRobotStatus_Request__FIELD_NAME__robot_id[] = "robot_id";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__GetRobotStatus_Request__FIELDS[] = {
  {
    {robot_central_interfaces__srv__GetRobotStatus_Request__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_central_interfaces__srv__GetRobotStatus_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__GetRobotStatus_Request__TYPE_NAME, 51, 51},
      {robot_central_interfaces__srv__GetRobotStatus_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_central_interfaces__srv__GetRobotStatus_Response__FIELD_NAME__success[] = "success";
static char robot_central_interfaces__srv__GetRobotStatus_Response__FIELD_NAME__message[] = "message";
static char robot_central_interfaces__srv__GetRobotStatus_Response__FIELD_NAME__robot_statuses[] = "robot_statuses";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__GetRobotStatus_Response__FIELDS[] = {
  {
    {robot_central_interfaces__srv__GetRobotStatus_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Response__FIELD_NAME__message, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Response__FIELD_NAME__robot_statuses, 14, 14},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_UNBOUNDED_SEQUENCE,
      0,
      0,
      {robot_central_interfaces__msg__RobotStatus__TYPE_NAME, 40, 40},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_central_interfaces__srv__GetRobotStatus_Response__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Point__TYPE_NAME, 23, 23},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Pose__TYPE_NAME, 22, 22},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Quaternion__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__msg__RobotStatus__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_central_interfaces__srv__GetRobotStatus_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__GetRobotStatus_Response__TYPE_NAME, 52, 52},
      {robot_central_interfaces__srv__GetRobotStatus_Response__FIELDS, 3, 3},
    },
    {robot_central_interfaces__srv__GetRobotStatus_Response__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Point__EXPECTED_HASH, geometry_msgs__msg__Point__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = geometry_msgs__msg__Point__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Pose__EXPECTED_HASH, geometry_msgs__msg__Pose__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = geometry_msgs__msg__Pose__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Quaternion__EXPECTED_HASH, geometry_msgs__msg__Quaternion__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = geometry_msgs__msg__Quaternion__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&robot_central_interfaces__msg__RobotStatus__EXPECTED_HASH, robot_central_interfaces__msg__RobotStatus__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = robot_central_interfaces__msg__RobotStatus__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_central_interfaces__srv__GetRobotStatus_Event__FIELD_NAME__info[] = "info";
static char robot_central_interfaces__srv__GetRobotStatus_Event__FIELD_NAME__request[] = "request";
static char robot_central_interfaces__srv__GetRobotStatus_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__GetRobotStatus_Event__FIELDS[] = {
  {
    {robot_central_interfaces__srv__GetRobotStatus_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_central_interfaces__srv__GetRobotStatus_Request__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_central_interfaces__srv__GetRobotStatus_Response__TYPE_NAME, 52, 52},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_central_interfaces__srv__GetRobotStatus_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Point__TYPE_NAME, 23, 23},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Pose__TYPE_NAME, 22, 22},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Quaternion__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__msg__RobotStatus__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Request__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__GetRobotStatus_Response__TYPE_NAME, 52, 52},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_central_interfaces__srv__GetRobotStatus_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__GetRobotStatus_Event__TYPE_NAME, 49, 49},
      {robot_central_interfaces__srv__GetRobotStatus_Event__FIELDS, 3, 3},
    },
    {robot_central_interfaces__srv__GetRobotStatus_Event__REFERENCED_TYPE_DESCRIPTIONS, 8, 8},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Point__EXPECTED_HASH, geometry_msgs__msg__Point__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = geometry_msgs__msg__Point__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Pose__EXPECTED_HASH, geometry_msgs__msg__Pose__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = geometry_msgs__msg__Pose__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Quaternion__EXPECTED_HASH, geometry_msgs__msg__Quaternion__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = geometry_msgs__msg__Quaternion__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&robot_central_interfaces__msg__RobotStatus__EXPECTED_HASH, robot_central_interfaces__msg__RobotStatus__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = robot_central_interfaces__msg__RobotStatus__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[5].fields = robot_central_interfaces__srv__GetRobotStatus_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[6].fields = robot_central_interfaces__srv__GetRobotStatus_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[7].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# Request\n"
  "string robot_id  # empty string for all robots\n"
  "---\n"
  "# Response\n"
  "bool success\n"
  "string message\n"
  "RobotStatus[] robot_statuses";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__GetRobotStatus__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__GetRobotStatus__TYPE_NAME, 43, 43},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 128, 128},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__GetRobotStatus_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__GetRobotStatus_Request__TYPE_NAME, 51, 51},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__GetRobotStatus_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__GetRobotStatus_Response__TYPE_NAME, 52, 52},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__GetRobotStatus_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__GetRobotStatus_Event__TYPE_NAME, 49, 49},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__GetRobotStatus__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[10];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 10, 10};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__GetRobotStatus__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *geometry_msgs__msg__Point__get_individual_type_description_source(NULL);
    sources[3] = *geometry_msgs__msg__Pose__get_individual_type_description_source(NULL);
    sources[4] = *geometry_msgs__msg__Quaternion__get_individual_type_description_source(NULL);
    sources[5] = *robot_central_interfaces__msg__RobotStatus__get_individual_type_description_source(NULL);
    sources[6] = *robot_central_interfaces__srv__GetRobotStatus_Event__get_individual_type_description_source(NULL);
    sources[7] = *robot_central_interfaces__srv__GetRobotStatus_Request__get_individual_type_description_source(NULL);
    sources[8] = *robot_central_interfaces__srv__GetRobotStatus_Response__get_individual_type_description_source(NULL);
    sources[9] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__GetRobotStatus_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__GetRobotStatus_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__GetRobotStatus_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__GetRobotStatus_Response__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *geometry_msgs__msg__Point__get_individual_type_description_source(NULL);
    sources[3] = *geometry_msgs__msg__Pose__get_individual_type_description_source(NULL);
    sources[4] = *geometry_msgs__msg__Quaternion__get_individual_type_description_source(NULL);
    sources[5] = *robot_central_interfaces__msg__RobotStatus__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__GetRobotStatus_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[9];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 9, 9};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__GetRobotStatus_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *geometry_msgs__msg__Point__get_individual_type_description_source(NULL);
    sources[3] = *geometry_msgs__msg__Pose__get_individual_type_description_source(NULL);
    sources[4] = *geometry_msgs__msg__Quaternion__get_individual_type_description_source(NULL);
    sources[5] = *robot_central_interfaces__msg__RobotStatus__get_individual_type_description_source(NULL);
    sources[6] = *robot_central_interfaces__srv__GetRobotStatus_Request__get_individual_type_description_source(NULL);
    sources[7] = *robot_central_interfaces__srv__GetRobotStatus_Response__get_individual_type_description_source(NULL);
    sources[8] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

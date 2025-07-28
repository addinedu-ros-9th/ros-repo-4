// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from robot_interfaces:srv/ChangeRobotStatus.idl
// generated code does not contain a copyright notice

#include "robot_interfaces/srv/detail/change_robot_status__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__ChangeRobotStatus__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xf5, 0xb5, 0x44, 0xe7, 0xc4, 0x19, 0x29, 0xbd,
      0xa8, 0xa8, 0x81, 0xb7, 0x3a, 0x1a, 0x47, 0x8f,
      0x47, 0xc0, 0x81, 0x86, 0x38, 0xda, 0xe3, 0xa2,
      0xf3, 0xb4, 0x08, 0x7e, 0x2d, 0x7f, 0x64, 0x17,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__ChangeRobotStatus_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xc0, 0xc1, 0xe2, 0x38, 0xee, 0x8e, 0xec, 0x1a,
      0xc7, 0x30, 0x2a, 0xc7, 0x6a, 0x04, 0x0d, 0xc1,
      0xf4, 0xf9, 0xe6, 0xd5, 0xa6, 0xb4, 0xfd, 0x06,
      0xd6, 0x40, 0xb4, 0x57, 0xeb, 0x9f, 0xca, 0x4b,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__ChangeRobotStatus_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x48, 0x31, 0xee, 0xe3, 0x69, 0xf9, 0x9f, 0xec,
      0x58, 0x6b, 0xde, 0x6f, 0xda, 0x68, 0xc0, 0x6a,
      0x92, 0x34, 0x46, 0xb8, 0xc2, 0xbc, 0x27, 0xf7,
      0xe4, 0x43, 0xd4, 0x37, 0x45, 0x95, 0x04, 0x32,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__ChangeRobotStatus_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xe9, 0xe5, 0xcc, 0xce, 0xa8, 0x2a, 0x32, 0x97,
      0xb6, 0xde, 0xac, 0x82, 0x1e, 0xa8, 0xf8, 0x69,
      0x83, 0x24, 0xc5, 0x4e, 0xb1, 0xc2, 0x15, 0x5e,
      0xb2, 0xc0, 0x19, 0x57, 0x63, 0xc0, 0x3b, 0x1f,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "service_msgs/msg/detail/service_event_info__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char robot_interfaces__srv__ChangeRobotStatus__TYPE_NAME[] = "robot_interfaces/srv/ChangeRobotStatus";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char robot_interfaces__srv__ChangeRobotStatus_Event__TYPE_NAME[] = "robot_interfaces/srv/ChangeRobotStatus_Event";
static char robot_interfaces__srv__ChangeRobotStatus_Request__TYPE_NAME[] = "robot_interfaces/srv/ChangeRobotStatus_Request";
static char robot_interfaces__srv__ChangeRobotStatus_Response__TYPE_NAME[] = "robot_interfaces/srv/ChangeRobotStatus_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char robot_interfaces__srv__ChangeRobotStatus__FIELD_NAME__request_message[] = "request_message";
static char robot_interfaces__srv__ChangeRobotStatus__FIELD_NAME__response_message[] = "response_message";
static char robot_interfaces__srv__ChangeRobotStatus__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__ChangeRobotStatus__FIELDS[] = {
  {
    {robot_interfaces__srv__ChangeRobotStatus__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_interfaces__srv__ChangeRobotStatus_Request__TYPE_NAME, 46, 46},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_interfaces__srv__ChangeRobotStatus_Response__TYPE_NAME, 47, 47},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_interfaces__srv__ChangeRobotStatus_Event__TYPE_NAME, 44, 44},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_interfaces__srv__ChangeRobotStatus__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Event__TYPE_NAME, 44, 44},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Request__TYPE_NAME, 46, 46},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Response__TYPE_NAME, 47, 47},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_interfaces__srv__ChangeRobotStatus__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__ChangeRobotStatus__TYPE_NAME, 38, 38},
      {robot_interfaces__srv__ChangeRobotStatus__FIELDS, 3, 3},
    },
    {robot_interfaces__srv__ChangeRobotStatus__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = robot_interfaces__srv__ChangeRobotStatus_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = robot_interfaces__srv__ChangeRobotStatus_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = robot_interfaces__srv__ChangeRobotStatus_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_interfaces__srv__ChangeRobotStatus_Request__FIELD_NAME__robot_id[] = "robot_id";
static char robot_interfaces__srv__ChangeRobotStatus_Request__FIELD_NAME__new_status[] = "new_status";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__ChangeRobotStatus_Request__FIELDS[] = {
  {
    {robot_interfaces__srv__ChangeRobotStatus_Request__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Request__FIELD_NAME__new_status, 10, 10},
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
robot_interfaces__srv__ChangeRobotStatus_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__ChangeRobotStatus_Request__TYPE_NAME, 46, 46},
      {robot_interfaces__srv__ChangeRobotStatus_Request__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_interfaces__srv__ChangeRobotStatus_Response__FIELD_NAME__success[] = "success";
static char robot_interfaces__srv__ChangeRobotStatus_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__ChangeRobotStatus_Response__FIELDS[] = {
  {
    {robot_interfaces__srv__ChangeRobotStatus_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Response__FIELD_NAME__message, 7, 7},
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
robot_interfaces__srv__ChangeRobotStatus_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__ChangeRobotStatus_Response__TYPE_NAME, 47, 47},
      {robot_interfaces__srv__ChangeRobotStatus_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_interfaces__srv__ChangeRobotStatus_Event__FIELD_NAME__info[] = "info";
static char robot_interfaces__srv__ChangeRobotStatus_Event__FIELD_NAME__request[] = "request";
static char robot_interfaces__srv__ChangeRobotStatus_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__ChangeRobotStatus_Event__FIELDS[] = {
  {
    {robot_interfaces__srv__ChangeRobotStatus_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_interfaces__srv__ChangeRobotStatus_Request__TYPE_NAME, 46, 46},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_interfaces__srv__ChangeRobotStatus_Response__TYPE_NAME, 47, 47},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_interfaces__srv__ChangeRobotStatus_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Request__TYPE_NAME, 46, 46},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__ChangeRobotStatus_Response__TYPE_NAME, 47, 47},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_interfaces__srv__ChangeRobotStatus_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__ChangeRobotStatus_Event__TYPE_NAME, 44, 44},
      {robot_interfaces__srv__ChangeRobotStatus_Event__FIELDS, 3, 3},
    },
    {robot_interfaces__srv__ChangeRobotStatus_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = robot_interfaces__srv__ChangeRobotStatus_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = robot_interfaces__srv__ChangeRobotStatus_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 robot_id\n"
  "string new_status  # \"idle\", \"assigned\", \"occupied\" (User GUI\\xec\\x97\\x90\\xec\\x84\\x9c \\xeb\\xb0\\x9b\\xec\\x9d\\x80 \\xec\\x83\\x81\\xed\\x83\\x9c\\xeb\\xa7\\x8c)\n"
  "---\n"
  "bool success\n"
  "string message ";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__ChangeRobotStatus__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__ChangeRobotStatus__TYPE_NAME, 38, 38},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 119, 119},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__ChangeRobotStatus_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__ChangeRobotStatus_Request__TYPE_NAME, 46, 46},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__ChangeRobotStatus_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__ChangeRobotStatus_Response__TYPE_NAME, 47, 47},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__ChangeRobotStatus_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__ChangeRobotStatus_Event__TYPE_NAME, 44, 44},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__ChangeRobotStatus__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__ChangeRobotStatus__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *robot_interfaces__srv__ChangeRobotStatus_Event__get_individual_type_description_source(NULL);
    sources[3] = *robot_interfaces__srv__ChangeRobotStatus_Request__get_individual_type_description_source(NULL);
    sources[4] = *robot_interfaces__srv__ChangeRobotStatus_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__ChangeRobotStatus_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__ChangeRobotStatus_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__ChangeRobotStatus_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__ChangeRobotStatus_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__ChangeRobotStatus_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__ChangeRobotStatus_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *robot_interfaces__srv__ChangeRobotStatus_Request__get_individual_type_description_source(NULL);
    sources[3] = *robot_interfaces__srv__ChangeRobotStatus_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

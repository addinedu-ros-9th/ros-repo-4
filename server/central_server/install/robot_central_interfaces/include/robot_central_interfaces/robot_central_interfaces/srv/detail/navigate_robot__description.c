// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from robot_central_interfaces:srv/NavigateRobot.idl
// generated code does not contain a copyright notice

#include "robot_central_interfaces/srv/detail/navigate_robot__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__NavigateRobot__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xdb, 0x7f, 0x53, 0xcf, 0xc9, 0x14, 0xa0, 0x56,
      0x67, 0xd4, 0xf7, 0x6a, 0x96, 0x27, 0xd9, 0x73,
      0x0c, 0x09, 0xf7, 0x82, 0x6f, 0x4d, 0x3d, 0x91,
      0x41, 0xb1, 0x93, 0x4e, 0xe1, 0xa4, 0x04, 0xc1,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__NavigateRobot_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xd2, 0x71, 0xb8, 0xc2, 0x91, 0xf6, 0x98, 0x20,
      0x4c, 0x73, 0x55, 0x77, 0x85, 0xda, 0xea, 0x44,
      0x6c, 0x1d, 0x24, 0x9c, 0x06, 0xbe, 0x7d, 0x89,
      0x4c, 0x1d, 0x71, 0xad, 0x7b, 0xbe, 0x36, 0x9b,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__NavigateRobot_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x25, 0x25, 0x88, 0x32, 0xed, 0x48, 0x8c, 0xdb,
      0x30, 0x72, 0xb4, 0x1b, 0xe6, 0x0e, 0xb8, 0x52,
      0xed, 0x31, 0x6c, 0x8e, 0x4c, 0xad, 0x74, 0x58,
      0xd0, 0xe5, 0xa2, 0x63, 0xa1, 0xf3, 0xdf, 0x65,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_central_interfaces
const rosidl_type_hash_t *
robot_central_interfaces__srv__NavigateRobot_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xa7, 0x1b, 0xb4, 0x6e, 0x74, 0x9b, 0x27, 0xbe,
      0x1b, 0xb5, 0x06, 0xf8, 0xc1, 0x98, 0x41, 0x0e,
      0x1b, 0x08, 0xa6, 0x9c, 0xbb, 0x99, 0x51, 0x35,
      0x88, 0xb3, 0xfe, 0x34, 0x21, 0x6e, 0x30, 0x6a,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

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

static char robot_central_interfaces__srv__NavigateRobot__TYPE_NAME[] = "robot_central_interfaces/srv/NavigateRobot";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char robot_central_interfaces__srv__NavigateRobot_Event__TYPE_NAME[] = "robot_central_interfaces/srv/NavigateRobot_Event";
static char robot_central_interfaces__srv__NavigateRobot_Request__TYPE_NAME[] = "robot_central_interfaces/srv/NavigateRobot_Request";
static char robot_central_interfaces__srv__NavigateRobot_Response__TYPE_NAME[] = "robot_central_interfaces/srv/NavigateRobot_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char robot_central_interfaces__srv__NavigateRobot__FIELD_NAME__request_message[] = "request_message";
static char robot_central_interfaces__srv__NavigateRobot__FIELD_NAME__response_message[] = "response_message";
static char robot_central_interfaces__srv__NavigateRobot__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__NavigateRobot__FIELDS[] = {
  {
    {robot_central_interfaces__srv__NavigateRobot__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_central_interfaces__srv__NavigateRobot_Request__TYPE_NAME, 50, 50},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_central_interfaces__srv__NavigateRobot_Response__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_central_interfaces__srv__NavigateRobot_Event__TYPE_NAME, 48, 48},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_central_interfaces__srv__NavigateRobot__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Event__TYPE_NAME, 48, 48},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Request__TYPE_NAME, 50, 50},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Response__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_central_interfaces__srv__NavigateRobot__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__NavigateRobot__TYPE_NAME, 42, 42},
      {robot_central_interfaces__srv__NavigateRobot__FIELDS, 3, 3},
    },
    {robot_central_interfaces__srv__NavigateRobot__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = robot_central_interfaces__srv__NavigateRobot_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = robot_central_interfaces__srv__NavigateRobot_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = robot_central_interfaces__srv__NavigateRobot_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_central_interfaces__srv__NavigateRobot_Request__FIELD_NAME__robot_id[] = "robot_id";
static char robot_central_interfaces__srv__NavigateRobot_Request__FIELD_NAME__waypoints[] = "waypoints";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__NavigateRobot_Request__FIELDS[] = {
  {
    {robot_central_interfaces__srv__NavigateRobot_Request__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Request__FIELD_NAME__waypoints, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_central_interfaces__srv__NavigateRobot_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__NavigateRobot_Request__TYPE_NAME, 50, 50},
      {robot_central_interfaces__srv__NavigateRobot_Request__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_central_interfaces__srv__NavigateRobot_Response__FIELD_NAME__success[] = "success";
static char robot_central_interfaces__srv__NavigateRobot_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__NavigateRobot_Response__FIELDS[] = {
  {
    {robot_central_interfaces__srv__NavigateRobot_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Response__FIELD_NAME__message, 7, 7},
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
robot_central_interfaces__srv__NavigateRobot_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__NavigateRobot_Response__TYPE_NAME, 51, 51},
      {robot_central_interfaces__srv__NavigateRobot_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_central_interfaces__srv__NavigateRobot_Event__FIELD_NAME__info[] = "info";
static char robot_central_interfaces__srv__NavigateRobot_Event__FIELD_NAME__request[] = "request";
static char robot_central_interfaces__srv__NavigateRobot_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field robot_central_interfaces__srv__NavigateRobot_Event__FIELDS[] = {
  {
    {robot_central_interfaces__srv__NavigateRobot_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_central_interfaces__srv__NavigateRobot_Request__TYPE_NAME, 50, 50},
    },
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_central_interfaces__srv__NavigateRobot_Response__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_central_interfaces__srv__NavigateRobot_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Request__TYPE_NAME, 50, 50},
    {NULL, 0, 0},
  },
  {
    {robot_central_interfaces__srv__NavigateRobot_Response__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_central_interfaces__srv__NavigateRobot_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_central_interfaces__srv__NavigateRobot_Event__TYPE_NAME, 48, 48},
      {robot_central_interfaces__srv__NavigateRobot_Event__FIELDS, 3, 3},
    },
    {robot_central_interfaces__srv__NavigateRobot_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = robot_central_interfaces__srv__NavigateRobot_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = robot_central_interfaces__srv__NavigateRobot_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# Request\n"
  "string robot_id\n"
  "string[] waypoints\n"
  "---\n"
  "# Response\n"
  "bool success\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__NavigateRobot__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__NavigateRobot__TYPE_NAME, 42, 42},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 87, 87},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__NavigateRobot_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__NavigateRobot_Request__TYPE_NAME, 50, 50},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__NavigateRobot_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__NavigateRobot_Response__TYPE_NAME, 51, 51},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_central_interfaces__srv__NavigateRobot_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_central_interfaces__srv__NavigateRobot_Event__TYPE_NAME, 48, 48},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__NavigateRobot__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__NavigateRobot__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *robot_central_interfaces__srv__NavigateRobot_Event__get_individual_type_description_source(NULL);
    sources[3] = *robot_central_interfaces__srv__NavigateRobot_Request__get_individual_type_description_source(NULL);
    sources[4] = *robot_central_interfaces__srv__NavigateRobot_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__NavigateRobot_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__NavigateRobot_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__NavigateRobot_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__NavigateRobot_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_central_interfaces__srv__NavigateRobot_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_central_interfaces__srv__NavigateRobot_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *robot_central_interfaces__srv__NavigateRobot_Request__get_individual_type_description_source(NULL);
    sources[3] = *robot_central_interfaces__srv__NavigateRobot_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

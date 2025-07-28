// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from robot_interfaces:srv/LoginAdmin.idl
// generated code does not contain a copyright notice

#include "robot_interfaces/srv/detail/login_admin__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__LoginAdmin__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xc3, 0x2b, 0xa3, 0xf2, 0xd1, 0xdf, 0xbe, 0x45,
      0xa7, 0x0b, 0x9f, 0x0d, 0x8c, 0x29, 0x33, 0x3b,
      0x0a, 0x5c, 0x52, 0x4f, 0x49, 0xeb, 0xa4, 0xb2,
      0x28, 0x80, 0xb0, 0x1b, 0xf9, 0x30, 0x54, 0x6f,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__LoginAdmin_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xeb, 0xa5, 0xef, 0x0c, 0xd2, 0x08, 0x05, 0x99,
      0xff, 0xca, 0x47, 0x24, 0x1e, 0x9b, 0x17, 0x31,
      0x91, 0x00, 0x50, 0x0a, 0xa8, 0xb6, 0xa3, 0x02,
      0x78, 0xd7, 0x6a, 0x1a, 0x79, 0x29, 0x7e, 0xe0,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__LoginAdmin_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xc8, 0x12, 0xbd, 0x79, 0xaa, 0xf9, 0x7a, 0xcc,
      0xdf, 0x02, 0x3f, 0x70, 0xd5, 0x8c, 0x48, 0xc3,
      0x0c, 0x08, 0x46, 0x96, 0xc2, 0x71, 0xc6, 0xe9,
      0x44, 0x2b, 0x61, 0xc6, 0x5b, 0xc9, 0xd5, 0xf2,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__srv__LoginAdmin_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x59, 0xb6, 0x4f, 0x04, 0x37, 0x64, 0x37, 0x60,
      0x83, 0x76, 0x4e, 0x24, 0xf4, 0x15, 0x09, 0x54,
      0xf3, 0x08, 0x9b, 0x59, 0xfd, 0x5f, 0x24, 0x00,
      0xdd, 0x2d, 0x90, 0xd2, 0xd8, 0x4d, 0xb0, 0xbc,
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

static char robot_interfaces__srv__LoginAdmin__TYPE_NAME[] = "robot_interfaces/srv/LoginAdmin";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char robot_interfaces__srv__LoginAdmin_Event__TYPE_NAME[] = "robot_interfaces/srv/LoginAdmin_Event";
static char robot_interfaces__srv__LoginAdmin_Request__TYPE_NAME[] = "robot_interfaces/srv/LoginAdmin_Request";
static char robot_interfaces__srv__LoginAdmin_Response__TYPE_NAME[] = "robot_interfaces/srv/LoginAdmin_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char robot_interfaces__srv__LoginAdmin__FIELD_NAME__request_message[] = "request_message";
static char robot_interfaces__srv__LoginAdmin__FIELD_NAME__response_message[] = "response_message";
static char robot_interfaces__srv__LoginAdmin__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__LoginAdmin__FIELDS[] = {
  {
    {robot_interfaces__srv__LoginAdmin__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_interfaces__srv__LoginAdmin_Request__TYPE_NAME, 39, 39},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_interfaces__srv__LoginAdmin_Response__TYPE_NAME, 40, 40},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {robot_interfaces__srv__LoginAdmin_Event__TYPE_NAME, 37, 37},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_interfaces__srv__LoginAdmin__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Event__TYPE_NAME, 37, 37},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Request__TYPE_NAME, 39, 39},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Response__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_interfaces__srv__LoginAdmin__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__LoginAdmin__TYPE_NAME, 31, 31},
      {robot_interfaces__srv__LoginAdmin__FIELDS, 3, 3},
    },
    {robot_interfaces__srv__LoginAdmin__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = robot_interfaces__srv__LoginAdmin_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = robot_interfaces__srv__LoginAdmin_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = robot_interfaces__srv__LoginAdmin_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_interfaces__srv__LoginAdmin_Request__FIELD_NAME__id[] = "id";
static char robot_interfaces__srv__LoginAdmin_Request__FIELD_NAME__password[] = "password";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__LoginAdmin_Request__FIELDS[] = {
  {
    {robot_interfaces__srv__LoginAdmin_Request__FIELD_NAME__id, 2, 2},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Request__FIELD_NAME__password, 8, 8},
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
robot_interfaces__srv__LoginAdmin_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__LoginAdmin_Request__TYPE_NAME, 39, 39},
      {robot_interfaces__srv__LoginAdmin_Request__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__success[] = "success";
static char robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__name[] = "name";
static char robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__email[] = "email";
static char robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__hospital_name[] = "hospital_name";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__LoginAdmin_Response__FIELDS[] = {
  {
    {robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__name, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__email, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Response__FIELD_NAME__hospital_name, 13, 13},
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
robot_interfaces__srv__LoginAdmin_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__LoginAdmin_Response__TYPE_NAME, 40, 40},
      {robot_interfaces__srv__LoginAdmin_Response__FIELDS, 4, 4},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char robot_interfaces__srv__LoginAdmin_Event__FIELD_NAME__info[] = "info";
static char robot_interfaces__srv__LoginAdmin_Event__FIELD_NAME__request[] = "request";
static char robot_interfaces__srv__LoginAdmin_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field robot_interfaces__srv__LoginAdmin_Event__FIELDS[] = {
  {
    {robot_interfaces__srv__LoginAdmin_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_interfaces__srv__LoginAdmin_Request__TYPE_NAME, 39, 39},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {robot_interfaces__srv__LoginAdmin_Response__TYPE_NAME, 40, 40},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_interfaces__srv__LoginAdmin_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Request__TYPE_NAME, 39, 39},
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__srv__LoginAdmin_Response__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_interfaces__srv__LoginAdmin_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__srv__LoginAdmin_Event__TYPE_NAME, 37, 37},
      {robot_interfaces__srv__LoginAdmin_Event__FIELDS, 3, 3},
    },
    {robot_interfaces__srv__LoginAdmin_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = robot_interfaces__srv__LoginAdmin_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = robot_interfaces__srv__LoginAdmin_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "string id\n"
  "string password\n"
  "---\n"
  "bool success\n"
  "string name\n"
  "string email\n"
  "string hospital_name ";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__LoginAdmin__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__LoginAdmin__TYPE_NAME, 31, 31},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 89, 89},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__LoginAdmin_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__LoginAdmin_Request__TYPE_NAME, 39, 39},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__LoginAdmin_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__LoginAdmin_Response__TYPE_NAME, 40, 40},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__srv__LoginAdmin_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__srv__LoginAdmin_Event__TYPE_NAME, 37, 37},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__LoginAdmin__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__LoginAdmin__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *robot_interfaces__srv__LoginAdmin_Event__get_individual_type_description_source(NULL);
    sources[3] = *robot_interfaces__srv__LoginAdmin_Request__get_individual_type_description_source(NULL);
    sources[4] = *robot_interfaces__srv__LoginAdmin_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__LoginAdmin_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__LoginAdmin_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__LoginAdmin_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__LoginAdmin_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__srv__LoginAdmin_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__srv__LoginAdmin_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *robot_interfaces__srv__LoginAdmin_Request__get_individual_type_description_source(NULL);
    sources[3] = *robot_interfaces__srv__LoginAdmin_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

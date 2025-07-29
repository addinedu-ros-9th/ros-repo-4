// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from robot_interfaces:msg/RobotFeedback.idl
// generated code does not contain a copyright notice

#include "robot_interfaces/msg/detail/robot_feedback__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_robot_interfaces
const rosidl_type_hash_t *
robot_interfaces__msg__RobotFeedback__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x3a, 0x53, 0xe7, 0x85, 0x91, 0x1b, 0x35, 0xfc,
      0x64, 0xbe, 0x86, 0x15, 0x6f, 0x72, 0x02, 0xaf,
      0x1b, 0x40, 0x94, 0x00, 0xd3, 0x13, 0xa0, 0xbe,
      0xd2, 0xb8, 0x03, 0xed, 0x60, 0xe3, 0x56, 0x08,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "std_msgs/msg/detail/header__functions.h"
#include "geometry_msgs/msg/detail/point__functions.h"
#include "geometry_msgs/msg/detail/pose_with_covariance__functions.h"
#include "geometry_msgs/msg/detail/quaternion__functions.h"
#include "geometry_msgs/msg/detail/pose_with_covariance_stamped__functions.h"
#include "geometry_msgs/msg/detail/pose__functions.h"

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
static const rosidl_type_hash_t geometry_msgs__msg__PoseWithCovariance__EXPECTED_HASH = {1, {
    0x9a, 0x7c, 0x0f, 0xd2, 0x34, 0xb7, 0xf4, 0x5c,
    0x60, 0x98, 0x74, 0x5e, 0xcc, 0xcd, 0x77, 0x3c,
    0xa1, 0x08, 0x56, 0x70, 0xe6, 0x41, 0x07, 0x13,
    0x53, 0x97, 0xae, 0xe3, 0x1c, 0x02, 0xe1, 0xbb,
  }};
static const rosidl_type_hash_t geometry_msgs__msg__PoseWithCovarianceStamped__EXPECTED_HASH = {1, {
    0x26, 0x43, 0x2f, 0x98, 0x03, 0xe4, 0x37, 0x27,
    0xd3, 0xc8, 0xf6, 0x68, 0xd1, 0xfd, 0xb3, 0xc6,
    0x30, 0xf5, 0x48, 0xaf, 0x63, 0x1e, 0x2f, 0x4e,
    0x31, 0x38, 0x23, 0x71, 0xbf, 0xea, 0x3b, 0x6e,
  }};
static const rosidl_type_hash_t geometry_msgs__msg__Quaternion__EXPECTED_HASH = {1, {
    0x8a, 0x76, 0x5f, 0x66, 0x77, 0x8c, 0x8f, 0xf7,
    0xc8, 0xab, 0x94, 0xaf, 0xcc, 0x59, 0x0a, 0x2e,
    0xd5, 0x32, 0x5a, 0x1d, 0x9a, 0x07, 0x6f, 0xff,
    0xf3, 0x8f, 0xbc, 0xe3, 0x6f, 0x45, 0x86, 0x84,
  }};
static const rosidl_type_hash_t std_msgs__msg__Header__EXPECTED_HASH = {1, {
    0xf4, 0x9f, 0xb3, 0xae, 0x2c, 0xf0, 0x70, 0xf7,
    0x93, 0x64, 0x5f, 0xf7, 0x49, 0x68, 0x3a, 0xc6,
    0xb0, 0x62, 0x03, 0xe4, 0x1c, 0x89, 0x1e, 0x17,
    0x70, 0x1b, 0x1c, 0xb5, 0x97, 0xce, 0x6a, 0x01,
  }};
#endif

static char robot_interfaces__msg__RobotFeedback__TYPE_NAME[] = "robot_interfaces/msg/RobotFeedback";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char geometry_msgs__msg__Point__TYPE_NAME[] = "geometry_msgs/msg/Point";
static char geometry_msgs__msg__Pose__TYPE_NAME[] = "geometry_msgs/msg/Pose";
static char geometry_msgs__msg__PoseWithCovariance__TYPE_NAME[] = "geometry_msgs/msg/PoseWithCovariance";
static char geometry_msgs__msg__PoseWithCovarianceStamped__TYPE_NAME[] = "geometry_msgs/msg/PoseWithCovarianceStamped";
static char geometry_msgs__msg__Quaternion__TYPE_NAME[] = "geometry_msgs/msg/Quaternion";
static char std_msgs__msg__Header__TYPE_NAME[] = "std_msgs/msg/Header";

// Define type names, field names, and default values
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__robot_id[] = "robot_id";
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__status[] = "status";
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__current_pose[] = "current_pose";
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__battery_percent[] = "battery_percent";
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__network_status[] = "network_status";
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__current_speed[] = "current_speed";
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__detected_objects[] = "detected_objects";
static char robot_interfaces__msg__RobotFeedback__FIELD_NAME__error_message[] = "error_message";

static rosidl_runtime_c__type_description__Field robot_interfaces__msg__RobotFeedback__FIELDS[] = {
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__robot_id, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__status, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__current_pose, 12, 12},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {geometry_msgs__msg__PoseWithCovarianceStamped__TYPE_NAME, 43, 43},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__battery_percent, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__network_status, 14, 14},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__current_speed, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__detected_objects, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {robot_interfaces__msg__RobotFeedback__FIELD_NAME__error_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription robot_interfaces__msg__RobotFeedback__REFERENCED_TYPE_DESCRIPTIONS[] = {
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
    {geometry_msgs__msg__PoseWithCovariance__TYPE_NAME, 36, 36},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__PoseWithCovarianceStamped__TYPE_NAME, 43, 43},
    {NULL, 0, 0},
  },
  {
    {geometry_msgs__msg__Quaternion__TYPE_NAME, 28, 28},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
robot_interfaces__msg__RobotFeedback__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {robot_interfaces__msg__RobotFeedback__TYPE_NAME, 34, 34},
      {robot_interfaces__msg__RobotFeedback__FIELDS, 8, 8},
    },
    {robot_interfaces__msg__RobotFeedback__REFERENCED_TYPE_DESCRIPTIONS, 7, 7},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Point__EXPECTED_HASH, geometry_msgs__msg__Point__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = geometry_msgs__msg__Point__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Pose__EXPECTED_HASH, geometry_msgs__msg__Pose__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = geometry_msgs__msg__Pose__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__PoseWithCovariance__EXPECTED_HASH, geometry_msgs__msg__PoseWithCovariance__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = geometry_msgs__msg__PoseWithCovariance__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__PoseWithCovarianceStamped__EXPECTED_HASH, geometry_msgs__msg__PoseWithCovarianceStamped__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = geometry_msgs__msg__PoseWithCovarianceStamped__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&geometry_msgs__msg__Quaternion__EXPECTED_HASH, geometry_msgs__msg__Quaternion__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[5].fields = geometry_msgs__msg__Quaternion__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[6].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "# \\xeb\\xa1\\x9c\\xeb\\xb4\\x87 \\xec\\xbb\\xa8\\xed\\x8a\\xb8\\xeb\\xa1\\xa4\\xeb\\x9f\\xac\\xec\\x97\\x90\\xec\\x84\\x9c \\xec\\xa4\\x91\\xec\\x95\\x99\\xec\\x84\\x9c\\xeb\\xb2\\x84\\xeb\\xa1\\x9c \\xeb\\xb3\\xb4\\xeb\\x82\\xb4\\xeb\\x8a\\x94 \\xed\\x94\\xbc\\xeb\\x93\\x9c\\xeb\\xb0\\xb1\n"
  "int32 robot_id\n"
  "string status  # \"moving\", \"standby\" (\\xeb\\xa1\\x9c\\xeb\\xb4\\x87\\xec\\x9d\\xb4 \\xec\\x8a\\xa4\\xec\\x8a\\xa4\\xeb\\xa1\\x9c \\xed\\x8c\\x90\\xeb\\x8b\\xa8\\xed\\x95\\x98\\xeb\\x8a\\x94 \\xec\\x83\\x81\\xed\\x83\\x9c\\xeb\\xa7\\x8c)\n"
  "geometry_msgs/PoseWithCovarianceStamped current_pose\n"
  "int32 battery_percent\n"
  "string network_status  # \"connected\", \"disconnected\", \"weak\"\n"
  "float32 current_speed  # \\xed\\x98\\x84\\xec\\x9e\\xac \\xec\\x9d\\xb4\\xeb\\x8f\\x99 \\xec\\x86\\x8d\\xeb\\x8f\\x84 (m/s)\n"
  "string[] detected_objects  # \\xea\\xb0\\x90\\xec\\xa7\\x80\\xeb\\x90\\x9c \\xea\\xb0\\x9d\\xec\\xb2\\xb4\\xeb\\x93\\xa4 [\"person\", \"obstacle\", \"chair\", ...]\n"
  "string error_message  # \\xec\\x98\\xa4\\xeb\\xa5\\x98 \\xeb\\xb0\\x9c\\xec\\x83\\x9d\\xec\\x8b\\x9c \\xeb\\xa9\\x94\\xec\\x8b\\x9c\\xec\\xa7\\x80 (\\xec\\xa0\\x95\\xec\\x83\\x81\\xec\\x8b\\x9c \\xeb\\xb9\\x88 \\xeb\\xac\\xb8\\xec\\x9e\\x90\\xec\\x97\\xb4) ";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
robot_interfaces__msg__RobotFeedback__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {robot_interfaces__msg__RobotFeedback__TYPE_NAME, 34, 34},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 394, 394},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
robot_interfaces__msg__RobotFeedback__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[8];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 8, 8};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *robot_interfaces__msg__RobotFeedback__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *geometry_msgs__msg__Point__get_individual_type_description_source(NULL);
    sources[3] = *geometry_msgs__msg__Pose__get_individual_type_description_source(NULL);
    sources[4] = *geometry_msgs__msg__PoseWithCovariance__get_individual_type_description_source(NULL);
    sources[5] = *geometry_msgs__msg__PoseWithCovarianceStamped__get_individual_type_description_source(NULL);
    sources[6] = *geometry_msgs__msg__Quaternion__get_individual_type_description_source(NULL);
    sources[7] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

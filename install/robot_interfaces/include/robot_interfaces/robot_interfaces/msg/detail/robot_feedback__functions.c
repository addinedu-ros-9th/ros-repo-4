// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from robot_interfaces:msg/RobotFeedback.idl
// generated code does not contain a copyright notice
#include "robot_interfaces/msg/detail/robot_feedback__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `status`
// Member `network_status`
// Member `detected_objects`
// Member `error_message`
#include "rosidl_runtime_c/string_functions.h"
// Member `current_pose`
#include "geometry_msgs/msg/detail/pose_with_covariance_stamped__functions.h"

bool
robot_interfaces__msg__RobotFeedback__init(robot_interfaces__msg__RobotFeedback * msg)
{
  if (!msg) {
    return false;
  }
  // robot_id
  // status
  if (!rosidl_runtime_c__String__init(&msg->status)) {
    robot_interfaces__msg__RobotFeedback__fini(msg);
    return false;
  }
  // current_pose
  if (!geometry_msgs__msg__PoseWithCovarianceStamped__init(&msg->current_pose)) {
    robot_interfaces__msg__RobotFeedback__fini(msg);
    return false;
  }
  // battery_percent
  // network_status
  if (!rosidl_runtime_c__String__init(&msg->network_status)) {
    robot_interfaces__msg__RobotFeedback__fini(msg);
    return false;
  }
  // current_speed
  // detected_objects
  if (!rosidl_runtime_c__String__Sequence__init(&msg->detected_objects, 0)) {
    robot_interfaces__msg__RobotFeedback__fini(msg);
    return false;
  }
  // error_message
  if (!rosidl_runtime_c__String__init(&msg->error_message)) {
    robot_interfaces__msg__RobotFeedback__fini(msg);
    return false;
  }
  return true;
}

void
robot_interfaces__msg__RobotFeedback__fini(robot_interfaces__msg__RobotFeedback * msg)
{
  if (!msg) {
    return;
  }
  // robot_id
  // status
  rosidl_runtime_c__String__fini(&msg->status);
  // current_pose
  geometry_msgs__msg__PoseWithCovarianceStamped__fini(&msg->current_pose);
  // battery_percent
  // network_status
  rosidl_runtime_c__String__fini(&msg->network_status);
  // current_speed
  // detected_objects
  rosidl_runtime_c__String__Sequence__fini(&msg->detected_objects);
  // error_message
  rosidl_runtime_c__String__fini(&msg->error_message);
}

bool
robot_interfaces__msg__RobotFeedback__are_equal(const robot_interfaces__msg__RobotFeedback * lhs, const robot_interfaces__msg__RobotFeedback * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // robot_id
  if (lhs->robot_id != rhs->robot_id) {
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->status), &(rhs->status)))
  {
    return false;
  }
  // current_pose
  if (!geometry_msgs__msg__PoseWithCovarianceStamped__are_equal(
      &(lhs->current_pose), &(rhs->current_pose)))
  {
    return false;
  }
  // battery_percent
  if (lhs->battery_percent != rhs->battery_percent) {
    return false;
  }
  // network_status
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->network_status), &(rhs->network_status)))
  {
    return false;
  }
  // current_speed
  if (lhs->current_speed != rhs->current_speed) {
    return false;
  }
  // detected_objects
  if (!rosidl_runtime_c__String__Sequence__are_equal(
      &(lhs->detected_objects), &(rhs->detected_objects)))
  {
    return false;
  }
  // error_message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->error_message), &(rhs->error_message)))
  {
    return false;
  }
  return true;
}

bool
robot_interfaces__msg__RobotFeedback__copy(
  const robot_interfaces__msg__RobotFeedback * input,
  robot_interfaces__msg__RobotFeedback * output)
{
  if (!input || !output) {
    return false;
  }
  // robot_id
  output->robot_id = input->robot_id;
  // status
  if (!rosidl_runtime_c__String__copy(
      &(input->status), &(output->status)))
  {
    return false;
  }
  // current_pose
  if (!geometry_msgs__msg__PoseWithCovarianceStamped__copy(
      &(input->current_pose), &(output->current_pose)))
  {
    return false;
  }
  // battery_percent
  output->battery_percent = input->battery_percent;
  // network_status
  if (!rosidl_runtime_c__String__copy(
      &(input->network_status), &(output->network_status)))
  {
    return false;
  }
  // current_speed
  output->current_speed = input->current_speed;
  // detected_objects
  if (!rosidl_runtime_c__String__Sequence__copy(
      &(input->detected_objects), &(output->detected_objects)))
  {
    return false;
  }
  // error_message
  if (!rosidl_runtime_c__String__copy(
      &(input->error_message), &(output->error_message)))
  {
    return false;
  }
  return true;
}

robot_interfaces__msg__RobotFeedback *
robot_interfaces__msg__RobotFeedback__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_interfaces__msg__RobotFeedback * msg = (robot_interfaces__msg__RobotFeedback *)allocator.allocate(sizeof(robot_interfaces__msg__RobotFeedback), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(robot_interfaces__msg__RobotFeedback));
  bool success = robot_interfaces__msg__RobotFeedback__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
robot_interfaces__msg__RobotFeedback__destroy(robot_interfaces__msg__RobotFeedback * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    robot_interfaces__msg__RobotFeedback__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
robot_interfaces__msg__RobotFeedback__Sequence__init(robot_interfaces__msg__RobotFeedback__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_interfaces__msg__RobotFeedback * data = NULL;

  if (size) {
    data = (robot_interfaces__msg__RobotFeedback *)allocator.zero_allocate(size, sizeof(robot_interfaces__msg__RobotFeedback), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = robot_interfaces__msg__RobotFeedback__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        robot_interfaces__msg__RobotFeedback__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
robot_interfaces__msg__RobotFeedback__Sequence__fini(robot_interfaces__msg__RobotFeedback__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      robot_interfaces__msg__RobotFeedback__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

robot_interfaces__msg__RobotFeedback__Sequence *
robot_interfaces__msg__RobotFeedback__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_interfaces__msg__RobotFeedback__Sequence * array = (robot_interfaces__msg__RobotFeedback__Sequence *)allocator.allocate(sizeof(robot_interfaces__msg__RobotFeedback__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = robot_interfaces__msg__RobotFeedback__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
robot_interfaces__msg__RobotFeedback__Sequence__destroy(robot_interfaces__msg__RobotFeedback__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    robot_interfaces__msg__RobotFeedback__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
robot_interfaces__msg__RobotFeedback__Sequence__are_equal(const robot_interfaces__msg__RobotFeedback__Sequence * lhs, const robot_interfaces__msg__RobotFeedback__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!robot_interfaces__msg__RobotFeedback__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
robot_interfaces__msg__RobotFeedback__Sequence__copy(
  const robot_interfaces__msg__RobotFeedback__Sequence * input,
  robot_interfaces__msg__RobotFeedback__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(robot_interfaces__msg__RobotFeedback);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    robot_interfaces__msg__RobotFeedback * data =
      (robot_interfaces__msg__RobotFeedback *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!robot_interfaces__msg__RobotFeedback__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          robot_interfaces__msg__RobotFeedback__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!robot_interfaces__msg__RobotFeedback__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

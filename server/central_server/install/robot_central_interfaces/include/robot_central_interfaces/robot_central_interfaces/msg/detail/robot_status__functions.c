// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from robot_central_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice
#include "robot_central_interfaces/msg/detail/robot_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `robot_id`
// Member `status`
// Member `current_task`
#include "rosidl_runtime_c/string_functions.h"
// Member `current_pose`
#include "geometry_msgs/msg/detail/pose__functions.h"
// Member `last_update`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
robot_central_interfaces__msg__RobotStatus__init(robot_central_interfaces__msg__RobotStatus * msg)
{
  if (!msg) {
    return false;
  }
  // robot_id
  if (!rosidl_runtime_c__String__init(&msg->robot_id)) {
    robot_central_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__init(&msg->status)) {
    robot_central_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // current_task
  if (!rosidl_runtime_c__String__init(&msg->current_task)) {
    robot_central_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // current_pose
  if (!geometry_msgs__msg__Pose__init(&msg->current_pose)) {
    robot_central_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // last_update
  if (!builtin_interfaces__msg__Time__init(&msg->last_update)) {
    robot_central_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  return true;
}

void
robot_central_interfaces__msg__RobotStatus__fini(robot_central_interfaces__msg__RobotStatus * msg)
{
  if (!msg) {
    return;
  }
  // robot_id
  rosidl_runtime_c__String__fini(&msg->robot_id);
  // status
  rosidl_runtime_c__String__fini(&msg->status);
  // current_task
  rosidl_runtime_c__String__fini(&msg->current_task);
  // current_pose
  geometry_msgs__msg__Pose__fini(&msg->current_pose);
  // last_update
  builtin_interfaces__msg__Time__fini(&msg->last_update);
}

bool
robot_central_interfaces__msg__RobotStatus__are_equal(const robot_central_interfaces__msg__RobotStatus * lhs, const robot_central_interfaces__msg__RobotStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // robot_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->robot_id), &(rhs->robot_id)))
  {
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->status), &(rhs->status)))
  {
    return false;
  }
  // current_task
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->current_task), &(rhs->current_task)))
  {
    return false;
  }
  // current_pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->current_pose), &(rhs->current_pose)))
  {
    return false;
  }
  // last_update
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->last_update), &(rhs->last_update)))
  {
    return false;
  }
  return true;
}

bool
robot_central_interfaces__msg__RobotStatus__copy(
  const robot_central_interfaces__msg__RobotStatus * input,
  robot_central_interfaces__msg__RobotStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // robot_id
  if (!rosidl_runtime_c__String__copy(
      &(input->robot_id), &(output->robot_id)))
  {
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__copy(
      &(input->status), &(output->status)))
  {
    return false;
  }
  // current_task
  if (!rosidl_runtime_c__String__copy(
      &(input->current_task), &(output->current_task)))
  {
    return false;
  }
  // current_pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->current_pose), &(output->current_pose)))
  {
    return false;
  }
  // last_update
  if (!builtin_interfaces__msg__Time__copy(
      &(input->last_update), &(output->last_update)))
  {
    return false;
  }
  return true;
}

robot_central_interfaces__msg__RobotStatus *
robot_central_interfaces__msg__RobotStatus__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_central_interfaces__msg__RobotStatus * msg = (robot_central_interfaces__msg__RobotStatus *)allocator.allocate(sizeof(robot_central_interfaces__msg__RobotStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(robot_central_interfaces__msg__RobotStatus));
  bool success = robot_central_interfaces__msg__RobotStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
robot_central_interfaces__msg__RobotStatus__destroy(robot_central_interfaces__msg__RobotStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    robot_central_interfaces__msg__RobotStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
robot_central_interfaces__msg__RobotStatus__Sequence__init(robot_central_interfaces__msg__RobotStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_central_interfaces__msg__RobotStatus * data = NULL;

  if (size) {
    data = (robot_central_interfaces__msg__RobotStatus *)allocator.zero_allocate(size, sizeof(robot_central_interfaces__msg__RobotStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = robot_central_interfaces__msg__RobotStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        robot_central_interfaces__msg__RobotStatus__fini(&data[i - 1]);
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
robot_central_interfaces__msg__RobotStatus__Sequence__fini(robot_central_interfaces__msg__RobotStatus__Sequence * array)
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
      robot_central_interfaces__msg__RobotStatus__fini(&array->data[i]);
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

robot_central_interfaces__msg__RobotStatus__Sequence *
robot_central_interfaces__msg__RobotStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_central_interfaces__msg__RobotStatus__Sequence * array = (robot_central_interfaces__msg__RobotStatus__Sequence *)allocator.allocate(sizeof(robot_central_interfaces__msg__RobotStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = robot_central_interfaces__msg__RobotStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
robot_central_interfaces__msg__RobotStatus__Sequence__destroy(robot_central_interfaces__msg__RobotStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    robot_central_interfaces__msg__RobotStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
robot_central_interfaces__msg__RobotStatus__Sequence__are_equal(const robot_central_interfaces__msg__RobotStatus__Sequence * lhs, const robot_central_interfaces__msg__RobotStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!robot_central_interfaces__msg__RobotStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
robot_central_interfaces__msg__RobotStatus__Sequence__copy(
  const robot_central_interfaces__msg__RobotStatus__Sequence * input,
  robot_central_interfaces__msg__RobotStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(robot_central_interfaces__msg__RobotStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    robot_central_interfaces__msg__RobotStatus * data =
      (robot_central_interfaces__msg__RobotStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!robot_central_interfaces__msg__RobotStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          robot_central_interfaces__msg__RobotStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!robot_central_interfaces__msg__RobotStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

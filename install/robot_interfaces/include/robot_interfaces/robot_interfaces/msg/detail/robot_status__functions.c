// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from robot_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice
#include "robot_interfaces/msg/detail/robot_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `status`
// Member `current_station`
// Member `destination_station`
// Member `network_status`
// Member `source`
#include "rosidl_runtime_c/string_functions.h"
// Member `amcl_pose`
#include "geometry_msgs/msg/detail/pose_with_covariance_stamped__functions.h"

bool
robot_interfaces__msg__RobotStatus__init(robot_interfaces__msg__RobotStatus * msg)
{
  if (!msg) {
    return false;
  }
  // robot_id
  // status
  if (!rosidl_runtime_c__String__init(&msg->status)) {
    robot_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // amcl_pose
  if (!geometry_msgs__msg__PoseWithCovarianceStamped__init(&msg->amcl_pose)) {
    robot_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // current_station
  if (!rosidl_runtime_c__String__init(&msg->current_station)) {
    robot_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // destination_station
  if (!rosidl_runtime_c__String__init(&msg->destination_station)) {
    robot_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // battery_percent
  // network_status
  if (!rosidl_runtime_c__String__init(&msg->network_status)) {
    robot_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // source
  if (!rosidl_runtime_c__String__init(&msg->source)) {
    robot_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  return true;
}

void
robot_interfaces__msg__RobotStatus__fini(robot_interfaces__msg__RobotStatus * msg)
{
  if (!msg) {
    return;
  }
  // robot_id
  // status
  rosidl_runtime_c__String__fini(&msg->status);
  // amcl_pose
  geometry_msgs__msg__PoseWithCovarianceStamped__fini(&msg->amcl_pose);
  // current_station
  rosidl_runtime_c__String__fini(&msg->current_station);
  // destination_station
  rosidl_runtime_c__String__fini(&msg->destination_station);
  // battery_percent
  // network_status
  rosidl_runtime_c__String__fini(&msg->network_status);
  // source
  rosidl_runtime_c__String__fini(&msg->source);
}

bool
robot_interfaces__msg__RobotStatus__are_equal(const robot_interfaces__msg__RobotStatus * lhs, const robot_interfaces__msg__RobotStatus * rhs)
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
  // amcl_pose
  if (!geometry_msgs__msg__PoseWithCovarianceStamped__are_equal(
      &(lhs->amcl_pose), &(rhs->amcl_pose)))
  {
    return false;
  }
  // current_station
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->current_station), &(rhs->current_station)))
  {
    return false;
  }
  // destination_station
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->destination_station), &(rhs->destination_station)))
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
  // source
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->source), &(rhs->source)))
  {
    return false;
  }
  return true;
}

bool
robot_interfaces__msg__RobotStatus__copy(
  const robot_interfaces__msg__RobotStatus * input,
  robot_interfaces__msg__RobotStatus * output)
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
  // amcl_pose
  if (!geometry_msgs__msg__PoseWithCovarianceStamped__copy(
      &(input->amcl_pose), &(output->amcl_pose)))
  {
    return false;
  }
  // current_station
  if (!rosidl_runtime_c__String__copy(
      &(input->current_station), &(output->current_station)))
  {
    return false;
  }
  // destination_station
  if (!rosidl_runtime_c__String__copy(
      &(input->destination_station), &(output->destination_station)))
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
  // source
  if (!rosidl_runtime_c__String__copy(
      &(input->source), &(output->source)))
  {
    return false;
  }
  return true;
}

robot_interfaces__msg__RobotStatus *
robot_interfaces__msg__RobotStatus__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_interfaces__msg__RobotStatus * msg = (robot_interfaces__msg__RobotStatus *)allocator.allocate(sizeof(robot_interfaces__msg__RobotStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(robot_interfaces__msg__RobotStatus));
  bool success = robot_interfaces__msg__RobotStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
robot_interfaces__msg__RobotStatus__destroy(robot_interfaces__msg__RobotStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    robot_interfaces__msg__RobotStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
robot_interfaces__msg__RobotStatus__Sequence__init(robot_interfaces__msg__RobotStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_interfaces__msg__RobotStatus * data = NULL;

  if (size) {
    data = (robot_interfaces__msg__RobotStatus *)allocator.zero_allocate(size, sizeof(robot_interfaces__msg__RobotStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = robot_interfaces__msg__RobotStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        robot_interfaces__msg__RobotStatus__fini(&data[i - 1]);
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
robot_interfaces__msg__RobotStatus__Sequence__fini(robot_interfaces__msg__RobotStatus__Sequence * array)
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
      robot_interfaces__msg__RobotStatus__fini(&array->data[i]);
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

robot_interfaces__msg__RobotStatus__Sequence *
robot_interfaces__msg__RobotStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_interfaces__msg__RobotStatus__Sequence * array = (robot_interfaces__msg__RobotStatus__Sequence *)allocator.allocate(sizeof(robot_interfaces__msg__RobotStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = robot_interfaces__msg__RobotStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
robot_interfaces__msg__RobotStatus__Sequence__destroy(robot_interfaces__msg__RobotStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    robot_interfaces__msg__RobotStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
robot_interfaces__msg__RobotStatus__Sequence__are_equal(const robot_interfaces__msg__RobotStatus__Sequence * lhs, const robot_interfaces__msg__RobotStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!robot_interfaces__msg__RobotStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
robot_interfaces__msg__RobotStatus__Sequence__copy(
  const robot_interfaces__msg__RobotStatus__Sequence * input,
  robot_interfaces__msg__RobotStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(robot_interfaces__msg__RobotStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    robot_interfaces__msg__RobotStatus * data =
      (robot_interfaces__msg__RobotStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!robot_interfaces__msg__RobotStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          robot_interfaces__msg__RobotStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!robot_interfaces__msg__RobotStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

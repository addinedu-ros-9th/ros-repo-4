// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from robot_central_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/msg/robot_status.hpp"


#ifndef ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_HPP_
#define ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'current_pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"
// Member 'last_update'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__robot_central_interfaces__msg__RobotStatus __attribute__((deprecated))
#else
# define DEPRECATED__robot_central_interfaces__msg__RobotStatus __declspec(deprecated)
#endif

namespace robot_central_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotStatus_
{
  using Type = RobotStatus_<ContainerAllocator>;

  explicit RobotStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : current_pose(_init),
    last_update(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = "";
      this->status = "";
      this->current_task = "";
    }
  }

  explicit RobotStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : robot_id(_alloc),
    status(_alloc),
    current_task(_alloc),
    current_pose(_alloc, _init),
    last_update(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = "";
      this->status = "";
      this->current_task = "";
    }
  }

  // field types and members
  using _robot_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _robot_id_type robot_id;
  using _status_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _status_type status;
  using _current_task_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _current_task_type current_task;
  using _current_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _current_pose_type current_pose;
  using _last_update_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _last_update_type last_update;

  // setters for named parameter idiom
  Type & set__robot_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->robot_id = _arg;
    return *this;
  }
  Type & set__status(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__current_task(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->current_task = _arg;
    return *this;
  }
  Type & set__current_pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->current_pose = _arg;
    return *this;
  }
  Type & set__last_update(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->last_update = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    robot_central_interfaces::msg::RobotStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const robot_central_interfaces::msg::RobotStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::msg::RobotStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::msg::RobotStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__robot_central_interfaces__msg__RobotStatus
    std::shared_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__robot_central_interfaces__msg__RobotStatus
    std::shared_ptr<robot_central_interfaces::msg::RobotStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotStatus_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->status != other.status) {
      return false;
    }
    if (this->current_task != other.current_task) {
      return false;
    }
    if (this->current_pose != other.current_pose) {
      return false;
    }
    if (this->last_update != other.last_update) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotStatus_

// alias to use template instance with default allocator
using RobotStatus =
  robot_central_interfaces::msg::RobotStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace robot_central_interfaces

#endif  // ROBOT_CENTRAL_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_HPP_

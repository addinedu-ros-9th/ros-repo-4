// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from robot_central_interfaces:srv/NavigateRobot.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "robot_central_interfaces/srv/navigate_robot.hpp"


#ifndef ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__STRUCT_HPP_
#define ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Request __attribute__((deprecated))
#else
# define DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Request __declspec(deprecated)
#endif

namespace robot_central_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct NavigateRobot_Request_
{
  using Type = NavigateRobot_Request_<ContainerAllocator>;

  explicit NavigateRobot_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = "";
    }
  }

  explicit NavigateRobot_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : robot_id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->robot_id = "";
    }
  }

  // field types and members
  using _robot_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _robot_id_type robot_id;
  using _waypoints_type =
    std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>>;
  _waypoints_type waypoints;

  // setters for named parameter idiom
  Type & set__robot_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->robot_id = _arg;
    return *this;
  }
  Type & set__waypoints(
    const std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>> & _arg)
  {
    this->waypoints = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Request
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Request
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateRobot_Request_ & other) const
  {
    if (this->robot_id != other.robot_id) {
      return false;
    }
    if (this->waypoints != other.waypoints) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateRobot_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateRobot_Request_

// alias to use template instance with default allocator
using NavigateRobot_Request =
  robot_central_interfaces::srv::NavigateRobot_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace robot_central_interfaces


#ifndef _WIN32
# define DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Response __attribute__((deprecated))
#else
# define DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Response __declspec(deprecated)
#endif

namespace robot_central_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct NavigateRobot_Response_
{
  using Type = NavigateRobot_Response_<ContainerAllocator>;

  explicit NavigateRobot_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  explicit NavigateRobot_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Response
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Response
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateRobot_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateRobot_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateRobot_Response_

// alias to use template instance with default allocator
using NavigateRobot_Response =
  robot_central_interfaces::srv::NavigateRobot_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace robot_central_interfaces


// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Event __attribute__((deprecated))
#else
# define DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Event __declspec(deprecated)
#endif

namespace robot_central_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct NavigateRobot_Event_
{
  using Type = NavigateRobot_Event_<ContainerAllocator>;

  explicit NavigateRobot_Event_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_init)
  {
    (void)_init;
  }

  explicit NavigateRobot_Event_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _info_type =
    service_msgs::msg::ServiceEventInfo_<ContainerAllocator>;
  _info_type info;
  using _request_type =
    rosidl_runtime_cpp::BoundedVector<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>>>;
  _request_type request;
  using _response_type =
    rosidl_runtime_cpp::BoundedVector<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>>>;
  _response_type response;

  // setters for named parameter idiom
  Type & set__info(
    const service_msgs::msg::ServiceEventInfo_<ContainerAllocator> & _arg)
  {
    this->info = _arg;
    return *this;
  }
  Type & set__request(
    const rosidl_runtime_cpp::BoundedVector<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<robot_central_interfaces::srv::NavigateRobot_Request_<ContainerAllocator>>> & _arg)
  {
    this->request = _arg;
    return *this;
  }
  Type & set__response(
    const rosidl_runtime_cpp::BoundedVector<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<robot_central_interfaces::srv::NavigateRobot_Response_<ContainerAllocator>>> & _arg)
  {
    this->response = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator> *;
  using ConstRawPtr =
    const robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Event
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__robot_central_interfaces__srv__NavigateRobot_Event
    std::shared_ptr<robot_central_interfaces::srv::NavigateRobot_Event_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateRobot_Event_ & other) const
  {
    if (this->info != other.info) {
      return false;
    }
    if (this->request != other.request) {
      return false;
    }
    if (this->response != other.response) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateRobot_Event_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateRobot_Event_

// alias to use template instance with default allocator
using NavigateRobot_Event =
  robot_central_interfaces::srv::NavigateRobot_Event_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace robot_central_interfaces

namespace robot_central_interfaces
{

namespace srv
{

struct NavigateRobot
{
  using Request = robot_central_interfaces::srv::NavigateRobot_Request;
  using Response = robot_central_interfaces::srv::NavigateRobot_Response;
  using Event = robot_central_interfaces::srv::NavigateRobot_Event;
};

}  // namespace srv

}  // namespace robot_central_interfaces

#endif  // ROBOT_CENTRAL_INTERFACES__SRV__DETAIL__NAVIGATE_ROBOT__STRUCT_HPP_

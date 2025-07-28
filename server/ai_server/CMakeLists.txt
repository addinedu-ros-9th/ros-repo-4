cmake_minimum_required(VERSION 3.8)
project(ai_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(robot_interfaces REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(include)

# Add executable
add_executable(ai_server_node
  src/main.cpp
  src/ai_server.cpp
  src/webcam_streamer.cpp
  src/udp_image_sender.cpp
)

ament_target_dependencies(ai_server_node 
  rclcpp 
  sensor_msgs 
  cv_bridge 
  image_transport
  robot_interfaces
)

target_link_libraries(ai_server_node ${OpenCV_LIBS})

# Install targets
install(TARGETS
  ai_server_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

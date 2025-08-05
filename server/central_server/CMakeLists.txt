cmake_minimum_required(VERSION 3.8)
project(central_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(yaml-cpp REQUIRED)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robot_interfaces REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(domain_bridge REQUIRED)

# HTTP 서버를 위한 라이브러리들
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# OpenSSL 라이브러리 찾기
find_package(OpenSSL REQUIRED)

# 실행 파일 추가
add_executable(central_server_node
  src/main.cpp
  src/central_server.cpp
  src/database_manager.cpp
  src/http_server.cpp
  src/robot_navigation_manager.cpp
)

# 의존성 연결
ament_target_dependencies(central_server_node
  rclcpp
  robot_interfaces
  nav_msgs
  domain_bridge
)

# MySQL 라이브러리 직접 링크
target_link_libraries(central_server_node 
  mysqlcppconn 
  mysqlclient 
  pthread
  ${JSONCPP_LIBRARIES}
  yaml-cpp
  OpenSSL::SSL
  OpenSSL::Crypto
)

# 헤더 파일 포함
target_include_directories(central_server_node PRIVATE 
  include
  ${JSONCPP_INCLUDE_DIRS}
)

# 설치
install(TARGETS
  central_server_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package() 
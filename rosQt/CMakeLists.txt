cmake_minimum_required(VERSION 3.16)
project(BasicQtApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt 찾기
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(/usr/lib/python3/dist-packages/numpy/core/include)

# ROS2 패키지들 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(robot_interfaces REQUIRED)


# Qt 자동 생성 활성화
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 소스 파일들
set(SOURCES
    main.cpp
    login.cpp
    layout.cpp
    dashboard.cpp
    log.cpp
    status.cpp
    status2.cpp
    status3.cpp
    style.cpp
    map.cpp
    udp_image_receiver.cpp
    control_popup1.cpp
    control_popup2.cpp
)

# UI 파일들
set(UI_FILES
    login.ui
    layout.ui
    dashboard.ui
    log.ui
    status.ui
    status2.ui
    status3.ui
    map.ui
    control_popup1.ui
    control_popup2.ui
)

# 실행 파일 생성
add_executable(BasicQtApp ${SOURCES} ${UI_FILES})

# 헤더 파일 경로
target_include_directories(BasicQtApp PRIVATE ${OpenCV_INCLUDE_DIRS})

# 라이브러리 링크
target_link_libraries(BasicQtApp
    ${Python3_LIBRARIES}
    Qt6::Core
    Qt6::Widgets
    ${OpenCV_LIBS}
    rclcpp::rclcpp
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${robot_interfaces_TARGETS}
)

# ROS2 의존성 설정
ament_target_dependencies(BasicQtApp
    rclcpp
    geometry_msgs
    sensor_msgs
    robot_interfaces
)
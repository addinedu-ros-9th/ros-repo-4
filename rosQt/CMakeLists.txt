cmake_minimum_required(VERSION 3.16)
project(BasicQtApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets UiTools)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

qt6_standard_project_setup()

# 소스 파일
set(SOURCES
    main.cpp
    login.cpp      
    layout.cpp     
    dashboard.cpp
    log.cpp 
    status.cpp
    style.cpp
    map.cpp
)

# 헤더 파일
set(HEADERS
    login.h
    layout.h      
    dashboard.h 
    log.h
    status.h
    style.h
    map.h
)

qt6_add_executable(BasicQtApp ${SOURCES} ${HEADERS})

target_link_libraries(BasicQtApp PRIVATE 
    Qt6::Core 
    Qt6::Widgets 
    Qt6::UiTools
    rclcpp::rclcpp
    ${geometry_msgs_TARGETS}
)
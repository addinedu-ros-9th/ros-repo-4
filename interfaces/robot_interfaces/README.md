# Robot Interfaces (로봇 공통 인터페이스)

병원 로봇 관리 시스템의 모든 컴포넌트가 공유하는 ROS2 인터페이스들입니다.

## 📋 포함된 인터페이스

### 📨 Messages (msg/)

#### 1. `RobotPose.msg`
```
int32 robot_id
float32 x
float32 y
```
- 로봇의 간단한 위치 정보

#### 2. `RobotStatus.msg`  
```
int32 robot_id
string status
geometry_msgs/PoseWithCovarianceStamped amcl_pose
string current_station
string destination_station
int32 battery_percent
string network_status
string source
```
- 관리자 GUI용 로봇 상태 (상세 정보)

#### 3. `RobotFeedback.msg`
```
int32 robot_id
string status  # "moving", "standby" (로봇이 스스로 판단하는 상태만)
geometry_msgs/PoseWithCovarianceStamped current_pose
int32 battery_percent
string network_status  # "connected", "disconnected", "weak"
float32 current_speed  # 현재 이동 속도 (m/s)
string[] detected_objects  # 감지된 객체들 ["person", "obstacle", "chair", ...]
string error_message  # 오류 발생시 메시지 (정상시 빈 문자열)
```
- 로봇 컨트롤러 → 중앙서버 피드백
- 로봇이 스스로 판단하는 정보만 포함

### 🔧 Services (srv/)

#### 1. `LoginAdmin.srv`
```
string id
string password
---
bool success
string name
string email
string hospital_name
```
- 관리자 로그인 인증

#### 2. `ChangeRobotStatus.srv`
```
int32 robot_id
string new_status  # "idle", "assigned", "occupied" (User GUI에서 받은 상태만)
---
bool success
string message
```
- 중앙서버 → 로봇 컨트롤러 상태 변경 명령
- User GUI에서 받은 상태만 전달

### ⚡ Actions (action/)

#### 1. `NavigateToGoal.action`
```
# Goal
int32 robot_id
geometry_msgs/PoseStamped target_pose
string target_station_name
bool use_station_coordinates
---
# Result
bool success
string message
geometry_msgs/PoseStamped final_pose
float32 total_distance
float32 total_time
---
# Feedback
float32 distance_remaining
float32 estimated_time_remaining
geometry_msgs/PoseStamped current_pose
string current_status
float32 progress_percentage
```
- 로봇 네비게이션 (장시간 실행되는 작업)

## 🏗️ 사용하는 패키지들

- `central_server` - 중앙서버
- `robot_controller` - 로봇 컨트롤러  
- `user_gui` - 사용자 GUI
- `admin_gui` - 관리자 GUI

## 🚀 빌드 및 사용

### 빌드
```bash
cd /home/ckim/ros-repo-4
colcon build --packages-select robot_interfaces
```

### 다른 패키지에서 사용
**package.xml에 추가:**
```xml
<depend>robot_interfaces</depend>
```

**CMakeLists.txt에 추가:**
```cmake
find_package(robot_interfaces REQUIRED)
ament_target_dependencies(your_target
  robot_interfaces
)
```

**C++ 코드에서 사용:**
```cpp
#include "robot_interfaces/msg/robot_status.hpp"
#include "robot_interfaces/srv/login_admin.hpp"
#include "robot_interfaces/action/navigate_to_goal.hpp"

// 메시지 생성
auto msg = robot_interfaces::msg::RobotStatus();
msg.robot_id = 1;
msg.status = "moving";
```

## 🔄 로봇 상태 흐름

```
idle → assigned (사용자 터치) → source="user_gui"
assigned → occupied (인증완료) → source="user_gui"  
occupied → moving (실제이동시작) → source="robot_controller"
moving → standby (목적지도착) → source="robot_controller"
standby → idle (복귀완료) → source="robot_controller"
```

### 상태별 설명:
- **idle**: 대기 상태 (User GUI → Central Server → Robot)
- **assigned**: 사용자 할당됨 (User GUI → Central Server → Robot)
- **occupied**: 인증 완료 (User GUI → Central Server → Robot)
- **moving**: 이동 중 (Robot → Central Server - 자동 판단)
- **standby**: 목적지 도착 (Robot → Central Server - 자동 판단)

### 통신 방향:
- **User GUI → Central → Robot**: `idle`, `assigned`, `occupied`
- **Robot → Central**: `moving`, `standby` (+ 센서 데이터)

## 🌐 인터페이스 맵

```
User GUI ←→ Central Server ←→ Robot Controller
    ↓              ↓              ↓
robot_interfaces 공유 (일관성 보장)
```

이렇게 하면 모든 컴포넌트가 동일한 메시지 구조를 사용해서 호환성 문제가 없습니다! 🎯 
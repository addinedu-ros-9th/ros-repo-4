# AI Server with Webcam Integration

이 프로젝트는 웹캠을 사용하여 영상을 캡처하고 Central Server로 전송하는 AI Server를 구현합니다.

## 시스템 구조

```
┌─────────────────┐    이미지 스트림     ┌─────────────────┐
│   AI Server     │ ──────────────────> │ Central Server   │
│                 │                     │                 │
│ - 웹캠 캡처       │    상태 메시지      │ - 이미지 수신   │
│ - 이미지 처리      │ ──────────────────> │ - 상태 모니터링 │
│ - ROS2 발행      │                     │ - HTTP API      │
└─────────────────┘                     └─────────────────┘
```

## 주요 기능

### AI Server
- 웹캠을 통한 실시간 비디오 캡처 (기본 해상도: 640x480, 30 FPS)
- OpenCV를 사용한 이미지 처리
- ROS2를 통한 이미지 스트리밍 (`webcam/image_raw` 토픽)
- 상태 정보 전송 (`robot_status` 토픽)
- 멀티스레드 구조로 안정적인 스트리밍

### Central Server
- AI Server로부터 이미지 스트림 수신
- 로봇 상태 모니터링
- HTTP API 서버 (포트 8080)
- MySQL 데이터베이스 연동
- 상태 변경 서비스 제공

## 필요한 의존성

### 시스템 의존성
```bash
# OpenCV 설치
sudo apt update
sudo apt install libopencv-dev python3-opencv

# ROS2 의존성
sudo apt install ros-humble-cv-bridge ros-humble-image-transport

# MySQL 개발 라이브러리 (Central Server용)
sudo apt install libmysqlcppconn-dev libmysqlclient-dev
```

### ROS2 패키지 의존성
- `rclcpp`
- `sensor_msgs`
- `cv_bridge`
- `image_transport`
- `robot_interfaces` (커스텀 인터페이스)

## 빌드 및 실행

### 자동 빌드
```bash
# 전체 자동 빌드
./build_and_run.sh
```

### 수동 빌드
```bash
# 워크스페이스로 이동
cd /home/wonho/ros-repo-4

# 인터페이스 먼저 빌드
colcon build --packages-select robot_interfaces

# AI Server 빌드
colcon build --packages-select ai_server

# Central Server 빌드  
colcon build --packages-select central_server

# 환경 설정
source install/setup.bash
```

### 실행

#### 터미널 1: Central Server 실행
```bash
source install/setup.bash
ros2 run central_server central_server_node
```

#### 터미널 2: AI Server 실행
```bash
source install/setup.bash
ros2 run ai_server ai_server_node
```

## 토픽 및 서비스

### 발행되는 토픽
- `webcam/image_raw` (sensor_msgs/Image): 웹캠 이미지 스트림
- `robot_status` (robot_interfaces/RobotStatus): AI Server 상태 정보

### 제공되는 서비스
- `change_robot_status` (robot_interfaces/ChangeRobotStatus): 상태 변경 서비스

## 모니터링 및 디버깅

### 토픽 확인
```bash
# 토픽 리스트 확인
ros2 topic list

# 이미지 토픽 확인
ros2 topic echo /webcam/image_raw

# 상태 토픽 확인  
ros2 topic echo /robot_status
```

### 이미지 시각화
```bash
# rqt 이미지 뷰어로 웹캠 영상 확인
ros2 run rqt_image_view rqt_image_view
```

### 로그 확인
```bash
# 실시간 로그 확인
ros2 topic echo /rosout
```

## 문제 해결

### 웹캠 관련 문제
1. **웹캠이 인식되지 않는 경우**
   ```bash
   # 웹캠 디바이스 확인
   ls /dev/video*
   
   # 권한 확인
   sudo chmod 666 /dev/video0
   ```

2. **다른 카메라 사용하는 경우**
   - `WebcamStreamer` 생성자의 `camera_id` 파라미터 수정

### 빌드 오류
1. **OpenCV 관련 오류**
   ```bash
   sudo apt install libopencv-dev python3-opencv
   ```

2. **의존성 오류**
   ```bash
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   ```

## 개발 참고사항

### 코드 구조
- `ai_server/src/main.cpp`: AI Server 메인 프로그램
- `ai_server/src/ai_server.cpp`: AI Server 핵심 로직
- `ai_server/src/webcam_streamer.cpp`: 웹캠 캡처 및 스트리밍
- `central_server/src/central_server.cpp`: Central Server 수정된 버전

### 확장 가능한 기능
- AI 기반 이미지 분석 (객체 감지, 얼굴 인식 등)
- 실시간 영상 압축 및 전송
- 웹 기반 영상 스트리밍
- 다중 카메라 지원

## 라이센스
Apache-2.0

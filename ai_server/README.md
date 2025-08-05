# AI Server

AI Server는 ROS2 기반의 카메라 스트리밍 및 AI 처리 서버입니다.

## 🚀 주요 기능

### 1. UDP 스트리밍
- 전면 카메라 이미지를 UDP로 실시간 스트리밍
- JPEG 압축을 통한 효율적인 데이터 전송
- 설정 가능한 압축 품질 및 패킷 크기

### 2. HTTP 카메라 전환 (NEW!)
- HTTP 요청을 통한 카메라 전환 기능
- 전면/후면 카메라 간 실시간 전환
- RESTful API 형태의 간단한 인터페이스

### 3. 이중 카메라 지원
- 전면 카메라: `/dev/video0`
- 후면 카메라: `/dev/video2`
- 동시 초기화 및 관리

## 📡 HTTP API

### 카메라 전환 (IF-01)
```
POST /change/camera
```

#### 요청
```json
{
  "robot_id": 3,
  "camera": "front"  // "front" 또는 "back"
}
```

#### 응답
```json
{
  "status": "success"
}
```

### 사용 예시
```bash
# 전면 카메라로 전환
curl -X POST http://localhost:7777/change/camera \
  -H "Content-Type: application/json" \
  -d '{"robot_id":3, "camera":"front"}'

# 후면 카메라로 전환
curl -X POST http://localhost:7777/change/camera \
  -H "Content-Type: application/json" \
  -d '{"robot_id":3, "camera":"back"}'
```

## 📡 UDP 프로토콜 (IF-01)

### 실시간 카메라 이미지 전송
- **IP/Port**: `192.168.0.74:7777`
- **전송 주기**: 30 FPS
- **프로토콜 구조**:

```
Header (10 bytes):
├── 1 byte: Start (0xAB)
├── 1 byte: 카메라 타입 (0x00=front, 0x01=back)
├── 4 bytes: 시퀀스 번호 (little-endian)
└── 4 bytes: 타임스탬프 (milliseconds, little-endian)

Payload:
└── JPEG 이미지 데이터
```

### 프로토콜 예시
```
Header: AB 00 01 00 00 00 64 00 00 00  (전면 카메라, 시퀀스 1, 타임스탬프 100ms)
Payload: [JPEG 이미지 바이너리 데이터]
```

## 🧪 테스트

### 웹 브라우저 테스트
1. `test_camera_switch.html` 파일을 웹 브라우저에서 열기
2. "전면 카메라" 또는 "후면 카메라" 버튼 클릭
3. HTTP API를 통한 카메라 전환 확인
4. UDP 스트리밍이 자동으로 변경됨

### 터미널 테스트
```bash
# 전면 카메라로 전환
curl -X POST http://localhost:7777/change/camera \
  -H "Content-Type: application/json" \
  -d '{"robot_id":3, "camera":"front"}'

# 후면 카메라로 전환
curl -X POST http://localhost:7777/change/camera \
  -H "Content-Type: application/json" \
  -d '{"robot_id":3, "camera":"back"}'

# UDP 패킷 수신 테스트 (다른 기기에서)
nc -ul 7777  # UDP 패킷 헥스 덤프 확인
```

## ⚙️ 설정

### config.yaml
```yaml
ai_server:
  ip: "192.168.0.27"
  port: 7777  # HTTP 서버 포트
  udp_target:
    ip: "192.168.0.74"
    port: 7777  # UDP 이미지 수신 포트
  max_packet_size: 60000
```

## 🔧 빌드 및 실행

### 빌드
```bash
cd ai_server
colcon build --packages-select ai_server
```

### 실행
```bash
source install/setup.bash
ros2 run ai_server ai_server
```

## 📊 성능 정보

- **UDP 스트리밍**: 30 FPS (전면 카메라)
- **HTTP 응답**: 요청 시 즉시 응답
- **카메라 전환**: 실시간 전환 (지연 < 100ms)
- **이미지 품질**: JPEG 80% 품질

## 🐛 문제 해결

### 카메라 인식 문제
```bash
# 카메라 장치 확인
ls -la /dev/video*

# 카메라 정보 확인
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### HTTP 서버 포트 충돌
- `config.yaml`에서 `port` 변경
- 또는 기존 프로세스 종료: `sudo lsof -ti:7777 | xargs kill`

## 📝 로그 확인

```bash
# ROS2 로그 확인
ros2 run ai_server ai_server --ros-args --log-level debug

# 카메라 전환 로그
ros2 run ai_server ai_server --ros-args --log-level info
```

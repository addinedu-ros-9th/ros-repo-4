# AI Server UDP 웹캠 이미지 전송 가이드

AI Server에서 UDP를 통해 웹캠 이미지를 실시간으로 전송하는 기능이 추가되었습니다.

## 🚀 기능 설명

### 새로 추가된 기능
- **UDP 이미지 전송**: 웹캠에서 캡처한 이미지를 UDP로 실시간 전송
- **이미지 압축**: JPEG 압축을 통한 효율적인 데이터 전송
- **패킷 분할**: 큰 이미지를 여러 UDP 패킷으로 분할 전송
- **설정 가능**: 압축 품질, 패킷 크기, 대상 주소 설정 가능

### 동작 방식
```
웹캠 → OpenCV 캡처 → JPEG 압축 → UDP 패킷 분할 → 네트워크 전송
```

## 📋 사용 방법

### 1. AI Server 실행
```bash
# 환경 설정
source install/setup.bash

# AI Server 실행 (기본: localhost:8888로 전송)
ros2 run ai_server ai_server_node
```

### 2. UDP 이미지 수신 테스트
다른 터미널에서 수신 클라이언트 실행:
```bash
cd /home/wonho/ros-repo-4/ai_server

# Python 수신 클라이언트 실행
python3 udp_image_receiver.py
```

### 3. 종료
- AI Server: `Ctrl+C`
- 수신 클라이언트: `q` 키 또는 `Ctrl+C`

## ⚙️ 설정 옵션

### AI Server 설정 (ai_server.cpp에서 수정 가능)
```cpp
// UDP 전송 대상 설정
udp_sender_ = std::make_unique<UdpImageSender>("192.168.1.100", 9999);

// 압축 품질 설정 (1-100, 기본값: 70)
udp_sender_->setCompressionQuality(80);

// 최대 패킷 크기 설정 (기본값: 60000 bytes)
udp_sender_->setMaxPacketSize(50000);
```

### 네트워크 설정
- **기본 IP**: `127.0.0.1` (localhost)
- **기본 포트**: `8888`
- **압축 품질**: `70%` (JPEG)
- **최대 패킷 크기**: `60KB`

## 🔧 문제 해결

### 1. 웹캠이 인식되지 않는 경우
```bash
# 웹캠 디바이스 확인
ls /dev/video*

# 권한 설정
sudo chmod 666 /dev/video0
```

### 2. UDP 포트가 사용 중인 경우
```bash
# 포트 사용 확인
sudo netstat -ulnp | grep 8888

# 프로세스 종료
sudo fuser -k 8888/udp
```

### 3. 방화벽 문제
```bash
# 포트 열기 (Ubuntu)
sudo ufw allow 8888/udp
```

### 4. 이미지가 수신되지 않는 경우
- AI Server 로그 확인
- 네트워크 연결 상태 확인
- IP 주소와 포트 번호 확인

## 📊 성능 정보

### 전송 성능
- **해상도**: 640x480 (기본)
- **프레임레이트**: ~30 FPS
- **압축 후 크기**: 약 15-30KB/프레임 (품질에 따라)
- **네트워크 대역폭**: 약 4-8 Mbps

### 지연시간
- **로컬 네트워크**: < 10ms
- **인터넷**: 네트워크 상황에 따라 가변

## 🔧 커스터마이징

### 다른 해상도 사용
`webcam_streamer.cpp`에서 해상도 변경:
```cpp
cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
```

### 다른 압축 포맷 사용
`udp_image_sender.cpp`에서 압축 포맷 변경:
```cpp
// PNG 압축 (무손실, 더 큰 파일)
cv::imencode(".png", image, compressed_data);

// WebP 압축 (더 좋은 압축률)
cv::imencode(".webp", image, compressed_data);
```

### 암호화 추가
보안이 필요한 경우 SSL/TLS나 자체 암호화 구현 가능

## 🌐 네트워크 사용 예시

### 원격 서버로 전송
```cpp
// 다른 컴퓨터로 전송
udp_sender_ = std::make_unique<UdpImageSender>("192.168.1.100", 8888);
```

### 여러 클라이언트에 전송
여러 UdpImageSender 인스턴스를 생성해서 다중 전송 가능

## 📝 로그 확인

### AI Server 로그
```bash
# ROS2 로그 확인
ros2 topic echo /rosout

# 터미널에서 직접 확인
ros2 run ai_server ai_server_node
```

### 예상 로그 출력
```
[INFO] [ai_server]: UDP Image Sender 초기화 완료! (타겟: 127.0.0.1:8888)
[INFO] [ai_server]: 이미지 압축 완료 - 원본: 640x480, 압축 크기: 18432 bytes
[INFO] [ai_server]: UDP 이미지 전송 - 프레임 #60
```

## 🚀 다음 개발 단계

### 가능한 확장 기능
1. **H.264 하드웨어 인코딩** - 더 효율적인 압축
2. **적응적 비트레이트** - 네트워크 상황에 따른 품질 조절
3. **다중 카메라 지원** - 여러 웹캠 동시 전송
4. **웹 스트리밍** - HTTP/WebRTC를 통한 브라우저 재생
5. **AI 분석 결과 오버레이** - 객체 감지 결과 표시

이제 AI Server가 UDP를 통해 웹캠 이미지를 실시간으로 전송할 수 있습니다! 🎉

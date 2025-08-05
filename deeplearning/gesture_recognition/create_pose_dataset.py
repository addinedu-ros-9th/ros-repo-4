"""
YOLO Pose 기반 제스처 데이터셋 생성 도구

주요 개선사항:
1. 영상 저장 속도 문제 해결:
   - 정확한 프레임레이트 제어를 위한 FrameRateController 클래스 추가
   - H.264, XVID, MJPG 코덱 자동 선택으로 안정적인 VideoWriter 생성
   - 프레임 간격 정확도 모니터링 및 통계 제공

2. 카운트다운 렉 문제 해결:
   - cv2.waitKey(1000) 대신 time.sleep(0.05)와 부드러운 업데이트 방식 사용
   - 50ms 간격으로 화면 업데이트하여 부드러운 카운트다운 구현
   - 실시간 남은 시간 표시

3. 동적 해상도 지원:
   - 카메라 최대 해상도 자동 감지
   - 해상도 프리셋 제공 (low/medium/high/full)
   - 실제 테스트 환경에 맞는 풀 해상도 대응
   - 해상도별 예상 파일 크기 계산

4. 추가 기능:
   - 실시간 FPS 모니터링
   - 프레임 정확도 및 간격 정확도 분석
   - 대안적인 녹화 방법 제공 (record_action_alternative)

해상도 설정:
- RESOLUTION_MODE 변수로 해상도 선택
- 'low': 640x480 (빠른 처리, 작은 파일)
- 'medium': 1280x720 (HD, 균형)
- 'high': 1920x1080 (Full HD, 고품질)
- 'full': 카메라 최대 해상도 (실제 테스트용)

사용법:
- 기본 녹화: record_action() 함수 사용
- 대안 녹화: record_action_alternative() 함수 사용
- 메인 함수에서 원하는 방법 선택 가능
"""

import cv2
import numpy as np
import time
import os
import threading

# 제스처 정의
actions = ['come', 'normal']
secs_for_action = 3  # 3초씩 촬영
num_samples = 50  # 각 액션당 50개 샘플

# 해상도 설정 옵션
RESOLUTION_PRESETS = {
    'low': (640, 480),      # 저해상도 (빠른 처리)
    'medium': (1280, 720),  # HD 해상도
    'high': (1920, 1080),   # Full HD 해상도
    'full': None            # 카메라 최대 해상도 사용
}

# 사용할 해상도 선택 (변경 가능)
RESOLUTION_MODE = 'full'  # 'low', 'medium', 'high', 'full' 중 선택

# 카메라 설정
cap = cv2.VideoCapture('/dev/video0')

def setup_camera_resolution(cap, mode='full'):
    """카메라 해상도 설정"""
    if mode in RESOLUTION_PRESETS:
        if RESOLUTION_PRESETS[mode] is None:  # full 모드
            # 카메라가 지원하는 최대 해상도 확인
            max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 일반적인 최대 해상도들을 시도
            test_resolutions = [
                (3840, 2160),  # 4K
                (2560, 1440),  # 2K
                (1920, 1080),  # Full HD
                (1280, 720),   # HD
                (640, 480)     # VGA (fallback)
            ]
            
            width, height = 640, 480  # 기본값
            
            for test_w, test_h in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, test_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_h)
                
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if actual_w == test_w and actual_h == test_h:
                    width, height = actual_w, actual_h
                    print(f"✅ 최대 해상도 설정 성공: {width}x{height}")
                    break
            
            if width == 640 and height == 480:
                print("⚠️ 최대 해상도 감지 실패, 640x480 사용")
                
        else:
            width, height = RESOLUTION_PRESETS[mode]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"✅ 해상도 설정: {width}x{height} ({mode} 모드)")
    else:
        print(f"❌ 알 수 없는 해상도 모드: {mode}")
        width, height = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 실제 설정된 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 실제 카메라 해상도: {actual_width}x{actual_height}")
    
    return actual_width, actual_height

# 카메라 해상도 설정
camera_width, camera_height = setup_camera_resolution(cap, RESOLUTION_MODE)

# 실제 카메라 프레임레이트 확인
actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps <= 0:
    actual_fps = 30.0  # 기본값
print(f"📹 카메라 프레임레이트: {actual_fps} FPS")

# 데이터셋 디렉토리 생성 (액션별 분리)
created_time = int(time.time())
dataset_dir = './deeplearning/gesture_recognition/pose_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# 액션별 폴더 생성
for action in actions:
    action_dir = os.path.join(dataset_dir, action)
    os.makedirs(action_dir, exist_ok=True)
    print(f"📁 폴더 생성: {action_dir}")

class FrameRateController:
    """정확한 프레임레이트 제어를 위한 클래스"""
    
    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0
        self.frame_count = 0
    
    def wait_for_next_frame(self):
        """다음 프레임까지 대기"""
        current_time = time.time()
        
        if self.last_frame_time == 0:
            self.last_frame_time = current_time
            return True
        
        elapsed = current_time - self.last_frame_time
        if elapsed >= self.frame_interval:
            self.last_frame_time = current_time
            self.frame_count += 1
            return True
        
        # 정확한 타이밍을 위해 짧게 대기
        time.sleep(max(0, self.frame_interval - elapsed - 0.001))
        return False
    
    def get_stats(self):
        """현재 통계 반환"""
        if self.frame_count > 0:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time + (self.frame_count * self.frame_interval)
            actual_fps = self.frame_count / elapsed
            return {
                'frame_count': self.frame_count,
                'elapsed_time': elapsed,
                'actual_fps': actual_fps,
                'target_fps': self.target_fps
            }
        return None

def create_video_writer(filename, fps, width=None, height=None):
    """안정적인 VideoWriter 생성 - 동적 해상도 지원"""
    # 기본값으로 카메라 해상도 사용
    if width is None or height is None:
        width, height = camera_width, camera_height
    
    print(f"🎬 VideoWriter 생성 중... ({width}x{height}, {fps:.1f} FPS)")
    
    # 코덱 우선순위: H264 > XVID > MJPG
    codecs = [
        ('H264', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    for codec, ext in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_path = filename.replace('.mp4', ext).replace('.avi', ext)
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if out.isOpened():
                print(f"✅ {codec} 코덱으로 VideoWriter 생성 성공")
                return out, video_path
            else:
                out.release()
        except Exception as e:
            print(f"⚠️ {codec} 코덱 실패: {e}")
            continue
    
    # 모든 코덱이 실패한 경우
    print("❌ 모든 코덱 실패, 기본 설정으로 시도")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(video_path, fourcc, fps, (width, height)), video_path

def record_action(action_name, sample_idx):
    """특정 액션을 녹화하고 영상 저장"""
    print(f"\n🎬 {action_name.upper()} 액션 녹화 시작 - 샘플 {sample_idx+1}/{num_samples}")
    
    frames = []
    frame_timestamps = []  # 프레임 타임스탬프 저장
    
    # 프레임레이트 컨트롤러 초기화
    fps_controller = FrameRateController(actual_fps)
    
    # 3초 카운트다운 - 렉 없는 방식
    print("카운트다운 시작...")
    countdown_start = time.time()
    
    for i in range(3, 0, -1):
        # 부드러운 카운트다운을 위해 0.1초씩 체크
        target_time = countdown_start + (3 - i)
        
        while time.time() < target_time:
            ret, img = cap.read()
            if ret:
                img = cv2.flip(img, 1)
                cv2.putText(img, f'Get ready for {action_name.upper()}: {i}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                cv2.putText(img, 'Recording will start soon...', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Recording', img)
            
            # 짧은 대기로 부드러운 업데이트
            if cv2.waitKey(50) & 0xFF == ord('q'):
                return False
            
            time.sleep(0.05)  # 50ms 대기로 부드러운 카운트다운
    
    # 녹화 시작 - 개선된 프레임레이트 제어
    start_time = time.time()
    
    while time.time() - start_time < secs_for_action:
        # 프레임레이트 컨트롤러로 정확한 타이밍 제어
        if fps_controller.wait_for_next_frame():
            ret, img = cap.read()
            if not ret:
                continue
            
            img = cv2.flip(img, 1)
            original_img = img.copy()
            
            # 프레임 타임스탬프 저장
            frame_timestamps.append(time.time() - start_time)
            
            # 녹화 정보 표시
            elapsed = time.time() - start_time
            remaining = secs_for_action - elapsed
            
            # FPS 컨트롤러 통계 가져오기
            stats = fps_controller.get_stats()
            current_fps = stats['actual_fps'] if stats else 0
            
            cv2.putText(img, f'Recording {action_name.upper()}: {elapsed:.1f}s / {secs_for_action}s', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f'Remaining: {remaining:.1f}s', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, f'Sample: {sample_idx+1}/{num_samples}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(img, f'Target FPS: {actual_fps:.1f}', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f'Actual FPS: {current_fps:.1f}', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f'Frames: {len(frames)}', 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 프레임 저장
            frames.append(original_img)
            
            cv2.imshow('Recording', img)
        
        # 키 입력 처리
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 영상 저장 - 개선된 VideoWriter 사용
    if len(frames) > 0:
        video_filename = f'video_{action_name}_{created_time}_{sample_idx:03d}.mp4'
        video_path = os.path.join(dataset_dir, action_name, video_filename)
        
        # 개선된 VideoWriter 생성
        out, final_video_path = create_video_writer(video_path, actual_fps)
        
        # 프레임 저장
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # 최종 통계 계산
        stats = fps_controller.get_stats()
        expected_frames = int(secs_for_action * actual_fps)
        
        print(f"✅ 영상 저장: {os.path.basename(final_video_path)}")
        print(f"   - 저장된 프레임: {len(frames)}")
        print(f"   - 예상 프레임: {expected_frames}")
        print(f"   - 목표 FPS: {actual_fps:.1f}")
        
        if stats:
            print(f"   - 실제 FPS: {stats['actual_fps']:.1f}")
            print(f"   - 녹화 시간: {stats['elapsed_time']:.2f}초")
            print(f"   - 프레임 정확도: {(len(frames) / expected_frames * 100):.1f}%")
        
        # 프레임 간격 분석
        if len(frame_timestamps) > 1:
            intervals = [frame_timestamps[i] - frame_timestamps[i-1] for i in range(1, len(frame_timestamps))]
            avg_interval = sum(intervals) / len(intervals)
            target_interval = 1.0 / actual_fps
            print(f"   - 평균 프레임 간격: {avg_interval:.3f}초 (목표: {target_interval:.3f}초)")
            print(f"   - 간격 정확도: {(target_interval / avg_interval * 100):.1f}%")
        
        return True
    else:
        print(f"❌ {action_name} 영상이 없습니다. 다시 시도해주세요.")
        return False

def record_action_alternative(action_name, sample_idx):
    """대안적인 녹화 방법 - 더 정확한 프레임레이트 제어"""
    print(f"\n🎬 {action_name.upper()} 액션 녹화 시작 (대안 방법) - 샘플 {sample_idx+1}/{num_samples}")
    
    frames = []
    frame_timestamps = []
    
    # 3초 카운트다운 - 렉 없는 방식
    print("카운트다운 시작 (대안 방법)...")
    countdown_start = time.time()
    
    for i in range(3, 0, -1):
        # 부드러운 카운트다운을 위해 0.1초씩 체크
        target_time = countdown_start + (3 - i)
        
        while time.time() < target_time:
            ret, img = cap.read()
            if ret:
                img = cv2.flip(img, 1)
                cv2.putText(img, f'Get ready for {action_name.upper()}: {i}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                cv2.putText(img, 'Alternative method - Recording will start soon...', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Recording', img)
            
            # 짧은 대기로 부드러운 업데이트
            if cv2.waitKey(50) & 0xFF == ord('q'):
                return False
            
            time.sleep(0.05)  # 50ms 대기로 부드러운 카운트다운
    
    # 녹화 시작 - 대안적 방법
    start_time = time.time()
    frame_interval = 1.0 / actual_fps
    next_frame_time = start_time
    
    while time.time() - start_time < secs_for_action:
        current_time = time.time()
        
        # 정확한 타이밍 제어
        if current_time >= next_frame_time:
            ret, img = cap.read()
            if not ret:
                continue
            
            img = cv2.flip(img, 1)
            original_img = img.copy()
            
            # 프레임 타임스탬프 저장
            frame_timestamps.append(current_time - start_time)
            
            # 녹화 정보 표시
            elapsed = current_time - start_time
            remaining = secs_for_action - elapsed
            current_fps = len(frames) / elapsed if elapsed > 0 else 0
            
            cv2.putText(img, f'Recording {action_name.upper()}: {elapsed:.1f}s / {secs_for_action}s', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f'Remaining: {remaining:.1f}s', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, f'Sample: {sample_idx+1}/{num_samples}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(img, f'Target FPS: {actual_fps:.1f}', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f'Current FPS: {current_fps:.1f}', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f'Frames: {len(frames)}', 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, 'ALT METHOD', 
                       (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 프레임 저장
            frames.append(original_img)
            
            cv2.imshow('Recording', img)
            
            # 다음 프레임 시간 계산 (누적 오차 방지)
            next_frame_time = start_time + (len(frames) * frame_interval)
        
        # 키 입력 처리
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 영상 저장
    if len(frames) > 0:
        video_filename = f'video_{action_name}_{created_time}_{sample_idx:03d}_alt.mp4'
        video_path = os.path.join(dataset_dir, action_name, video_filename)
        
        # 개선된 VideoWriter 생성
        out, final_video_path = create_video_writer(video_path, actual_fps)
        
        # 프레임 저장
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # 통계 계산
        expected_frames = int(secs_for_action * actual_fps)
        actual_fps_calculated = len(frames) / secs_for_action
        
        print(f"✅ 영상 저장 (대안 방법): {os.path.basename(final_video_path)}")
        print(f"   - 저장된 프레임: {len(frames)}")
        print(f"   - 예상 프레임: {expected_frames}")
        print(f"   - 목표 FPS: {actual_fps:.1f}")
        print(f"   - 실제 FPS: {actual_fps_calculated:.1f}")
        print(f"   - 프레임 정확도: {(len(frames) / expected_frames * 100):.1f}%")
        
        # 프레임 간격 분석
        if len(frame_timestamps) > 1:
            intervals = [frame_timestamps[i] - frame_timestamps[i-1] for i in range(1, len(frame_timestamps))]
            avg_interval = sum(intervals) / len(intervals)
            target_interval = 1.0 / actual_fps
            print(f"   - 평균 프레임 간격: {avg_interval:.3f}초 (목표: {target_interval:.3f}초)")
            print(f"   - 간격 정확도: {(target_interval / avg_interval * 100):.1f}%")
        
        return True
    else:
        print(f"❌ {action_name} 영상이 없습니다. 다시 시도해주세요.")
        return False

def main():
    """메인 녹화 프로세스"""
    print("🎬 YOLO Pose 기반 제스처 데이터셋 생성")
    print("=" * 50)
    print(f"📹 카메라 해상도: {camera_width}x{camera_height} ({RESOLUTION_MODE} 모드)")
    print(f"📹 카메라 프레임레이트: {actual_fps} FPS")
    print("COME: 한 손 들고 흔들기 (호출 제스처)")
    print("NORMAL: 지나가기, 팔짱끼기, 아무것도 안하기 등")
    print(f"📹 녹화 시간: {secs_for_action}초")
    print(f"📊 각 액션당: {num_samples}개 샘플")
    print(f"🎬 예상 프레임 수: {int(secs_for_action * actual_fps)} 프레임/영상")
    
    # 해상도별 예상 파일 크기 계산
    estimated_frame_size = camera_width * camera_height * 3  # RGB 기준
    estimated_video_size_mb = (estimated_frame_size * int(secs_for_action * actual_fps)) / (1024 * 1024)
    total_videos = num_samples * len(actions)
    total_size_gb = (estimated_video_size_mb * total_videos) / 1024
    
    print(f"💾 예상 영상 크기: {estimated_video_size_mb:.1f}MB/영상")
    print(f"💾 총 예상 용량: {total_size_gb:.1f}GB ({total_videos}개 영상)")
    
    # 해상도 변경 가이드
    print("\n⚙️ 해상도 변경하려면:")
    print("   파일 상단의 RESOLUTION_MODE 변수를 수정하세요")
    print("   - 'low': 640x480 (빠른 처리)")
    print("   - 'medium': 1280x720 (HD)")
    print("   - 'high': 1920x1080 (Full HD)")
    print("   - 'full': 카메라 최대 해상도")
    
    # 정확한 시간 계산
    total_recording_time = num_samples * len(actions) * secs_for_action  # 녹화 시간
    total_countdown_time = num_samples * len(actions) * 3  # 카운트다운 시간
    total_wait_time = (num_samples - 1) * len(actions) * 1  # 다음 샘플 대기 시간 (1초로 단축)
    total_time = total_recording_time + total_countdown_time + total_wait_time
    
    print(f"⏱️ 총 예상 시간: {total_time / 60:.1f}분")
    print(f"   - 녹화: {total_recording_time / 60:.1f}분")
    print(f"   - 대기: {(total_countdown_time + total_wait_time) / 60:.1f}분")
    print("=" * 50)
    
    print("\n🎯 데이터 증강 가이드:")
    print("COME 제스처 다양성:")
    print("  - 왼손/오른손/양손으로 흔들기")
    print("  - 높이: 어깨/머리/가슴 높이")
    print("  - 속도: 천천히/보통/빠르게")
    print("  - 방향: 좌우/상하/원형")
    print("  - 거리: 가까이/멀리")
    print("  - 각도: 45도/90도/180도")
    print("\nNORMAL 행동 다양성:")
    print("  - 걷기: 앞/뒤/옆으로")
    print("  - 서있기: 팔짱/손주머니/팔벌리기")
    print("  - 앉기: 의자/바닥")
    print("  - 기타: 전화/책보기/음료마시기")
    print("=" * 50)
    
    for action in actions:
        print(f"\n📹 {action.upper()} 액션 녹화 시작")
        print(f"각 액션당 {num_samples}개 샘플을 {secs_for_action}초씩 녹화합니다.")
        
        successful_samples = 0
        
        for sample_idx in range(num_samples):
            print(f"\n--- {action.upper()} 샘플 {sample_idx+1}/{num_samples} ---")
            
            # 안내 메시지
            if action == 'come':
                print("💡 한 손을 들고 흔들어주세요 (호출 제스처)")
            else:
                print("💡 평상시 행동을 해주세요 (지나가기, 팔짱끼기, 아무것도 안하기 등)")
            
            # 녹화 시작
            if record_action(action, sample_idx):
                successful_samples += 1
            
            # 다음 샘플 준비 (1초만 대기) - 렉 없는 방식
            if sample_idx < num_samples - 1:
                print("⏳ 다음 샘플 준비 중... (1초)")
                wait_start = time.time()
                
                while time.time() - wait_start < 1.0:
                    ret, img = cap.read()
                    if ret:
                        img = cv2.flip(img, 1)
                        remaining = 1.0 - (time.time() - wait_start)
                        cv2.putText(img, f'Next sample in: {remaining:.1f}s', 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.putText(img, f'Sample {sample_idx+2}/{num_samples} coming up...', 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow('Recording', img)
                    
                    # 짧은 대기로 부드러운 업데이트
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    
                    time.sleep(0.05)  # 50ms 대기
        
        print(f"\n✅ {action.upper()} 완료: {successful_samples}/{num_samples} 성공")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 데이터셋 생성 완료!")
    print(f"📁 저장 위치: {dataset_dir}/")
    print(f"📂 COME 영상: {dataset_dir}/come/")
    print(f"📂 NORMAL 영상: {dataset_dir}/normal/")
    print(f"📊 총 샘플: {num_samples * len(actions)}개")
    print(f"🎬 다음 단계: 영상 편집 후 관절점 추출")

if __name__ == "__main__":
    main() 
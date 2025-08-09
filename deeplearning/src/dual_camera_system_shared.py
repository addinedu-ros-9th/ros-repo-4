"""
공유 메모리를 사용하는 듀얼 카메라 통합 시스템
AI 서버에서 공유 메모리에 저장한 프레임을 읽어와서 딥러닝 처리를 수행합니다.
"""
import cv2
import time
import os
import threading
import numpy as np
from collections import deque
import sys
import requests # 추가된 임포트
import gc # 가비지 컬렉션 추가
from flask import Flask, request, jsonify  # Flask 추가

# CUDA 메모리 최적화 설정 (디버그 시에만 강제 동기화/캐시 비활성화)
if os.environ.get('GPU_DEBUG', '0') == '1':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.4'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
# 기본 메모리 제한만 유지
os.environ['CUDA_MEMORY_FRACTION'] = '0.7'  # GPU 메모리의 70%만 사용

# GPU 메모리 할당 제어
def setup_gpu_memory_allocation():
    """GPU 메모리 할당을 제어하여 PersonTracker와 GestureRecognizer가 나눠서 사용"""
    try:
        import torch
        if torch.cuda.is_available():
            # GPU 메모리 할당 제한 설정
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # PersonTracker용 메모리 (40%)
            person_tracker_memory = int(total_memory * 0.4)
            # GestureRecognizer용 메모리 (30%)
            gesture_recognizer_memory = int(total_memory * 0.3)
            # 나머지 30%는 여유분
            
            print(f"🎮 GPU 메모리 할당 설정:")
            print(f"  📊 총 메모리: {total_memory / 1024**3:.1f}GB")
            print(f"  👥 PersonTracker: {person_tracker_memory / 1024**3:.1f}GB (40%)")
            print(f"  🤲 GestureRecognizer: {gesture_recognizer_memory / 1024**3:.1f}GB (30%)")
            print(f"  📦 여유분: {(total_memory - person_tracker_memory - gesture_recognizer_memory) / 1024**3:.1f}GB (30%)")
            
            # CUDA 메모리 할당 제한 설정
            torch.cuda.set_per_process_memory_fraction(0.7, 0)  # 70% 제한
            
            return True
        else:
            print("⚠️ CUDA 사용 불가, CPU 모드로 실행")
            return False
    except Exception as e:
        print(f"❌ GPU 메모리 설정 실패: {e}")
        return False

# GPU 메모리 할당 설정 실행
GPU_MEMORY_SETUP = setup_gpu_memory_allocation()

# SlidingShiftGCN 모델 import를 위한 경로 추가
sys.path.append('/home/ckim/ros-repo-4/deeplearning/gesture_recognition/src')

# 모듈 import
from person_tracker import PersonTracker
from gesture_recognizer import GestureRecognizer
from shared_memory_reader import DualCameraSharedMemoryReader

# 가벼운 모델 설정
USE_LIGHTWEIGHT_MODELS = True  # 가벼운 모델 사용
if USE_LIGHTWEIGHT_MODELS:
    pass

# 성능 최적화 설정 추가
PERFORMANCE_MODE = True  # 성능 최적화 모드
if PERFORMANCE_MODE:
    # 프레임 처리 주기 조정
    FRAME_PROCESS_INTERVAL = 1  # 매프레임 처리
    # 메모리 정리 주기 완화
    MEMORY_CLEANUP_INTERVAL = 60  # 60초마다
    # PersonTracker 초기화 주기 완화
    TRACKER_RESET_INTERVAL = 300  # 300프레임마다 (약 10초@30FPS)
else:
    FRAME_PROCESS_INTERVAL = 1
    MEMORY_CLEANUP_INTERVAL = 60
    TRACKER_RESET_INTERVAL = 300

# GUI/GStreamer 오류 방지를 위한 환경 변수 설정
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_DEBUG_PLUGINS'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'

# 로컬 AI 서버 주소
AI_SERVER_BASE = "http://localhost:5006"
CAMERA_HFOV_DEGREES = 60.0 # 카메라 수평 화각 (가정치)

# CPU 모드 강제 설정 (CUDA 메모리 문제 해결용)
FORCE_CPU_MODE = os.environ.get("FORCE_CPU_MODE", "false").lower() == "true"
HYBRID_MODE = False  # 하이브리드 모드 완전 비활성화

if FORCE_CPU_MODE:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 비활성화
    print("⚠️ CPU 모드로 실행됩니다 (CUDA 메모리 문제 해결)")
else:
    print("🚀 GPU 최적화 모드로 실행됩니다")

def cleanup_gpu_memory():
    """GPU 메모리 정리 함수"""
    try:
        import torch
        if torch.cuda.is_available():
            # 모든 CUDA 캐시 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 가비지 컬렉션 강제 실행
            gc.collect()
    except Exception as e:
        print(f"GPU 메모리 정리 중 오류: {e}")


class SharedPersonTracker:
    """전면/후면 카메라 간 사람 매칭을 위한 공유 추적기 (도플갱어 방지)"""
    def __init__(self):
        print("🔄 공유 사람 추적기 초기화")
        
        # 통합된 사람 데이터 (도플갱어 방지)
        self.unified_people = {}
        self.next_unified_id = 0
        
        # 최대 기억할 사람 수 제한
        self.max_people = 15
        
        # 사용 가능한 ID 풀 (0-14)
        self.available_ids = set(range(15))
        
        # 카메라별 최신 감지 결과 (원본)
        self.camera_raw_detections = {
            'front': [],
            'back': []
        }
        
        # 매칭 설정
        self.cross_camera_match_threshold = 0.7
        self.match_timeout = 15.0
        
        self.lock = threading.Lock()
        print("✅ 공유 사람 추적기 초기화 완료")
        
    def add_frame(self, frame, frame_id, elapsed_time, camera_name):
        """특정 카메라의 프레임 추가 (원본 감지 결과만 저장)"""
        # 원본 PersonTracker로 감지 (GPU 모드)
        if camera_name == 'front':
            if not hasattr(self, 'front_tracker'):
                self.front_tracker = PersonTracker()
            self.front_tracker.add_frame(frame, frame_id, elapsed_time)
            self.camera_raw_detections['front'] = self.front_tracker.get_latest_detections()
        else:
            if not hasattr(self, 'back_tracker'):
                self.back_tracker = PersonTracker()
            self.back_tracker.add_frame(frame, frame_id, elapsed_time)
            self.camera_raw_detections['back'] = self.back_tracker.get_latest_detections()
        
        if frame_id % MEMORY_CLEANUP_INTERVAL == 0:
            self.cleanup_old_mappings(elapsed_time)
        
        # 주기적으로 PersonTracker 초기화 (메모리 관리)
        if frame_id % 1800 == 0:  # 300 → 1800으로 늘림 (30초 → 3분)
            print(f"🔄 {camera_name} 카메라 PersonTracker 초기화")
            if camera_name == 'front':
                self.front_tracker = PersonTracker()
            else:
                self.back_tracker = PersonTracker()
        
    def get_latest_detections(self, camera_name, elapsed_time):
        """특정 카메라의 최신 감지 결과 반환 (도플갱어 방지)"""
        with self.lock:
            # 1단계: 현재 카메라의 원본 감지 결과 가져오기
            raw_detections = self.camera_raw_detections.get(camera_name, [])
            
            # 도플갱어 방지: 같은 프레임에서 중복 ID 제거
            seen_unified_ids = set()
            
            # 2단계: 기존 통합 ID와 매칭 시도
            unified_detections = []
            used_unified_ids = set()
            
            for detection in raw_detections:
                person_id = detection['id']
                bbox = detection['bbox']
                
                # 기존 통합 ID와 매칭 시도
                matched_unified_id = self._find_existing_match(detection, camera_name, elapsed_time)
                
                if matched_unified_id and matched_unified_id not in seen_unified_ids:
                    # 기존 통합 ID 사용
                    unified_detection = detection.copy()
                    unified_detection['id'] = matched_unified_id
                    unified_detections.append(unified_detection)
                    seen_unified_ids.add(matched_unified_id)
                    used_unified_ids.add(matched_unified_id)
                    
                    # 매칭 정보 업데이트
                    self.unified_people[matched_unified_id]['last_seen'][camera_name] = elapsed_time
                    self.unified_people[matched_unified_id]['bbox'][camera_name] = bbox
                elif matched_unified_id is None:
                    # 새로운 통합 ID 할당 (도플갱어 방지)
                    new_unified_id = self._assign_new_unified_id(detection, camera_name, elapsed_time)
                    if new_unified_id not in seen_unified_ids:
                        unified_detection = detection.copy()
                        unified_detection['id'] = new_unified_id
                        unified_detections.append(unified_detection)
                        seen_unified_ids.add(new_unified_id)
                        used_unified_ids.add(new_unified_id)
                # else: 이미 사용된 ID는 건너뜀 (도플갱어 방지)
            
            return unified_detections
    
    def _find_existing_match(self, detection, camera_name, elapsed_time):
        """기존 통합 ID와 매칭 시도"""
        person_id = detection['id']
        bbox = detection['bbox']
        
        for unified_id, data in self.unified_people.items():
            if camera_name in data['camera_ids'] and data['camera_ids'][camera_name] == person_id:
                return unified_id
        
        return None
    
    def _assign_new_unified_id(self, detection, camera_name, elapsed_time):
        """새로운 통합 ID 할당 (도플갱어 방지)"""
        # 최대 인원 제한 확인
        if len(self.unified_people) >= self.max_people:
            oldest_id = min(self.unified_people.keys(), key=lambda x: self.unified_people[x]['created_time'])
            removed_person = self.unified_people.pop(oldest_id)
            removed_id_num = int(oldest_id.split('_')[1])
            self.available_ids.add(removed_id_num)
            print(f"🗑️ 최대 인원 초과로 오래된 사람 제거: {oldest_id}")
        
        # 사용 가능한 ID 중 가장 작은 번호 선택
        if self.available_ids:
            next_id = min(self.available_ids)
            self.available_ids.remove(next_id)
        else:
            next_id = 0
            print(f"⚠️ 모든 ID가 사용 중, 0부터 재시작")
        
        # 새로운 통합 ID 생성
        unified_id = f"Person_{next_id}"
        
        self.unified_people[unified_id] = {
            'camera_ids': {camera_name: detection['id']},
            'last_seen': {camera_name: elapsed_time},
            'bbox': {camera_name: detection['bbox']},
            'created_time': elapsed_time
        }
        
        print(f"🆕 새로운 통합 ID 생성: {detection['id']} → {unified_id}")
        return unified_id
    
    def cleanup_old_mappings(self, elapsed_time):
        """오래된 매핑 정리 (최대 인원 제한 고려)"""
        with self.lock:
            people_to_remove = []
            for unified_id, data in self.unified_people.items():
                # 모든 카메라에서 일정 시간 이상 감지되지 않은 사람 제거
                all_old = True
                for camera_name, last_seen in data['last_seen'].items():
                    if elapsed_time - last_seen < self.match_timeout:
                        all_old = False
                        break
                
                if all_old:
                    people_to_remove.append(unified_id)
            
            # 오래된 사람들 제거
            for unified_id in people_to_remove:
                removed_person = self.unified_people.pop(unified_id)
                # 제거된 ID를 사용 가능한 ID 풀에 추가
                removed_id_num = int(unified_id.split('_')[1])
                self.available_ids.add(removed_id_num)


class SingleCameraProcessor:
    """단일 카메라 스트림 처리기 (공유 메모리 사용)"""
    def __init__(self, name, shared_tracker):
        self.name = name
        self.shared_tracker = shared_tracker
        self.gesture_recognizer = GestureRecognizer()
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # 프레임 간 지연시간 계산을 위한 변수
        self.last_frame_time = None
        self.current_delay = 0.0
        
        # predict_webcam_realtime.py처럼 안정적인 초기값 설정
        self.current_gesture = "NORMAL"
        self.current_confidence = 0.5
        self.last_gesture_update_frame = 0  # 마지막 제스처 업데이트 프레임
        self.gesture_changed = False  # 제스처 변경 플래그
        self.first_prediction_received = False  # 첫 예측 결과 수신 플래그
        
        self.person_tracker_skip = 1
        self.gesture_recognizer_skip = 2
        self.color_palette = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        self.last_tracking_update_time = 0.0 # 추적 업데이트 주기 제어
        self.last_come_sent_ts = 0.0
        self.last_come_person_id = None
        
        # come 제스처 상태 관리 (ai_server와 동기화)
        self.come_gesture_active = False
        self.last_come_gesture_time = 0.0

    # 비동기 POST 유틸
    def _post_async(self, url, payload, timeout=0.2):
        def _worker():
            try:
                requests.post(url, json=payload, timeout=timeout)
            except Exception:
                pass
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def process_frame(self, frame):
        if frame is None:
            return None
            
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        current_time = time.time()
        
        # 프레임 간 지연시간 계산
        if self.last_frame_time is not None:
            self.current_delay = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # AI 모델용으로 프레임 리사이즈 (640x480)
        ai_frame = cv2.resize(frame, (640, 480))
        camera_name = 'front' if 'Front' in self.name else 'back'
        if self.frame_count % self.person_tracker_skip == 0:
            self.shared_tracker.add_frame(ai_frame.copy(), self.frame_count, elapsed_time, camera_name)
            latest_detections = self.shared_tracker.get_latest_detections(camera_name, elapsed_time)
        else:
            latest_detections = self.shared_tracker.get_latest_detections(camera_name, elapsed_time)

        # GestureRecognizer: 2프레임마다 실행
        if self.frame_count % self.gesture_recognizer_skip == 0:
            self.gesture_recognizer.add_frame(ai_frame.copy(), self.frame_count, elapsed_time, latest_detections)
            gesture_prediction, gesture_confidence, keypoints_detected, current_keypoints = self.gesture_recognizer.get_latest_gesture()
            
            # predict_webcam_realtime.py와 동일: 30프레임 윈도우에서 실제 예측이 나왔을 때만 UI 업데이트
            # GestureRecognizer 내부에서 30프레임 쌓였을 때만 예측하므로, 그때만 UI 변경
            if (keypoints_detected and current_keypoints is not None and 
                gesture_confidence > 0.0 and 
                (not self.first_prediction_received or gesture_prediction != self.current_gesture)):
                # predict_webcam_realtime.py와 동일: 제스처가 실제로 변경된 경우에만 업데이트
                if not self.first_prediction_received:
                    print(f"[{self.name}] 🎯 첫 제스처 예측: {gesture_prediction} (신뢰도: {gesture_confidence:.3f})")
                    self.first_prediction_received = True
                else:
                    print(f"[{self.name}] 🎯 제스처 변경: {self.current_gesture} → {gesture_prediction} (신뢰도: {gesture_confidence:.3f})")
                self.current_gesture = gesture_prediction
                self.current_confidence = gesture_confidence
                self.last_gesture_update_frame = self.frame_count
                self.gesture_changed = True
            # 그 외의 경우: UI 값 변경 없음 (안정적 표시)
        # 스킵된 프레임에서는 이전 제스처 결과 유지
        
        annotated = frame.copy()
        
        if latest_detections:
            for i, person in enumerate(latest_detections):
                # AI 모델 결과를 원본 해상도로 스케일링
                x1, y1, x2, y2 = map(int, person['bbox'])
                # 640x480 → 원본 해상도 스케일링
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 480
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                person_id = person['id']
                color = self.color_palette[int(person_id.split('_')[-1]) % len(self.color_palette)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID:{person_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 현재 프레임의 가장 큰 사람 ID 계산 (tracking/update용)
        largest_person_id = None
        if latest_detections:
            largest_area = -1
            for det in latest_detections:
                x1, y1, x2, y2 = det['bbox']
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_person_id = det['id']
        
        if current_time - self.last_tracking_update_time > 1.0:
            largest_person_visible = bool(largest_person_id is not None)
            self._post_async(f"{AI_SERVER_BASE}/tracking/update", {"person_visible": largest_person_visible, "person_id": largest_person_id}, timeout=0.2)
            self.last_tracking_update_time = current_time

        # 키포인트 시각화 (GestureRecognizer가 실행된 프레임에서만)
        if self.frame_count % self.gesture_recognizer_skip == 0 and keypoints_detected and current_keypoints is not None:
            # 키포인트도 원본 해상도로 스케일링
            scaled_keypoints = current_keypoints.copy()
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            scaled_keypoints[:, 0] *= scale_x
            scaled_keypoints[:, 1] *= scale_y
            annotated = self.gesture_recognizer.draw_visualization(annotated, scaled_keypoints, self.current_gesture, self.current_confidence)

            # COME 인식 시 각도 계산 및 전송
            if self.current_gesture == "COME" and self.current_confidence >= 0.8:
                # come 제스처가 이미 활성화되어 있는지 확인
                if self.come_gesture_active:
                    # return_command가 올 때까지 어떤 사람이든 come 제스처를 중앙에 보내지 않도록 수정하고, continue 오류도 수정합니다.
                    print(f"[{self.name}] come 제스처 이미 활성화됨, return_command 대기 중")
                    return annotated
                
                # 가장 큰 바운딩 박스를 가진 사람의 ID 사용
                person_id = None
                bbox = None
                if latest_detections:
                    # 가장 큰 바운딩 박스 찾기
                    largest_area = 0
                    for det in latest_detections:
                        x1, y1, x2, y2 = det['bbox']
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            person_id = det['id']
                            bbox = det['bbox']
                
                if bbox:
                    # 5초 쿨다운
                    if current_time - self.last_come_sent_ts > 5.0 or self.last_come_person_id != person_id:
                        left_angle, right_angle = self._calculate_person_angles(bbox, ai_frame.shape[1])
                        payload = {
                            "robot_id": 3,
                            "person_id": person_id,
                            "left_angle": f"{left_angle:.1f}",
                            "right_angle": f"{right_angle:.1f}",
                            "timestamp": int(current_time)
                        }
                        # 비동기 전송
                        self._post_async(f"{AI_SERVER_BASE}/gesture/come_local", payload, timeout=0.2)
                        self.last_come_sent_ts = current_time
                        self.last_come_person_id = person_id
                        
                        # come 제스처 활성화 상태 설정
                        self.come_gesture_active = True
                        self.last_come_gesture_time = current_time

        # 간단한 정보 표시: 딜레이, 제스처, confidence
        cv2.putText(annotated, f"Delay: {self.current_delay*1000:.0f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 제스처 색상: COME은 빨간색, NORMAL은 초록색
        gesture_color = (0, 0, 255) if self.current_gesture == "COME" else (0, 255, 0)
        cv2.putText(annotated, f"Gesture: {self.current_gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        cv2.putText(annotated, f"Conf: {self.current_confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)

        # 디버그: 30프레임 예측 상태 표시
        frames_since_update = self.frame_count - self.last_gesture_update_frame
        cv2.putText(annotated, f"Last Update: {frames_since_update}f ago", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated

    def _calculate_person_angles(self, bbox, frame_width):
        """Bbox를 사용해 카메라 중심 기준 사람의 좌우 각도를 계산합니다."""
        if bbox is None:
            return 0.0, 0.0
        camera_center_x = frame_width / 2.0
        degrees_per_pixel = CAMERA_HFOV_DEGREES / frame_width
        x1, _, x2, _ = bbox
        left_angle = (x1 - camera_center_x) * degrees_per_pixel
        right_angle = (x2 - camera_center_x) * degrees_per_pixel
        return left_angle, right_angle


class DualCameraSystemShared:
    def __init__(self):
        # 공유 메모리 리더 초기화
        self.shared_memory_reader = DualCameraSharedMemoryReader()
        
        # 공유 추적기 생성
        self.shared_tracker = SharedPersonTracker()
        
        # 공유 추적기를 사용하는 프로세서들
        self.front_processor = SingleCameraProcessor("Front", self.shared_tracker)
        self.back_processor = SingleCameraProcessor("Back", self.shared_tracker)
        self.latest_angles = {'front': 180.0, 'back': 180.0}
        self.last_come_sent_ts = 0.0
        self.last_come_person_id = None
        
        # Flask 서버 초기화 (return_command 수신용)
        self.app = Flask(__name__)
        self.setup_flask_routes()
        
        # Flask 서버를 별도 스레드에서 실행
        self.flask_thread = threading.Thread(target=self.run_flask_server, daemon=True)
        self.flask_thread.start()
        
    def setup_flask_routes(self):
        """Flask 라우트 설정"""
        @self.app.route("/gesture/return_command", methods=["POST"])
        def gesture_return_command():
            """return_command 수신 시 come 제스처 상태 리셋"""
            try:
                data = request.get_json(force=True)
            except Exception:
                return jsonify({"status_code": 400, "error": "invalid_json"}), 400

            robot_id = data.get("robot_id")
            if robot_id is None:
                return jsonify({"status_code": 400, "error": "robot_id_required"}), 400
            self.front_processor.come_gesture_active = False
            self.front_processor.last_come_person_id = None
            self.back_processor.come_gesture_active = False
            self.back_processor.last_come_person_id = None
            return jsonify({"status_code": 200})
            
        @self.app.route("/health", methods=["GET"])
        def health():
            """상태 확인"""
            return jsonify({
                "status": "ok",
                "front_come_active": self.front_processor.come_gesture_active,
                "back_come_active": self.back_processor.come_gesture_active,
                "front_last_person": self.front_processor.last_come_person_id,
                "back_last_person": self.back_processor.last_come_person_id,
                "server_type": "dual_camera_system_shared"
            })
    
    def run_flask_server(self):
        """Flask 서버 실행"""
        try:
            self.app.run(host="0.0.0.0", port=5008, debug=False)
        except Exception as e:
            print(f"Flask 서버 실행 실패: {e}")

    def run_system(self):
        if not self.shared_memory_reader.is_available():
            print("🛑 공유 메모리를 사용할 수 없습니다.")
            print("💡 AI 서버가 실행 중인지 확인해주세요.")
            return

        window_front = "Front Camera (Shared Memory)"
        window_back = "Back Camera (Shared Memory)"
        cv2.namedWindow(window_front, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_back, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_front, 50, 50)
        cv2.moveWindow(window_back, 700, 50)
        
        # 창 크기 설정
        cv2.resizeWindow(window_front, 960, 540)
        cv2.resizeWindow(window_back, 960, 540)
        while True:
            # 공유 메모리에서 프레임 읽기 (최적화)
            try:
                front_frame, back_frame = self.shared_memory_reader.read_frames()
            except Exception as e:
                print(f"공유메모리 읽기 오류: {e}")
                time.sleep(0.01)
                continue

            if front_frame is not None:
                annotated_front = self.front_processor.process_frame(front_frame)
                if annotated_front is not None:
                    cv2.imshow(window_front, annotated_front)
            
            if back_frame is not None:
                annotated_back = self.back_processor.process_frame(back_frame)
                if annotated_back is not None:
                    cv2.imshow(window_back, annotated_back)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
        self.stop()

    def stop(self):
        self.shared_memory_reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    dual_system = DualCameraSystemShared()
    dual_system.run_system() 
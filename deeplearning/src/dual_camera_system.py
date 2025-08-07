"""
듀얼 카메라에 통합 시스템(Person Tracking + Gesture Recognition)을 적용합니다.

- WebcamStream: 각 카메라의 프레임을 비동기적으로 읽어오는 스레드 (I/O 블로킹 방지)
- SingleCameraProcessor: 단일 카메라 스트림에 대한 모든 처리 로직(추적, 인식, 시각화)을 캡슐화
- DualCameraSystem: 두 개의 WebcamStream과 두 개의 SingleCameraProcessor를 관리하여 전체 시스템을 운영
- SharedPersonTracker: 전면/후면 카메라 간 사람 매칭을 위한 공유 추적기
"""
import cv2
import time
import os
import threading
import numpy as np
from collections import deque
import sys

# SlidingShiftGCN 모델 import를 위한 경로 추가
sys.path.append('/home/ckim/ros-repo-4/deeplearning/gesture_recognition/src')

# 모듈 import
from person_tracker import PersonTracker
from gesture_recognizer import GestureRecognizer

# GUI/GStreamer 오류 방지를 위한 환경 변수 설정
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_DEBUG_PLUGINS'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'


class SharedPersonTracker:
    """전면/후면 카메라 간 사람 매칭을 위한 공유 추적기 (도플갱어 방지)"""
    def __init__(self):
        print("🔄 공유 사람 추적기 초기화")
        
        # 통합된 사람 데이터 (도플갱어 방지)
        self.unified_people = {}
        self.next_unified_id = 0
        
        # 최대 기억할 사람 수 제한
        self.max_people = 10
        
        # 사용 가능한 ID 풀 (0-9)
        self.available_ids = set(range(10))
        
        # 카메라별 최신 감지 결과 (원본)
        self.camera_raw_detections = {
            'front': [],
            'back': []
        }
        
        # 매칭 설정
        self.cross_camera_match_threshold = 0.4
        self.match_timeout = 8.0
        
        self.lock = threading.Lock()
        print("✅ 공유 사람 추적기 초기화 완료")
        
    def add_frame(self, frame, frame_id, elapsed_time, camera_name):
        """특정 카메라의 프레임 추가 (원본 감지 결과만 저장)"""
        # 원본 PersonTracker로 감지
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
        
        # 주기적으로 오래된 매핑 정리
        if frame_id % 60 == 0:
            self.cleanup_old_mappings(elapsed_time)
        
        # 주기적으로 PersonTracker 초기화 (메모리 관리)
        if frame_id % 300 == 0:
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
            
            # 2단계: 기존 통합 ID와 매칭 시도
            unified_detections = []
            used_unified_ids = set()  # 현재 카메라에서 사용된 통합 ID 추적
            
            for detection in raw_detections:
                person_id = detection['id']
                bbox = detection['bbox']
                
                # 기존 통합 ID와 매칭 시도
                matched_unified_id = self._find_existing_match(detection, camera_name, elapsed_time)
                
                if matched_unified_id:
                    # 기존 통합 ID 사용
                    unified_detection = detection.copy()
                    unified_detection['id'] = matched_unified_id
                    unified_detections.append(unified_detection)
                    used_unified_ids.add(matched_unified_id)
                    
                    # 매칭 정보 업데이트
                    self.unified_people[matched_unified_id]['last_seen'][camera_name] = elapsed_time
                    self.unified_people[matched_unified_id]['bbox'][camera_name] = bbox
                else:
                    # 새로운 통합 ID 할당 (도플갱어 방지)
                    new_unified_id = self._assign_new_unified_id(detection, camera_name, elapsed_time)
                    unified_detection = detection.copy()
                    unified_detection['id'] = new_unified_id
                    unified_detections.append(unified_detection)
                    used_unified_ids.add(new_unified_id)
            
            # 3단계: 다른 카메라에서 현재 카메라로 이동한 사람 확인
            other_camera = 'back' if camera_name == 'front' else 'front'
            other_detections = self.camera_raw_detections.get(other_camera, [])
            
            for other_detection in other_detections:
                other_person_id = other_detection['id']
                other_bbox = other_detection['bbox']
                
                # 이미 현재 카메라에서 사용된 통합 ID는 제외
                for unified_id, data in self.unified_people.items():
                    if (unified_id not in used_unified_ids and 
                        other_camera in data['camera_ids'] and 
                        data['camera_ids'][other_camera] == other_person_id):
                        
                        # 시간 차이 확인
                        other_last_seen = data['last_seen'].get(other_camera, 0)
                        time_diff = elapsed_time - other_last_seen
                        
                        if time_diff <= self.match_timeout:
                            # 현재 카메라의 원본 감지 결과에서 매칭할 대상 찾기
                            for current_detection in raw_detections:
                                current_bbox = current_detection['bbox']
                                # 공간적 매칭 시도
                                score = self._calculate_cross_camera_similarity(other_bbox, current_bbox, other_camera, camera_name)
                                if score > self.cross_camera_match_threshold:
                                    # 매칭 성공 - 기존 통합 ID 사용
                                    unified_detection = current_detection.copy()
                                    unified_detection['id'] = unified_id
                                    unified_detections.append(unified_detection)
                                    used_unified_ids.add(unified_id)
                                    
                                    # 매칭 정보 업데이트
                                    data['camera_ids'][camera_name] = current_detection['id']
                                    data['last_seen'][camera_name] = elapsed_time
                                    data['bbox'][camera_name] = current_detection['bbox']
                                    
                                    print(f"✅ 카메라 간 이동 감지: {other_camera} → {camera_name} ({unified_id})")
                                    break
            
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
            # 가장 오래된 사람 제거
            oldest_id = min(self.unified_people.keys(), 
                          key=lambda x: self.unified_people[x]['created_time'])
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
    
    def _calculate_cross_camera_similarity(self, bbox1, bbox2, camera1, camera2):
        """두 카메라 간 바운딩 박스 유사도 계산 (적당한 기준)"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 면적 비율 유사도 (적당하게)
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # 면적 차이가 너무 크면 매칭 거부 (더 관대하게)
        if area_ratio < 0.1:  # 면적 차이가 90% 이상이면 매칭 거부 (0.3 → 0.1로 더 관대하게)
            return 0.0
        
        # 위치 유사도 (카메라별 특성 고려)
        if camera1 == 'front' and camera2 == 'back':
            # 전면-후면: 전면에서는 상단, 후면에서는 하단에 있을 가능성
            front_center_y = (y1_1 + y2_1) / 2
            back_center_y = (y1_2 + y2_2) / 2
            height_ratio = 480  # 프레임 높이
            
            # 전면 상단 ↔ 후면 하단 매칭
            front_normalized = front_center_y / height_ratio
            back_normalized = (height_ratio - back_center_y) / height_ratio
            position_similarity = 1.0 - abs(front_normalized - back_normalized)
            
            # 위치 차이가 너무 크면 매칭 거부 (더 관대하게)
            if position_similarity < 0.2:  # 위치 차이가 80% 이상이면 매칭 거부 (0.4 → 0.2로 더 관대하게)
                return 0.0
        else:
            position_similarity = 0.5  # 기본값
        
        # 종합 점수 (면적 비율에 더 높은 가중치, 적당하게)
        total_score = (area_ratio * 0.6 + position_similarity * 0.4)
        
        # 디버깅 정보 (15프레임마다 - 더 자주 출력)
        if hasattr(self, 'debug_counter') and self.debug_counter % 15 == 0:
            print(f"🔍 카메라 간 매칭 시도: {camera1} ↔ {camera2}")
            print(f"   - 면적 비율: {area_ratio:.3f}")
            print(f"   - 위치 유사도: {position_similarity:.3f}")
            print(f"   - 종합 점수: {total_score:.3f}")
            print(f"   - 임계값: {self.cross_camera_match_threshold}")
            if total_score < self.cross_camera_match_threshold:
                print(f"   ❌ 매칭 거부: 점수 부족")
            else:
                print(f"   ✅ 매칭 가능: 점수 충분")
        
        return total_score
    
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
                print(f"🗑️ 타임아웃으로 오래된 사람 제거: {unified_id}")
                print(f"   - 제거된 사람 정보: {removed_person['camera_ids']}")
            
            # 현재 인원 수 출력 (디버깅용)
            if len(self.unified_people) > 0:
                print(f"📊 현재 추적 중인 사람: {len(self.unified_people)}/{self.max_people}명")
                print(f"   사용 가능한 ID 풀: {sorted(self.available_ids)}")


class WebcamStream:
    """특정 카메라를 위한 비동기 프레임 읽기 스레드"""
    def __init__(self, src, name):
        print(f"🔄 {name} ({src}) 스트림 초기화 시도...")
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.stream.isOpened():
            raise IOError(f"카메라를 열 수 없습니다: {name} at {src}")

        self.name = name
        fps = 15 if "Back" in self.name else 30
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        print(f"   - {name} FPS 설정: {fps}")

        (self.ret, self.frame) = self.stream.read()
        if not self.ret:
            raise IOError(f"초기 프레임을 읽을 수 없습니다: {name}")
        
        print(f"✅ {name} ({src}) 스트림 초기화 성공!")

        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            (ret, frame) = self.stream.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
        self.stream.release()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1)

class SingleCameraProcessor:
    """단일 카메라 스트림 처리기"""
    def __init__(self, name, shared_tracker):
        print(f"🚀 {name} Processor 초기화")
        self.name = name
        self.shared_tracker = shared_tracker  # 공유 추적기 사용
        self.gesture_recognizer = GestureRecognizer()
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # 프레임 간 지연시간 계산을 위한 변수
        self.last_frame_time = None
        self.current_delay = 0.0
        
        self.current_gesture = "NORMAL"
        self.current_confidence = 0.5
        
        self.color_palette = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        print(f"✅ {name} Processor 초기화 완료")

    def process_frame(self, frame):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        current_time = time.time()
        
        # 프레임 간 지연시간 계산
        if self.last_frame_time is not None:
            self.current_delay = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # AI 모델용으로 프레임 리사이즈 (640x480)
        ai_frame = cv2.resize(frame, (640, 480))

        # 공유 추적기 사용
        camera_name = 'front' if 'Front' in self.name else 'back'
        self.shared_tracker.add_frame(ai_frame.copy(), self.frame_count, elapsed_time, camera_name)
        latest_detections = self.shared_tracker.get_latest_detections(camera_name, elapsed_time)

        self.gesture_recognizer.add_frame(ai_frame.copy(), self.frame_count, elapsed_time, latest_detections)
        gesture_prediction, gesture_confidence, keypoints_detected, current_keypoints = self.gesture_recognizer.get_latest_gesture()

        # 제스처와 confidence 업데이트
        if keypoints_detected and current_keypoints is not None:
            # gesture_prediction이 이미 "COME" 또는 "NORMAL" 문자열
            self.current_gesture = gesture_prediction
            self.current_confidence = gesture_confidence
        else:
            self.current_gesture = "NORMAL"
            self.current_confidence = 0.0  # keypoints가 없으면 confidence 0
        
        annotated = frame.copy()  # 원본 해상도로 표시
        
        if latest_detections:
            for i, person in enumerate(latest_detections):
                # AI 모델 결과를 원본 해상도로 스케일링
                x1, y1, x2, y2 = map(int, person['bbox'])
                # 640x480 → 1280x720 스케일링
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

        if keypoints_detected and current_keypoints is not None:
            # 키포인트도 원본 해상도로 스케일링
            scaled_keypoints = current_keypoints.copy()
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            scaled_keypoints[:, 0] *= scale_x
            scaled_keypoints[:, 1] *= scale_y
            annotated = self.gesture_recognizer.draw_visualization(annotated, scaled_keypoints, self.current_gesture, self.current_confidence)

        # 간단한 정보 표시: 딜레이, 제스처, confidence
        cv2.putText(annotated, f"Delay: {self.current_delay*1000:.0f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 제스처 색상: COME은 빨간색, NORMAL은 초록색
        gesture_color = (0, 0, 255) if self.current_gesture == "COME" else (0, 255, 0)
        cv2.putText(annotated, f"Gesture: {self.current_gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        cv2.putText(annotated, f"Conf: {self.current_confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)

        return annotated


class DualCameraSystem:
    def __init__(self):
        print("🚀 이중 카메라 통합 시스템 초기화")
        self.front_stream = None
        self.back_stream = None
        
        # 공유 추적기 생성
        self.shared_tracker = SharedPersonTracker()
        
        # 공유 추적기를 사용하는 프로세서들
        self.front_processor = SingleCameraProcessor("Front", self.shared_tracker)
        self.back_processor = SingleCameraProcessor("Back", self.shared_tracker)

    def run_system(self):
        try:
            self.front_stream = WebcamStream(src="/dev/video0", name="Front Camera").start()
            time.sleep(1.0)
            self.back_stream = WebcamStream(src="/dev/video2", name="Back Camera").start()
        except IOError as e:
            print(f"🛑 카메라 초기화 실패: {e}")
            self.stop()
            return

        window_front = "Front Camera"
        window_back = "Back Camera"
        cv2.namedWindow(window_front, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_back, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_front, 50, 50)
        cv2.moveWindow(window_back, 700, 50)
        
        # 창 크기 설정 (1280x720 해상도에 맞춰 더 크게)
        cv2.resizeWindow(window_front, 960, 540)  # 16:9 비율로 크게
        cv2.resizeWindow(window_back, 960, 540)   # 16:9 비율로 크게

        print("\n🚀 양쪽 통합 시스템 가동 시작. 'q'를 누르면 종료됩니다.")
        print("📝 이제 전면/후면 카메라에서 같은 사람이 같은 ID로 인식됩니다!")
        
        while True:
            if self.front_stream.stopped or self.back_stream.stopped:
                break

            ret_front, frame_front = self.front_stream.read()
            ret_back, frame_back = self.back_stream.read()

            if ret_front:
                annotated_front = self.front_processor.process_frame(frame_front)
                cv2.imshow(window_front, annotated_front)
            
            if ret_back:
                annotated_back = self.back_processor.process_frame(frame_back)
                cv2.imshow(window_back, annotated_back)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 메인 루프에 휴식을 주어 CPU 과부하 방지 (카메라 FPS에 맞춰 조정)
            time.sleep(0.01)  # 10ms 대기 (약 100 FPS 제한)
        
        self.stop()

    def stop(self):
        print("🧹 시스템 정리 시작...")
        if self.front_stream: self.front_stream.stop()
        if self.back_stream: self.back_stream.stop()
        cv2.destroyAllWindows()
        print("✅ 시스템 정리 완료.")


if __name__ == "__main__":
    dual_system = DualCameraSystem()
    dual_system.run_system() 
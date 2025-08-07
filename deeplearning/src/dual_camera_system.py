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

# 모듈 import
from person_tracker import PersonTracker
from gesture_recognizer import GestureRecognizer

# GUI/GStreamer 오류 방지를 위한 환경 변수 설정
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_DEBUG_PLUGINS'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'


class SharedPersonTracker:
    """전면/후면 카메라 간 사람 매칭을 위한 공유 추적기"""
    def __init__(self):
        print("🔄 공유 사람 추적기 초기화")
        
        # 카메라별 독립적인 추적기 (각각의 히스토그램 기억)
        self.front_tracker = PersonTracker()
        self.back_tracker = PersonTracker()
        
        # 통합된 사람 데이터 (카메라 간 매칭)
        self.unified_people = {}
        self.next_unified_id = 0
        
        # 카메라별 최신 감지 결과
        self.camera_detections = {
            'front': [],
            'back': []
        }
        
        # 매칭 설정 (더 엄격하게)
        self.cross_camera_match_threshold = 0.6  # 0.4 → 0.6으로 더 엄격하게
        self.match_timeout = 10.0 
        
        self.lock = threading.Lock()
        print("✅ 공유 사람 추적기 초기화 완료")
        
    def add_frame(self, frame, frame_id, elapsed_time, camera_name):
        """특정 카메라의 프레임 추가"""
        if camera_name == 'front':
            self.front_tracker.add_frame(frame, frame_id, elapsed_time)
        else:
            self.back_tracker.add_frame(frame, frame_id, elapsed_time)
        
    def get_latest_detections(self, camera_name, elapsed_time):
        """특정 카메라의 최신 감지 결과 반환 (통합 ID 적용)"""
        with self.lock:
            # 해당 카메라의 원본 감지 결과 가져오기
            if camera_name == 'front':
                raw_detections = self.front_tracker.get_latest_detections()
            else:
                raw_detections = self.back_tracker.get_latest_detections()
            
            # 통합 ID 매핑 적용
            unified_detections = []
            for detection in raw_detections:
                unified_id = self._get_unified_id(detection, camera_name, elapsed_time)
                unified_detection = detection.copy()
                unified_detection['id'] = unified_id
                unified_detections.append(unified_detection)
            
            # 카메라별 결과 저장
            self.camera_detections[camera_name] = unified_detections
            
            return unified_detections
    
    def _get_unified_id(self, detection, camera_name, elapsed_time):
        """감지된 사람에 대한 통합 ID 반환 (카메라 간 매칭)"""
        person_id = detection['id']
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # 현재 카메라에서 이미 매핑된 ID가 있는지 확인
        for unified_id, data in self.unified_people.items():
            if camera_name in data['camera_ids'] and data['camera_ids'][camera_name] == person_id:
                # 기존 매핑 업데이트
                data['last_seen'][camera_name] = elapsed_time
                data['bbox'][camera_name] = bbox
                return unified_id
        
        # 다른 카메라에서 매칭 가능한 사람 찾기
        best_match_id = None
        best_score = 0.0
        
        for unified_id, data in self.unified_people.items():
            # 다른 카메라에서 감지된 사람과 매칭 시도
            other_camera = 'back' if camera_name == 'front' else 'front'
            if other_camera in data['camera_ids']:
                # 현재 카메라에 이미 매핑된 통합 ID는 제외 (중복 매칭 방지)
                if camera_name in data['camera_ids']:
                    continue
                    
                other_bbox = data['bbox'].get(other_camera)
                if other_bbox:
                    # 시간 차이 확인 (너무 오래된 매칭은 제외)
                    other_last_seen = data['last_seen'].get(other_camera, 0)
                    time_diff = elapsed_time - other_last_seen
                    
                    if time_diff <= self.match_timeout:  # 타임아웃 내에만 매칭
                        # 공간적 매칭
                        score = self._calculate_cross_camera_similarity(bbox, other_bbox, camera_name, other_camera)
                        if score > best_score and score > self.cross_camera_match_threshold:
                            best_score = score
                            best_match_id = unified_id
        
        if best_match_id:
            # 기존 통합 ID에 현재 카메라 정보 추가
            self.unified_people[best_match_id]['camera_ids'][camera_name] = person_id
            self.unified_people[best_match_id]['last_seen'][camera_name] = elapsed_time
            self.unified_people[best_match_id]['bbox'][camera_name] = bbox
            
            # 매칭 성공 디버깅
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
                
            if self.debug_counter % 5 == 0:  # 10 → 5로 더 자주 출력
                print(f"✅ 카메라 간 매칭 성공: {person_id} → {best_match_id} (점수: {best_score:.3f})")
                print(f"   현재 매핑 상태: {self.unified_people[best_match_id]['camera_ids']}")
                print(f"   전체 통합 ID 개수: {len(self.unified_people)}")
            
            return best_match_id
        else:
            # 새로운 통합 ID 생성
            unified_id = f"Person_{self.next_unified_id}"
            self.next_unified_id += 1
            
            self.unified_people[unified_id] = {
                'camera_ids': {camera_name: person_id},
                'last_seen': {camera_name: elapsed_time},
                'bbox': {camera_name: bbox},
                'created_time': elapsed_time
            }
            
            # 새 ID 생성 디버깅
            if hasattr(self, 'debug_counter') and self.debug_counter % 5 == 0:  # 10 → 5로 더 자주 출력
                print(f"🆕 새로운 통합 ID 생성: {person_id} → {unified_id}")
                if best_score > 0:
                    print(f"   - 최고 매칭 점수: {best_score:.3f} (임계값: {self.cross_camera_match_threshold})")
                print(f"   현재 전체 매핑: {len(self.unified_people)}개 통합 ID")
                print(f"   전체 매핑 상태:")
                for uid, data in self.unified_people.items():
                    print(f"     {uid}: {data['camera_ids']}")
            
            return unified_id
    
    def _calculate_cross_camera_similarity(self, bbox1, bbox2, camera1, camera2):
        """두 카메라 간 바운딩 박스 유사도 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 면적 비율 유사도 (더 관대하게)
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
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
        else:
            position_similarity = 0.5  # 기본값
        
        # 종합 점수 (면적 비율에 더 높은 가중치)
        total_score = (area_ratio * 0.8 + position_similarity * 0.2)
        
        # 디버깅 정보 (30프레임마다)
        if hasattr(self, 'debug_counter') and self.debug_counter % 30 == 0:
            print(f"🔍 카메라 간 매칭 시도: {camera1} ↔ {camera2}")
            print(f"   - 면적 비율: {area_ratio:.3f}")
            print(f"   - 위치 유사도: {position_similarity:.3f}")
            print(f"   - 종합 점수: {total_score:.3f}")
            print(f"   - 임계값: {self.cross_camera_match_threshold}")
        
        return total_score
    
    def cleanup_old_mappings(self, elapsed_time):
        """오래된 매핑 정리"""
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
            
            for unified_id in people_to_remove:
                del self.unified_people[unified_id]


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
            annotated = self.gesture_recognizer.draw_keypoints(annotated, scaled_keypoints, (0, 255, 255))

        # 간단한 정보 표시: 딜레이, 제스처, confidence
        cv2.putText(annotated, f"Delay: {self.current_delay*1000:.0f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"Gesture: {self.current_gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Conf: {self.current_confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
"""
듀얼 카메라에 통합 시스템(Person Tracking + Gesture Recognition)을 적용합니다.

- WebcamStream: 각 카메라의 프레임을 비동기적으로 읽어오는 스레드 (I/O 블로킹 방지)
- SingleCameraProcessor: 단일 카메라 스트림에 대한 모든 처리 로직(추적, 인식, 시각화)을 캡슐화
- DualCameraSystem: 두 개의 WebcamStream과 두 개의 SingleCameraProcessor를 관리하여 전체 시스템을 운영
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
    def __init__(self, name):
        print(f"🚀 {name} Processor 초기화")
        self.name = name
        self.person_tracker = PersonTracker()
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

        self.person_tracker.add_frame(ai_frame.copy(), self.frame_count, elapsed_time)
        latest_detections = self.person_tracker.get_latest_detections()

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
        self.front_processor = SingleCameraProcessor("Front")
        self.back_processor = SingleCameraProcessor("Back")

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
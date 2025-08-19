import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque
import threading
import queue
import time
import os
import sys

# SlidingShiftGCN 모델 import를 위한 경로 추가
sys.path.append('/home/ckim/ros-repo-4/deeplearning/gesture_recognition/src')
from train_sliding_shift_gcn import SlidingShiftGCN

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU 메모리 제한 설정 (GestureRecognizer용 30%)
if torch.cuda.is_available():
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        gesture_recognizer_memory = int(total_memory * 0.3)
        torch.cuda.set_per_process_memory_fraction(0.3, 0)  # 30% 제한
        print(f"🎮 GestureRecognizer GPU 메모리 제한: {gesture_recognizer_memory / 1024**3:.1f}GB")
    except Exception as e:
        print(f"⚠️ GPU 메모리 제한 설정 실패: {e}")

class GestureRecognizer:
    """predict_webcam_realtime.py 기반 제스처 인식 시스템"""
    
    def __init__(self):
        print("🚀 Gesture Recognizer 초기화 (predict_webcam_realtime.py 기반)")
        
        # predict_webcam_realtime.py와 동일한 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 사용 디바이스: {self.device}")
        
        # YOLO 모델 로드
        yolo_path = '/home/ckim/ros-repo-4/deeplearning/yolov8n-pose.pt'
        self.yolo_model = YOLO(yolo_path)
        if torch.cuda.is_available():
            self.yolo_model.to(self.device)
        
        # 관절점 인덱스 (predict_webcam_realtime.py와 동일)
        self.joint_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12]
        self.num_joints = len(self.joint_indices)
        
        # GCN 모델 로드
        self.load_shift_gcn_model()
        
        # 버퍼 및 설정 (predict_webcam_realtime.py와 동일)
        self.window_size = 30
        self.keypoint_buffer = deque(maxlen=self.window_size)
        
        # 액션 라벨 및 색상 (predict_webcam_realtime.py와 동일)
        self.action_labels = {0: 'COME', 1: 'NORMAL'}
        self.actions = ['COME', 'NORMAL']  # 호환성 유지
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}
        self.last_prediction = None
        
        # 결과 저장용
        self.current_gesture = "NORMAL"
        self.current_confidence = 0.5
        
        # 쓰레드 설정
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.lock = threading.Lock()
        
        # 결과 저장
        self.latest_gesture = ("NORMAL", 0.5, False, None)
        
        # 워커 스레드 시작
        self.worker_thread = threading.Thread(target=self.recognition_worker)
        self.worker_thread.start()
        print(f"✅ 워커 스레드 시작됨 (스레드 ID: {self.worker_thread.ident})")
        
        print("✅ Gesture Recognizer 초기화 완료")
    
    def load_shift_gcn_model(self):
        """predict_webcam_realtime.py와 동일한 GCN 모델 로딩"""
        model_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/models/sliding_shift_gcn_model.pth'
        
        # 인접 행렬 로드
        adj_matrix_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/shift_gcn_data_sliding/adjacency_matrix.npy'
        adjacency_matrix = np.load(adj_matrix_path) if os.path.exists(adj_matrix_path) else np.eye(self.num_joints)
        
        # 모델 생성 및 로드
        self.model = SlidingShiftGCN(num_classes=2, num_joints=self.num_joints, dropout=0.3)
        self.model.set_adjacency_matrix(adjacency_matrix)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 모델 로드 완료: {model_path}")
    
    def normalize_keypoints(self, keypoints):
        """predict_webcam_realtime.py와 완전히 동일한 정규화"""
        normalized = keypoints.copy()
        valid_joints = keypoints[:, 2] > 0
        if not np.any(valid_joints): 
            return None
        valid_points = keypoints[valid_joints, :2]
        center = np.mean(valid_points, axis=0)
        scale = np.std(valid_points)
        if scale == 0: 
            scale = 1.0
        normalized[:, :2] = (keypoints[:, :2] - center) / scale
        return normalized

    def predict_gesture(self):
        """predict_webcam_realtime.py와 완전히 동일한 예측"""
        if len(self.keypoint_buffer) < self.window_size:
            return None, 0.0
        
        window_data = np.array(list(self.keypoint_buffer))
        shift_gcn_input = window_data.transpose(2, 0, 1)[np.newaxis, :, :, :, np.newaxis]
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(shift_gcn_input).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze()
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
        return prediction, confidence

    def draw_visualization(self, frame, keypoints, prediction, confidence):
        """predict_webcam_realtime.py와 동일한 시각화 (호환성용)"""
        if keypoints is not None:
            # 관절점 연결선 정의 (predict_webcam_realtime.py와 동일)
            connections = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)]
            for i, j in connections:
                idx1, idx2 = self.joint_indices[i], self.joint_indices[j]
                if keypoints[idx1, 2] > 0.3 and keypoints[idx2, 2] > 0.3:
                    p1 = (int(keypoints[idx1, 0]), int(keypoints[idx1, 1]))
                    p2 = (int(keypoints[idx2, 0]), int(keypoints[idx2, 1]))
                    cv2.line(frame, p1, p2, (255, 255, 255), 2)
        
            # 관절점 그리기 (predict_webcam_realtime.py와 동일)
            for idx in self.joint_indices:
                if keypoints[idx, 2] > 0.3:
                    center = (int(keypoints[idx, 0]), int(keypoints[idx, 1]))
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)

        # 제스처 예측 결과 표시 (prediction이 문자열인 경우 고려)
        if prediction is None or (isinstance(prediction, str) and prediction == "NORMAL" and confidence == 0.5):
            text = f"Collecting frames... [{len(self.keypoint_buffer)}/{self.window_size}]"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # prediction이 문자열이면 그대로 사용, 숫자면 변환
            if isinstance(prediction, str):
                label = prediction
                color = (0, 0, 255) if label == "COME" else (0, 255, 0)  # BGR
            else:
                label = self.action_labels[prediction]
                color = self.colors[prediction]  # BGR 색상
            
            text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (10, 10), (350, 60), color, -1)
            cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return frame

    def recognition_worker(self):
        """predict_webcam_realtime.py 로직 기반 워커"""
        print("🔄 제스처 인식 워커 스레드 시작됨")
        frame_count = 0
        
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    print("🛑 워커 스레드 종료 신호 수신")
                    break
                
                frame, frame_id, elapsed_time, latest_detections = frame_data
                frame_count += 1
                
                # predict_webcam_realtime.py와 동일한 처리
                all_keypoints = None
                keypoints_detected = False
                
                # YOLO로 전체 프레임에서 포즈 검출 (predict_webcam_realtime.py와 동일)
                results = self.yolo_model(frame, verbose=False)
                if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                    all_keypoints = results[0].keypoints.data[0].cpu().numpy()
                    selected_keypoints = all_keypoints[self.joint_indices]
                    
                    # predict_webcam_realtime.py와 동일한 검증
                    if np.all(selected_keypoints[:, 2] > 0.3):
                        normalized = self.normalize_keypoints(selected_keypoints)
                        if normalized is not None:
                            self.keypoint_buffer.append(normalized)
                            keypoints_detected = True
                
                # predict_webcam_realtime.py와 동일: 매 프레임 예측
                prediction, confidence = self.predict_gesture()
                
                if prediction is not None:
                    gesture_name = self.action_labels[prediction]
                    
                    # predict_webcam_realtime.py와 동일한 변경 감지
                    if self.last_prediction != prediction:
                        print(f"🎯 제스처 변경: {gesture_name} (신뢰도: {confidence:.3f})")
                        self.last_prediction = prediction
                    
                    # 결과 업데이트
                    with self.lock:
                        self.latest_gesture = (gesture_name, confidence, keypoints_detected, all_keypoints)
                        self.current_gesture = gesture_name
                        self.current_confidence = confidence

                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 제스처 인식 워커 오류: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def get_latest_gesture(self):
        """최신 제스처 결과 반환"""
        with self.lock:
            return self.latest_gesture
    
    def add_frame(self, frame, frame_id, elapsed_time, latest_detections):
        """프레임 추가 (비동기)"""
        try:
            self.frame_queue.put_nowait((frame, frame_id, elapsed_time, latest_detections))
        except queue.Full:
            if frame_id % 120 == 0:
                print(f"⚠️ 제스처 인식기 큐가 가득참 - 프레임 {frame_id} 건너뜀")
        except Exception as e:
            if frame_id % 120 == 0:
                print(f"❌ 제스처 인식기 프레임 전달 오류: {e}")
    
    def stop(self):
        """인식기 중지"""
        self.running = False
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join()
        print("🛑 Gesture Recognizer 중지") 
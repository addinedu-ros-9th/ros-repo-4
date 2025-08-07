"""
실시간 웹캠 제스처 분류
- 웹캠 피드를 받아 실시간으로 관절점 추출
- 30프레임 누적 시마다 즉시 예측 및 결과 표시
- Matplotlib 창으로 실시간 결과 확인
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import os
from collections import deque
from train_sliding_shift_gcn import SlidingShiftGCN

# RealtimeGesturePredictor 클래스는 그대로 유지됩니다.
class RealtimeGesturePredictor:
    """실시간 제스처 분류기 (이전 코드 재사용)"""
    
    def __init__(self, model_path, yolo_path='yolov8n-pose.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 사용 디바이스: {self.device}")
        
        self.yolo_model = YOLO(yolo_path)
        if torch.cuda.is_available():
            self.yolo_model.to(self.device)
        
        self.joint_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12]
        self.num_joints = len(self.joint_indices)
        
        self.load_shift_gcn_model(model_path)
        
        self.window_size = 30
        self.keypoint_buffer = deque(maxlen=self.window_size)
        
        self.action_labels = {0: 'COME', 1: 'NORMAL'}
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}
        self.last_prediction = None
        
        print("✅ 실시간 예측기 초기화 완료")
    
    def load_shift_gcn_model(self, model_path):
        adj_matrix_path = os.path.join(os.path.dirname(model_path), '../shift_gcn_data_sliding/adjacency_matrix.npy')
        adjacency_matrix = np.load(adj_matrix_path) if os.path.exists(adj_matrix_path) else np.eye(self.num_joints)
        
        self.model = SlidingShiftGCN(num_classes=2, num_joints=self.num_joints, dropout=0.3)
        self.model.set_adjacency_matrix(adjacency_matrix)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 모델 로드 완료: {model_path}")

    def normalize_keypoints(self, keypoints):
        normalized = keypoints.copy()
        valid_joints = keypoints[:, 2] > 0
        if not np.any(valid_joints): return None
        valid_points = keypoints[valid_joints, :2]
        center = np.mean(valid_points, axis=0)
        scale = np.std(valid_points)
        if scale == 0: scale = 1.0
        normalized[:, :2] = (keypoints[:, :2] - center) / scale
        return normalized

    def predict_gesture(self):
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
        # 이 함수는 이제 BGR 프레임을 직접 처리합니다.
        if keypoints is not None:
            connections = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)]
            for i, j in connections:
                idx1, idx2 = self.joint_indices[i], self.joint_indices[j]
                if keypoints[idx1, 2] > 0.3 and keypoints[idx2, 2] > 0.3:
                    p1 = (int(keypoints[idx1, 0]), int(keypoints[idx1, 1]))
                    p2 = (int(keypoints[idx2, 0]), int(keypoints[idx2, 1]))
                    cv2.line(frame, p1, p2, (255, 255, 255), 2)
            for idx in self.joint_indices:
                if keypoints[idx, 2] > 0.3:
                    center = (int(keypoints[idx, 0]), int(keypoints[idx, 1]))
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)

        if prediction is None:
            text = f"Collecting frames... [{len(self.keypoint_buffer)}/{self.window_size}]"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            label = self.action_labels[prediction]
            color = self.colors[prediction] # BGR 색상
            text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (10, 10), (350, 60), color, -1)
            cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            if self.last_prediction != prediction:
                print(f"--> 예측 변경: {label} (신뢰도: {confidence:.2f})")
                self.last_prediction = prediction
        return frame

    def process_frame(self, frame):
        # BGR 프레임을 직접 받아서 처리
        results = self.yolo_model(frame, verbose=False)
        all_keypoints = None
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            all_keypoints = results[0].keypoints.data[0].cpu().numpy()
            selected_keypoints = all_keypoints[self.joint_indices]
            if np.all(selected_keypoints[:, 2] > 0.3):
                normalized = self.normalize_keypoints(selected_keypoints)
                if normalized is not None:
                    self.keypoint_buffer.append(normalized)
        
        prediction, confidence = self.predict_gesture()
        vis_frame = self.draw_visualization(frame, all_keypoints, prediction, confidence)
        return vis_frame

def run_webcam_prediction_cv2(video_source=4):
    """cv2.imshow를 사용하여 실시간 예측 실행"""
    model_path = "./models/sliding_shift_gcn_model.pth"
    predictor = RealtimeGesturePredictor(model_path)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ 웹캠을 열 수 없습니다: {video_source}")
        return
        
    print("\n🚀 실시간 웹캠 예측을 시작합니다. 'q'를 누르면 종료됩니다.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ℹ️ 웹캠 스트림이 종료되었습니다.")
            break
        
        # 프레임 처리 (BGR 프레임 전달)
        processed_frame = predictor.process_frame(frame)
        
        # 화면에 표시
        cv2.imshow('Realtime Gesture Prediction - Press Q to Exit', processed_frame)
        
        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 실시간 예측 종료")

if __name__ == "__main__":
    run_webcam_prediction_cv2(2) # 1번 웹캠 사용 
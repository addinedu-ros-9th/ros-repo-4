"""
실시간 제스처 예측 결과를 동영상 파일로 저장
- `RealtimeGesturePredictor`를 사용하여 프레임별로 예측 수행
- 처리된 각 프레임을 `cv2.VideoWriter`를 사용하여 동영상으로 저장
- 새로운 비디오 샘플을 사용하여 테스트
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import os
import glob
from collections import deque
from train_sliding_shift_gcn import SlidingShiftGCN

class RealtimeGesturePredictor:
    """실시간 제스처 분류기 (predict_realtime.py에서 가져옴)"""
    
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
            color = self.colors[prediction]
            text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (10, 10), (350, 60), color, -1)
            cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            if self.last_prediction != prediction:
                print(f"--> 예측 변경: {label} (신뢰도: {confidence:.2f})")
                self.last_prediction = prediction
        return frame

    def process_frame(self, frame):
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
        frame = self.draw_visualization(frame, all_keypoints, prediction, confidence)
        return frame

def save_prediction_to_video(predictor, video_path, output_path):
    """예측 과정을 비디오 파일로 저장"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        return

    # 비디오 정보 추출 및 VideoWriter 설정
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n🎬 영상 처리 시작: {os.path.basename(video_path)}")
    print(f"   -> 저장 경로: {output_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = predictor.process_frame(frame)
        out.write(processed_frame)
        frame_count += 1

    cap.release()
    out.release()
    predictor.keypoint_buffer.clear() # 다음 영상을 위해 버퍼 초기화
    print(f"✅ 영상 저장 완료 ({frame_count} 프레임)")

if __name__ == "__main__":
    model_path = "./models/sliding_shift_gcn_model.pth"
    output_dir = "./visualization_results/all_videos"  # 결과 저장을 위한 하위 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 예측기 인스턴스 생성
    predictor = RealtimeGesturePredictor(model_path)
    
    # 처리할 모든 비디오 찾기
    all_videos = sorted(glob.glob("./pose_dataset/**/*.avi", recursive=True))

    if not all_videos:
        print("❌ 처리할 비디오를 찾을 수 없습니다.")
    else:
        print(f"✅ 총 {len(all_videos)}개의 비디오에 대한 예측을 시작합니다.")
        
        for video_path in all_videos:
            # 출력 파일 경로 설정 (하위 폴더 구조 유지)
            relative_path = os.path.relpath(video_path, "./pose_dataset")
            output_filename = f"predicted_{os.path.splitext(relative_path.replace(os.sep, '_'))[0]}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            save_prediction_to_video(predictor, video_path, output_path)

        print("\n🎉 모든 영상 처리가 완료되었습니다!")
        print(f"📁 저장 폴더: {output_dir}") 
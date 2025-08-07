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

class GestureRecognizer:
    """제스처 인식 시스템"""
    
    def __init__(self):
        print("🚀 Gesture Recognizer 초기화")
        
        # YOLO Pose 모델 (경로 수정)
        pose_model_path = '/home/ckim/ros-repo-4/deeplearning/yolov8n-pose.pt'  # 절대 경로로 수정
        if os.path.exists(pose_model_path):
            try:
                self.pose_model = YOLO(pose_model_path)
                print(f"✅ YOLO Pose 모델 로드 성공: {pose_model_path}")
                
                # 모델 테스트
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                test_results = self.pose_model(test_image, verbose=False)
                print(f"✅ YOLO Pose 모델 테스트 성공: {len(test_results)} 결과")
                
            except Exception as e:
                print(f"❌ YOLO Pose 모델 로드 실패: {e}")
                self.pose_model = None
        else:
            print(f"❌ YOLO Pose 모델 파일 없음: {pose_model_path}")
            # 상대 경로로 시도
            try:
                self.pose_model = YOLO('yolov8n-pose.pt')
                print("⚠️ 상대 경로로 YOLO Pose 모델 로드 성공")
            except Exception as e:
                print(f"❌ 상대 경로 YOLO Pose 모델 로드 실패: {e}")
                self.pose_model = None
        
        # Shift-GCN 모델 로드
        self.load_gesture_model()
        
        # 제스처 인식 설정 (SlidingShiftGCN 모델에 맞춤)
        self.gesture_frame_buffer = deque(maxlen=30)  # 90 → 30으로 변경 (SlidingShiftGCN 모델에 맞춤)
        self.actions = ['COME', 'NORMAL']  # 2클래스
        self.min_gesture_frames = 30  # 90 → 30으로 변경 (SlidingShiftGCN 모델에 맞춤)
        
        # 1초 단위 판단 설정 (SlidingShiftGCN 모델에 맞춤)
        self.gesture_decision_interval = 30  # 90 → 30으로 변경 (1초마다 판단)
        self.last_gesture_decision_frame = 0
        self.current_gesture_confidence = 0.5
        
        # COME 제스처 인식 개선 설정 (임계값 조정)
        self.come_detection_threshold = 0.08  # 0.10 → 0.08로 더 낮춤 (COME 감지 개선)
        self.normal_detection_threshold = 0.30  # 0.35 → 0.30로 낮춤 (더 관대하게)
        self.min_keypoints_for_gesture = 2  # 3 → 2로 낮춤 (더 관대하게)
        
        # 실시간 감지 설정 (SlidingShiftGCN 모델에 맞춰 활성화)
        self.realtime_detection_enabled = True  # 실시간 감지 활성화 (30프레임 학습 모델에 맞춤)
        self.realtime_confidence_threshold = 0.20  # 0.25 → 0.20로 낮춤 (더 관대하게)
        
        # 상체 관절점 (YOLO Pose 17개 중 상체 9개)
        self.upper_body_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]  # 코+어깨+팔+엉덩이
        
        # 인접 행렬 생성
        self.adjacency_matrix = self.create_adjacency_matrix()
        if hasattr(self, 'gesture_model'):
            self.gesture_model.set_adjacency_matrix(self.adjacency_matrix)
        
        # 쓰레드 설정
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.lock = threading.Lock()
        
        # 결과 저장
        self.latest_gesture = ("NORMAL", self.current_gesture_confidence, False, None)
        self.last_valid_keypoints = None
        
        # 워커 스레드 자동 시작
        self.worker_thread = threading.Thread(target=self.recognition_worker)
        self.worker_thread.start()
        print(f"✅ 워커 스레드 시작됨 (스레드 ID: {self.worker_thread.ident})")
        
        print("✅ Gesture Recognizer 초기화 완료")
    
    def start(self):
        """인식기 시작"""
        self.worker_thread = threading.Thread(target=self.recognition_worker)
        self.worker_thread.start()
        print("🚀 Gesture Recognizer 시작")
    
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
    
    def load_gesture_model(self):
        """Shift-GCN 모델 로드"""
        model_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/models/sliding_shift_gcn_model.pth'
        
        if os.path.exists(model_path):
            self.gesture_model = SlidingShiftGCN(num_classes=2, num_joints=9, dropout=0.3)
            self.gesture_model.load_state_dict(torch.load(model_path, map_location=device))
            self.gesture_model = self.gesture_model.to(device)
            self.gesture_model.eval()
            self.enable_gesture_recognition = True
            print(f"✅ SlidingShift-GCN 모델 로드 성공: {model_path}")
            
            # 학습 시 사용된 인접 행렬 로드 시도
            adj_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/shift_gcn_data_sliding/adjacency_matrix.npy'
            if os.path.exists(adj_path):
                self.adjacency_matrix = np.load(adj_path)
                print(f"✅ 학습 시 인접 행렬 로드: {self.adjacency_matrix.shape}")
            else:
                print("⚠️ 학습 시 인접 행렬 없음 - 기본값 사용")
                self.adjacency_matrix = self.create_adjacency_matrix()
            
            self.gesture_model.set_adjacency_matrix(self.adjacency_matrix)
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
            self.enable_gesture_recognition = False
    
    def create_adjacency_matrix(self):
        """인접 행렬 생성 (학습 시와 동일한 방식)"""
        num_joints = len(self.upper_body_joints)
        A = np.zeros((num_joints, num_joints))
        
        # self-connection (대각선)
        for i in range(num_joints):
            A[i, i] = 1
        
        # 관절점 연결 정의 (학습 시와 동일)
        connections = [
            (0, 1), (0, 2),  # nose - shoulders
            (1, 2),  # shoulders
            (1, 3), (2, 4),  # shoulders - elbows
            (3, 5), (4, 6),  # elbows - wrists
            (1, 7), (2, 8),  # shoulders - hips
            (7, 8),  # hips
        ]
        
        # 매핑: 원본 관절점 → 상체 배열 인덱스
        joint_mapping = {joint: i for i, joint in enumerate(self.upper_body_joints)}
        
        for joint1, joint2 in connections:
            if joint1 in joint_mapping and joint2 in joint_mapping:
                i, j = joint_mapping[joint1], joint_mapping[joint2]
                A[i, j] = 1
                A[j, i] = 1
        
        return A
    
    def normalize_keypoints(self, keypoints):
        """키포인트 정규화 (학습 시와 완전히 동일한 방식)"""
        if keypoints.shape[0] == 0:
            return keypoints
        
        normalized_keypoints = keypoints.copy()
        
        for t in range(keypoints.shape[0]):
            frame_keypoints = keypoints[t]  # (V, 3)
            
            # 유효한 관절점만 사용 (학습 시와 동일한 임계값)
            valid_joints = frame_keypoints[:, 2] > 0.3  # 0.1 → 0.3으로 복원 (학습 시와 동일)
            
            if np.any(valid_joints):
                valid_points = frame_keypoints[valid_joints]
                
                # 중심점 계산 (유효한 관절점들의 평균) - 학습 시와 동일
                center_x = np.mean(valid_points[:, 0])
                center_y = np.mean(valid_points[:, 1])
                
                # 스케일 계산 (유효한 관절점들의 표준편차) - 학습 시와 동일
                scale = np.std(valid_points[:, :2])
                if scale == 0:
                    scale = 1.0
                
                # 정규화 적용 (학습 시와 동일한 방식)
                normalized_keypoints[t, :, 0] = (frame_keypoints[:, 0] - center_x) / scale
                normalized_keypoints[t, :, 1] = (frame_keypoints[:, 1] - center_y) / scale
                # confidence는 그대로 유지
            else:
                # 유효한 관절점이 없으면 기본값 사용
                normalized_keypoints[t, :, 0] = 0.0
                normalized_keypoints[t, :, 1] = 0.0
        
        return normalized_keypoints
    
    def normalize_keypoints_with_training_stats(self, keypoints):
        """키포인트 정규화 (학습 데이터 통계 사용)"""
        if keypoints is None:
            return keypoints
        
        # 학습 데이터 통계 (실제 학습된 데이터 기반)
        # COME: 평균=0.315, 표준편차=0.798, 범위=[-2.396, 2.119]
        # NORMAL: 평균=0.307, 표준편차=0.699, 범위=[-2.566, 1.955]
        
        # 학습 데이터의 전체 통계
        training_mean = 0.311  # (0.315 + 0.307) / 2
        training_std = 0.749   # (0.798 + 0.699) / 2
        
        normalized_keypoints = keypoints.copy()
        
        for t in range(keypoints.shape[0]):
            frame_keypoints = keypoints[t]  # (V, 3)
            
            # 유효한 관절점만 사용
            valid_joints = frame_keypoints[:, 2] > 0.3
            
            if np.any(valid_joints):
                valid_points = frame_keypoints[valid_joints]
                
                # 중심점 계산
                center_x = np.mean(valid_points[:, 0])
                center_y = np.mean(valid_points[:, 1])
                
                # 스케일 계산
                scale = np.std(valid_points[:, :2])
                if scale == 0:
                    scale = 1.0
                
                # 정규화 적용
                normalized_keypoints[t, :, 0] = (frame_keypoints[:, 0] - center_x) / scale
                normalized_keypoints[t, :, 1] = (frame_keypoints[:, 1] - center_y) / scale
                
                # 학습 데이터 통계에 맞춰 추가 정규화
                normalized_keypoints[t, :, 0] = (normalized_keypoints[t, :, 0] - training_mean) / training_std
                normalized_keypoints[t, :, 1] = (normalized_keypoints[t, :, 1] - training_mean) / training_std
            else:
                normalized_keypoints[t, :, 0] = 0.0
                normalized_keypoints[t, :, 1] = 0.0
        
        return normalized_keypoints
    
    def normalize_keypoints_exact_training(self, keypoints):
        """키포인트 정규화 (학습 시와 완전히 동일)"""
        if keypoints is None:
            return keypoints
        
        normalized_keypoints = keypoints.copy()
        
        for t in range(keypoints.shape[0]):
            frame_keypoints = keypoints[t]  # (V, 3)
            
            # 유효한 관절점만 사용 (학습 시와 동일한 임계값)
            valid_joints = frame_keypoints[:, 2] > 0.3  # 0.1 → 0.3으로 복원 (학습 시와 동일)
            
            if np.any(valid_joints):
                valid_points = frame_keypoints[valid_joints]
                
                # 중심점 계산 (유효한 관절점들의 평균) - 학습 시와 동일
                center_x = np.mean(valid_points[:, 0])
                center_y = np.mean(valid_points[:, 1])
                
                # 스케일 계산 (유효한 관절점들의 표준편차) - 학습 시와 동일
                scale = np.std(valid_points[:, :2])
                if scale == 0:
                    scale = 1.0
                
                # 정규화 적용 (학습 시와 동일한 방식)
                normalized_keypoints[t, :, 0] = (frame_keypoints[:, 0] - center_x) / scale
                normalized_keypoints[t, :, 1] = (frame_keypoints[:, 1] - center_y) / scale
                # confidence는 그대로 유지
            else:
                # 유효한 관절점이 없으면 기본값 사용
                normalized_keypoints[t, :, 0] = 0.0
                normalized_keypoints[t, :, 1] = 0.0
        
        return normalized_keypoints
    
    def normalize_keypoints_distance_compensated(self, keypoints):
        """키포인트 정규화 (거리 차이 보상)"""
        if keypoints is None:
            return keypoints
        
        normalized_keypoints = keypoints.copy()
        
        for t in range(keypoints.shape[0]):
            frame_keypoints = keypoints[t]  # (V, 3)
            
            # 유효한 관절점만 사용
            valid_joints = frame_keypoints[:, 2] > 0.3
            
            if np.any(valid_joints):
                valid_points = frame_keypoints[valid_joints]
                
                # 중심점 계산
                center_x = np.mean(valid_points[:, 0])
                center_y = np.mean(valid_points[:, 1])
                
                # 스케일 계산 (거리 보상 적용)
                scale = np.std(valid_points[:, :2])
                if scale == 0:
                    scale = 1.0
                
                # 거리 보상: 현재 스케일을 학습 시 스케일로 조정
                # 학습 시 평균 스케일: 약 0.8, 현재 스케일: 약 170
                # 보상 계수 = 학습_스케일 / 현재_스케일 (더 보수적으로)
                if scale > 50:  # 스케일이 매우 큰 경우만 보상
                    compensation_factor = min(0.8 / scale, 0.1)  # 최대 0.1로 제한
                elif scale > 10:  # 중간 스케일
                    compensation_factor = min(0.8 / scale, 0.3)  # 최대 0.3으로 제한
                else:  # 작은 스케일
                    compensation_factor = 1.0  # 보상 없음
                
                adjusted_scale = scale * compensation_factor
                
                # 정규화 적용 (거리 보상 포함)
                normalized_keypoints[t, :, 0] = (frame_keypoints[:, 0] - center_x) / adjusted_scale
                normalized_keypoints[t, :, 1] = (frame_keypoints[:, 1] - center_y) / adjusted_scale
                # confidence는 그대로 유지
            else:
                normalized_keypoints[t, :, 0] = 0.0
                normalized_keypoints[t, :, 1] = 0.0
        
        return normalized_keypoints
    
    def draw_visualization(self, frame, keypoints, prediction, confidence):
        """관절점 시각화 (predict_webcam_realtime.py와 동일한 방식)"""
        if keypoints is not None:
            # 관절점 연결선 정의 (predict_webcam_realtime.py와 동일)
            connections = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)]
            for i, j in connections:
                idx1, idx2 = self.upper_body_joints[i], self.upper_body_joints[j]
                if keypoints[idx1, 2] > 0.3 and keypoints[idx2, 2] > 0.3:
                    p1 = (int(keypoints[idx1, 0]), int(keypoints[idx1, 1]))
                    p2 = (int(keypoints[idx2, 0]), int(keypoints[idx2, 1]))
                    cv2.line(frame, p1, p2, (255, 255, 255), 2)
        
            # 관절점 그리기 (predict_webcam_realtime.py와 동일)
            for idx in self.upper_body_joints:
                if keypoints[idx, 2] > 0.3:
                    center = (int(keypoints[idx, 0]), int(keypoints[idx, 1]))
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)

        # 제스처 예측 결과 표시 (COME: 빨간색, NORMAL: 초록색)
        if prediction is None:
            text = f"Collecting frames... [{len(self.gesture_frame_buffer)}/{self.min_gesture_frames}]"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            label = prediction  # 이미 문자열로 되어 있음
            # COME: 빨간색, NORMAL: 초록색 (BGR 색상)
            if label == "COME":
                color = (0, 0, 255)  # 빨간색
            else:  # NORMAL
                color = (0, 255, 0)  # 초록색
            
            text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (10, 10), (350, 60), color, -1)
            cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return frame
    
    def recognition_worker(self):
        """제스처 인식 워커 (가장 큰 사람의 pose만 감지)"""
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
                
                # 프레임 수신 확인 (60프레임마다)
                if frame_count % 60 == 0:
                    print(f"🔄 워커 스레드: 프레임 {frame_id} 처리 중 (총 {frame_count}개 처리됨)")
                    print(f"   - 프레임 크기: {frame.shape}")
                    print(f"   - 감지된 사람: {len(latest_detections)}명")
                    print(f"   - 제스처 인식 활성화: {self.enable_gesture_recognition}")
                
                if not self.enable_gesture_recognition:
                    self.frame_queue.task_done()
                    continue
                
                # 기본값 유지 (새로운 판단이 없으면 이전 값 유지)
                with self.lock:
                    prediction, confidence, keypoints_detected, current_keypoints = self.latest_gesture
        
                # 가장 큰 사람(주요 대상) 찾기
                target_person = None
                if latest_detections:
                    # 면적이 가장 큰 사람 선택
                    target_person = max(latest_detections, key=lambda p: 
                        (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))
                    
                    if frame_id % 60 == 0:  # 60프레임마다 디버깅
                        area = (target_person['bbox'][2] - target_person['bbox'][0]) * (target_person['bbox'][3] - target_person['bbox'][1])
                        print(f"🎯 주요 대상: {target_person['id']} (면적: {area:.0f})")
                        print(f"   - 바운딩 박스: {target_person['bbox']}")
                else:
                    if frame_id % 60 == 0:
                        print(f"🎯 감지된 사람 없음")
                
                # YOLO Pose로 키포인트 추출 (가장 큰 사람만)
                if target_person is not None:
                    # 주요 대상의 바운딩 박스로 ROI 설정
                    x1, y1, x2, y2 = map(int, target_person['bbox'])
                    
                    # ROI 확장 (더 넓은 영역에서 pose 감지)
                    margin = 100  # 50 → 100으로 늘림
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    if frame_id % 60 == 0:
                        print(f"   - ROI 설정: [{x1}, {y1}, {x2}, {y2}] (마진: {margin})")
                    
                    # ROI 추출
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:  # ROI가 유효한 경우
                        if frame_id % 60 == 0:
                            print(f"   - ROI 크기: {roi.shape}")
                        
                        # ROI에서 pose 감지 (설정 개선)
                        results = self.pose_model(roi, imgsz=256, conf=0.01,  # 0.05 → 0.01로 더 낮춤
                                                verbose=False, device=0)
                        
                        keypoints_data = []
                        
                        for result in results:
                            if result.keypoints is not None:
                                keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                                
                                if frame_id % 60 == 0:  # 60프레임마다 디버깅
                                    print(f"   🔍 YOLO Pose 결과: {len(keypoints)}명 감지")
                                
                                for person_idx, person_kpts in enumerate(keypoints):
                                    # 상체 관절점만 추출
                                    upper_body_kpts = person_kpts[self.upper_body_joints]  # (9, 3)
                                    
                                    # 신뢰도 체크 (임계값 낮춤)
                                    valid_joints = upper_body_kpts[:, 2] >= 0.01  # 0.05 → 0.01로 더 낮춤
                                    valid_count = np.sum(valid_joints)
                                    
                                    if frame_id % 60 == 0:
                                        print(f"   주요 대상 {target_person['id']}: {valid_count}/9 키포인트")
                                        print(f"      - 어깨: {person_kpts[5][2]:.2f}/{person_kpts[6][2]:.2f}")
                                        print(f"      - 팔꿈치: {person_kpts[7][2]:.2f}/{person_kpts[8][2]:.2f}")
                                        print(f"      - 손목: {person_kpts[9][2]:.2f}/{person_kpts[10][2]:.2f}")
                                    
                                    if valid_count >= self.min_keypoints_for_gesture:  # 3 → 2로 낮춤
                                        # ROI 좌표를 전체 프레임 좌표로 변환
                                        person_kpts[:, 0] += x1
                                        person_kpts[:, 1] += y1
                                        
                                        keypoints_data.append(upper_body_kpts)
                                        keypoints_detected = True
                                        current_keypoints = person_kpts  # 전체 17개 키포인트 저장 (시각화용)
                                        
                                        if frame_id % 60 == 0:
                                            print(f"   ✅ 키포인트 감지 성공! {valid_count}개 유효")
                                        break  # 첫 번째 사람만 사용
                                    else:
                                        if frame_id % 60 == 0:
                                            print(f"   ⚠️ 키포인트 부족: {valid_count}/{self.min_keypoints_for_gesture} (임계값 미달)")
                            else:
                                if frame_id % 60 == 0:
                                    print(f"   ❌ YOLO Pose 결과에 키포인트 없음")
                        
                        # 키포인트가 감지되지 않으면 더 자세한 디버깅
                        if len(keypoints_data) == 0 and frame_id % 60 == 0:
                            print(f"   ❌ 키포인트 감지 실패 - ROI 크기: {roi.shape}")
                            print(f"      - 바운딩 박스: [{x1}, {y1}, {x2}, {y2}]")
                            print(f"      - ROI 크기: {roi.size}")
                            print(f"      - YOLO Pose 결과 개수: {len(results)}")
                            if len(results) > 0:
                                print(f"      - 첫 번째 결과 키포인트: {results[0].keypoints is not None}")
                        
                        # 키포인트가 감지되면 처리
                        if len(keypoints_data) > 0:
                            current_keypoints_data = keypoints_data[0]
                            keypoints_detected = True
                            current_keypoints = current_keypoints  # 전체 17개 키포인트 저장 (시각화용)
                        else:
                            # 키포인트가 감지되지 않으면 제스처 판단 중단
                            if frame_id % 60 == 0:
                                print(f"   ⚠️ 주요 대상 키포인트 부족 - 제스처 판단 중단")
                            keypoints_detected = False
                            current_keypoints = None
                            # 이전 프레임 데이터 사용하지 않음
                            current_keypoints_data = None
                        
                        # 유효한 키포인트 저장
                        if len(keypoints_data) > 0:
                            self.last_valid_keypoints = keypoints_data[0]
                    else:
                        # ROI가 유효하지 않은 경우
                        if frame_id % 60 == 0:
                            print(f"   ❌ ROI가 유효하지 않음 - 크기: {roi.size}")
                        keypoints_detected = False
                        current_keypoints = None
                        current_keypoints_data = None
                else:
                    # 감지된 사람이 없는 경우
                    if frame_id % 60 == 0:
                        print(f"   ❌ 감지된 사람 없음 - 키포인트 감지 불가")
                    keypoints_detected = False
                    current_keypoints = None
                    current_keypoints_data = None
                
                # 제스처 분석 (키포인트가 충분히 감지된 경우에만)
                if keypoints_detected and current_keypoints_data is not None:
                    # 최소 4개 이상의 키포인트가 있어야 제스처 판단 (6 → 4로 낮춤)
                    valid_joints = current_keypoints_data[:, 2] >= 0.05  # 0.1 → 0.05로 낮춤
                    valid_count = np.sum(valid_joints)
                    
                    if valid_count >= 4:  # 최소 4개 키포인트 필요 (6 → 4로 낮춤)
                        self.gesture_frame_buffer.append(current_keypoints_data)
                        
                        if frame_id % 60 == 0:
                            print(f"   ✅ 제스처 분석 진행: {valid_count}/9 키포인트")
                    else:
                        if frame_id % 60 == 0:
                            print(f"   ⚠️ 키포인트 부족으로 제스처 분석 중단: {valid_count}/4 (최소 필요)")
                else:
                    if frame_id % 60 == 0:
                        print(f"   ❌ 키포인트 없음으로 제스처 분석 중단")
                
                # 실시간 감지 (SlidingShiftGCN 모델에 맞춰 활성화)
                if (self.realtime_detection_enabled and 
                    len(self.gesture_frame_buffer) >= 30 and
                    keypoints_detected and current_keypoints_data is not None):
                    
                    print(f"🎯 [Frame {frame_id}] 실시간 제스처 판단 시작!")
                    
                    try:
                        # 키포인트 시퀀스 전처리 (SlidingShiftGCN 모델에 맞춤)
                        keypoints_sequence = list(self.gesture_frame_buffer)
                        keypoints_array = np.array(keypoints_sequence)  # (T, 9, 3)
                        
                        # 키포인트 품질 검증
                        valid_frames = []
                        for i, frame_kpts in enumerate(keypoints_array):
                            valid_joints = frame_kpts[:, 2] >= 0.01
                            valid_count = np.sum(valid_joints)
                            if valid_count >= 4:
                                valid_frames.append(frame_kpts)
                        
                        if len(valid_frames) < 30:  # 최소 30프레임 필요 (SlidingShiftGCN 모델에 맞춤)
                            print(f"   ⚠️ 유효한 키포인트 프레임 부족: {len(valid_frames)}/30")
                            self.frame_queue.task_done()
                            continue
                        
                        # 유효한 프레임들만 사용
                        keypoints_array = np.array(valid_frames)
                        
                        # SlidingShiftGCN 모델에 맞는 전처리
                        T, V, C = keypoints_array.shape
                        target_frames = 30  # SlidingShiftGCN 모델은 30프레임
                    
                        if T != target_frames:
                            old_indices = np.linspace(0, T-1, T)
                            new_indices = np.linspace(0, T-1, target_frames)
                            
                            resampled_keypoints = np.zeros((target_frames, V, C))
                            for v in range(V):
                                for c in range(C):
                                    resampled_keypoints[:, v, c] = np.interp(new_indices, old_indices, keypoints_array[:, v, c])
                            
                            keypoints_array = resampled_keypoints
                        
                        # 정규화 적용 (중심점 기반)
                        normalized_keypoints = self.normalize_keypoints(keypoints_array)
                        
                        # SlidingShiftGCN 입력 형태로 변환
                        shift_gcn_data = np.zeros((C, target_frames, V, 1))
                        shift_gcn_data[:, :, :, 0] = normalized_keypoints.transpose(2, 0, 1)
                        
                        # 모델 예측
                        input_tensor = torch.FloatTensor(shift_gcn_data).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            self.gesture_model.eval()
                            output = self.gesture_model(input_tensor)
                            probabilities = F.softmax(output, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0, prediction].item()
                        
                        # 결과 해석
                        gesture_name = self.actions[prediction]
                        
                        # 실시간 판단 결과 업데이트
                        with self.lock:
                            self.latest_gesture = (gesture_name, confidence, keypoints_detected, current_keypoints)
                            self.current_gesture_confidence = confidence
                        
                        print(f"🎯 실시간 제스처 결과: {gesture_name} (신뢰도: {confidence:.3f})")
                        
                        # 버퍼 초기화 (새로운 실시간 구간 시작)
                        self.gesture_frame_buffer.clear()
                        
                    except Exception as e:
                        print(f"❌ 실시간 제스처 인식 오류: {e}")
                
                # 1초(30프레임) 단위로 정기 판단 (SlidingShiftGCN 모델에 맞춤)
                frames_since_last_decision = frame_id - self.last_gesture_decision_frame
                
                if (len(self.gesture_frame_buffer) >= self.min_gesture_frames and 
                    frames_since_last_decision >= self.gesture_decision_interval):
                    
                    print(f"🎯 [Frame {frame_id}] 1초 단위 제스처 판단 시작!")
                    
                    try:
                        # 키포인트 시퀀스 전처리 (SlidingShiftGCN 모델에 맞춤)
                        keypoints_sequence = list(self.gesture_frame_buffer)
                        keypoints_array = np.array(keypoints_sequence)  # (T, 9, 3)
                        
                        # 키포인트 품질 검증
                        valid_frames = []
                        for i, frame_kpts in enumerate(keypoints_array):
                            valid_joints = frame_kpts[:, 2] >= 0.01
                            valid_count = np.sum(valid_joints)
                            if valid_count >= 4:
                                valid_frames.append(frame_kpts)
                        
                        if len(valid_frames) < 30:  # 최소 30프레임 필요 (SlidingShiftGCN 모델에 맞춤)
                            print(f"   ⚠️ 유효한 키포인트 프레임 부족: {len(valid_frames)}/30")
                            self.frame_queue.task_done()
                            continue
                        
                        # 유효한 프레임들만 사용
                        keypoints_array = np.array(valid_frames)
                        
                        # SlidingShiftGCN 모델에 맞는 전처리
                        T, V, C = keypoints_array.shape
                        target_frames = 30  # SlidingShiftGCN 모델은 30프레임
                    
                        if T != target_frames:
                            old_indices = np.linspace(0, T-1, T)
                            new_indices = np.linspace(0, T-1, target_frames)
                            
                            resampled_keypoints = np.zeros((target_frames, V, C))
                            for v in range(V):
                                for c in range(C):
                                    resampled_keypoints[:, v, c] = np.interp(new_indices, old_indices, keypoints_array[:, v, c])
                            
                            keypoints_array = resampled_keypoints
                        
                        # 정규화 적용 (중심점 기반)
                        normalized_keypoints = self.normalize_keypoints(keypoints_array)
                        
                        # SlidingShiftGCN 입력 형태로 변환
                        shift_gcn_data = np.zeros((C, target_frames, V, 1))
                        shift_gcn_data[:, :, :, 0] = normalized_keypoints.transpose(2, 0, 1)
                        
                        # 모델 예측
                        input_tensor = torch.FloatTensor(shift_gcn_data).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            self.gesture_model.eval()
                            output = self.gesture_model(input_tensor)
                            probabilities = F.softmax(output, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0, prediction].item()
                        
                        # 결과 해석
                        gesture_name = self.actions[prediction]
                        
                        # 정기 판단 결과 업데이트
                        with self.lock:
                            self.latest_gesture = (gesture_name, confidence, keypoints_detected, current_keypoints)
                            self.current_gesture_confidence = confidence
                        
                        self.last_gesture_decision_frame = frame_id
                        
                        print(f"🎯 1초 제스처 결과: {gesture_name} (신뢰도: {confidence:.3f})")
                        
                        # 버퍼 초기화 (새로운 1초 구간 시작)
                        self.gesture_frame_buffer.clear()
                        
                    except Exception as e:
                        print(f"❌ 1초 제스처 인식 오류: {e}")
                
                # 결과 저장 (키포인트 정보 포함)
                # 키포인트가 감지된 경우에만 업데이트
                if keypoints_detected and current_keypoints is not None:
                    # 실시간 감지와 1초 단위 판단 결과 모두 사용 (SlidingShiftGCN 모델에 맞춤)
                    # prediction이 이미 문자열인지 확인
                    if isinstance(prediction, str):
                        prediction_label = prediction
                    else:
                        prediction_label = self.actions[prediction] if prediction is not None else "NORMAL"
                    with self.lock:
                        self.latest_gesture = (prediction_label, confidence, True, current_keypoints)
                # 키포인트가 없으면 이전 값 유지 (업데이트 안 함)
                elif not keypoints_detected and frame_id % 120 == 0:
                    # 키포인트가 감지되지 않을 때 디버깅
                    print(f"   ⚠️ 키포인트 없음 - 이전 결과 유지: {self.latest_gesture[0]} ({self.latest_gesture[1]:.3f})")
                
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
            
            # 프레임 전달 확인 (120프레임마다)
            if frame_id % 120 == 0:
                print(f"📤 제스처 인식기에 프레임 {frame_id} 전달됨")
                print(f"   - 큐 크기: {self.frame_queue.qsize()}")
                print(f"   - 감지된 사람: {len(latest_detections)}명")
                
        except queue.Full:
            if frame_id % 120 == 0:
                print(f"⚠️ 제스처 인식기 큐가 가득참 - 프레임 {frame_id} 건너뜀")
        except Exception as e:
            if frame_id % 120 == 0:
                print(f"❌ 제스처 인식기 프레임 전달 오류: {e}") 
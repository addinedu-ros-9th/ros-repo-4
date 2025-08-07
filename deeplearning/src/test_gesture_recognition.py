"""
학습 데이터에서 랜덤으로 3개 영상을 뽑아서 제스처 인식을 테스트합니다.
시각화와 함께 결과를 보여줍니다.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque
import time
import os
import random
import glob

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleShiftGCN(nn.Module):
    """Shift-GCN 모델"""
    
    def __init__(self, num_classes, num_joints, in_channels=3, dropout=0.5):
        super(SimpleShiftGCN, self).__init__()
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        
        # Adjacency matrix
        self.register_buffer('A', torch.eye(num_joints))
        
        # 입력 정규화
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # 3층 구조
        self.conv1 = nn.Conv2d(in_channels, 32, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 1) 
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Temporal convolution
        self.tcn = nn.Conv2d(128, 128, (3, 1), padding=(1, 0))
        self.tcn_bn = nn.BatchNorm2d(128)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
        
    def set_adjacency_matrix(self, A):
        self.A = torch.FloatTensor(A)
        if next(self.parameters()).is_cuda:
            self.A = self.A.cuda()
    
    def forward(self, x):
        N, C, T, V, M = x.size()
        
        # Focus on first person
        x = x[:, :, :, :, 0]  # (N, C, T, V)
        
        # Data normalization
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, V, T)
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T)
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, T, V)
        
        # Graph convolution layers
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply graph adjacency
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply graph adjacency
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Temporal convolution
        x = self.tcn(x)
        x = self.tcn_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])  # (N, 128, 1, 1)
        x = x.view(N, -1)  # (N, 128)
        
        # Classification
        x = self.fc(x)
        
        return x

class TestGestureRecognizer:
    """제스처 인식 테스트 시스템"""
    
    def __init__(self):
        print("🚀 제스처 인식 테스트 시스템 초기화")
        
        # YOLO Pose 모델
        pose_model_path = '/home/ckim/ros-repo-4/deeplearning/yolov8n-pose.pt'
        if os.path.exists(pose_model_path):
            self.pose_model = YOLO(pose_model_path)
            print(f"✅ YOLO Pose 모델 로드 성공")
        else:
            # 상대 경로로 시도
            try:
                self.pose_model = YOLO('yolov8n-pose.pt')
                print("⚠️ 상대 경로로 YOLO Pose 모델 로드 성공")
            except Exception as e:
                print(f"❌ YOLO Pose 모델 로드 실패: {e}")
                self.pose_model = None
        
        # Shift-GCN 모델 로드
        self.load_gesture_model()
        
        # 제스처 인식 설정
        self.gesture_frame_buffer = deque(maxlen=30)  # 90 → 30으로 줄임 (1초)
        self.actions = ['COME', 'NORMAL']
        self.upper_body_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]  # 코+어깨+팔+엉덩이
        
        # 인접 행렬 생성
        self.adjacency_matrix = self.create_adjacency_matrix()
        if hasattr(self, 'gesture_model'):
            self.gesture_model.set_adjacency_matrix(self.adjacency_matrix)
        
        print("✅ 제스처 인식 테스트 시스템 초기화 완료")
    
    def load_gesture_model(self):
        """Shift-GCN 모델 로드"""
        model_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/models/simple_shift_gcn_model.pth'
        
        if os.path.exists(model_path):
            self.gesture_model = SimpleShiftGCN(num_classes=2, num_joints=9, dropout=0.5)
            self.gesture_model.load_state_dict(torch.load(model_path, map_location=device))
            self.gesture_model = self.gesture_model.to(device)
            self.gesture_model.eval()
            print(f"✅ Shift-GCN 모델 로드 성공")
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
            self.gesture_model = None
    
    def create_adjacency_matrix(self):
        """인접 행렬 생성"""
        num_joints = len(self.upper_body_joints)
        A = np.zeros((num_joints, num_joints))
        
        # self-connection (대각선)
        for i in range(num_joints):
            A[i, i] = 1
        
        # 관절점 연결 정의
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
        """키포인트 정규화 (학습 시와 동일한 방식)"""
        if keypoints.shape[0] == 0:
            return keypoints
        
        normalized_keypoints = keypoints.copy()
        
        for t in range(keypoints.shape[0]):
            frame_keypoints = keypoints[t]  # (V, 3)
            
            # 유효한 관절점만 사용
            valid_joints = frame_keypoints[:, 2] > 0.3
            
            if np.any(valid_joints):
                valid_points = frame_keypoints[valid_joints]
                
                # 중심점 계산 (유효한 관절점들의 평균)
                center_x = np.mean(valid_points[:, 0])
                center_y = np.mean(valid_points[:, 1])
                
                # 스케일 계산 (유효한 관절점들의 표준편차)
                scale = np.std(valid_points[:, :2])
                if scale == 0:
                    scale = 1.0
                
                # 정규화 적용
                normalized_keypoints[t, :, 0] = (frame_keypoints[:, 0] - center_x) / scale
                normalized_keypoints[t, :, 1] = (frame_keypoints[:, 1] - center_y) / scale
                # confidence는 그대로 유지
            else:
                # 유효한 관절점이 없으면 기본값 사용
                normalized_keypoints[t, :, 0] = 0.0
                normalized_keypoints[t, :, 1] = 0.0
        
        return normalized_keypoints
    
    def draw_keypoints(self, frame, keypoints, color=(0, 255, 0)):
        """키포인트 시각화"""
        if keypoints is None:
            return frame
        
        annotated = frame.copy()
        
        # 관절점 연결 정의
        connections = [
            (0, 1), (0, 2),  # nose - shoulders
            (1, 2),  # shoulders
            (1, 3), (2, 4),  # shoulders - elbows
            (3, 5), (4, 6),  # elbows - wrists
            (1, 7), (2, 8),  # shoulders - hips
            (7, 8),  # hips
        ]
        
        # 관절점 그리기
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(annotated, (int(x), int(y)), 3, color, -1)
                cv2.putText(annotated, str(i), (int(x)+5, int(y)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 관절점 연결선 그리기
        for connection in connections:
            start_idx, end_idx = connection
            if (keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(annotated, start_point, end_point, color, 2)
        
        return annotated
    
    def process_video(self, video_path, expected_gesture):
        """비디오 처리 및 제스처 인식"""
        print(f"\n🎬 비디오 처리 시작: {os.path.basename(video_path)}")
        print(f"   예상 제스처: {expected_gesture}")
        print(f"   파일 경로: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 비디오를 열 수 없습니다: {video_path}")
            return None
        
        # 비디오 정보 출력
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"   비디오 정보: {frame_count_total}프레임, {fps:.1f} FPS")
        
        frame_count = 0
        self.gesture_frame_buffer.clear()
        
        # 결과 저장
        results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 30프레임마다 진행상황 출력
            if frame_count % 30 == 0:
                print(f"   처리 중: {frame_count}/{frame_count_total} 프레임")
            
            # YOLO Pose로 키포인트 추출
            results_pose = self.pose_model(frame, imgsz=256, conf=0.01, verbose=False, device=0)
            
            keypoints_detected = False
            current_keypoints = None
            
            for result in results_pose:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                    
                    for person_idx, person_kpts in enumerate(keypoints):
                        # 상체 관절점만 추출
                        upper_body_kpts = person_kpts[self.upper_body_joints]  # (9, 3)
                        
                        # 신뢰도 체크
                        valid_joints = upper_body_kpts[:, 2] >= 0.01
                        valid_count = np.sum(valid_joints)
                        
                        if valid_count >= 4:  # 최소 4개 키포인트 필요
                            keypoints_detected = True
                            current_keypoints = person_kpts  # 전체 17개 키포인트 저장
                            
                            # 제스처 분석을 위해 버퍼에 추가
                            self.gesture_frame_buffer.append(upper_body_kpts)
                            break
                    
                    if keypoints_detected:
                        break
            
            # 90프레임(3초) 단위로 제스처 판단
            if len(self.gesture_frame_buffer) >= 30: # 30프레임(1초) 단위로 판단
                print(f"   🎯 제스처 판단 시작 (프레임 {frame_count})")
                try:
                    # 키포인트 시퀀스 전처리
                    keypoints_sequence = list(self.gesture_frame_buffer)
                    keypoints_array = np.array(keypoints_sequence)  # (T, 9, 3)
                    
                    # 키포인트 품질 검증
                    valid_frames = []
                    for i, frame_kpts in enumerate(keypoints_array):
                        valid_joints = frame_kpts[:, 2] >= 0.01
                        valid_count = np.sum(valid_joints)
                        if valid_count >= 4:
                            valid_frames.append(frame_kpts)
                    
                    print(f"   유효한 키포인트 프레임: {len(valid_frames)}/30")
                    
                    if len(valid_frames) >= 30: # 30프레임(1초) 단위로 판단
                        # 유효한 프레임들만 사용
                        keypoints_array = np.array(valid_frames)
                        
                        # 학습 시와 동일한 전처리
                        T, V, C = keypoints_array.shape
                        target_frames = 30 # 30프레임(1초)
                    
                        if T != target_frames:
                            old_indices = np.linspace(0, T-1, T)
                            new_indices = np.linspace(0, T-1, target_frames)
                            
                            resampled_keypoints = np.zeros((target_frames, V, C))
                            for v in range(V):
                                for c in range(C):
                                    resampled_keypoints[:, v, c] = np.interp(new_indices, old_indices, keypoints_array[:, v, c])
                            
                            keypoints_array = resampled_keypoints
                        
                        # 학습 시와 동일한 정규화 적용
                        normalized_keypoints = self.normalize_keypoints(keypoints_array)
                        
                        # Shift-GCN 입력 형태로 변환
                        shift_gcn_data = np.zeros((C, target_frames, V, 1))
                        shift_gcn_data[:, :, :, 0] = normalized_keypoints.transpose(2, 0, 1)
                        
                        # 모델 예측
                        input_tensor = torch.FloatTensor(shift_gcn_data).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = self.gesture_model(input_tensor)
                            probabilities = F.softmax(output, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0, prediction].item()
                        
                        # 결과 해석
                        gesture_name = self.actions[prediction]
                        
                        results.append({
                            'frame': frame_count,
                            'gesture': gesture_name,
                            'confidence': confidence,
                            'expected': expected_gesture,
                            'correct': gesture_name == expected_gesture
                        })
                        
                        print(f"   프레임 {frame_count}: {gesture_name} (신뢰도: {confidence:.3f}) - {'✅' if gesture_name == expected_gesture else '❌'}")
                        
                        # 버퍼 초기화
                        self.gesture_frame_buffer.clear()
                    else:
                        print(f"   ⚠️ 유효한 키포인트 프레임 부족: {len(valid_frames)}/30")
                        self.gesture_frame_buffer.clear()
                        
                except Exception as e:
                    print(f"   ❌ 제스처 인식 오류: {e}")
                    self.gesture_frame_buffer.clear()
            
            # 시각화 (키포인트가 감지된 경우)
            if keypoints_detected and current_keypoints is not None:
                annotated_frame = self.draw_keypoints(frame, current_keypoints)
                
                # 제스처 정보 표시
                if results:
                    latest_result = results[-1]
                    cv2.putText(annotated_frame, f"Gesture: {latest_result['gesture']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Confidence: {latest_result['confidence']:.3f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Expected: {latest_result['expected']}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                cv2.imshow('Gesture Recognition Test', annotated_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"   비디오 처리 완료: {frame_count}프레임 처리됨")
        
        # 결과 요약
        if results:
            correct_count = sum(1 for r in results if r['correct'])
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            print(f"\n📊 결과 요약:")
            print(f"   총 판단 횟수: {total_count}")
            print(f"   정확한 판단: {correct_count}")
            print(f"   정확도: {accuracy:.2%}")
            
            for i, result in enumerate(results):
                status = "✅" if result['correct'] else "❌"
                print(f"   {i+1}. 프레임 {result['frame']}: {result['gesture']} (신뢰도: {result['confidence']:.3f}) {status}")
        else:
            print(f"   ⚠️ 제스처 판단 결과 없음")
        
        return results

def main():
    """메인 함수"""
    print("🎯 학습 데이터 제스처 인식 테스트")
    
    # 테스트 시스템 초기화
    recognizer = TestGestureRecognizer()
    
    if recognizer.gesture_model is None:
        print("❌ 모델 로드 실패")
        return
    
    # 학습 데이터 경로
    come_dir = "/home/ckim/ros-repo-4/deeplearning/gesture_recognition/pose_dataset/come"
    normal_dir = "/home/ckim/ros-repo-4/deeplearning/gesture_recognition/pose_dataset/normal"
    
    # 랜덤으로 3개 영상 선택
    come_videos = glob.glob(os.path.join(come_dir, "*.avi"))
    normal_videos = glob.glob(os.path.join(normal_dir, "*.avi"))
    
    # COME 영상 2개, NORMAL 영상 1개 선택
    selected_videos = []
    selected_videos.extend(random.sample(come_videos, 2))
    selected_videos.extend(random.sample(normal_videos, 1))
    random.shuffle(selected_videos)  # 순서 섞기
    
    print(f"\n📁 선택된 영상:")
    for i, video_path in enumerate(selected_videos):
        gesture_type = "COME" if "come" in video_path else "NORMAL"
        print(f"   {i+1}. {os.path.basename(video_path)} ({gesture_type})")
    
    # 각 영상 처리
    all_results = []
    for video_path in selected_videos:
        gesture_type = "COME" if "come" in video_path else "NORMAL"
        results = recognizer.process_video(video_path, gesture_type)
        if results:
            all_results.extend(results)
    
    # 전체 결과 요약
    if all_results:
        correct_count = sum(1 for r in all_results if r['correct'])
        total_count = len(all_results)
        overall_accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n🎯 전체 결과 요약:")
        print(f"   총 판단 횟수: {total_count}")
        print(f"   정확한 판단: {correct_count}")
        print(f"   전체 정확도: {overall_accuracy:.2%}")
        
        # 제스처별 정확도
        come_results = [r for r in all_results if r['expected'] == 'COME']
        normal_results = [r for r in all_results if r['expected'] == 'NORMAL']
        
        if come_results:
            come_accuracy = sum(1 for r in come_results if r['correct']) / len(come_results)
            print(f"   COME 정확도: {come_accuracy:.2%} ({len(come_results)}개 판단)")
        
        if normal_results:
            normal_accuracy = sum(1 for r in normal_results if r['correct']) / len(normal_results)
            print(f"   NORMAL 정확도: {normal_accuracy:.2%} ({len(normal_results)}개 판단)")

if __name__ == "__main__":
    main() 
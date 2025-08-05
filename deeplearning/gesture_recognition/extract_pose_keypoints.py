"""
YOLO Pose를 사용한 관절점 추출 및 Shift-GCN 데이터 생성

Shift-GCN 데이터 형식:
- Input shape: (C, T, V, M)
  - C: 채널 (3 - x, y, confidence)
  - T: 시간 프레임 수
  - V: 관절점 수 (어깨부터 손까지 선택 가능)
  - M: 사람 수 (기본 1)

COCO 17 관절점 인덱스 (YOLO Pose 기준):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

어깨부터 손까지: [5, 6, 7, 8, 9, 10] (6개 관절점)
상체 전체: [0, 5, 6, 7, 8, 9, 10, 11, 12] (9개 관절점)
"""

import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
import time
from pathlib import Path

class YOLOPoseExtractor:
    """YOLO Pose를 사용한 관절점 추출기"""
    
    def __init__(self, model_path='yolov8n-pose.pt', target_joints='upper_body'):
        """
        초기화
        
        Args:
            model_path: YOLO Pose 모델 경로
            target_joints: 추출할 관절점 범위
                - 'arms_only': 어깨부터 손까지 (6개)
                - 'upper_body': 상체 전체 (9개)
                - 'full_body': 전신 (17개)
        """
        self.model = YOLO(model_path)
        self.target_joints = target_joints
        
        # 관절점 인덱스 정의
        self.joint_indices = {
            'arms_only': [5, 6, 7, 8, 9, 10],  # 어깨, 팔꿈치, 손목
            'upper_body': [0, 5, 6, 7, 8, 9, 10, 11, 12],  # 코, 어깨, 팔꿈치, 손목, 엉덩이
            'full_body': list(range(17))  # 전신
        }
        
        # 관절점 연결 정의 (Shift-GCN adjacency matrix용)
        self.edges = {
            'arms_only': [
                (0, 1),  # left_shoulder - right_shoulder
                (0, 2),  # left_shoulder - left_elbow
                (1, 3),  # right_shoulder - right_elbow
                (2, 4),  # left_elbow - left_wrist
                (3, 5),  # right_elbow - right_wrist
            ],
            'upper_body': [
                (0, 1), (0, 2),  # nose - shoulders
                (1, 2),  # shoulders
                (1, 3), (2, 4),  # shoulders - elbows
                (3, 5), (4, 6),  # elbows - wrists
                (1, 7), (2, 8),  # shoulders - hips
                (7, 8),  # hips
            ],
            'full_body': [
                (0, 1), (0, 2), (1, 3), (2, 4),  # head
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # arms
                (5, 11), (6, 12), (11, 12),  # torso
                (11, 13), (12, 14), (13, 15), (14, 16)  # legs
            ]
        }
        
        self.selected_indices = self.joint_indices[target_joints]
        self.num_joints = len(self.selected_indices)
        
        print(f"✅ YOLO Pose 추출기 초기화 완료")
        print(f"   - 모델: {model_path}")
        print(f"   - 관절점 범위: {target_joints} ({self.num_joints}개)")
        print(f"   - 선택된 관절점: {self.selected_indices}")
    
    def extract_keypoints_from_video(self, video_path):
        """
        비디오에서 관절점 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            keypoints: (T, V, C) 형태의 관절점 데이터
            valid_frames: 유효한 프레임 인덱스 리스트
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 비디오 열기 실패: {video_path}")
            return None, []
        
        keypoints_sequence = []
        valid_frames = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO Pose 추론
            results = self.model(frame, verbose=False)
            
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # 첫 번째 사람의 관절점만 사용 (가장 신뢰도 높은 것)
                all_keypoints = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
                
                # 선택된 관절점만 추출
                selected_keypoints = all_keypoints[self.selected_indices]  # (V, 3)
                
                # 신뢰도 체크 (모든 관절점의 신뢰도가 0.3 이상)
                if np.all(selected_keypoints[:, 2] > 0.3):
                    keypoints_sequence.append(selected_keypoints)
                    valid_frames.append(frame_idx)
                else:
                    # 신뢰도가 낮은 경우 이전 프레임 복사 (있다면)
                    if len(keypoints_sequence) > 0:
                        keypoints_sequence.append(keypoints_sequence[-1].copy())
                        valid_frames.append(frame_idx)
            else:
                # 관절점을 찾지 못한 경우 이전 프레임 복사 (있다면)
                if len(keypoints_sequence) > 0:
                    keypoints_sequence.append(keypoints_sequence[-1].copy())
                    valid_frames.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        if len(keypoints_sequence) == 0:
            print(f"❌ 관절점 추출 실패: {video_path}")
            return None, []
        
        keypoints_array = np.array(keypoints_sequence)  # (T, V, 3)
        print(f"✅ 관절점 추출 완료: {video_path}")
        print(f"   - 총 프레임: {frame_idx}")
        print(f"   - 유효 프레임: {len(valid_frames)}")
        print(f"   - 관절점 shape: {keypoints_array.shape}")
        
        return keypoints_array, valid_frames
    
    def normalize_keypoints(self, keypoints):
        """
        관절점 정규화
        
        Args:
            keypoints: (T, V, 3) 형태의 관절점 데이터
            
        Returns:
            normalized_keypoints: 정규화된 관절점 데이터
        """
        if keypoints is None:
            return None
        
        normalized_keypoints = keypoints.copy()
        
        for t in range(keypoints.shape[0]):
            frame_keypoints = keypoints[t]  # (V, 3)
            
            # 유효한 관절점만 사용 (confidence > 0)
            valid_joints = frame_keypoints[:, 2] > 0
            
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
                
        return normalized_keypoints
    
    def create_adjacency_matrix(self):
        """
        Shift-GCN용 adjacency matrix 생성
        
        Returns:
            adjacency_matrix: (V, V) 형태의 인접 행렬
        """
        A = np.zeros((self.num_joints, self.num_joints))
        
        # self-connection
        for i in range(self.num_joints):
            A[i, i] = 1
        
        # joint connections
        edges = self.edges[self.target_joints]
        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1
        
        return A
    
    def convert_to_shift_gcn_format(self, keypoints, target_frames=64):
        """
        Shift-GCN 입력 형태로 변환
        
        Args:
            keypoints: (T, V, 3) 형태의 관절점 데이터
            target_frames: 목표 프레임 수 (시간 정규화)
            
        Returns:
            shift_gcn_data: (C, T, V, M) 형태의 데이터
        """
        if keypoints is None:
            return None
        
        T, V, C = keypoints.shape
        M = 1  # 사람 수
        
        # 시간 정규화 (target_frames로 리샘플링)
        if T != target_frames:
            # 선형 보간을 사용한 시간 정규화
            old_indices = np.linspace(0, T-1, T)
            new_indices = np.linspace(0, T-1, target_frames)
            
            resampled_keypoints = np.zeros((target_frames, V, C))
            for v in range(V):
                for c in range(C):
                    resampled_keypoints[:, v, c] = np.interp(new_indices, old_indices, keypoints[:, v, c])
            
            keypoints = resampled_keypoints
        
        # (T, V, C) -> (C, T, V, M) 변환
        shift_gcn_data = np.zeros((C, target_frames, V, M))
        shift_gcn_data[:, :, :, 0] = keypoints.transpose(2, 0, 1)  # (C, T, V)
        
        return shift_gcn_data

def process_gesture_videos(video_dir, output_dir, target_joints='upper_body', target_frames=64):
    """
    제스처 비디오들을 처리하여 Shift-GCN 데이터 생성
    
    Args:
        video_dir: 비디오가 있는 디렉토리
        output_dir: 출력 디렉토리
        target_joints: 추출할 관절점 범위
        target_frames: 목표 프레임 수
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # YOLO Pose 추출기 초기화
    extractor = YOLOPoseExtractor(target_joints=target_joints)
    
    # 비디오 파일 찾기
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))
    
    print(f"📹 찾은 비디오 파일: {len(video_files)}개")
    
    if len(video_files) == 0:
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_dir}")
        return
    
    # 액션별 데이터 수집
    action_data = {}
    
    for video_path in video_files:
        print(f"\n🔍 처리 중: {os.path.basename(video_path)}")
        
        # 액션명 추출 (파일명 또는 폴더명에서)
        path_parts = Path(video_path).parts
        action_name = None
        
        # 폴더명에서 액션 찾기 (come, normal 등)
        for part in reversed(path_parts):
            if part.lower() in ['come', 'normal']:
                action_name = part.lower()
                break
        
        # 파일명에서 액션 찾기
        if action_name is None:
            filename = os.path.basename(video_path).lower()
            if 'come' in filename:
                action_name = 'come'
            elif 'normal' in filename:
                action_name = 'normal'
        
        if action_name is None:
            print(f"⚠️ 액션명을 찾을 수 없습니다: {video_path}")
            continue
        
        # 관절점 추출
        keypoints, valid_frames = extractor.extract_keypoints_from_video(video_path)
        
        if keypoints is not None:
            # 정규화
            normalized_keypoints = extractor.normalize_keypoints(keypoints)
            
            # Shift-GCN 형태로 변환
            shift_gcn_data = extractor.convert_to_shift_gcn_format(normalized_keypoints, target_frames)
            
            if action_name not in action_data:
                action_data[action_name] = []
            
            action_data[action_name].append({
                'data': shift_gcn_data,
                'video_path': video_path,
                'valid_frames': len(valid_frames)
            })
    
    # 데이터 저장
    print(f"\n💾 데이터 저장 중...")
    
    for action_name, samples in action_data.items():
        print(f"📊 {action_name}: {len(samples)}개 샘플")
        
        # 액션별 데이터 합치기
        all_data = []
        labels = []
        
        for i, sample in enumerate(samples):
            all_data.append(sample['data'])
            labels.append(action_name)
        
        if len(all_data) > 0:
            # numpy 배열로 변환
            data_array = np.array(all_data)  # (N, C, T, V, M)
            
            # 저장
            data_file = os.path.join(output_dir, f'{action_name}_pose_data.npy')
            labels_file = os.path.join(output_dir, f'{action_name}_labels.npy')
            
            np.save(data_file, data_array)
            np.save(labels_file, labels)
            
            print(f"✅ 저장 완료: {data_file}")
            print(f"   - Shape: {data_array.shape}")
    
    # 인접 행렬 저장
    adjacency_matrix = extractor.create_adjacency_matrix()
    adj_file = os.path.join(output_dir, 'adjacency_matrix.npy')
    np.save(adj_file, adjacency_matrix)
    print(f"✅ 인접 행렬 저장: {adj_file}")
    print(f"   - Shape: {adjacency_matrix.shape}")
    
    # 메타데이터 저장
    metadata = {
        'target_joints': target_joints,
        'num_joints': extractor.num_joints,
        'joint_indices': extractor.selected_indices,
        'target_frames': target_frames,
        'actions': list(action_data.keys()),
        'total_samples': sum(len(samples) for samples in action_data.values())
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.npy')
    np.save(metadata_file, metadata)
    print(f"✅ 메타데이터 저장: {metadata_file}")

if __name__ == "__main__":
    # 이미 찍은 영상에 맞는 설정
    video_dir = "./deeplearning/gesture_recognition/pose_dataset"
    output_dir = "./deeplearning/gesture_recognition/shift_gcn_data"
    
    # 이미 찍은 영상 설정 (수정 불가)
    RECORDING_TIME = 3.0    # 이미 찍은 영상: 3초
    ESTIMATED_FPS = 30.0    # 예상 FPS
    ESTIMATED_FRAMES = int(RECORDING_TIME * ESTIMATED_FPS)  # 약 90 프레임
    
    # Shift-GCN 옵션
    TARGET_FRAMES_OPTIONS = {
        'use_original': ESTIMATED_FRAMES,  # 90 프레임 (원본 그대로)
        'standard_64': 64,                 # 64 프레임 (일반적)
        'standard_128': 128                # 128 프레임 (고품질)
    }
    
    # 선택: 원본 프레임 수 사용 vs 표준 프레임 수 사용
    SELECTED_OPTION = 'use_original'  # 'use_original', 'standard_64', 'standard_128' 중 선택
    TARGET_FRAMES = TARGET_FRAMES_OPTIONS[SELECTED_OPTION]
    
    print("🚀 YOLO Pose 기반 Shift-GCN 데이터 생성 시작")
    print("=" * 60)
    print(f"📹 이미 찍은 영상 정보:")
    print(f"   - 녹화 시간: {RECORDING_TIME}초")
    print(f"   - 예상 FPS: {ESTIMATED_FPS}")
    print(f"   - 예상 프레임: {ESTIMATED_FRAMES}")
    print(f"🎯 Shift-GCN 설정:")
    print(f"   - 선택된 옵션: {SELECTED_OPTION}")
    print(f"   - 목표 프레임: {TARGET_FRAMES}")
    
    if TARGET_FRAMES != ESTIMATED_FRAMES:
        change_percent = ((TARGET_FRAMES - ESTIMATED_FRAMES) / ESTIMATED_FRAMES) * 100
        print(f"   - 프레임 변화: {change_percent:+.1f}%")
        if abs(change_percent) > 20:
            print(f"   ⚠️ 큰 변화 감지! 데이터 손실 가능성 있음")
        else:
            print(f"   ✅ 적절한 변화 범위")
    else:
        print(f"   ✅ 원본 프레임 수 유지 (변화 없음)")
    
    # 상체 관절점으로 데이터 생성
    process_gesture_videos(
        video_dir=video_dir,
        output_dir=output_dir,
        target_joints='upper_body',  # 'arms_only', 'upper_body', 'full_body' 중 선택
        target_frames=TARGET_FRAMES  # 이미 찍은 영상에 맞게 조정
    )
    
    print("\n🎉 Shift-GCN 데이터 생성 완료!")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"💡 Tip: 성능이 좋지 않으면 target_frames를 조정해보세요") 
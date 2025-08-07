"""
Sliding Window 방식으로 제스처 데이터 생성
- 원본 영상에서 관절점 추출 (프레임 수 그대로 유지)
- 짧은 시퀀스 (30프레임)로 sliding window 적용
- 더 많은 샘플 생성
"""

import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
from pathlib import Path

class SlidingWindowPoseExtractor:
    """Sliding Window 방식의 관절점 추출기"""
    
    def __init__(self, model_path='yolov8n-pose.pt', target_joints='upper_body'):
        # GPU 사용 설정
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 사용 중인 디바이스: {self.device}")
        
        self.model = YOLO(model_path)
        # YOLO 모델을 GPU로 이동
        if torch.cuda.is_available():
            self.model.to(self.device)
        
        self.target_joints = target_joints
        
        # 관절점 인덱스 정의
        self.joint_indices = {
            'arms_only': [5, 6, 7, 8, 9, 10],  # 어깨, 팔꿈치, 손목
            'upper_body': [0, 5, 6, 7, 8, 9, 10, 11, 12],  # 코, 어깨, 팔꿈치, 손목, 엉덩이
            'full_body': list(range(17))  # 전신
        }
        
        # 관절점 연결 정의
        self.edges = {
            'arms_only': [
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5)
            ],
            'upper_body': [
                (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
                (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)
            ],
            'full_body': [
                (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), 
                (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
        }
        
        self.selected_indices = self.joint_indices[target_joints]
        self.num_joints = len(self.selected_indices)
        
        print(f"✅ Sliding Window Pose 추출기 초기화")
        print(f"   - 관절점 범위: {target_joints} ({self.num_joints}개)")
    
    def extract_keypoints_from_video(self, video_path):
        """비디오에서 관절점 추출 (원본 프레임 수 유지)"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 비디오 열기 실패: {video_path}")
            return None
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        keypoints_sequence = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO Pose 추론
            results = self.model(frame, verbose=False)
            
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # 첫 번째 사람의 관절점만 사용
                all_keypoints = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
                
                # 선택된 관절점만 추출
                selected_keypoints = all_keypoints[self.selected_indices]  # (V, 3)
                
                # 신뢰도 체크
                if np.all(selected_keypoints[:, 2] > 0.3):
                    keypoints_sequence.append(selected_keypoints)
                else:
                    # 신뢰도가 낮은 경우 이전 프레임 복사
                    if len(keypoints_sequence) > 0:
                        keypoints_sequence.append(keypoints_sequence[-1].copy())
                    else:
                        # 첫 프레임인 경우 일단 저장
                        keypoints_sequence.append(selected_keypoints)
            else:
                # 관절점을 찾지 못한 경우 이전 프레임 복사
                if len(keypoints_sequence) > 0:
                    keypoints_sequence.append(keypoints_sequence[-1].copy())
                else:
                    # 첫 프레임인 경우 영벡터 저장
                    keypoints_sequence.append(np.zeros((self.num_joints, 3)))
            
            frame_idx += 1
        
        cap.release()
        
        if len(keypoints_sequence) == 0:
            print(f"❌ 관절점 추출 실패: {video_path}")
            return None
        
        keypoints_array = np.array(keypoints_sequence)  # (T, V, 3)
        print(f"✅ 관절점 추출: {os.path.basename(video_path)}")
        print(f"   - 실제 프레임: {keypoints_array.shape[0]} (예상: {total_frames})")
        print(f"   - FPS: {fps:.1f}")
        
        return keypoints_array
    
    def normalize_keypoints(self, keypoints):
        """관절점 정규화"""
        if keypoints is None:
            return None
        
        normalized_keypoints = keypoints.copy()
        
        for t in range(keypoints.shape[0]):
            frame_keypoints = keypoints[t]  # (V, 3)
            
            # 유효한 관절점만 사용
            valid_joints = frame_keypoints[:, 2] > 0
            
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
        
        return normalized_keypoints
    
    def create_sliding_windows(self, keypoints, window_size=30, stride=1):
        """
        Sliding window로 데이터 분할
        
        Args:
            keypoints: (T, V, 3) 형태의 관절점 데이터
            window_size: 윈도우 크기 (프레임 수)
            stride: 슬라이딩 간격
            
        Returns:
            windows: 윈도우들의 리스트
        """
        if keypoints is None:
            return []
        
        T, V, C = keypoints.shape
        windows = []
        
        # Sliding window 생성
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = keypoints[start:end]  # (window_size, V, 3)
            windows.append(window)
        
        print(f"   - 원본 프레임: {T}")
        print(f"   - 윈도우 크기: {window_size}")
        print(f"   - 생성된 윈도우: {len(windows)}개")
        
        return windows
    
    def convert_to_shift_gcn_format(self, window):
        """
        윈도우를 Shift-GCN 형태로 변환
        
        Args:
            window: (T, V, 3) 형태의 윈도우 데이터
            
        Returns:
            shift_gcn_data: (3, T, V, 1) 형태의 데이터
        """
        T, V, C = window.shape
        M = 1  # 사람 수
        
        # (T, V, C) -> (C, T, V, M) 변환
        shift_gcn_data = np.zeros((C, T, V, M))
        shift_gcn_data[:, :, :, 0] = window.transpose(2, 0, 1)  # (C, T, V)
        
        return shift_gcn_data
    
    def create_adjacency_matrix(self):
        """Shift-GCN용 adjacency matrix 생성"""
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

def process_videos_sliding_window(video_dir, output_dir, target_joints='upper_body', 
                                window_size=30, stride=1):
    """
    Sliding window 방식으로 제스처 비디오 처리
    
    Args:
        video_dir: 비디오가 있는 디렉토리
        output_dir: 출력 디렉토리
        target_joints: 추출할 관절점 범위
        window_size: 윈도우 크기 (프레임 수)
        stride: 슬라이딩 간격
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 추출기 초기화
    extractor = SlidingWindowPoseExtractor(target_joints=target_joints)
    
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
        
        # 액션명 추출
        path_parts = Path(video_path).parts
        action_name = None
        
        for part in reversed(path_parts):
            if part.lower() in ['come', 'normal']:
                action_name = part.lower()
                break
        
        if action_name is None:
            filename = os.path.basename(video_path).lower()
            if 'come' in filename:
                action_name = 'come'
            elif 'normal' in filename:
                action_name = 'normal'
        
        if action_name is None:
            print(f"⚠️ 액션명을 찾을 수 없습니다: {video_path}")
            continue
        
        # 관절점 추출 (원본 프레임 수 유지)
        keypoints = extractor.extract_keypoints_from_video(video_path)
        
        if keypoints is not None:
            # 정규화
            normalized_keypoints = extractor.normalize_keypoints(keypoints)
            
            # Sliding window 생성
            windows = extractor.create_sliding_windows(
                normalized_keypoints, 
                window_size=window_size, 
                stride=stride
            )
            
            if action_name not in action_data:
                action_data[action_name] = []
            
            # 각 윈도우를 Shift-GCN 형태로 변환
            for window in windows:
                shift_gcn_data = extractor.convert_to_shift_gcn_format(window)
                action_data[action_name].append({
                    'data': shift_gcn_data,
                    'video_path': video_path
                })
    
    # 데이터 저장
    print(f"\n💾 데이터 저장 중...")
    
    total_samples = 0
    for action_name, samples in action_data.items():
        print(f"📊 {action_name}: {len(samples)}개 윈도우")
        total_samples += len(samples)
        
        # 액션별 데이터 합치기
        all_data = []
        labels = []
        
        for sample in samples:
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
    
    # 메타데이터 저장
    metadata = {
        'target_joints': target_joints,
        'num_joints': extractor.num_joints,
        'joint_indices': extractor.selected_indices,
        'window_size': window_size,
        'stride': stride,
        'actions': list(action_data.keys()),
        'total_samples': total_samples
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.npy')
    np.save(metadata_file, metadata)
    print(f"✅ 메타데이터 저장: {metadata_file}")
    
    print(f"\n🎉 Sliding Window 데이터 생성 완료!")
    print(f"   - 총 샘플: {total_samples}개 (기존 100개 -> {total_samples}개)")
    print(f"   - 윈도우 크기: {window_size} 프레임 (기존 90 프레임)")
    print(f"   - 데이터 증강: {total_samples/100:.1f}배 증가")

if __name__ == "__main__":
    video_dir = "./pose_dataset"
    output_dir = "./shift_gcn_data_sliding"
    
    print("🚀 Sliding Window 방식 데이터 생성 시작")
    print("=" * 60)
    print("📋 설정:")
    print("   - 윈도우 크기: 30 프레임 (1초)")
    print("   - 슬라이딩 간격: 1 프레임")
    print("   - 예상 증가: 각 3초 영상 → 약 36개 윈도우")
    print("   - 총 예상 샘플: 100개 영상 → 약 3600개 윈도우")
    
    process_videos_sliding_window(
        video_dir=video_dir,
        output_dir=output_dir,
        target_joints='upper_body',
        window_size=30,  # 30 프레임 (약 1초)
        stride=1         # 1 프레임씩 슬라이딩
    ) 
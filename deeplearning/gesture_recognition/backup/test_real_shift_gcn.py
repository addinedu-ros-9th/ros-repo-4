import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 진짜 Shift-GCN 모델 클래스 (학습된 모델과 동일)
class ShiftGCNLayer(nn.Module):
    """Shift-GCN 레이어"""
    def __init__(self, in_channels, out_channels, adjacency_matrix, num_adj=8):
        super(ShiftGCNLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_adj = num_adj
        
        # 인접 행렬 분할
        A_list = self._split_adjacency_matrix(adjacency_matrix)
        # 리스트를 텐서로 변환하여 저장
        self.register_buffer('A_list', torch.stack(A_list))
        
        # 각 분할에 대한 컨볼루션
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // num_adj, 
                     kernel_size=1, bias=False)
            for _ in range(num_adj)
        ])
        
        # 잔차 연결
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def _split_adjacency_matrix(self, A):
        """인접 행렬을 여러 개로 분할"""
        A_list = []
        
        # 기본 인접 행렬
        A_list.append(A)
        
        # 거리에 따른 인접 행렬들 (1-hop, 2-hop, ...)
        A_power = A.clone()
        for _ in range(self.num_adj - 1):
            A_power = torch.mm(A_power, A)
            A_list.append(A_power)
        
        return A_list
    
    def forward(self, x):
        # x shape: (batch_size, in_channels, num_frames, num_joints)
        batch_size, in_channels, num_frames, num_joints = x.size()
        
        # Shift-GCN 연산
        out_list = []
        for i in range(self.num_adj):
            A = self.A_list[i]  # (num_joints, num_joints)
            conv = self.conv_list[i]
            
            # 그래프 컨볼루션 - einsum 사용
            # x: (batch_size, in_channels, num_frames, num_joints)
            # A: (num_joints, num_joints)
            # einsum('bcfj,jk->bcfk', x, A): (batch_size, in_channels, num_frames, num_joints)
            graph_conv = torch.einsum('bcfj,jk->bcfk', x, A)
            conv_out = conv(graph_conv)
            out_list.append(conv_out)
        
        # 결과 결합
        out = torch.cat(out_list, dim=1)  # (batch_size, out_channels, num_frames, num_joints)
        
        # 잔차 연결
        residual = self.residual(x)
        out = out + residual
        
        # BatchNorm과 ReLU
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class RealShiftGCN(nn.Module):
    """진짜 Shift-GCN 모델 구현"""
    def __init__(self, num_classes=3, num_frames=60, num_joints=21, num_features=3):
        super(RealShiftGCN, self).__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_features = num_features  # x, y, z
        
        # 손 관절 그래프 구조 정의 (MediaPipe Hands 기준)
        # 21개 관절의 연결 관계
        self.hand_connections = [
            # 엄지
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 검지
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 중지
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 약지
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 새끼
            (0, 17), (17, 18), (18, 19), (19, 20),
            # 손바닥 연결
            (5, 9), (9, 13), (13, 17)
        ]
        
        # 인접 행렬 생성
        self.register_buffer('A', self._build_adjacency_matrix())
        
        # Shift-GCN 레이어들
        self.gcn_layers = nn.ModuleList([
            ShiftGCNLayer(3, 64, self.A),    # 3 -> 64
            ShiftGCNLayer(64, 128, self.A),  # 64 -> 128
            ShiftGCNLayer(128, 256, self.A), # 128 -> 256
        ])
        
        # Temporal CNN
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _build_adjacency_matrix(self):
        """인접 행렬 생성"""
        A = torch.zeros(self.num_joints, self.num_joints)
        for i, j in self.hand_connections:
            A[i, j] = 1
            A[j, i] = 1  # 무방향 그래프
        # 자기 자신과의 연결
        A += torch.eye(self.num_joints)
        return A
    
    def forward(self, x):
        # x shape: (batch_size, num_frames, num_joints, num_features)
        batch_size, num_frames, num_joints, num_features = x.size()
        
        # (batch_size, num_frames, num_joints, num_features) -> (batch_size, num_features, num_frames, num_joints)
        x = x.permute(0, 3, 1, 2)  # (batch_size, 3, num_frames, num_joints)
        
        # GCN 레이어들 통과
        gcn_out = x
        for gcn_layer in self.gcn_layers:
            gcn_out = gcn_layer(gcn_out)
        
        # Temporal modeling
        # (batch_size, 256, num_frames, num_joints) -> (batch_size, 256, num_frames)
        gcn_out = gcn_out.mean(dim=3)  # 관절 차원 평균
        
        # Temporal CNN
        temporal_out = self.temporal_conv(gcn_out)
        
        # Global pooling
        pooled = self.global_pool(temporal_out)  # (batch_size, 512, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, 512)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

def calculate_angles(landmarks):
    """관절 각도 계산"""
    angles = []
    
    # 손가락 관절 각도들 (15개)
    finger_joints = [
        # 엄지
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        # 검지
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        # 중지
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        # 약지
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        # 새끼
        (0, 17, 18), (17, 18, 19), (18, 19, 20)
    ]
    
    for joint in finger_joints:
        p1, p2, p3 = joint
        
        # 3D 벡터 계산
        v1 = landmarks[p1] - landmarks[p2]
        v2 = landmarks[p3] - landmarks[p2]
        
        # 각도 계산
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        angles.append(angle)
    
    return np.array(angles)

def process_frame_for_prediction(frame_data):
    """프레임 데이터를 모델 입력 형태로 변환"""
    frames = len(frame_data)
    
    if frames < 60:
        # 부족한 프레임은 0으로 패딩
        padding = np.zeros((60 - frames, 21, 3), dtype=np.float32)
        x = np.vstack([frame_data, padding])
    elif frames > 60:
        # 초과하는 프레임은 중앙에서 60개 추출
        start = (frames - 60) // 2
        x = frame_data[start:start + 60]
    else:
        x = frame_data
    
    # (60, 21, 3) -> (1, 60, 21, 3) - 배치 차원 추가
    x = torch.FloatTensor(x).unsqueeze(0)
    
    return x

def main():
    """진짜 Shift-GCN 실시간 제스처 인식 테스트"""
    print("🚀 진짜 Shift-GCN 실시간 제스처 인식 테스트")
    print("💡 그래프 컨볼루션 + Shift 연산으로 학습된 모델")
    
    # 모델 로드
    model = RealShiftGCN(num_classes=3, num_frames=60, num_joints=21, num_features=3)
    model.load_state_dict(torch.load('best_real_shift_gcn_model.pth'))
    model = model.to(device)
    model.eval()
    
    print("✅ 모델 로드 완료")
    
    # MediaPipe 설정
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 카메라 설정
    cap = cv2.VideoCapture('/dev/video0')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 프레임 버퍼 (60프레임)
    frame_buffer = deque(maxlen=60)
    prediction_buffer = deque(maxlen=5)  # 예측 결과 스무딩
    
    # 제스처 라벨
    gesture_labels = ['come', 'away', 'stop']
    
    print("🎥 실시간 제스처 인식 시작 (q: 종료)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 좌우 반전
        frame = cv2.flip(frame, 1)
        
        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 랜드마크 좌표 추출
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                landmarks = np.array(landmarks)
                
                # 프레임 버퍼에 추가
                frame_buffer.append(landmarks)
        
        # 충분한 프레임이 모이면 예측
        if len(frame_buffer) >= 30:  # 최소 30프레임
            try:
                # 프레임 데이터를 모델 입력 형태로 변환
                frame_data = list(frame_buffer)
                x = process_frame_for_prediction(frame_data)
                x = x.to(device)
                
                # 모델 예측
                with torch.no_grad():
                    outputs = model(x)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # 예측 결과 버퍼에 추가
                prediction_buffer.append(predicted_class)
                
                # 최빈값으로 최종 예측
                if len(prediction_buffer) >= 3:
                    final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                    gesture = gesture_labels[final_prediction]
                    
                    # 결과 표시
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/60", (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
            except Exception as e:
                print(f"예측 오류: {e}")
        
        # 프레임 표시
        cv2.imshow('Real Shift-GCN Gesture Recognition', frame)
        
        # q 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 테스트 종료")

if __name__ == "__main__":
    main() 
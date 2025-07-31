import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 기존 모델 클래스 (학습된 모델과 동일)
class GestureCNN(nn.Module):
    """제스처 인식 CNN 모델 (특징 엔지니어링 + CNN)"""
    def __init__(self, num_classes=3, num_frames=60, num_joints=21):
        super(GestureCNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        
        # 입력 특징 계산
        self.basic_features = 99  # 랜드마크(84) + 각도(15)
        self.gesture_features = 260  # 동적(252) + 손모양(8)
        self.total_features = self.basic_features * num_frames + self.gesture_features
        
        # 1. 기본 시퀀스 처리
        self.sequence_encoder = nn.Sequential(
            nn.Linear(self.basic_features * num_frames, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 2. 제스처 특성 처리
        self.gesture_encoder = nn.Sequential(
            nn.Linear(self.gesture_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. 결합된 특징 처리
        self.combined_encoder = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 기본 시퀀스 특징 추출
        sequence_features = x[:, :self.basic_features * self.num_frames]
        sequence_encoded = self.sequence_encoder(sequence_features)
        
        # 제스처 특성 추출
        gesture_features = x[:, self.basic_features * self.num_frames:]
        gesture_encoded = self.gesture_encoder(gesture_features)
        
        # 특징 결합
        combined_features = torch.cat([sequence_encoded, gesture_encoded], dim=1)
        
        # 최종 분류
        output = self.combined_encoder(combined_features)
        
        return output

def extract_gesture_features(data):
    """제스처 특성 추출 (학습 시와 동일)"""
    landmarks = data[:, :84]
    angles = data[:, 84:99]
    
    # 1. 동적 특성
    motion_features = []
    for i in range(1, len(data)):
        landmark_diff = landmarks[i] - landmarks[i-1]
        motion_features.append(landmark_diff)
    
    if len(motion_features) > 0:
        motion_features = np.array(motion_features)
        motion_mean = np.mean(motion_features, axis=0)
        motion_std = np.std(motion_features, axis=0)
        motion_max = np.max(np.abs(motion_features), axis=0)
    else:
        motion_mean = np.zeros(84)
        motion_std = np.zeros(84)
        motion_max = np.zeros(84)
    
    # 2. 손 모양 특성
    finger_tips = [8, 12, 16, 20]
    palm_center = 0
    
    hand_shape_features = []
    for frame in landmarks:
        frame_reshaped = frame.reshape(-1, 4)
        finger_distances = []
        for tip_idx in finger_tips:
            tip_pos = frame_reshaped[tip_idx][:3]
            palm_pos = frame_reshaped[palm_center][:3]
            distance = np.linalg.norm(tip_pos - palm_pos)
            finger_distances.append(distance)
        hand_shape_features.append(finger_distances)
    
    hand_shape_features = np.array(hand_shape_features)
    hand_shape_mean = np.mean(hand_shape_features, axis=0)
    hand_shape_std = np.std(hand_shape_features, axis=0)
    
    # 3. 제스처별 특성 벡터
    gesture_specific = np.concatenate([
        motion_mean, motion_std, motion_max,
        hand_shape_mean, hand_shape_std
    ])
    
    return gesture_specific

def main():
    """간단한 실시간 제스처 인식 테스트"""
    print("🚀 간단한 실시간 제스처 인식 테스트")
    print("💡 모델이 학습한 패턴만 사용합니다")
    
    # 모델 로드
    model = GestureCNN(num_classes=3, num_frames=60, num_joints=21)
    model.load_state_dict(torch.load('best_gesture_cnn_model.pth'))
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
    
    # 프레임 버퍼
    frame_buffer = deque(maxlen=60)
    
    # 제스처 라벨
    actions = ['COME', 'AWAY', 'STOP']
    
    # 예측 결과 버퍼
    prediction_buffer = deque(maxlen=5)
    
    print("📹 카메라 시작...")
    print("💡 제스처를 해보세요:")
    print("   - COME: 손을 까딱까딱 움직이기")
    print("   - STOP: 주먹 쥐기")
    print("   - AWAY: 손바닥 펴기")
    print("   - q: 종료, r: 리셋")
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
            
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        prediction = "WAITING..."
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 랜드마크 데이터 추출
                joint = np.zeros((21, 4))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                
                # 각도 계산
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                angle = np.degrees(angle)
                
                # 특징 벡터 생성
                d = np.concatenate([joint.flatten(), angle])
                frame_buffer.append(d)
                
                # 충분한 프레임이 모이면 예측
                if len(frame_buffer) >= 30:
                    try:
                        # 모델 입력 준비 (학습 시와 동일한 방식)
                        frame_data = np.array(list(frame_buffer))
                        
                        # 프레임 수를 60으로 패딩 또는 자르기
                        frames = len(frame_data)
                        if frames < 60:
                            padding = np.zeros((60 - frames, 99), dtype=np.float32)
                            x = np.vstack([frame_data, padding])
                        elif frames > 60:
                            start = (frames - 60) // 2
                            x = frame_data[start:start + 60]
                        else:
                            x = frame_data
                        
                        # 제스처 특성 추출
                        gesture_features = extract_gesture_features(x)
                        
                        # 기본 특징 + 제스처 특성 결합
                        x_with_gesture = np.concatenate([
                            x.flatten(),
                            gesture_features
                        ])
                        
                        model_input = torch.FloatTensor(x_with_gesture).unsqueeze(0).to(device)
                        
                        # 예측
                        with torch.no_grad():
                            outputs = model(model_input)
                            probabilities = torch.softmax(outputs, dim=1)
                            predicted = torch.argmax(outputs, dim=1).item()
                            confidence = probabilities[0][predicted].item()
                        
                        prediction = actions[predicted]
                        prediction_buffer.append(predicted)
                        
                    except Exception as e:
                        print(f"예측 오류: {e}")
                        prediction = "ERROR"
        
        # 부드러운 예측 결과
        if len(prediction_buffer) >= 3:
            from collections import Counter
            most_common = Counter(prediction_buffer).most_common(1)[0]
            final_prediction = actions[most_common[0]]
            final_confidence = most_common[1] / len(prediction_buffer)
        else:
            final_prediction = prediction
            final_confidence = confidence
        
        # 화면에 정보 표시
        cv2.putText(img, f'Prediction: {final_prediction}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Confidence: {final_confidence:.2f}', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f'Buffer: {len(frame_buffer)}/60', 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 제스처 가이드
        cv2.putText(img, 'COME: Move hand', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, 'STOP: Fist', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, 'AWAY: Open palm', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Simple Gesture Recognition', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_buffer.clear()
            prediction_buffer.clear()
            print("🔄 버퍼 리셋")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 테스트 종료")

if __name__ == "__main__":
    main() 
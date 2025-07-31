import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HandGestureDataset(Dataset):
    """손 제스처 데이터셋 클래스 (제스처 특성 반영)"""
    def __init__(self, data_paths, labels):
        self.data_paths = data_paths
        self.labels = labels
        self.samples = []
        
        # 각 파일에서 시퀀스들을 개별 샘플로 분리
        for file_path, label in zip(data_paths, labels):
            data = np.load(file_path)
            # data shape: (num_sequences, 60, 100)
            for i in range(data.shape[0]):
                self.samples.append((data[i], label))  # (sequence, label)
        
    def __len__(self):
        return len(self.samples)
    
    def extract_gesture_features(self, data):
        """제스처 특성 추출"""
        # 기본 특징: 랜드마크 + 각도
        landmarks = data[:, :84]  # 21*4 = 84
        angles = data[:, 84:99]   # 15
        
        # 1. 동적 특성 (움직임 분석)
        motion_features = []
        for i in range(1, len(data)):
            # 이전 프레임과의 랜드마크 변화량
            landmark_diff = landmarks[i] - landmarks[i-1]
            motion_features.append(landmark_diff)
        
        if len(motion_features) > 0:
            motion_features = np.array(motion_features)
            # 움직임 통계: 평균, 표준편차, 최대값
            motion_mean = np.mean(motion_features, axis=0)
            motion_std = np.std(motion_features, axis=0)
            motion_max = np.max(np.abs(motion_features), axis=0)
        else:
            motion_mean = np.zeros(84)
            motion_std = np.zeros(84)
            motion_max = np.zeros(84)
        
        # 2. 손 모양 특성 (주먹 vs 손바닥)
        # 손가락 끝점들 (8, 12, 16, 20)과 손바닥 중심점(0)의 거리
        finger_tips = [8, 12, 16, 20]  # 검지, 중지, 약지, 새끼 손가락 끝
        palm_center = 0  # 손바닥 중심
        
        hand_shape_features = []
        for frame in landmarks:
            frame_reshaped = frame.reshape(-1, 4)  # (21, 4)
            
            # 손가락 끝점들의 손바닥 중심으로부터의 거리
            finger_distances = []
            for tip_idx in finger_tips:
                tip_pos = frame_reshaped[tip_idx][:3]  # x, y, z
                palm_pos = frame_reshaped[palm_center][:3]
                distance = np.linalg.norm(tip_pos - palm_pos)
                finger_distances.append(distance)
            
            # 손가락들이 펴져있으면 거리가 멀고, 주먹이면 거리가 가까움
            hand_shape_features.append(finger_distances)
        
        hand_shape_features = np.array(hand_shape_features)
        hand_shape_mean = np.mean(hand_shape_features, axis=0)
        hand_shape_std = np.std(hand_shape_features, axis=0)
        
        # 3. 제스처별 특성 벡터
        # come: 동적 + 손바닥
        # stop: 정적 + 주먹  
        # away: 정적 + 손바닥
        gesture_specific = np.concatenate([
            motion_mean, motion_std, motion_max,  # 동적 특성 (84*3 = 252)
            hand_shape_mean, hand_shape_std       # 손 모양 특성 (4*2 = 8)
        ])
        
        return gesture_specific
    
    def __getitem__(self, idx):
        # 개별 시퀀스와 라벨 가져오기
        sequence, label = self.samples[idx]
        
        # 데이터 형태: (60, 100) - 60프레임, 100특징 (99 + 1라벨)
        frames = sequence.shape[0]  # 60
        features = sequence.shape[1] - 1  # 99 (라벨 제외)
        
        # 입력 데이터와 라벨 분리
        x = sequence[:, :-1].astype(np.float32)  # 특징 데이터 (60, 99)
        y = int(sequence[0, -1])  # 라벨 (첫 번째 프레임의 라벨 사용)
        
        # 제스처 특성 추출
        gesture_features = self.extract_gesture_features(x)
        
        # 기본 특징 + 제스처 특성 결합
        # 기본: 60프레임 × 99특징
        # 제스처 특성: 260 (252 + 8)
        # 최종: 60프레임 × 99특징 + 260 제스처 특성
        x_with_gesture = np.concatenate([
            x.flatten(),  # 60*99 = 5940
            gesture_features  # 260
        ])
        
        return torch.FloatTensor(x_with_gesture), torch.LongTensor([y])

class GestureCNN(nn.Module):
    """제스처 인식 CNN 모델 (특징 엔지니어링 + CNN)"""
    def __init__(self, num_classes=3, num_frames=60, num_joints=21):
        super(GestureCNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames  # 60프레임 (2초)
        self.num_joints = num_joints
        
        # 입력 특징 계산
        self.basic_features = 99  # 랜드마크(84) + 각도(15)
        self.gesture_features = 260  # 동적(252) + 손모양(8)
        self.total_features = self.basic_features * num_frames + self.gesture_features  # 5940 + 260 = 6200
        
        # 1. 기본 시퀀스 처리 (60프레임 × 99특징)
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
        # x shape: (batch_size, total_features)
        batch_size = x.size(0)
        
        # 기본 시퀀스 특징 추출 (처음 5940개)
        sequence_features = x[:, :self.basic_features * self.num_frames]
        sequence_encoded = self.sequence_encoder(sequence_features)
        
        # 제스처 특성 추출 (나머지 260개)
        gesture_features = x[:, self.basic_features * self.num_frames:]
        gesture_encoded = self.gesture_encoder(gesture_features)
        
        # 특징 결합
        combined_features = torch.cat([sequence_encoded, gesture_encoded], dim=1)
        
        # 최종 분류
        output = self.combined_encoder(combined_features)
        
        return output

def load_dataset(data_dir):
    """데이터셋 로드"""
    print("📁 데이터셋 로드 중...")
    
    # 시퀀스 데이터 파일들 찾기
    seq_files = glob.glob(os.path.join(data_dir, 'seq_*.npy'))
    
    if not seq_files:
        print("⚠️ 시퀀스 데이터 파일을 찾을 수 없습니다. 원시 데이터를 사용합니다.")
        # 원시 데이터 파일들 찾기
        raw_files = glob.glob(os.path.join(data_dir, 'raw_*.npy'))
        if not raw_files:
            raise FileNotFoundError("데이터 파일을 찾을 수 없습니다!")
        data_files = raw_files
    else:
        data_files = seq_files
    
    print(f"📊 발견된 데이터 파일: {len(data_files)}개")
    
    # 파일별로 라벨 추출
    data_paths = []
    labels = []
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        
        # 라벨 추출 (come=0, away=1, stop=2)
        if 'come' in filename:
            label = 0
        elif 'away' in filename:
            label = 1
        elif 'stop' in filename:
            label = 2
        else:
            continue
            
        data_paths.append(file_path)
        labels.append(label)
    
    print(f"📈 클래스별 데이터 수:")
    for i, action in enumerate(['come', 'away', 'stop']):
        count = labels.count(i)
        print(f"  {action}: {count}개")
    
    return data_paths, labels

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """모델 학습"""
    print("🚀 모델 학습 시작...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 학습
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.squeeze().to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # 통계 계산
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 학습률 조정
        scheduler.step()
        
        # 결과 출력
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gesture_cnn_model.pth')
            print(f"💾 최고 성능 모델 저장 (검증 정확도: {val_acc:.2f}%)")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """학습 히스토리 플롯"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 정확도 그래프
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader):
    """모델 평가"""
    print("📊 모델 평가 중...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze().to(device)
            
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # 정확도 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"🎯 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 분류 리포트
    print("\n📋 분류 리포트:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['come', 'away', 'stop']))
    
    return accuracy

def main():
    """메인 함수"""
    print("🚀 제스처 인식 CNN 모델 학습 (특징 엔지니어링 + CNN)")
    
    # 데이터셋 로드
    data_paths, labels = load_dataset('dataset')
    
    print(f"📊 원본 데이터:")
    for i, action in enumerate(['come', 'away', 'stop']):
        count = labels.count(i)
        print(f"  {action}: {count}개")
    
    # 데이터셋 생성 (시퀀스들이 개별 샘플로 분리됨)
    train_dataset = HandGestureDataset(data_paths, labels)
    
    # 전체 샘플 수 계산
    total_samples = len(train_dataset)
    print(f"📊 총 시퀀스 샘플: {total_samples}개")
    
    # 클래스별 샘플 수 계산
    class_counts = [0, 0, 0]
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    print(f"📈 클래스별 시퀀스 수:")
    for i, action in enumerate(['come', 'away', 'stop']):
        print(f"  {action}: {class_counts[i]}개")
    
    # 데이터 분할 (stratify 사용 가능)
    all_samples = list(range(total_samples))
    all_labels = [label for _, label in train_dataset.samples]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_samples, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"📊 데이터 분할:")
    print(f"  학습: {len(X_train)}개")
    print(f"  검증: {len(X_val)}개")
    print(f"  테스트: {len(X_test)}개")
    
    # 서브셋 데이터셋 생성
    class SubsetDataset(Dataset):
        def __init__(self, full_dataset, indices):
            self.full_dataset = full_dataset
            self.indices = indices
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            return self.full_dataset[self.indices[idx]]
    
    train_subset = SubsetDataset(train_dataset, X_train)
    val_subset = SubsetDataset(train_dataset, X_val)
    test_subset = SubsetDataset(train_dataset, X_test)
    
    # 데이터 로더 생성
    batch_size = 32  # 충분한 데이터가 있으므로 배치 크기 증가
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    # 모델 생성 (제스처 특성 반영)
    model = GestureCNN(num_classes=3, num_frames=60, num_joints=21)
    model = model.to(device)
    
    print(f"📈 모델 구조 (특징 엔지니어링 + CNN):")
    print(f"  - 기본 특징: 60프레임 × 99특징 (5940)")
    print(f"  - 제스처 특성: 260 (동적 252 + 손모양 8)")
    print(f"  - 총 입력: 6200 특징")
    print(f"  - 출력: 3개 클래스 (come, away, stop)")
    print(f"  - 제스처별 특성:")
    print(f"    • come: 동적 + 손바닥 (움직임 많음)")
    print(f"    • stop: 정적 + 주먹 (움직임 적음, 주먹 모양)")
    print(f"    • away: 정적 + 손바닥 (움직임 적음, 손바닥 펴짐)")
    print(model)
    
    # 모델 학습
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=50, learning_rate=0.001
    )
    
    # 학습 히스토리 플롯
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 최고 성능 모델 로드
    if os.path.exists('best_gesture_cnn_model.pth'):
        model.load_state_dict(torch.load('best_gesture_cnn_model.pth'))
        
        # 모델 평가
        test_accuracy = evaluate_model(model, test_loader)
        
        print("🎉 학습 완료!")
        print(f"📁 저장된 파일:")
        print(f"  - best_gesture_cnn_model.pth (제스처 인식 CNN 모델)")
        print(f"  - training_history.png (학습 히스토리 그래프)")
        print(f"💡 제스처 특성 분석:")
        print(f"  - 동적 특성: 움직임 패턴 분석")
        print(f"  - 손 모양: 주먹 vs 손바닥 구분")
        print(f"  - 실시간 인식: 2초 단위로 정확한 제스처 판단")
    else:
        print("⚠️ 모델 파일이 생성되지 않았습니다. 더 많은 데이터가 필요할 수 있습니다.")

if __name__ == "__main__":
    main() 
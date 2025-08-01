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
    """손 제스처 데이터셋 클래스 (진짜 Shift-GCN용)"""
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
    
    def __getitem__(self, idx):
        # 개별 시퀀스와 라벨 가져오기
        sequence, label = self.samples[idx]
        
        # 데이터 형태: (60, 100) - 60프레임, 100특징 (99 + 1라벨)
        frames = sequence.shape[0]  # 60
        features = sequence.shape[1] - 1  # 99 (라벨 제외)
        
        # 입력 데이터와 라벨 분리
        x = sequence[:, :-1].astype(np.float32)  # 특징 데이터 (60, 99)
        y = int(sequence[0, -1])  # 라벨 (첫 번째 프레임의 라벨 사용)
        
        # 랜드마크 좌표만 추출 (x, y, z)
        # 21개 관절 × 4 (x, y, z, visibility) = 84
        landmarks = x[:, :84]  # (60, 84)
        
        # (60, 84) -> (60, 21, 4) -> (60, 21, 3) - visibility 제거
        landmarks = landmarks.reshape(60, 21, 4)  # (60, 21, 4)
        landmarks = landmarks[:, :, :3]  # (60, 21, 3) - x, y, z만 사용
        
        # 디버그 출력 (첫 번째 샘플만)
        if idx == 0:
            print(f"🔍 데이터셋 디버그:")
            print(f"  원본 sequence shape: {sequence.shape}")
            print(f"  특징 데이터 x shape: {x.shape}")
            print(f"  랜드마크 shape: {landmarks.shape}")
            print(f"  라벨: {y}")
        
        return torch.FloatTensor(landmarks), torch.LongTensor([y])

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
        
        # 디버그 출력 (첫 번째 배치만)
        if batch_size > 0:
            print(f"🔍 모델 입력 디버그:")
            print(f"  입력 x shape: {x.shape}")
            print(f"  batch_size: {batch_size}, num_frames: {num_frames}, num_joints: {num_joints}, num_features: {num_features}")
        
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
            torch.save(model.state_dict(), 'best_real_shift_gcn_model.pth')
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
    plt.savefig('real_shift_gcn_training_history.png', dpi=300, bbox_inches='tight')
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
    print("🚀 진짜 Shift-GCN 모델 학습")
    print("💡 그래프 컨볼루션 + Shift 연산 + 관절 관계 학습")
    
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
    batch_size = 16  # GCN은 더 많은 메모리 사용
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    # 모델 생성 (진짜 Shift-GCN)
    model = RealShiftGCN(num_classes=3, num_frames=60, num_joints=21, num_features=3)
    model = model.to(device)
    
    print(f"📈 모델 구조 (진짜 Shift-GCN):")
    print(f"  - 입력: 60프레임 × 21관절 × 3좌표")
    print(f"  - 그래프 구조: 21개 관절 연결 관계")
    print(f"  - Shift 연산: 8개 인접 행렬 분할")
    print(f"  - 출력: 3개 클래스 (come, away, stop)")
    print(f"  - 특징:")
    print(f"    • Graph Convolution: 관절 간 관계 학습")
    print(f"    • Shift 연산: 시간적 패턴 학습")
    print(f"    • 자동 특징 학습: 수동 특성 없음")
    print(model)
    
    # 모델 학습
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=50, learning_rate=0.001
    )
    
    # 학습 히스토리 플롯
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 최고 성능 모델 로드
    if os.path.exists('best_real_shift_gcn_model.pth'):
        model.load_state_dict(torch.load('best_real_shift_gcn_model.pth'))
        
        # 모델 평가
        test_accuracy = evaluate_model(model, test_loader)
        
        print("🎉 학습 완료!")
        print(f"📁 저장된 파일:")
        print(f"  - best_real_shift_gcn_model.pth (진짜 Shift-GCN 모델)")
        print(f"  - real_shift_gcn_training_history.png (학습 히스토리 그래프)")
        print(f"💡 진짜 Shift-GCN 특징:")
        print(f"  - 그래프 컨볼루션: 관절 간 공간적 관계 학습")
        print(f"  - Shift 연산: 시간적 변화 패턴 학습")
        print(f"  - 자동 특징 발견: 모델이 스스로 패턴 학습")
    else:
        print("⚠️ 모델 파일이 생성되지 않았습니다. 더 많은 데이터가 필요할 수 있습니다.")

if __name__ == "__main__":
    main() 
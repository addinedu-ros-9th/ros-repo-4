"""
데이터 크기에 최적화된 Shift-GCN 학습
- 작은 데이터셋에 맞춘 경량 모델
- 과적합 방지 강화
- 최적화된 하이퍼파라미터
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class LightShiftGraphConv(nn.Module):
    """경량화된 Shift Graph Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5):
        super(LightShiftGraphConv, self).__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 1)
        self.bn = nn.BatchNorm2d(out_channels * kernel_size)  # 차원 수정
        self.dropout = nn.Dropout(dropout)
        
        # 더 강한 정규화를 위한 Shift 비율 조정
        self.temporal_shift_ratio = 0.25  # 50% → 25% (경량화)
        self.spatial_shift_ratio = 0.125   # 25% → 12.5% (경량화)
        
    def forward(self, x, A):
        N, C, T, V = x.size()
        
        # Apply shifts
        x = self.apply_shifts(x)
        
        # Graph convolution
        x = self.conv(x)
        
        # Batch normalization before reshape
        x = self.bn(x)
        x = F.relu(x)
        
        x = x.view(N, self.kernel_size, -1, T, V)
        
        # Apply adjacency matrix
        x = torch.einsum('nkctv,vw->nkctw', x, A)
        x = x.contiguous().view(N, -1, T, V)
        
        # Dropout
        x = self.dropout(x)
        
        return x
    
    def apply_shifts(self, x):
        """경량화된 Shift 연산"""
        N, C, T, V = x.size()
        
        temporal_shift_channels = int(C * self.temporal_shift_ratio)
        spatial_shift_channels = int(C * self.spatial_shift_ratio)
        
        out = x.clone()
        
        # Temporal shift (left and right)
        if temporal_shift_channels > 0 and T > 1:
            # Left shift
            out[:, :temporal_shift_channels//2, 1:, :] = x[:, :temporal_shift_channels//2, :-1, :]
            out[:, :temporal_shift_channels//2, 0, :] = 0
            
            # Right shift
            out[:, temporal_shift_channels//2:temporal_shift_channels, :-1, :] = x[:, temporal_shift_channels//2:temporal_shift_channels, 1:, :]
            out[:, temporal_shift_channels//2:temporal_shift_channels, -1, :] = 0
        
        # Spatial shift
        if spatial_shift_channels > 0 and V > 1:
            start_idx = temporal_shift_channels
            end_idx = start_idx + spatial_shift_channels
            out[:, start_idx:end_idx, :, 1:] = x[:, start_idx:end_idx, :, :-1]
            out[:, start_idx:end_idx, :, 0] = 0
        
        return out

class OptimizedShiftGCN(nn.Module):
    """작은 데이터셋에 최적화된 Shift-GCN"""
    
    def __init__(self, num_classes, num_joints, in_channels=3, temporal_kernel_size=5, dropout=0.5):
        super(OptimizedShiftGCN, self).__init__()
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        
        # Adjacency matrix
        self.register_buffer('A', torch.eye(num_joints))
        
        # 더 강한 정규화
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # 경량화된 Graph convolution blocks (10개 → 6개)
        self.l1 = LightShiftGraphConv(in_channels, 32, 3, dropout=dropout)  # 64 → 32
        self.l2 = LightShiftGraphConv(32, 32, 3, dropout=dropout)
        self.l3 = LightShiftGraphConv(32, 64, 3, dropout=dropout)  # stride=2 제거
        self.l4 = LightShiftGraphConv(64, 64, 3, dropout=dropout)
        self.l5 = LightShiftGraphConv(64, 128, 3, dropout=dropout)
        self.l6 = LightShiftGraphConv(128, 128, 3, dropout=dropout)
        
        # 더 작은 Temporal convolution
        self.tcn = nn.Sequential(
            nn.Conv2d(128, 128, (temporal_kernel_size, 1), padding=(temporal_kernel_size//2, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 더 강한 정규화를 위한 중간 층 추가
        self.fc_intermediate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
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
        
        # Graph convolution layers (경량화)
        x = self.l1(x, self.A)
        x = self.l2(x, self.A)
        x = self.l3(x, self.A)
        x = self.l4(x, self.A)
        x = self.l5(x, self.A)
        x = self.l6(x, self.A)
        
        # Temporal convolution
        x = self.tcn(x)
        
        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])  # (N, 128, 1, 1)
        x = x.view(N, -1)  # (N, 128)
        
        # Classification with intermediate layer
        x = self.fc_intermediate(x)
        
        return x

class GestureDataset(Dataset):
    """제스처 데이터셋 - 데이터 증강 추가"""
    
    def __init__(self, data_dir, split='train', test_size=0.2, random_state=42, augment=True):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment and (split == 'train')  # 훈련 시에만 증강
        
        # Load data and labels
        self.data, self.labels, self.action_to_idx = self.load_data()
        
        # Split train/test
        if len(self.data) > 0:
            # Stratified split for balanced classes
            train_data, test_data, train_labels, test_labels = train_test_split(
                self.data, self.labels, test_size=test_size, random_state=random_state, 
                stratify=self.labels
            )
            
            if split == 'train':
                self.data = train_data
                self.labels = train_labels
            else:
                self.data = test_data
                self.labels = test_labels
        
        print(f"✅ {split} 데이터셋 로드 완료: {len(self.data)}개 샘플")
        if self.augment:
            print(f"   🔄 데이터 증강 활성화 (훈련 데이터만)")
    
    def load_data(self):
        all_data = []
        all_labels = []
        action_to_idx = {}
        
        data_files = glob.glob(os.path.join(self.data_dir, '*_pose_data.npy'))
        
        if len(data_files) == 0:
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {self.data_dir}")
            return [], [], {}
        
        for data_file in data_files:
            action_name = os.path.basename(data_file).replace('_pose_data.npy', '')
            
            if action_name not in action_to_idx:
                action_to_idx[action_name] = len(action_to_idx)
            
            action_data = np.load(data_file)
            action_labels = [action_to_idx[action_name]] * len(action_data)
            
            all_data.extend(action_data)
            all_labels.extend(action_labels)
            
            print(f"📊 {action_name}: {len(action_data)}개 샘플, shape: {action_data.shape}")
        
        return np.array(all_data), np.array(all_labels), action_to_idx
    
    def augment_data(self, data):
        """간단한 데이터 증강"""
        # Random temporal jittering (시간축 변형)
        if np.random.rand() < 0.3:
            # Random frame dropping
            T = data.shape[1]
            keep_ratio = 0.9
            keep_frames = int(T * keep_ratio)
            indices = np.sort(np.random.choice(T, keep_frames, replace=False))
            data = data[:, indices, :, :]
            
            # Pad back to original length
            if data.shape[1] < T:
                padding = T - data.shape[1]
                last_frame = data[:, -1:, :, :].repeat(padding, axis=1)
                data = np.concatenate([data, last_frame], axis=1)
        
        # Random spatial jittering (공간축 변형)
        if np.random.rand() < 0.3:
            noise_factor = 0.02
            noise = np.random.normal(0, noise_factor, data.shape)
            data = data + noise
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx].copy()
        
        if self.augment:
            data = self.augment_data(data)
        
        data = torch.FloatTensor(data)
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

def train_optimized_shift_gcn(data_dir, model_save_path, epochs=200, batch_size=8, lr=0.0005):
    """최적화된 Shift-GCN 학습"""
    
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리가 없습니다: {data_dir}")
        return
    
    # Load metadata
    metadata_file = os.path.join(data_dir, 'metadata.npy')
    if os.path.exists(metadata_file):
        metadata = np.load(metadata_file, allow_pickle=True).item()
        num_joints = metadata['num_joints']
        target_joints = metadata['target_joints']
        print(f"📊 메타데이터 로드: {target_joints}, {num_joints}개 관절점")
    else:
        print("⚠️ 메타데이터가 없습니다. 기본값 사용")
        num_joints = 9
    
    # Load adjacency matrix
    adj_file = os.path.join(data_dir, 'adjacency_matrix.npy')
    if os.path.exists(adj_file):
        adjacency_matrix = np.load(adj_file)
        print(f"📊 인접 행렬 로드: {adjacency_matrix.shape}")
    else:
        print("⚠️ 인접 행렬이 없습니다. 단위 행렬 사용")
        adjacency_matrix = np.eye(num_joints)
    
    # Create datasets with augmentation
    train_dataset = GestureDataset(data_dir, split='train', test_size=0.2, augment=True)
    test_dataset = GestureDataset(data_dir, split='test', test_size=0.2, augment=False)
    
    if len(train_dataset) == 0:
        print("❌ 학습 데이터가 없습니다.")
        return
    
    print(f"\n📊 데이터셋 분석:")
    print(f"   - 훈련 샘플: {len(train_dataset)}")
    print(f"   - 테스트 샘플: {len(test_dataset)}")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 총 배치 수: {len(train_dataset) // batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get number of classes
    num_classes = len(train_dataset.action_to_idx)
    print(f"📊 클래스 수: {num_classes}")
    print(f"📊 액션 매핑: {train_dataset.action_to_idx}")
    
    # Create optimized model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedShiftGCN(
        num_classes=num_classes, 
        num_joints=num_joints, 
        temporal_kernel_size=5,  # 9 → 5 (경량화)
        dropout=0.5  # 0.3 → 0.5 (강한 정규화)
    )
    model.set_adjacency_matrix(adjacency_matrix)
    model = model.to(device)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🚀 최적화된 모델 생성 완료: {device}")
    print(f"   - 총 파라미터: {total_params:,}")
    print(f"   - 훈련 가능 파라미터: {trainable_params:,}")
    
    # Loss and optimizer (더 보수적인 설정)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # weight_decay 증가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20
    )  # verbose 제거
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 50  # 조기 종료 patience
    
    print(f"\n🚀 최적화된 Shift-GCN 학습 시작")
    print(f"   - 에포크: {epochs}")
    print(f"   - 학습률: {lr}")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 조기 종료: {early_stopping_patience} 에포크")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        test_accuracies.append(test_acc)
        
        # Update scheduler
        scheduler.step(test_acc)
        
        # Save best model and early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 더 자주 출력
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Acc: {test_acc:.2f}%, Best: {best_test_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Patience: {patience_counter}/{early_stopping_patience}')
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️ 조기 종료: {early_stopping_patience} 에포크 동안 개선 없음")
            break
    
    print(f"\n🎉 학습 완료!")
    print(f"   최고 테스트 정확도: {best_test_acc:.2f}%")
    print(f"   실제 에포크: {epoch + 1}/{epochs}")
    print(f"   모델 저장: {model_save_path}")
    
    # Final evaluation (이후 코드는 기존과 동일)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    idx_to_action = {v: k for k, v in train_dataset.action_to_idx.items()}
    action_names = [idx_to_action[i] for i in range(num_classes)]
    
    print(f"\n📊 최종 성능 평가:")
    print(classification_report(all_labels, all_predictions, target_names=action_names))
    
    # Save plots
    plots_dir = os.path.dirname(model_save_path)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=action_names, yticklabels=action_names)
    plt.title('Optimized Shift-GCN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'optimized_confusion_matrix.png'))
    plt.close()
    
    # Training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train', alpha=0.7)
    plt.plot(test_accuracies, label='Test', alpha=0.7)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    overfitting_gap = [train_accuracies[i] - test_accuracies[i] for i in range(len(train_accuracies))]
    plt.plot(overfitting_gap, color='red', alpha=0.7)
    plt.title('Overfitting Gap (Train - Test)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'optimized_training_history.png'))
    plt.close()
    
    print(f"📊 최적화된 평가 결과 저장: {plots_dir}")

if __name__ == "__main__":
    # 최적화된 학습 설정
    data_dir = "./shift_gcn_data"
    model_save_path = "./models/optimized_shift_gcn_model.pth"  # models 폴더로 변경
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print("🚀 최적화된 Shift-GCN 제스처 인식 학습 시작")
    print("=" * 60)
    
    # 작은 데이터셋에 최적화된 설정
    train_optimized_shift_gcn(
        data_dir=data_dir,
        model_save_path=model_save_path,
        epochs=200,      # 더 많은 에포크 (조기 종료로 제어)
        batch_size=8,    # 더 작은 배치 (안정적 학습)
        lr=0.0005        # 더 작은 학습률 (안정적 수렴)
    ) 
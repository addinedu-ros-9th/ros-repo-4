"""
100개 샘플에 최적화된 간단한 Shift-GCN
- 안정적인 아키텍처
- 강한 정규화
- 과적합 방지
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

class SimpleShiftGCN(nn.Module):
    """간단하고 안정적인 Shift-GCN"""
    
    def __init__(self, num_classes, num_joints, in_channels=3, dropout=0.5):
        super(SimpleShiftGCN, self).__init__()
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        
        # Adjacency matrix
        self.register_buffer('A', torch.eye(num_joints))
        
        # 입력 정규화
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # 간단한 3층 구조
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

class GestureDataset(Dataset):
    """제스처 데이터셋"""
    
    def __init__(self, data_dir, split='train', test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.split = split
        
        # Load data and labels
        self.data, self.labels, self.action_to_idx = self.load_data()
        
        # Split train/test
        if len(self.data) > 0:
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

def train_simple_shift_gcn(data_dir, model_save_path, epochs=150, batch_size=8, lr=0.001):
    """간단한 Shift-GCN 학습"""
    
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
    
    # Create datasets
    train_dataset = GestureDataset(data_dir, split='train', test_size=0.2)
    test_dataset = GestureDataset(data_dir, split='test', test_size=0.2)
    
    if len(train_dataset) == 0:
        print("❌ 학습 데이터가 없습니다.")
        return
    
    print(f"\n📊 데이터셋 분석:")
    print(f"   - 훈련 샘플: {len(train_dataset)}")
    print(f"   - 테스트 샘플: {len(test_dataset)}")
    print(f"   - 배치 크기: {batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get number of classes
    num_classes = len(train_dataset.action_to_idx)
    print(f"📊 클래스 수: {num_classes}")
    print(f"📊 액션 매핑: {train_dataset.action_to_idx}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleShiftGCN(
        num_classes=num_classes, 
        num_joints=num_joints, 
        dropout=0.5
    )
    model.set_adjacency_matrix(adjacency_matrix)
    model = model.to(device)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 간단한 모델 생성 완료: {device}")
    print(f"   - 총 파라미터: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 30
    
    print(f"\n🚀 간단한 Shift-GCN 학습 시작")
    print(f"   - 에포크: {epochs}")
    print(f"   - 학습률: {lr}")
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
            
            # Gradient clipping
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
        scheduler.step()
        
        # Save best model and early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Acc: {test_acc:.2f}%, Best: {best_test_acc:.2f}%')
            print(f'  Patience: {patience_counter}/{early_stopping_patience}')
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️ 조기 종료: {early_stopping_patience} 에포크 동안 개선 없음")
            break
    
    print(f"\n🎉 학습 완료!")
    print(f"   최고 테스트 정확도: {best_test_acc:.2f}%")
    print(f"   실제 에포크: {epoch + 1}/{epochs}")
    print(f"   모델 저장: {model_save_path}")
    
    # Final evaluation
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
    plt.title('Simple Shift-GCN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'simple_confusion_matrix.png'))
    plt.close()
    
    # Training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train', alpha=0.7)
    plt.plot(test_accuracies, label='Test', alpha=0.7)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'simple_training_history.png'))
    plt.close()
    
    print(f"📊 평가 결과 저장: {plots_dir}")

if __name__ == "__main__":
    # 간단하고 안정적인 학습 설정
    data_dir = "./shift_gcn_data"
    model_save_path = "./models/simple_shift_gcn_model.pth"
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print("🚀 간단한 Shift-GCN 제스처 인식 학습 시작")
    print("=" * 60)
    
    # 100개 샘플에 안정적인 설정
    train_simple_shift_gcn(
        data_dir=data_dir,
        model_save_path=model_save_path,
        epochs=150,
        batch_size=8,
        lr=0.001
    ) 
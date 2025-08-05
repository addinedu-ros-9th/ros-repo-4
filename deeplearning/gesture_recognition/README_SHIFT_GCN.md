# 🤖 Shift-GCN을 사용한 제스처 인식 시스템

YOLO Pose + Shift-GCN을 활용한 고성능 제스처 인식 시스템입니다.

## 📋 목차
1. [개요](#개요)
2. [시스템 구조](#시스템-구조)
3. [설치 및 설정](#설치-및-설정)
4. [사용법](#사용법)
5. [성능 최적화](#성능-최적화)
6. [트러블슈팅](#트러블슈팅)

## 🎯 개요

### Shift-GCN의 장점
- **ST-GCN 개선**: 기존 ST-GCN에 Shift 연산 추가로 성능 향상
- **효율적인 시공간 모델링**: Temporal Shift와 Spatial Shift로 관절점 간의 관계를 더 잘 학습
- **적은 파라미터**: ST-GCN 대비 비슷한 성능에 더 적은 파라미터 사용

### 데이터 형식
```
Shift-GCN 입력: (C, T, V, M)
- C: 채널 (3 - x, y, confidence)
- T: 시간 프레임 수 (64)
- V: 관절점 수 (선택 가능)
- M: 사람 수 (1)
```

### 관절점 옵션
- **'arms_only'**: 어깨부터 손까지 (6개) - 빠른 처리
- **'upper_body'**: 상체 전체 (9개) - 권장
- **'full_body'**: 전신 (17개) - 고품질

## 🏗️ 시스템 구조

```
영상 생성 → 관절점 추출 → 데이터 전처리 → 모델 학습 → 추론
    ↓           ↓            ↓           ↓        ↓
create_pose_  extract_     Shift-GCN   train_   실시간
dataset.py    pose_        data        shift_   인식
             keypoints.py  format      gcn.py
```

## 🚀 설치 및 설정

### 필요한 패키지
```bash
pip install ultralytics torch torchvision opencv-python numpy scikit-learn matplotlib seaborn
```

### YOLO Pose 모델 다운로드
```bash
# YOLOv8 Pose 모델 자동 다운로드됨 (첫 실행 시)
# 또는 수동 다운로드:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
```

## 📖 사용법

### 1단계: 영상 데이터 생성 ✅ (완료)

```bash
python create_pose_dataset.py
```

현재 설정:
- 해상도: 풀 해상도 (카메라 최대)
- 액션: 'come', 'normal'
- 샘플: 각 50개
- 영상 길이: 3초

### 2단계: 관절점 추출 및 Shift-GCN 데이터 생성

```bash
python extract_pose_keypoints.py
```

**관절점 선택 가이드:**
- **어깨부터 손까지만**: `target_joints='arms_only'`
- **상체 전체 (권장)**: `target_joints='upper_body'`  
- **전신**: `target_joints='full_body'`

```python
# extract_pose_keypoints.py 설정 변경 예시
process_gesture_videos(
    video_dir="./pose_dataset",
    output_dir="./shift_gcn_data", 
    target_joints='upper_body',  # 👈 여기서 변경
    target_frames=64
)
```

**생성되는 파일:**
```
shift_gcn_data/
├── come_pose_data.npy      # come 액션 데이터
├── normal_pose_data.npy    # normal 액션 데이터  
├── adjacency_matrix.npy    # 관절점 연결 정보
└── metadata.npy            # 메타데이터
```

### 3단계: Shift-GCN 모델 학습

```bash
python train_shift_gcn.py
```

**학습 설정:**
```python
train_shift_gcn(
    data_dir="./shift_gcn_data",
    model_save_path="./shift_gcn_model.pth",
    epochs=100,
    batch_size=16,
    lr=0.001
)
```

**학습 결과:**
```
shift_gcn_model.pth        # 학습된 모델
confusion_matrix.png       # 혼동 행렬
training_history.png       # 학습 곡선
```

### 4단계: 실시간 추론 (예시)

```python
import torch
import numpy as np
from train_shift_gcn import ShiftGCN
from extract_pose_keypoints import YOLOPoseExtractor

# 모델 로드
model = ShiftGCN(num_classes=2, num_joints=9)
model.load_state_dict(torch.load('./shift_gcn_model.pth'))
model.eval()

# 관절점 추출기
extractor = YOLOPoseExtractor(target_joints='upper_body')

# 실시간 추론
def predict_gesture(video_frame):
    keypoints, _ = extractor.extract_keypoints_from_video(video_frame)
    if keypoints is not None:
        normalized = extractor.normalize_keypoints(keypoints)
        shift_gcn_data = extractor.convert_to_shift_gcn_format(normalized)
        
        with torch.no_grad():
            output = model(torch.FloatTensor(shift_gcn_data).unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()
            
        return ['come', 'normal'][prediction]
    return 'unknown'
```

## ⚡ 성능 최적화

### 1. 관절점 선택 최적화

```python
# 빠른 처리가 필요한 경우
target_joints = 'arms_only'  # 6개 관절점

# 성능과 속도의 균형
target_joints = 'upper_body'  # 9개 관절점 (권장)

# 최고 성능이 필요한 경우  
target_joints = 'full_body'  # 17개 관절점
```

### 2. 프레임 수 조정

```python
# 빠른 처리
target_frames = 32

# 기본 설정 (권장)
target_frames = 64

# 고품질 (더 긴 시퀀스)
target_frames = 128
```

### 3. 모델 크기 조정

```python
# 가벼운 모델
model = ShiftGCN(num_classes=2, num_joints=9, dropout=0.5)

# 성능 중심 모델
model = ShiftGCN(num_classes=2, num_joints=9, dropout=0.1)
```

## 🔧 트러블슈팅

### Q1: 관절점 추출이 실패하는 경우
**원인**: 사람이 잘 보이지 않거나 해상도가 낮음
**해결**: 
- 영상 품질 확인
- 카메라 거리 조정
- 조명 개선

### Q2: 메모리 부족 오류
**원인**: 배치 사이즈가 너무 크거나 프레임 수가 많음
**해결**:
```python
# 배치 사이즈 줄이기
batch_size = 8  # 16에서 8로

# 프레임 수 줄이기  
target_frames = 32  # 64에서 32로
```

### Q3: 학습 정확도가 낮은 경우
**원인**: 데이터 부족, 액션 구분이 어려움
**해결**:
- 더 많은 데이터 수집
- 액션 간 차이를 더 명확히
- Data Augmentation 추가

### Q4: 추론 속도가 느린 경우
**해결**:
```python
# 모델 경량화
target_joints = 'arms_only'
target_frames = 32

# GPU 사용 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 📊 예상 성능

| 관절점 | 프레임 | 정확도 | 추론시간 | 메모리 |
|--------|--------|--------|----------|--------|
| 6개    | 32     | ~85%   | 5ms     | 낮음   |
| 9개    | 64     | ~90%   | 8ms     | 보통   | 
| 17개   | 128    | ~95%   | 15ms    | 높음   |

## 🎯 다음 단계

1. **데이터 증강**: 회전, 스케일링, 노이즈 추가
2. **멀티모달**: RGB + Pose 정보 결합
3. **실시간 최적화**: TensorRT, ONNX 변환
4. **추가 액션**: 더 많은 제스처 추가

## 📚 참고자료

- [Shift-GCN 논문](https://arxiv.org/abs/2003.14111)
- [ST-GCN 논문](https://arxiv.org/abs/1801.07455) 
- [YOLO Pose](https://docs.ultralytics.com/tasks/pose/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

**💡 Tip**: 처음 사용하시는 경우 'upper_body' 관절점으로 시작하는 것을 권장합니다! 
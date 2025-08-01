## 🎯 인식 가능한 객체

현재 COCO 데이터셋 기반으로 다음 객체들을 인식합니다:

| 객체 | 설명 | 색상 |
|------|------|------|
| person | 사람 | 초록색 |
| chair | 의자 | 파란색 |
| dining-table | 책상/테이블 | 빨간색 |
| handbag | 가방 | 청록색 |
| suitcase | 캐리어/큰 가방 | 마젠타색 |

## ⚙️ 설정 변경

### 신뢰도 임계값 조정
`simple_detector.py`의 53번째 줄에서 변경 가능:
```python
if class_id in self.target_classes and confidence > 0.5:  # 0.5 → 다른 값으로 변경
```

### 새로운 객체 클래스 추가
`target_classes` 딕셔너리에 COCO 클래스 추가:
```python
self.target_classes = {
    0: 'person',
    56: 'chair',
    # 새로운 클래스 추가
    62: 'tv',  # 예시: TV
}
```

## 🔧 다음 단계

1. **문 인식 모듈** 개발
2. **커스텀 데이터셋** 학습 (강의실 특화)
3. **ROS2 연동** 
4. **실시간 성능 최적화**

## 🐛 문제 해결

### 웹캠이 인식되지 않는 경우
```python
cap = cv2.VideoCapture(0)  # 0을 1, 2로 변경해보세요
```

### GPU 사용하고 싶은 경우
CUDA가 설치되어 있다면 자동으로 GPU를 사용합니다.
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")
``` 
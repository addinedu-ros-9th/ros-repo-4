"""
🚀 완전 새로운 통합 시스템
- MediaPipe 수준의 완벽한 색상 매칭
- YOLO Pose + Shift-GCN 제스처 인식  
- 멀티스레딩으로 고성능 달성
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import os
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Qt 오류 방지
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_DEBUG_PLUGINS'] = '0'

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

class SimpleShiftGCN(nn.Module):
    """Shift-GCN 모델 (학습된 구조와 동일)"""
    
    def __init__(self, num_classes, num_joints, in_channels=3, dropout=0.5):
        super(SimpleShiftGCN, self).__init__()
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        
        # Adjacency matrix
        self.register_buffer('A', torch.eye(num_joints))
        
        # 입력 정규화
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # 3층 구조
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

class NewIntegratedSystem:
    def __init__(self):
        print("🚀 새로운 통합 시스템 초기화 시작")
        
        # YOLO 모델들
        self.person_model = YOLO('./yolov8s-seg.pt')  # 사람 감지 + 세그멘테이션
        self.pose_model = YOLO('../yolov8n-pose.pt')   # 포즈 감지
        print("✅ YOLO 모델 로드 완료")
        
        # Shift-GCN 모델 로드
        self.load_gesture_model()
        
        # MediaPipe 수준의 완벽한 색상 매칭 설정
        self.people_data = {}
        self.next_id = 0
        self.match_threshold = 0.50
        self.reentry_threshold = 0.45
        self.min_detection_confidence = 0.5
        self.min_person_area = 4000
        self.max_histograms_per_person = 20
        
        # 제스처 인식 설정
        self.gesture_frame_buffer = deque(maxlen=90)  # 90 프레임
        self.gesture_prediction_buffer = deque(maxlen=5)
        self.actions = ['COME', 'NORMAL']  # 2클래스
        self.min_gesture_frames = 90  # 90프레임이 모여야 판단 (3초 단위)
        self.gesture_confidence_threshold = 0.6
        
        # 3초 단위 판단 설정
        self.gesture_decision_interval = 90  # 90프레임(3초)마다 새로운 판단
        self.last_gesture_decision_frame = 0
        self.current_gesture_confidence = 0.5
        
        # 빠른 NORMAL 복구를 위한 설정
        self.static_detection_frames = 0
        self.static_threshold = 60  # 30 → 60프레임(2초)으로 증가
        # 보호기간 제거
        
        # 상체 관절점 (YOLO Pose 17개 중 상체 9개) - 학습과 동일하게
        self.upper_body_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]  # 코+어깨+팔+엉덩이
        self.display_joints = [5, 6, 7, 8, 9, 10]  # 화면 표시용 (어깨+팔만)
        
        # 인접 행렬 생성
        self.adjacency_matrix = self.create_adjacency_matrix()
        if hasattr(self, 'gesture_model'):
            self.gesture_model.set_adjacency_matrix(self.adjacency_matrix)
        
        # 멀티스레딩 설정
        self.setup_threading()
        
        # 모델 테스트 (COME/NORMAL 구분 능력 확인)
        self.test_model_performance()
        
        print("🎉 새로운 통합 시스템 초기화 완료!")
    
    def load_gesture_model(self):
        """Shift-GCN 모델 로드"""
        # 절대 경로로 수정
        model_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/models/simple_shift_gcn_model.pth'
        
        if os.path.exists(model_path):
            self.gesture_model = SimpleShiftGCN(num_classes=2, num_joints=9, dropout=0.5)
            self.gesture_model.load_state_dict(torch.load(model_path, map_location=device))
            self.gesture_model = self.gesture_model.to(device)
            self.gesture_model.eval()
            self.enable_gesture_recognition = True
            print(f"✅ Shift-GCN 모델 로드 성공: {model_path}")
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
            self.enable_gesture_recognition = False
        
    def get_person_color(self, person_id):
        """Person ID별 고유 색상 생성"""
        # 다양한 색상 팔레트
        colors = [
            (0, 255, 0),    # 초록
            (255, 0, 0),    # 빨강  
            (0, 0, 255),    # 파랑
            (255, 255, 0),  # 노랑
            (255, 0, 255),  # 마젠타
            (0, 255, 255),  # 시안
            (255, 128, 0),  # 주황
            (128, 0, 255),  # 보라
            (0, 128, 255),  # 하늘
            (255, 128, 128) # 분홍
        ]
        
        # Person ID에서 숫자 추출
        if isinstance(person_id, str) and '_' in person_id:
            try:
                id_num = int(person_id.split('_')[-1])
                return colors[id_num % len(colors)]
            except:
                pass
        
        # 기본 색상
        return (0, 255, 0)
    
    def draw_keypoints(self, frame, keypoints, color=(0, 255, 0)):
        """관절점 시각화 (상체만)"""
        if keypoints is None or len(keypoints) == 0:
            return frame
            
        # 상체 관절점 인덱스 (YOLO Pose 17개 중)
        # 5: 왼쪽 어깨, 6: 오른쪽 어깨, 7: 왼쪽 팔꿈치, 8: 오른쪽 팔꿈치
        # 9: 왼쪽 손목, 10: 오른쪽 손목, 0: 코
        display_joints = [0, 5, 6, 7, 8, 9, 10]  # 얼굴 + 상체
        
        # 관절점 연결선 정의
        connections = [
            (5, 6),   # 어깨 연결
            (5, 7),   # 왼쪽 어깨-팔꿈치
            (7, 9),   # 왼쪽 팔꿈치-손목
            (6, 8),   # 오른쪽 어깨-팔꿈치
            (8, 10),  # 오른쪽 팔꿈치-손목
        ]
        
        # 키포인트 그리기
        for joint_idx in display_joints:
            if joint_idx < len(keypoints):
                x, y, conf = keypoints[joint_idx]
                if conf > 0.3:  # 신뢰도 0.3 이상만
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                    cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 1)
        
        # 연결선 그리기
        for joint1, joint2 in connections:
            if joint1 < len(keypoints) and joint2 < len(keypoints):
                x1, y1, conf1 = keypoints[joint1]
                x2, y2, conf2 = keypoints[joint2]
                if conf1 > 0.3 and conf2 > 0.3:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        return frame
    
    def create_adjacency_matrix(self):
        """인접 행렬 생성 (상체 관절점 연결)"""
        num_joints = len(self.upper_body_joints)
        A = np.eye(num_joints)
        
        # 관절점 연결 정의
        connections = [
            (5, 6),   # 어깨
            (5, 7),   # 왼쪽 어깨-팔꿈치  
            (7, 9),   # 왼쪽 팔꿈치-손목
            (6, 8),   # 오른쪽 어깨-팔꿈치
            (8, 10),  # 오른쪽 팔꿈치-손목
            (0, 1),   # 얼굴 연결
            (0, 2),
            (1, 2)
        ]
        
        # 매핑: 원본 관절점 → 상체 배열 인덱스
        joint_mapping = {joint: i for i, joint in enumerate(self.upper_body_joints)}
        
        for joint1, joint2 in connections:
            if joint1 in joint_mapping and joint2 in joint_mapping:
                i, j = joint_mapping[joint1], joint_mapping[joint2]
                A[i, j] = 1
                A[j, i] = 1
        
        return A
    
    def setup_threading(self):
        """멀티스레딩 설정"""
        self.frame_queue = queue.Queue(maxsize=3)
        self.detection_queue = queue.Queue(maxsize=3)
        self.gesture_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.lock = threading.Lock()
        
        # 공유 결과 (키포인트 정보 포함)
        self.latest_detections = []
        self.latest_gesture = ("NORMAL", self.current_gesture_confidence, False, None)
        
        print("🔧 멀티스레딩 설정 완료")
    
    # MediaPipe의 완벽한 색상 매칭 로직 (그대로 가져옴)
    def extract_histogram(self, img, mask, bins=16):
        """MediaPipe 수준의 완벽한 히스토그램 추출"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 전체 마스크에서 히스토그램 추출
        h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])
        
        # 상체 부분 강조 (MediaPipe 방식)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 상체 부분 마스크 (상위 60%)
            upper_mask = np.zeros_like(mask)
            upper_y = y + int(h * 0.6)
            upper_mask[y:upper_y, x:x+w] = mask[y:upper_y, x:x+w]
            
            # 상체 히스토그램
            h_hist_upper = cv2.calcHist([hsv], [0], upper_mask, [bins], [0, 180])
            s_hist_upper = cv2.calcHist([hsv], [1], upper_mask, [bins], [0, 256])
            v_hist_upper = cv2.calcHist([hsv], [2], upper_mask, [bins], [0, 256])
        else:
            h_hist_upper = h_hist.copy()
            s_hist_upper = s_hist.copy()
            v_hist_upper = v_hist.copy()
        
        # 정규화
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        h_hist_upper = cv2.normalize(h_hist_upper, h_hist_upper).flatten()
        s_hist_upper = cv2.normalize(s_hist_upper, s_hist_upper).flatten()
        v_hist_upper = cv2.normalize(v_hist_upper, v_hist_upper).flatten()
        
        # 전체 + 상체 결합 (상체 70% 가중치)
        combined_hist = np.concatenate([
            h_hist * 0.3, s_hist * 0.3, v_hist * 0.3,
            h_hist_upper * 0.7, s_hist_upper * 0.7, v_hist_upper * 0.7
        ])
        
        return combined_hist, h_hist, s_hist, v_hist
    
    def calculate_similarity_metrics(self, hist1, hist2):
        """다양한 유사도 메트릭 계산 (MediaPipe 방식)"""
        bhatt_dist = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        
        dot_product = np.dot(hist1, hist2)
        norm1 = np.linalg.norm(hist1)
        norm2 = np.linalg.norm(hist2)
        cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        corr = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
        chi_square = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CHISQR)
        intersection = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_INTERSECT)
        
        return {
            'bhattacharyya': bhatt_dist,
            'cosine_similarity': cosine_sim,
            'correlation': corr,
            'chi_square': chi_square,
            'intersection': intersection
        }
    
    def find_best_match(self, current_hist, current_bbox, used_ids, elapsed_time=0.0):
        """가장 유사한 사람 찾기 (MediaPipe 방식)"""
        best_match_id = None
        best_score = 0.0
        best_metrics = {}
        
        x1, y1, x2, y2 = current_bbox
        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        current_area = (x2 - x1) * (y2 - y1)
        
        for pid, pdata in self.people_data.items():
            if pid in used_ids:
                continue
                
            if len(pdata['histograms']) == 0:
                continue
            
            # 1. 히스토그램 유사도 계산 (가장 중요)
            hist_scores = []
            for i, stored_hist in enumerate(pdata['histograms'][-self.max_histograms_per_person:]):
                metrics = self.calculate_similarity_metrics(current_hist, stored_hist)
                # Bhattacharyya 거리와 코사인 유사도를 모두 고려
                hist_score = max(1.0 - metrics['bhattacharyya'], metrics['cosine_similarity'])
                hist_scores.append(hist_score)
            
            # 최고 히스토그램 점수 (가장 유사한 히스토그램 사용)
            best_hist_score = max(hist_scores) if hist_scores else 0.0
            
            # 2. 공간적 유사도 계산
            latest_bbox = pdata['bboxes'][-1]
            stored_center = ((latest_bbox[0] + latest_bbox[2]) // 2, (latest_bbox[1] + latest_bbox[3]) // 2)
            stored_area = (latest_bbox[2] - latest_bbox[0]) * (latest_bbox[3] - latest_bbox[1])
            
            # 중심점 거리
            center_distance = np.sqrt((current_center[0] - stored_center[0])**2 + 
                                     (current_center[1] - stored_center[1])**2)
            max_distance = np.sqrt(640**2 + 480**2)
            spatial_score = 1.0 - (center_distance / max_distance)
            
            # 3. 크기 유사도 (사람이 갑자기 크게 변하지 않음)
            area_ratio = min(current_area, stored_area) / max(current_area, stored_area)
            size_score = area_ratio
            
            # 4. 시간적 연속성 (최근에 본 사람에게 더 높은 가중치)
            if pdata['timestamps']:
                time_since_last_seen = elapsed_time - pdata['timestamps'][-1]
                time_score = max(0.0, 1.0 - (time_since_last_seen / 10.0))  # 10초 내에 본 사람에게 높은 점수
            else:
                time_score = 0.0
            
            # 5. 종합 점수 계산 (가중치 조정)
            total_score = (
                0.7 * best_hist_score +      # 히스토그램 (가장 중요)
                0.15 * spatial_score +       # 공간적 위치
                0.1 * size_score +           # 크기 유사도
                0.05 * time_score            # 시간적 연속성
            )
            
            # 6. 최소 임계값 검사 (히스토그램 점수가 너무 낮으면 제외)
            if best_hist_score < 0.35:  # 0.3 → 0.35로 증가 (더 엄격한 히스토그램 매칭)
                continue
            
            # 7. 더 높은 점수를 가진 매칭 선택
            if total_score > best_score:
                best_score = total_score
                best_match_id = pid
                best_metrics = {
                    'hist_score': best_hist_score,
                    'spatial_score': spatial_score,
                    'size_score': size_score,
                    'time_score': time_score,
                    'hist_scores': hist_scores
                }
        
        return best_match_id, best_score, best_metrics
    
    def person_detection_worker(self):
        """사람 감지 워커 (MediaPipe 매칭 로직 사용)"""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    break
                
                frame, frame_id, elapsed_time = frame_data
                
                # 사람 감지
                results = self.person_model(frame, classes=[0], 
                                          imgsz=256, conf=0.6, verbose=False, device=0)
                
                detections = []
                for result in results:
                    if result.masks is None:
                        continue
                        
                    for i in range(len(result.boxes)):
                        seg = result.masks.data[i]
                        box = result.boxes[i]
                        confidence = box.conf[0].item()
                        
                        if confidence < self.min_detection_confidence:
                            continue
                        
                        # 마스크 처리
                        mask = seg.cpu().numpy().astype(np.uint8) * 255
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                        
                        # 노이즈 제거
                        kernel = np.ones((5,5), np.uint8)
                        mask_cleaned = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
                        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
                        
                        # MediaPipe 수준 히스토그램
                        combined_hist, h_hist, s_hist, v_hist = self.extract_histogram(frame, mask_cleaned)
                        
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area < self.min_person_area:
                            continue
                        
                        detections.append({
                            'hist': combined_hist,
                            'bbox': bbox,
                            'mask': mask_cleaned,
                            'confidence': confidence,
                            'area': area
                        })
                
                # 매칭 처리 (MediaPipe 방식)
                detections.sort(key=lambda x: x['area'], reverse=True)  # 큰 사람부터
                used_ids = set()
                matched_people = []
                
                for detection in detections:
                    matched_id, match_score, metrics = self.find_best_match(
                        detection['hist'], detection['bbox'], used_ids, elapsed_time)
                    
                    if matched_id is not None and match_score > self.match_threshold:
                        # 기존 사람 업데이트
                        self.people_data[matched_id]['histograms'].append(detection['hist'])
                        self.people_data[matched_id]['bboxes'].append(detection['bbox'])
                        self.people_data[matched_id]['timestamps'].append(elapsed_time)
                        used_ids.add(matched_id)
                        
                        # 히스토그램 메모리 관리
                        if len(self.people_data[matched_id]['histograms']) > self.max_histograms_per_person:
                            self.people_data[matched_id]['histograms'].pop(0)
                            self.people_data[matched_id]['bboxes'].pop(0)
                            self.people_data[matched_id]['timestamps'].pop(0)
                        
                        person_id = matched_id
                        color = self.get_person_color(matched_id)
                        
                        # 매칭 정보 디버깅 (주기적으로)
                        if frame_id % 120 == 0 and metrics:
                            print(f"   🔍 매칭: {matched_id} (점수: {match_score:.3f})")
                            print(f"      - 히스토그램: {metrics['hist_score']:.3f}")
                            print(f"      - 공간적: {metrics['spatial_score']:.3f}")
                            print(f"      - 크기: {metrics['size_score']:.3f}")
                            print(f"      - 시간: {metrics['time_score']:.3f}")
                    else:
                        # 새로운 사람
                        new_id = f"Person_{self.next_id}"
                        self.people_data[new_id] = {
                            'histograms': [detection['hist']],
                            'bboxes': [detection['bbox']],
                            'timestamps': [elapsed_time]
                        }
                        self.next_id += 1
                        used_ids.add(new_id)
                        
                        person_id = new_id
                        color = self.get_person_color(new_id)
                        
                        if frame_id % 120 == 0:
                            print(f"   🆕 새 사람: {new_id} (매칭 실패)")
                    
                    matched_people.append({
                        'id': person_id,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'color': color,
                        'match_score': match_score if matched_id else 0.0
                    })
                
                # 결과 저장
                with self.lock:
                    self.latest_detections = matched_people
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 사람 감지 워커 오류: {e}")
                break
    
    def normalize_keypoints(self, keypoints):
        """키포인트 정규화 (학습 시와 동일)"""
        if keypoints.shape[0] == 0:
            return keypoints
        
        valid_mask = keypoints[:, :, 2] >= 0.3  # 신뢰도 0.3 이상
        
        for frame_idx in range(keypoints.shape[0]):
            frame_kpts = keypoints[frame_idx]
            valid_joints = valid_mask[frame_idx]
            
            if np.any(valid_joints):
                valid_coords = frame_kpts[valid_joints][:, :2]
                
                # 중심점과 스케일 계산
                center = np.mean(valid_coords, axis=0)
                distances = np.linalg.norm(valid_coords - center, axis=1)
                scale = np.mean(distances) + 1e-6
                
                # 정규화 적용
                frame_kpts[:, :2] = (frame_kpts[:, :2] - center) / scale
                keypoints[frame_idx] = frame_kpts
        
        return keypoints
    
    def detect_static_pose(self, current_keypoints, previous_keypoints):
        """정적 자세 감지 (움직임이 거의 없으면 True)"""
        if previous_keypoints is None:
            return False
        
        # 주요 관절점들의 움직임 계산
        movement_threshold = 25.0  # 10.0 → 25.0으로 증가 (덜 민감하게)
        total_movement = 0.0
        valid_joints = 0
        
        for joint_idx in [5, 6, 7, 8, 9, 10]:  # 어깨, 팔꿈치, 손목
            if (current_keypoints[joint_idx][2] > 0.3 and 
                previous_keypoints[joint_idx][2] > 0.3):
                
                curr_pos = current_keypoints[joint_idx][:2]
                prev_pos = previous_keypoints[joint_idx][:2]
                movement = np.linalg.norm(curr_pos - prev_pos)
                total_movement += movement
                valid_joints += 1
        
        if valid_joints == 0:
            return False
        
        avg_movement = total_movement / valid_joints
        is_static = avg_movement < movement_threshold
        
        # 디버깅용 (30프레임마다 출력)
        if hasattr(self, 'debug_frame_count') and self.debug_frame_count % 30 == 0:
            print(f"   움직임 분석: 평균 {avg_movement:.1f}px, 임계값 {movement_threshold}px → {'정적' if is_static else '동적'}")
        
        return is_static

    def gesture_recognition_worker(self):
        """제스처 인식 워커 (가장 큰 사람의 pose만 감지)"""
        previous_keypoints = None
        self.debug_frame_count = 0  # 디버깅용 카운트
        
        while self.running:
            try:
                frame_data = self.gesture_queue.get(timeout=1.0)
                if frame_data is None:
                    break
                
                frame, frame_id, elapsed_time = frame_data
                self.debug_frame_count = frame_id  # 프레임 카운트 동기화
                
                if not self.enable_gesture_recognition:
                    self.gesture_queue.task_done()
                    continue
                
                # 기본값 유지 (새로운 판단이 없으면 이전 값 유지)
                with self.lock:
                    prediction, confidence, keypoints_detected, current_keypoints = self.latest_gesture
                    latest_detections = self.latest_detections.copy()
        
                # 가장 큰 사람(주요 대상) 찾기
                target_person = None
                if latest_detections:
                    # 면적이 가장 큰 사람 선택
                    target_person = max(latest_detections, key=lambda p: 
                        (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))
                    
                    if frame_id % 120 == 0:
                        print(f"🎯 주요 대상: {target_person['id']} (면적: {(target_person['bbox'][2] - target_person['bbox'][0]) * (target_person['bbox'][3] - target_person['bbox'][1]):.0f})")
                
                # YOLO Pose로 키포인트 추출 (가장 큰 사람만)
                if target_person is not None:
                    # 주요 대상의 바운딩 박스로 ROI 설정
                    x1, y1, x2, y2 = map(int, target_person['bbox'])
                    
                    # ROI 확장 (더 넓은 영역에서 pose 감지)
                    margin = 50
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    
                    # ROI 추출
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:  # ROI가 유효한 경우
                        # ROI에서 pose 감지
                        results = self.pose_model(roi, imgsz=256, conf=0.3, 
                                                verbose=False, device=0)
                        
                        keypoints_data = []
                        
                        for result in results:
                            if result.keypoints is not None:
                                keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                                
                                for person_idx, person_kpts in enumerate(keypoints):
                                    # 상체 관절점만 추출
                                    upper_body_kpts = person_kpts[self.upper_body_joints]  # (9, 3)
                                    
                                    # 신뢰도 체크
                                    valid_joints = upper_body_kpts[:, 2] >= 0.2
                                    valid_count = np.sum(valid_joints)
                                    
                                    if frame_id % 120 == 0:
                                        print(f"   주요 대상 {target_person['id']}: {valid_count}/9 키포인트 (어깨:{person_kpts[5][2]:.2f}/{person_kpts[6][2]:.2f})")
                                    
                                    if valid_count >= 3:
                                        # ROI 좌표를 전체 프레임 좌표로 변환
                                        person_kpts[:, 0] += x1
                                        person_kpts[:, 1] += y1
                                        
                                        keypoints_data.append(upper_body_kpts)
                                        keypoints_detected = True
                                        current_keypoints = person_kpts  # 전체 17개 키포인트 저장 (시각화용)
                                        break  # 첫 번째 사람만 사용
                        
                        # 키포인트가 감지되면 처리
                        if len(keypoints_data) > 0:
                            current_keypoints_data = keypoints_data[0]
                            # 이전 키포인트 저장
                            previous_keypoints = current_keypoints.copy()
                        else:
                            # 키포인트가 감지되지 않으면 이전 프레임 사용 (패딩)
                            if hasattr(self, 'last_valid_keypoints') and self.last_valid_keypoints is not None:
                                current_keypoints_data = self.last_valid_keypoints
                                if frame_id % 120 == 0:
                                    print(f"   ⚠️ 주요 대상 키포인트 없음 - 이전 프레임 사용")
                            else:
                                # 아예 처음이면 기본값 생성
                                current_keypoints_data = np.zeros((9, 3))
                                if frame_id % 120 == 0:
                                    print(f"   ⚠️ 주요 대상 키포인트 없음 - 기본값 사용")
                        
                        # 유효한 키포인트 저장
                        if len(keypoints_data) > 0:
                            self.last_valid_keypoints = keypoints_data[0]
                    else:
                        # ROI가 유효하지 않은 경우
                        if hasattr(self, 'last_valid_keypoints') and self.last_valid_keypoints is not None:
                            current_keypoints_data = self.last_valid_keypoints
                        else:
                            current_keypoints_data = np.zeros((9, 3))
                        keypoints_detected = False
                        current_keypoints = None
                else:
                    # 감지된 사람이 없는 경우
                    if hasattr(self, 'last_valid_keypoints') and self.last_valid_keypoints is not None:
                        current_keypoints_data = self.last_valid_keypoints
                    else:
                        current_keypoints_data = np.zeros((9, 3))
                    keypoints_detected = False
                    current_keypoints = None
                
                # 정상적인 제스처 분석 (항상 실행 - 패딩된 데이터라도 처리)
                self.gesture_frame_buffer.append(current_keypoints_data)
                
                # 3초(90프레임) 단위로 판단
                frames_since_last_decision = frame_id - self.last_gesture_decision_frame
                
                if (len(self.gesture_frame_buffer) >= self.min_gesture_frames and 
                    frames_since_last_decision >= self.gesture_decision_interval):
                    
                    print(f"🎯 [Frame {frame_id}] 3초 단위 제스처 판단 시작!")
                    
                    try:
                        # 키포인트 시퀀스 전처리
                        keypoints_sequence = list(self.gesture_frame_buffer)
                        keypoints_array = np.array(keypoints_sequence)  # (T, 9, 3)
                    
                        # 정규화
                        normalized_keypoints = self.normalize_keypoints(keypoints_array)
                    
                        # 정확히 90프레임으로 맞추기
                        T, V, C = normalized_keypoints.shape
                        target_frames = 90
                    
                        if T < target_frames:
                            # 패딩
                            padding_frames = target_frames - T
                            last_frame = normalized_keypoints[-1:].repeat(padding_frames, axis=0)
                            normalized_keypoints = np.concatenate([normalized_keypoints, last_frame], axis=0)
                        elif T > target_frames:
                            # 최신 90프레임 사용
                            normalized_keypoints = normalized_keypoints[-target_frames:]
                    
                        # Shift-GCN 입력 형태로 변환: (C, T, V, M)
                        shift_gcn_data = np.zeros((C, target_frames, V, 1))
                        shift_gcn_data[:, :, :, 0] = normalized_keypoints.transpose(2, 0, 1)
                    
                        # 모델 예측
                        input_tensor = torch.FloatTensor(shift_gcn_data).unsqueeze(0).to(device)
                    
                        with torch.no_grad():
                            outputs = self.gesture_model(input_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            predicted_class = torch.argmax(outputs, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                    
                        prediction = self.actions[predicted_class]
                        self.current_gesture_confidence = confidence
                        
                        # 결정 기록
                        self.last_gesture_decision_frame = frame_id
                        
                        # 버퍼 초기화 (새로운 3초 구간 시작)
                        self.gesture_frame_buffer.clear()
                        self.static_detection_frames = 0  # 정적 카운터 리셋
                        
                        # COME 판단 시 보호 기간 설정
                        if prediction == "COME":
                            print(f"   🛡️ COME 판단 - 보호 기간 없음")
                        
                        print(f"🎯 [3초 판단] {prediction} ({confidence:.3f}) | Raw: [{outputs[0][0]:.2f}, {outputs[0][1]:.2f}]")
                        
                        # COME 제스처 디버깅 정보 추가
                        if prediction == "COME":
                            print(f"   🔍 COME 감지! 상세 분석:")
                            print(f"      - Raw outputs: [{outputs[0][0]:.4f}, {outputs[0][1]:.4f}]")
                            print(f"      - Softmax probs: [{probabilities[0][0]:.4f}, {probabilities[0][1]:.4f}]")
                            print(f"      - Predicted class: {predicted_class} -> {prediction}")
                        elif prediction == "NORMAL" and confidence < 0.7:  # 낮은 신뢰도일 때
                            print(f"   🔍 낮은 신뢰도 NORMAL 감지:")
                            print(f"      - Raw outputs: [{outputs[0][0]:.4f}, {outputs[0][1]:.4f}]")
                            print(f"      - Softmax probs: [{probabilities[0][0]:.4f}, {probabilities[0][1]:.4f}]")
                            print(f"      - Predicted class: {predicted_class} -> {prediction}")
                        
                    except Exception as e:
                        print(f"❌ 제스처 예측 오류: {e}")
                        prediction = "NORMAL"
                        confidence = 0.5
                
                # 결과 저장 (키포인트 정보 포함)
                # 키포인트가 감지된 경우에만 업데이트
                if keypoints_detected and current_keypoints is not None:
                    # 새로운 판단이 있는 경우에만 저장
                    with self.lock:
                        self.latest_gesture = (prediction, confidence, True, current_keypoints)
                # 키포인트가 없으면 이전 값 유지 (업데이트 안 함)
                
                self.gesture_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 제스처 인식 워커 오류: {e}")
                break
            
    def warmup_models(self, frame):
        """모델 워밍업 (첫 추론 속도 개선)"""
        print("🔥 모델 워밍업 시작...")
        
        # YOLO 모델들 워밍업
        try:
            _ = self.person_model(frame, classes=[0], imgsz=256, conf=0.6, verbose=False, device=0)
            _ = self.pose_model(frame, imgsz=256, conf=0.3, verbose=False, device=0)  # 320→256으로 축소
            print("✅ YOLO 모델 워밍업 완료")
        except Exception as e:
            print(f"⚠️ YOLO 워밍업 오류: {e}")
        
        # Shift-GCN 모델 워밍업
        if self.enable_gesture_recognition:
            try:
                dummy_input = torch.randn(1, 3, 90, 9, 1).to(device)
                with torch.no_grad():
                    _ = self.gesture_model(dummy_input)
                print("✅ Shift-GCN 모델 워밍업 완료")
            except Exception as e:
                print(f"⚠️ Shift-GCN 워밍업 오류: {e}")
        
        print("🚀 모든 모델 워밍업 완료!")

    def wait_for_workers_ready(self, timeout=5.0):
        """워커들이 준비될 때까지 대기"""
        print("⏳ 워커 스레드 준비 대기...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                # 워커가 최소 한 번은 실행되었는지 확인
                if len(self.latest_detections) > 0 or self.latest_gesture[2]:  # detections 있거나 keypoints 감지됨
                    print("✅ 워커 스레드 준비 완료!")
                    return True
            time.sleep(0.1)
        
        print("⚠️ 워커 준비 타임아웃 (정상 진행)")
        return False
    
    def run_system(self, camera_device="/dev/video0"):
        """🚀 새로운 통합 시스템 실행"""
        print("🚀 완전 새로운 통합 시스템 시작!")
        print(f"📹 카메라: {camera_device}")
        print("⚡ MediaPipe 매칭 + YOLO Pose + Shift-GCN + 멀티스레딩")
        print("🔧 초기 감지 지연 해결 버전!")
        
        cap = cv2.VideoCapture(camera_device)
        if not cap.isOpened():
            print(f"❌ 카메라 연결 실패: {camera_device}")
            return
        
        # 창 설정 추가 (문제 해결)
        window_name = "🚀 NEW Integrated System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, 800, 600)  # 적당한 창 크기
        cv2.moveWindow(window_name, 50, 50)
        print("🖼️  창 설정 완료!")
        
        # 카메라 해상도를 VGA로 설정 (성능 우선)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 자동 노출 설정
        
        # 실제 설정된 해상도 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 카메라 해상도: {actual_width}x{actual_height} (성능 최적화)")
        print("📹 카메라 설정 최적화 완료!")
        
        # 첫 프레임으로 창 테스트
        ret, test_frame = cap.read()
        if ret:
            cv2.putText(test_frame, "🚀 System Starting...", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow(window_name, test_frame)
            cv2.waitKey(100)  # 100ms 대기
            print("🖼️  초기 창 표시 완료!")
            
            # 모델 워밍업 (초기 지연 해결)
            self.warmup_models(test_frame)
        
        # 워커 스레드 시작
        workers = []
        workers.append(self.executor.submit(self.person_detection_worker))
        workers.append(self.executor.submit(self.gesture_recognition_worker))
        print("🔥 워커 스레드 시작!")
        
        # 워커 준비 대기 (초기 감지 지연 해결)
        workers_ready = self.wait_for_workers_ready(timeout=3.0)
        if not workers_ready:
            print("⚠️ 워커 준비 완료 전에 진행 (일부 초기 프레임에서 감지 안 될 수 있음)")
        
        frame_count = 0
        start_time = datetime.now()
        fps_times = deque(maxlen=30)
        
        current_gesture = "NORMAL"
        current_confidence = self.current_gesture_confidence
        
        try:
            while cap.isOpened():
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                elapsed_time = (datetime.now() - start_time).total_seconds()
                
                # 프레임을 워커들에게 전달
                frame_data = (frame.copy(), frame_count, elapsed_time)
                
                # 사람 감지 워커에 프레임 전달 (2프레임마다 - 성능 최적화)
                if frame_count % 2 == 0:
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        pass
                
                # 제스처 인식 워커에 프레임 전달 (매 프레임 - 학습과 동일)
                try:
                    self.gesture_queue.put_nowait(frame_data)
                except queue.Full:
                    pass
                
                # 최신 결과 가져오기
                with self.lock:
                    latest_detections = self.latest_detections.copy()
                    latest_gesture = self.latest_gesture
                
                # 제스처 결과 업데이트
                gesture_prediction, gesture_confidence, keypoints_detected, current_keypoints = latest_gesture
                if gesture_prediction != current_gesture and gesture_confidence > 0.1:  # 0.3 → 0.1으로 더 낮춤
                    current_gesture = gesture_prediction
                    current_confidence = gesture_confidence
                    print(f"✋ 제스처 변화: {gesture_prediction} ({gesture_confidence:.2f})")
                
                # 화면 구성 (항상 원본 프레임 기반)
                annotated = frame.copy()
                
                # 사람 시각화 (워커 결과가 있는 경우에만)
                if latest_detections:
                    for person in latest_detections:
                        x1, y1, x2, y2 = map(int, person['bbox'])
                        color = person['color']
                        
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, person['id'], (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(annotated, f"Conf: {person['confidence']:.2f}", 
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        if person['match_score'] > 0:
                            cv2.putText(annotated, f"Match: {person['match_score']:.2f}", 
                                       (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 관절점 시각화 (키포인트가 감지된 경우)
                if keypoints_detected and current_keypoints is not None:
                    # 첫 번째 사람의 색상 사용 (있는 경우)
                    keypoint_color = latest_detections[0]['color'] if latest_detections else (0, 255, 255)
                    annotated = self.draw_keypoints(annotated, current_keypoints, keypoint_color)
                    
                    # 제스처 인식용 키포인트 개수 표시 (9개 기준)
                    gesture_keypoints = current_keypoints[self.upper_body_joints]  # 9개 추출
                    valid_gesture_keypoints = np.sum(gesture_keypoints[:, 2] > 0.3)
                    cv2.putText(annotated, f"Gesture KPts: {valid_gesture_keypoints}/9", 
                               (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, keypoint_color, 2)
                
                # FPS 계산
                frame_time = time.time() - frame_start
                fps_times.append(frame_time)
                fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
                
                # 시스템 정보 표시 (크게 표시)
                cv2.putText(annotated, f"🚀 NEW System FPS: {fps:.1f}", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated, f"People: {len(latest_detections)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated, f"Gesture: {current_gesture}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
                cv2.putText(annotated, f"Confidence: {current_confidence:.2f}", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(annotated, f"Keypoints: {'OK' if keypoints_detected else 'NONE'}", 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if keypoints_detected else (0, 0, 255), 2)
                cv2.putText(annotated, f"Frame: {frame_count}", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 3초 단위 판단 정보 표시
                frames_to_next_decision = self.gesture_decision_interval - (frame_count - self.last_gesture_decision_frame)
                if frames_to_next_decision > 0:
                    cv2.putText(annotated, f"Next Decision: {frames_to_next_decision}f", 
                               (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(annotated, f"Ready for Decision", 
                               (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 워커 상태 표시
                worker_status = f"Workers Ready: {workers_ready}" if frame_count < 100 else "Workers: Running"
                cv2.putText(annotated, worker_status, (10, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 화면 표시 (매 프레임, 창은 이미 설정됨)
                cv2.imshow(window_name, annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
                # 성능 모니터링
                if frame_count % 120 == 0:  # 60→120프레임마다만 출력
                    print(f"📊 FPS: {fps:.1f} | People: {len(latest_detections)} | Gesture: {current_gesture}")
                    print(f"   화면 확인: 밝기 {np.mean(annotated):.1f}, 크기 {annotated.shape}")
                    print(f"   제스처 상태: 버퍼 {len(self.gesture_frame_buffer)}/90, 다음 판단까지 {frames_to_next_decision}프레임")
        
        except KeyboardInterrupt:
            print("🛑 사용자 중단")
        
        finally:
            # 정리
            self.running = False
            
            # 종료 신호
            try:
                self.frame_queue.put_nowait(None)
                self.gesture_queue.put_nowait(None)
            except queue.Full:
                pass
            
            # 워커 대기
            for worker in workers:
                try:
                    worker.result(timeout=2)
                except:
                    pass
            
            self.executor.shutdown(wait=True)
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🎉 시스템 종료")
            print(f"   - 총 프레임: {frame_count}")
            print(f"   - 최종 FPS: {fps:.1f}")
            print(f"   - 감지된 사람: {len(self.people_data)}")

    def test_model_performance(self):
        """학습된 모델의 성능 테스트"""
        print("🧪 모델 성능 테스트 시작...")
        
        if not self.enable_gesture_recognition:
            print("❌ 제스처 인식이 비활성화되어 있습니다.")
            return
        
        try:
            # 테스트 데이터 로드
            come_data_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/shift_gcn_data/come_pose_data.npy'
            normal_data_path = '/home/ckim/ros-repo-4/deeplearning/gesture_recognition/shift_gcn_data/normal_pose_data.npy'
            
            if os.path.exists(come_data_path) and os.path.exists(normal_data_path):
                come_data = np.load(come_data_path)
                normal_data = np.load(normal_data_path)
                
                print(f"📊 테스트 데이터 로드: COME {come_data.shape}, NORMAL {normal_data.shape}")
                
                # 각 클래스에서 몇 개 샘플 테스트
                test_samples = 5
                come_correct = 0
                normal_correct = 0
                
                # COME 샘플 테스트
                for i in range(min(test_samples, len(come_data))):
                    sample = come_data[i]  # (3, 90, 9, 1)
                    input_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = self.gesture_model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(outputs, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    prediction = self.actions[predicted_class]
                    if prediction == "COME":
                        come_correct += 1
                    
                    print(f"   COME 샘플 {i+1}: {prediction} ({confidence:.3f}) | Raw: [{outputs[0][0]:.2f}, {outputs[0][1]:.2f}]")
                
                # NORMAL 샘플 테스트
                for i in range(min(test_samples, len(normal_data))):
                    sample = normal_data[i]  # (3, 90, 9, 1)
                    input_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = self.gesture_model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(outputs, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    prediction = self.actions[predicted_class]
                    if prediction == "NORMAL":
                        normal_correct += 1
                    
                    print(f"   NORMAL 샘플 {i+1}: {prediction} ({confidence:.3f}) | Raw: [{outputs[0][0]:.2f}, {outputs[0][1]:.2f}]")
                
                come_acc = come_correct / test_samples * 100
                normal_acc = normal_correct / test_samples * 100
                
                print(f"📊 모델 테스트 결과:")
                print(f"   - COME 정확도: {come_acc:.1f}% ({come_correct}/{test_samples})")
                print(f"   - NORMAL 정확도: {normal_acc:.1f}% ({normal_correct}/{test_samples})")
                
                if come_acc < 60 or normal_acc < 60:
                    print("⚠️ 모델 성능이 낮습니다. 재학습이 필요할 수 있습니다.")
                else:
                    print("✅ 모델 성능이 양호합니다.")
                    
            else:
                print("❌ 테스트 데이터를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ 모델 테스트 중 오류: {e}")

if __name__ == "__main__":
    system = NewIntegratedSystem()
    system.run_system(camera_device="/dev/video0")
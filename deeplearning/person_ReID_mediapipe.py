import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque
import os
from datetime import datetime

# Qt 플랫폼 플러그인 오류 방지
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_DEBUG_PLUGINS'] = '0'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 제스처 인식 CNN 모델 클래스
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

def estimate_distance(bbox_height, ref_height=300, ref_distance=1.0):
    distance = ref_height / (bbox_height + 1e-6) * ref_distance
    return round(distance, 2)

def estimate_distance_from_mask(mask, ref_height=300, ref_distance=1.0):
    """세그멘테이션 마스크를 사용한 더 정확한 거리 계산"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        person_height = h
        distance = ref_height / (person_height + 1e-6) * ref_distance
        return round(distance, 2)
    
    return 2.0

def estimate_distance_advanced(mask, ref_height=300, ref_distance=1.0):
    """고급 거리 계산 - 마스크의 실제 픽셀 수와 형태 고려"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        person_pixels = cv2.contourArea(cnt)
        bbox_area = w * h
        density = person_pixels / (bbox_area + 1e-6)
        adjusted_height = h * density
        distance = ref_height / (adjusted_height + 1e-6) * ref_distance
        
        return round(distance, 2), {
            'person_height': h,
            'person_pixels': person_pixels,
            'bbox_area': bbox_area,
            'density': density,
            'adjusted_height': adjusted_height
        }
    
    return 2.0, {}

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

class IntegratedPersonGestureSystem:
    def __init__(self):
        # YOLO 모델 초기화
        self.model = YOLO('./deeplearning/yolov8s-seg.pt')
        
        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 제스처 모델 초기화
        self.gesture_model = GestureCNN(num_classes=3, num_frames=60, num_joints=21)
        model_path = 'gesture_recognition/best_gesture_cnn_model.pth'
        if os.path.exists(model_path):
            self.gesture_model.load_state_dict(torch.load(model_path))
            print(f"✅ 제스처 모델 로드 성공: {model_path}")
        else:
            print(f"❌ 제스처 모델 파일을 찾을 수 없습니다: {model_path}")
            print("제스처 인식이 비활성화됩니다.")
            self.enable_gesture_recognition = False
        self.gesture_model = self.gesture_model.to(device)
        self.gesture_model.eval()
        
        # 사람 재식별 데이터
        self.people_data = {}
        self.next_id = 0
        
        # 성능 최적화 설정
        self.process_every_n_frames = 3
        self.frame_skip_counter = 0
        
        # 매칭 관련 설정 (더 엄격한 설정 - 모자/의상 구분 강화)
        self.match_threshold = 0.50  # 0.45 → 0.50로 증가 (더 엄격한 매칭)
        self.reentry_threshold = 0.45  # 0.40 → 0.45로 증가
        self.min_detection_confidence = 0.6
        self.min_person_area = 5000
        self.max_frames_without_seen = 300
        
        # 히스토그램 기억 설정 (더 많은 히스토그램 저장)
        self.max_histograms_per_person = 20  # 15 → 20으로 증가 (더 많은 샘플)
        self.histogram_memory_duration = 30
        
        # 제스처 인식 설정 (더 엄격한 조건)
        self.enable_gesture_recognition = True
        self.gesture_frame_buffer = deque(maxlen=60)
        self.gesture_prediction_buffer = deque(maxlen=5)
        self.actions = ['COME', 'AWAY', 'STOP']
        
        # 제스처 인식 임계값 (더 엄격하게)
        self.gesture_confidence_threshold = 0.75  # 0.75 이상일 때만 제스처로 인식
        self.gesture_buffer_threshold = 4  # 4번 이상 같은 제스처가 나와야 인식
        self.min_gesture_frames = 40  # 최소 40프레임이 모여야 예측 시작
        
        # 분석 결과 저장 디렉토리
        self.analysis_dir = "./deeplearning/integrated_analysis"
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
            print(f"📁 통합 분석 디렉토리 생성: {self.analysis_dir}")
    
    def extract_histogram(self, img, mask, bins=16):
        """HSV의 모든 채널(H, S, V)을 고려한 히스토그램 추출 (개선된 버전)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 전체 마스크에서 히스토그램 추출
        h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])
        
        # 상체 부분 (머리/모자/상의)에 집중한 히스토그램 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 상체 부분 마스크 생성 (전체 높이의 상위 60%)
            upper_mask = np.zeros_like(mask)
            upper_y = y + int(h * 0.6)  # 상위 60% 부분
            upper_mask[y:upper_y, x:x+w] = mask[y:upper_y, x:x+w]
            
            # 상체 부분 히스토그램
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
        
        # 전체 + 상체 히스토그램 결합 (상체에 더 높은 가중치)
        combined_hist = np.concatenate([
            h_hist * 0.3, s_hist * 0.3, v_hist * 0.3,  # 전체 (30%)
            h_hist_upper * 0.7, s_hist_upper * 0.7, v_hist_upper * 0.7  # 상체 (70%)
        ])
        
        return combined_hist, h_hist, s_hist, v_hist
    
    def calculate_similarity_metrics(self, hist1, hist2):
        """다양한 유사도 메트릭 계산"""
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
        """가장 유사한 사람 찾기 (개선된 버전)"""
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
    
    def process_gesture(self, frame):
        """제스처 인식 처리"""
        if not self.enable_gesture_recognition:
            return "DISABLED", 0.0
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        prediction = "WAITING..."
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
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
                self.gesture_frame_buffer.append(d)
                
                # 충분한 프레임이 모이면 예측
                if len(self.gesture_frame_buffer) >= self.min_gesture_frames:
                    try:
                        frame_data = np.array(list(self.gesture_frame_buffer))
                        
                        frames = len(frame_data)
                        if frames < 60:
                            padding = np.zeros((60 - frames, 99), dtype=np.float32)
                            x = np.vstack([frame_data, padding])
                        elif frames > 60:
                            start = (frames - 60) // 2
                            x = frame_data[start:start + 60]
                        else:
                            x = frame_data
                        
                        gesture_features = extract_gesture_features(x)
                        x_with_gesture = np.concatenate([x.flatten(), gesture_features])
                        model_input = torch.FloatTensor(x_with_gesture).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = self.gesture_model(model_input)
                            probabilities = torch.softmax(outputs, dim=1)
                            predicted = torch.argmax(outputs, dim=1).item()
                            confidence = probabilities[0][predicted].item()
                        
                        prediction = self.actions[predicted]
                        self.gesture_prediction_buffer.append(predicted)
                        
                    except Exception as e:
                        print(f"제스처 예측 오류: {e}")
                        prediction = "ERROR"
        
        # 부드러운 예측 결과
        if len(self.gesture_prediction_buffer) >= self.gesture_buffer_threshold:
            from collections import Counter
            most_common = Counter(self.gesture_prediction_buffer).most_common(1)[0]
            final_prediction = self.actions[most_common[0]]
            final_confidence = most_common[1] / len(self.gesture_prediction_buffer)
            
            # 신뢰도가 임계값을 넘지 않으면 WAITING으로 처리
            if final_confidence < 0.8:  # 80% 이상 일관성이 있어야 제스처로 인식
                final_prediction = "WAITING..."
                final_confidence = 0.0
        else:
            final_prediction = prediction
            final_confidence = confidence
        
        # 최종 신뢰도 검사 (모델 신뢰도도 고려)
        if final_confidence < self.gesture_confidence_threshold:
            final_prediction = "WAITING..."
            final_confidence = 0.0
        
        return final_prediction, final_confidence
    
    def run_integrated_system(self, camera_device="/dev/video2"):
        """통합 시스템 실행 (무한 실행)"""
        print("🚀 통합 사람 재식별 + 제스처 인식 시스템 시작")
        print(f"📹 카메라: {camera_device}")
        print("⏱️ 무한 실행 모드 (q키로 종료)")
        
        cap = cv2.VideoCapture(camera_device)
        
        if not cap.isOpened():
            print(f"❌ 카메라 연결 실패: {camera_device}")
            return
        
        frame_count = 0
        start_time = datetime.now()
        
        # 제스처 인식 결과 추적
        current_gesture = "WAITING..."
        current_gesture_confidence = 0.0
        gesture_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # 프레임 처리 간격 조절
            if self.frame_skip_counter < self.process_every_n_frames - 1:
                self.frame_skip_counter += 1
                continue
            
            self.frame_skip_counter = 0
            
            # 사람 감지
            results = self.model(frame, classes=[0])
            annotated = frame.copy()
            
            # 제스처 인식 (실시간 업데이트)
            gesture_prediction, gesture_confidence = self.process_gesture(annotated)
            
            # 제스처 결과 업데이트 (더 엄격한 조건)
            if (gesture_prediction != "WAITING..." and 
                gesture_prediction != "ERROR" and 
                gesture_confidence >= self.gesture_confidence_threshold):
                
                current_gesture = gesture_prediction
                current_gesture_confidence = gesture_confidence
                
                # 제스처 히스토리 저장
                gesture_history.append({
                    'gesture': gesture_prediction,
                    'confidence': gesture_confidence,
                    'timestamp': elapsed_time,
                    'frame': frame_count
                })
                
                # 콘솔에 제스처 인식 결과 출력
                print(f"✋ 제스처 인식: {gesture_prediction} (신뢰도: {gesture_confidence:.2f})")
            elif gesture_prediction == "WAITING...":
                # WAITING 상태일 때는 이전 제스처를 유지하되 신뢰도는 낮춤
                current_gesture_confidence = max(0.0, current_gesture_confidence - 0.1)
            
            # 현재 프레임에서 감지된 모든 사람의 정보를 먼저 수집
            current_detections = []
            
            for result in results:
                for i in range(len(result.boxes)):
                    seg = result.masks.data[i]
                    box = result.boxes[i]
                    confidence = box.conf[0].item()
                    
                    if confidence < self.min_detection_confidence:
                        continue
                    
                    mask = seg.cpu().numpy().astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 노이즈 제거
                    kernel = np.ones((5,5), np.uint8)
                    mask_cleaned = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
                    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
                    
                    # HSV 히스토그램 추출
                    combined_hist, h_hist, s_hist, v_hist = self.extract_histogram(frame, mask_cleaned)
                    
                    # 바운딩 박스 추출
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area < self.min_person_area:
                        continue
                    
                    # 거리 추정
                    est_dist, dist_info = estimate_distance_advanced(mask_cleaned, ref_height=300, ref_distance=1.0)
                    
                    current_detections.append({
                        'hist': combined_hist,
                        'bbox': bbox,
                        'mask': mask_cleaned,
                        'confidence': confidence,
                        'area': area,
                        'distance': est_dist,
                        'dist_info': dist_info
                    })
            
            # 감지된 사람들을 영역 크기 순으로 정렬
            current_detections.sort(key=lambda x: x['area'], reverse=True)
            
            # 이미 매칭된 ID들을 추적
            used_ids = set()
            
            # 각 감지된 사람에 대해 매칭 수행
            for detection in current_detections:
                combined_hist = detection['hist']
                bbox = detection['bbox']
                
                # 매칭 시도
                matched_id, match_score, metrics = self.find_best_match(combined_hist, bbox, used_ids, elapsed_time)
                
                # 매칭 성공 여부에 따른 처리
                if matched_id is not None and match_score > self.match_threshold:
                    # 기존 사람 업데이트
                    self.people_data[matched_id]['histograms'].append(combined_hist)
                    self.people_data[matched_id]['bboxes'].append(bbox)
                    self.people_data[matched_id]['timestamps'].append(elapsed_time)
                    used_ids.add(matched_id)
                    
                    # 히스토그램 메모리 관리
                    if len(self.people_data[matched_id]['histograms']) > self.max_histograms_per_person:
                        self.people_data[matched_id]['histograms'].pop(0)
                        self.people_data[matched_id]['bboxes'].pop(0)
                        self.people_data[matched_id]['timestamps'].pop(0)
                    
                    color = (0, 255, 0)  # 초록색 (기존 사람)
                    print(f"🔄 기존 사람 재식별: {matched_id} (점수: {match_score:.3f})")
                    print(f"   - 히스토그램 점수: {metrics['hist_score']:.3f}")
                    print(f"   - 공간적 점수: {metrics['spatial_score']:.3f}")
                    print(f"   - 크기 점수: {metrics['size_score']:.3f}")
                    print(f"   - 시간 점수: {metrics['time_score']:.3f}")
                    print(f"   - 저장된 히스토그램 수: {len(self.people_data[matched_id]['histograms'])}")
                    
                else:
                    # 새로운 사람 (매칭 실패 또는 임계값 미달)
                    new_id = f"Person_{self.next_id}"
                    self.people_data[new_id] = {
                        'histograms': [combined_hist],
                        'bboxes': [bbox],
                        'timestamps': [elapsed_time],
                        'images': []
                    }
                    self.next_id += 1
                    used_ids.add(new_id)
                    
                    color = (0, 0, 255)  # 빨간색 (새로운 사람)
                    print(f"🆕 새로운 사람 감지: {new_id}")
                    if matched_id is not None:
                        print(f"   - 최고 매칭 점수: {match_score:.3f} (임계값: {self.match_threshold:.3f})")
                        print(f"   - 매칭된 ID: {matched_id}")
                        print(f"   - 히스토그램 점수: {metrics['hist_score']:.3f}")
                    else:
                        print(f"   - 매칭 가능한 사람 없음")
                        print(f"   - 매칭 실패로 인한 새로운 사람 등록")
                
                # 시각화
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # ID 표시
                id_text = matched_id if matched_id else new_id
                cv2.putText(annotated, id_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Confidence 표시
                conf_text = f"Conf: {detection['confidence']:.2f}"
                cv2.putText(annotated, conf_text, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 거리 표시
                distance_text = f"Dist: {detection['distance']:.2f}m"
                cv2.putText(annotated, distance_text, (x1, y2+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 밀도 정보 표시
                if detection['dist_info']:
                    density_text = f"Density: {detection['dist_info']['density']:.2f}"
                    cv2.putText(annotated, density_text, (x1, y2+60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 유사도 점수 표시
                score_text = f"{match_score:.3f}"
                cv2.putText(annotated, score_text, (x1, y2+80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 제스처 정보 표시 (실시간 업데이트된 결과)
            gesture_text = f"Gesture: {current_gesture}"
            cv2.putText(annotated, gesture_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            gesture_conf_text = f"Gesture Conf: {current_gesture_confidence:.2f}"
            cv2.putText(annotated, gesture_conf_text, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # 제스처 버퍼 상태 표시
            buffer_text = f"Buffer: {len(self.gesture_prediction_buffer)}/{self.gesture_buffer_threshold}"
            cv2.putText(annotated, buffer_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 시스템 정보 표시
            info_text = f"People: {len(self.people_data)} | Frame: {frame_count} | Time: {elapsed_time:.1f}s"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 제스처 가이드 (더 명확하게)
            cv2.putText(annotated, 'COME: Move hand quickly', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, 'STOP: Make a fist', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, 'AWAY: Open palm clearly', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f'Threshold: {self.gesture_confidence_threshold}', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 성능 최적화: 5프레임마다만 화면 업데이트
            if frame_count % 5 == 0:
                cv2.imshow("Integrated Person + Gesture Recognition", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 분석 결과 생성 및 저장
        print(f"\n📊 통합 분석 완료!")
        print(f"   - 총 프레임: {frame_count}")
        print(f"   - 감지된 사람 수: {len(self.people_data)}")
        print(f"   - 최종 제스처 인식: {current_gesture} (신뢰도: {current_gesture_confidence:.2f})")
        
        # 분석 결과 저장
        self.save_analysis_results(frame_count, gesture_history, start_time)
    
    def save_analysis_results(self, frame_count, gesture_history, start_time):
        """분석 결과를 파일로 저장"""
        try:
            # 분석 결과 파일 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.analysis_dir, f"analysis_results_{timestamp}.txt")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=== 통합 사람 재식별 + 제스처 인식 분석 결과 ===\n\n")
                f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"총 프레임: {frame_count}\n")
                f.write(f"감지된 사람 수: {len(self.people_data)}\n")
                f.write(f"제스처 인식 횟수: {len(gesture_history)}\n\n")
                
                # 사람별 정보
                f.write("=== 사람별 정보 ===\n")
                for pid, pdata in self.people_data.items():
                    f.write(f"\n{pid}:\n")
                    f.write(f"  - 감지 횟수: {len(pdata['histograms'])}\n")
                    f.write(f"  - 첫 감지: {pdata['timestamps'][0]:.1f}초\n")
                    f.write(f"  - 마지막 감지: {pdata['timestamps'][-1]:.1f}초\n")
                
                # 제스처 히스토리
                if gesture_history:
                    f.write("\n=== 제스처 인식 히스토리 ===\n")
                    for i, gesture in enumerate(gesture_history):
                        f.write(f"{i+1}. {gesture['gesture']} (신뢰도: {gesture['confidence']:.2f}) - {gesture['timestamp']:.1f}초\n")
                
                # 제스처 통계
                if gesture_history:
                    gestures = [g['gesture'] for g in gesture_history]
                    from collections import Counter
                    gesture_counts = Counter(gestures)
                    f.write("\n=== 제스처 통계 ===\n")
                    for gesture, count in gesture_counts.items():
                        f.write(f"  - {gesture}: {count}회\n")
            
            print(f"✅ 분석 결과 저장: {results_file}")
            
        except Exception as e:
            print(f"❌ 분석 결과 저장 실패: {e}")

if __name__ == "__main__":
    system = IntegratedPersonGestureSystem()
    system.run_integrated_system(camera_device="/dev/video2") 
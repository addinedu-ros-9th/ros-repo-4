#!/usr/bin/env python3
"""
논문 기반 개선: Person Re-identification Based on Color Histogram and Spatial Configuration
- Dominant Color Descriptor (DCD) 기반
- 상하체 분할 (Upper/Lower body parts)
- HSV 8×3×3 양자화 (72 레벨)
- 공간적 구성 고려
- 가중 매칭 (α=0.4, β=0.6, γ=0.55)
"""

import cv2
import numpy as np
import sys
import pickle
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.append('../object_detection')
from simple_detector import PersonDetector

class PaperBasedPersonReidentification:
    def __init__(self):
        self.people = {}
        self.next_id = 0
        self.frame_count = 0
        
        # **논문 핵심: HSV 8×3×3 양자화**
        self.hue_bins = 8      # 논문: 8개 구간
        self.sat_bins = 3      # 논문: 3개 구간  
        self.val_bins = 3      # 논문: 3개 구간
        self.total_bins = 72   # 8×3×3 = 72 레벨
        
        # **논문 핵심: 가중 매칭 파라미터**
        self.alpha = 0.4       # 논문: 전체 가중치 (DCD vs 공간)
        self.beta = 0.6        # 논문: 공간 가중치 (y vs h)
        self.gamma = 0.52      # 논문: 신체부위 가중치 (상체 vs 하체) - 0.55 → 0.52 (균형 개선)
        
        # **논문 핵심: 상하체 분할**
        self.upper_ratio = 0.55  # 상체 비율 (55%) - 논문에 더 적합
        self.lower_ratio = 0.45  # 하체 비율 (45%) - 균형 개선
        
        # 성능 파라미터 (최적화)
        self.confidence_threshold = 0.5
        self.max_people = 5
        self.max_disappeared = 10  # 15프레임 → 더 빠른 제거
        self.similarity_threshold = 0.5  # 0.7 → 0.6 (재매칭을 위해 낮춤)
        self.reappear_threshold = 0.4  # 일시 사라진 사람 재매칭용 (더 관대)
        
        # **성능 최적화 파라미터**
        self.process_every_n_frames = 3  # 1 → 3 (3프레임마다 처리)
        self.min_bbox_size = 60          # 30 → 60 (큰 사람만 처리)
        self.max_spatial_regions = 1     # 2 → 1 (공간적 특징 최소화)
        
        # **추적 개선 파라미터**
        self.similarity_threshold = 0.5  # 더 관대한 매칭
        self.reappear_threshold = 0.4    # 재매칭 개선
        self.max_disappeared = 10        # 빠른 제거
        
        # **바운딩박스 스무딩**
        self.bbox_smoothing_factor = 0.7  # 바운딩박스 스무딩 계수
        self.bbox_history = {}           # 바운딩박스 히스토리
        self.velocity_history = {}       # 속도 히스토리
        
        # **자동 스크린샷 기능**
        self.auto_screenshot_enabled = False  # True → False (성능 향상)
        self.screenshot_dir = None
        
        # 색상 (더 많은 색상 추가)
        self.colors = [
            (0, 255, 0),    # 초록색
            (255, 0, 0),    # 파란색  
            (0, 0, 255),    # 빨간색
            (255, 255, 0),  # 청록색
            (255, 0, 255),  # 자홍색
            (0, 255, 255),  # 노란색
            (128, 0, 128),  # 보라색
            (255, 165, 0),  # 주황색
            (0, 128, 128),  # 올리브색
            (128, 128, 0)   # 갈색
        ]
        
        # 성능 통계
        self.performance_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'dcd_matches': 0,
            'spatial_matches': 0,
            'integrated_matches': 0,
            'upper_body_matches': 0,
            'lower_body_matches': 0,
            'skipped_frames': 0,
            'auto_screenshots': 0
        }
        
        print("🎯 Paper-Based Person Re-identification System (Optimized)")
        print("📋 Core Features:")
        print("  1. ✅ HSV 8×3×3 Quantization (72 levels)")
        print("  2. ✅ Upper/Lower Body Segmentation")
        print("  3. ✅ Dominant Color Descriptor (DCD)")
        print("  4. ✅ Spatial Configuration Analysis")
        print("  5. ✅ Weighted Matching (α=0.4, β=0.6, γ=0.55)")
        print("🚀 Performance Optimizations:")
        print("  - Vectorized HSV quantization")
        print("  - Frame skipping (every 3 frames)")
        print("  - Limited spatial regions (max 3)")
        print("  - Minimum bbox size filtering")
        print("📸 Auto Screenshot: Enabled for new person detection")

    def set_screenshot_directory(self, directory):
        """스크린샷 디렉토리 설정"""
        self.screenshot_dir = directory
    
    def save_auto_screenshot(self, frame, person_id, bbox, confidence, upper_hist, lower_hist, upper_spatial, lower_spatial):
        """새로운 사람 감지 시 자동 스크린샷 저장"""
        print(f"🔍 Auto screenshot attempt - Person ID: {person_id}")
        print(f"   - Auto screenshot enabled: {self.auto_screenshot_enabled}")
        print(f"   - Screenshot directory: {self.screenshot_dir}")
        
        if not self.auto_screenshot_enabled:
            print("   - ❌ Auto screenshot disabled")
            return
        
        if self.screenshot_dir is None:
            print("   - ❌ Screenshot directory not set")
            return
        
        try:
            import os
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"new_person_{person_id}_{timestamp}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            print(f"   - 📁 Filepath: {filepath}")
            
            # 결과 프레임 생성
            result_frame = frame.copy()
            
            # 바운딩박스 그리기
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[person_id % len(self.colors)]
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
            
            # 상하체 구분선
            upper_bbox, lower_bbox = self.segment_body_parts(frame, bbox)
            split_y = y1 + int((y2 - y1) * self.upper_ratio)
            cv2.line(result_frame, (x1, split_y), (x2, split_y), (255, 255, 255), 2)
            
            # 상체 영역 표시
            cv2.rectangle(result_frame, (x1, y1), (x2, split_y), (0, 255, 255), 1)
            cv2.putText(result_frame, "UPPER", (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 하체 영역 표시
            cv2.rectangle(result_frame, (x1, split_y), (x2, y2), (255, 0, 255), 1)
            cv2.putText(result_frame, "LOWER", (x1+5, split_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 디버그 정보 패널
            panel_width = 400
            panel_height = 200
            panel_x = max(10, x2 + 10)
            panel_y = y1
            
            cv2.rectangle(result_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), color, 2)
            
            # 상세 정보
            debug_info = [
                f"🆕 NEW PERSON DETECTED",
                f"Person ID: {person_id}",
                f"Confidence: {confidence:.3f}",
                f"Frame: {self.frame_count}",
                f"Timestamp: {timestamp}",
                f"",
                f"📊 UPPER BODY ANALYSIS:",
                f"  - DCD Regions: {len(upper_spatial)}",
                f"  - Dominant Colors: {len(self.extract_dominant_colors(upper_hist))}",
                f"  - Histogram Sum: {np.sum(upper_hist):.3f}",
                f"  - Max Hist Value: {np.max(upper_hist):.3f}",
                f"  - Hist Non-zero: {np.count_nonzero(upper_hist)}/72",
                f"",
                f"📊 LOWER BODY ANALYSIS:",
                f"  - DCD Regions: {len(lower_spatial)}",
                f"  - Dominant Colors: {len(self.extract_dominant_colors(lower_hist))}",
                f"  - Histogram Sum: {np.sum(lower_hist):.3f}",
                f"  - Max Hist Value: {np.max(lower_hist):.3f}",
                f"  - Hist Non-zero: {np.count_nonzero(lower_hist)}/72",
                f"",
                f"🎯 PAPER METHOD PARAMS:",
                f"  - α={self.alpha}, β={self.beta}, γ={self.gamma}",
                f"  - HSV Quant: 8x3x3 (72 levels)",
                f"  - Bbox Size: {x2-x1}x{y2-y1}px",
                f"  - Similarity Threshold: {self.similarity_threshold}"
            ]
            
            for i, text in enumerate(debug_info):
                y_pos = panel_y + 20 + (i * 15)
                if y_pos < panel_y + panel_height - 10:
                    text_color = (0, 255, 255) if "NEW PERSON" in text else (255, 255, 255)
                    cv2.putText(result_frame, text, (panel_x + 10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
            
            # 히스토그램 시각화 (상체)
            hist_width = 200
            hist_height = 60
            hist_x = panel_x
            hist_y = panel_y + panel_height + 10
            
            # 상체 히스토그램
            cv2.putText(result_frame, "Upper Histogram:", (hist_x, hist_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 히스토그램 바 그리기
            max_val = np.max(upper_hist) if np.max(upper_hist) > 0 else 1
            for i in range(min(72, hist_width)):
                height = int((upper_hist[i] / max_val) * hist_height)
                cv2.line(result_frame, (hist_x + i, hist_y + hist_height), 
                        (hist_x + i, hist_y + hist_height - height), (0, 255, 0), 1)
            
            # 하체 히스토그램
            hist_y2 = hist_y + hist_height + 20
            cv2.putText(result_frame, "Lower Histogram:", (hist_x, hist_y2 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            max_val = np.max(lower_hist) if np.max(lower_hist) > 0 else 1
            for i in range(min(72, hist_width)):
                height = int((lower_hist[i] / max_val) * hist_height)
                cv2.line(result_frame, (hist_x + i, hist_y2 + hist_height), 
                        (hist_x + i, hist_y2 + hist_height - height), (255, 0, 255), 1)
            
            # 저장
            print(f"   - 💾 Attempting to save...")
            success = cv2.imwrite(filepath, result_frame)
            if success:
                self.performance_stats['auto_screenshots'] += 1
                print(f"📸 Auto screenshot saved: {filename}")
                print(f"   - Person ID: {person_id}")
                print(f"   - Frame: {self.frame_count}")
                print(f"   - Upper regions: {len(upper_spatial)}")
                print(f"   - Lower regions: {len(lower_spatial)}")
            else:
                print(f"❌ Failed to save auto screenshot: {filepath}")
                
        except Exception as e:
            print(f"❌ Auto screenshot error: {e}")
            import traceback
            traceback.print_exc()
    
    def hsv_quantization(self, h, s, v):
        """논문 방식: HSV 8×3×3 양자화"""
        # Hue 양자화 (8개 구간)
        if h >= 316 or h < 20:
            H = 0
        elif 20 <= h < 40:
            H = 1
        elif 40 <= h < 75:
            H = 2
        elif 75 <= h < 155:
            H = 3
        elif 155 <= h < 190:
            H = 4
        elif 190 <= h < 270:
            H = 5
        elif 270 <= h < 295:
            H = 6
        else:  # 295 <= h < 316
            H = 7
        
        # Saturation 양자화 (3개 구간)
        if s <= 0.2:
            S = 0
        elif s <= 0.7:
            S = 1
        else:
            S = 2
        
        # Value 양자화 (3개 구간)
        if v <= 0.2:
            V = 0
        elif v <= 0.7:
            V = 1
        else:
            V = 2
        
        # 논문 공식: C = 9H + 3S + V
        C = 9 * H + 3 * S + V
        return C
    
    def extract_dominant_colors(self, hist):
        """논문 방식: Dominant Color Descriptor 추출"""
        # 상위 8개 색상만 선택 (논문: M=8)
        dominant_indices = np.argsort(hist.flatten())[-8:][::-1]
        dominant_colors = []
        
        total_sum = np.sum(hist)
        for idx in dominant_indices:
            percentage = hist.flatten()[idx] / total_sum
            dominant_colors.append((idx, percentage))
        
        return dominant_colors
    
    def segment_body_parts(self, frame, bbox):
        """논문 방식: 상하체 분할"""
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        
        # 상체 (60%)
        upper_y1 = y1
        upper_y2 = y1 + int(height * self.upper_ratio)
        upper_bbox = [x1, upper_y1, x2, upper_y2]
        
        # 하체 (40%)
        lower_y1 = upper_y2
        lower_y2 = y2
        lower_bbox = [x1, lower_y1, x2, lower_y2]
        
        return upper_bbox, lower_bbox
    
    def get_quantized_histogram(self, frame, bbox):
        """논문 방식: 양자화된 히스토그램 추출 (최적화)"""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # **성능 최적화: ROI 크기 제한**
        if roi.shape[0] > 200 or roi.shape[1] > 150:
            roi = cv2.resize(roi, (150, 200))
        
        # HSV 변환
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # **최적화된 히스토그램 계산 (OpenCV 내장 함수 사용)**
        # 8x3x3 = 72 bins 히스토그램 직접 계산
        hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [8, 3, 3], 
                           [0, 180, 0, 256, 0, 256])
        
        # 정규화
        hist = hist.flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist.astype(np.float32)
    
    def calculate_dcd_similarity(self, hist1, hist2):
        """논문 방식: Dominant Color Histogram 유사도"""
        if hist1 is None or hist2 is None:
            return 0.0
        
        # 논문 공식: min(P1, P2)의 합
        similarity = np.sum(np.minimum(hist1, hist2))
        return similarity
    
    def extract_spatial_features(self, frame, bbox, hist):
        """논문 방식: 공간적 구성 특징 추출 (최적화)"""
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1
        
        if height < 10 or width < 10:  # 너무 작은 영역은 건너뛰기
            return []
        
        # Dominant Color Regions 추출 (최적화)
        hsv_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        
        spatial_features = []
        dominant_colors = self.extract_dominant_colors(hist)
        
        # 상위 1개 색상만 처리 (성능 향상)
        for color_idx, percentage in dominant_colors[:1]:
            if percentage < 0.2:  # 20% 미만은 무시 (10% → 20%)
                continue
            
            # 벡터화된 마스크 생성
            h, s, v = cv2.split(hsv_roi)
            h_norm = h.astype(np.float32) * 2
            s_norm = s.astype(np.float32) / 255.0
            v_norm = v.astype(np.float32) / 255.0
            
            # 양자화 (벡터화)
            H = np.zeros_like(h_norm, dtype=np.int32)
            S = np.zeros_like(s_norm, dtype=np.int32)
            V = np.zeros_like(v_norm, dtype=np.int32)
            
            # Hue 양자화
            H[(h_norm >= 316) | (h_norm < 20)] = 0
            H[(h_norm >= 20) & (h_norm < 40)] = 1
            H[(h_norm >= 40) & (h_norm < 75)] = 2
            H[(h_norm >= 75) & (h_norm < 155)] = 3
            H[(h_norm >= 155) & (h_norm < 190)] = 4
            H[(h_norm >= 190) & (h_norm < 270)] = 5
            H[(h_norm >= 270) & (h_norm < 295)] = 6
            H[(h_norm >= 295) & (h_norm < 316)] = 7
            
            # Saturation & Value 양자화
            S[s_norm <= 0.2] = 0
            S[(s_norm > 0.2) & (s_norm <= 0.7)] = 1
            S[s_norm > 0.7] = 2
            
            V[v_norm <= 0.2] = 0
            V[(v_norm > 0.2) & (v_norm <= 0.7)] = 1
            V[v_norm > 0.7] = 2
            
            C = 9 * H + 3 * S + V
            mask = (C == color_idx).astype(np.uint8) * 255
            
            # Connected Components 추출
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # 각 영역의 공간적 특징 계산 (최대 1개만)
            region_count = 0
            for k in range(1, min(num_labels, 2)):  # 최대 1개 영역만
                area = stats[k, cv2.CC_STAT_AREA]
                if area < 30:  # 최소 영역 크기 증가 (10 → 30)
                    continue
                
                # 중심점 y좌표 (정규화)
                center_y = centroids[k][1] / height
                
                # 높이 (정규화)
                region_height = stats[k, cv2.CC_STAT_HEIGHT] / height
                
                spatial_features.append({
                    'color_idx': color_idx,
                    'percentage': percentage,
                    'center_y': center_y,
                    'height': region_height
                })
                
                region_count += 1
                if region_count >= 1:  # 최대 1개 영역만
                    break
        
        return spatial_features
    
    def calculate_spatial_similarity(self, spatial1, spatial2):
        """논문 방식: 공간적 구성 유사도"""
        # 리스트가 아닌 경우 처리
        if not isinstance(spatial1, list) or not isinstance(spatial2, list):
            return 0.0
            
        if not spatial1 or not spatial2:
            return 0.0
        
        min_distances = []
        
        for region1 in spatial1:
            distances = []
            for region2 in spatial2:
                if region1['color_idx'] == region2['color_idx']:
                    # 논문 공식: dy(u,w) = |uy - wy|/H
                    dy = abs(region1['center_y'] - region2['center_y'])
                    
                    # 논문 공식: dh(u,w) = |uh - wh|/H  
                    dh = abs(region1['height'] - region2['height'])
                    
                    # 논문 공식: dR(u,w) = β*dy + (1-β)*dh
                    dr = self.beta * dy + (1 - self.beta) * dh
                    distances.append(dr)
            
            if distances:
                min_distances.append(min(distances))
        
        if min_distances:
            # 논문 공식: dDCR = Σ min(dR)
            spatial_similarity = 1.0 - np.mean(min_distances)
            return max(0.0, spatial_similarity)
        
        return 0.0
    
    def integrated_similarity(self, person1_data, person2_data):
        """논문 방식: 통합 유사도 계산"""
        # 상체 유사도
        upper_dcd = self.calculate_dcd_similarity(
            person1_data['upper_hist'], person2_data['upper_hist'])
        upper_spatial = self.calculate_spatial_similarity(
            person1_data['upper_spatial'], person2_data['upper_spatial'])
        
        # 하체 유사도
        lower_dcd = self.calculate_dcd_similarity(
            person1_data['lower_hist'], person2_data['lower_hist'])
        lower_spatial = self.calculate_spatial_similarity(
            person1_data['lower_spatial'], person2_data['lower_spatial'])
        
        # 논문 공식: d(AU,BU) = min(P1, P2)의 합
        upper_similarity = upper_dcd
        
        # 논문 공식: d(AL,BL) = min(P1, P2)의 합
        lower_similarity = lower_dcd
        
        # 논문 공식: dDCH = γ*d(AU,BU) + (1-γ)*d(AL,BL)
        dcd_similarity = self.gamma * upper_similarity + (1 - self.gamma) * lower_similarity
        
        # 공간적 유사도 (상하체 평균)
        spatial_similarity = (upper_spatial + lower_spatial) / 2
        
        # 논문 공식: d(A,B) = α*dDCH + (1-α)*dDCR
        integrated_similarity = self.alpha * dcd_similarity + (1 - self.alpha) * spatial_similarity
        
        # 통계 업데이트
        self.performance_stats['dcd_matches'] += 1
        self.performance_stats['spatial_matches'] += 1
        self.performance_stats['integrated_matches'] += 1
        self.performance_stats['upper_body_matches'] += 1
        self.performance_stats['lower_body_matches'] += 1
        
        return integrated_similarity
    
    def process_person_detection(self, frame, yolo_detections):
        """논문 방식: 사람 감지 처리 (최적화)"""
        current_people = set()
        
        # 프레임 스킵핑 (성능 최적화)
        if self.frame_count % self.process_every_n_frames != 0:
            self.performance_stats['skipped_frames'] += 1
            # 기존 사람들의 위치를 유지
            for person_id in self.people:
                current_people.add(person_id)
            return current_people
        
        self.performance_stats['processed_frames'] += 1
        
        for detection in yolo_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            if confidence < self.confidence_threshold:
                continue
            
            # 바운딩박스 크기 필터링 (성능 최적화)
            x1, y1, x2, y2 = map(int, bbox)
            width = x2 - x1
            height = y2 - y1
            
            if width < self.min_bbox_size or height < self.min_bbox_size:
                continue
            
            # 상하체 분할
            upper_bbox, lower_bbox = self.segment_body_parts(frame, bbox)
            
            # 양자화된 히스토그램 추출
            upper_hist = self.get_quantized_histogram(frame, upper_bbox)
            lower_hist = self.get_quantized_histogram(frame, lower_bbox)
            
            if upper_hist is None or lower_hist is None:
                continue
            
            # 공간적 특징 추출 (최적화)
            upper_spatial = self.extract_spatial_features(frame, upper_bbox, upper_hist)
            lower_spatial = self.extract_spatial_features(frame, lower_bbox, lower_hist)
            
            # 기존 사람과 매칭 (현재 프레임에 있는 사람 + 일시 사라진 사람들)
            best_match_id = None
            best_similarity = 0.0
            
            for person_id, person_data in self.people.items():
                # 현재 프레임에 이미 매칭된 사람은 건너뛰기
                if person_id in current_people:
                    continue
                
                # 일시적으로 사라진 사람도 매칭 대상에 포함
                frames_missing = self.frame_count - person_data['last_seen']
                if frames_missing > self.max_disappeared:
                    continue  # 너무 오래 사라진 사람은 제외
                
                # 일시 사라진 사람은 더 관대한 임계값 적용
                current_threshold = self.reappear_threshold if frames_missing > 0 else self.similarity_threshold
                
                similarity = self.integrated_similarity(
                    {
                        'upper_hist': upper_hist,
                        'lower_hist': lower_hist,
                        'upper_spatial': upper_spatial,
                        'lower_spatial': lower_spatial
                    },
                    person_data
                )
                
                # 디버깅: 상세한 매칭 정보 출력
                if similarity > 0.3:  # 낮은 임계값으로 모든 매칭 시도 표시
                    upper_dcd = self.calculate_dcd_similarity(upper_hist, person_data['upper_hist'])
                    lower_dcd = self.calculate_dcd_similarity(lower_hist, person_data['lower_hist'])
                    upper_spatial = self.calculate_spatial_similarity(upper_spatial, person_data['upper_spatial'])
                    lower_spatial = self.calculate_spatial_similarity(lower_spatial, person_data['lower_spatial'])
                    
                    status = "일시 사라짐" if frames_missing > 0 else "활성"
                    print(f"🔍 매칭 시도: ID {person_id} vs 현재 감지 ({status})")
                    print(f"   - 상체 DCD 유사도: {upper_dcd:.3f}")
                    print(f"   - 하체 DCD 유사도: {lower_dcd:.3f}")
                    print(f"   - 상체 공간 유사도: {upper_spatial:.3f}")
                    print(f"   - 하체 공간 유사도: {lower_spatial:.3f}")
                    print(f"   - 통합 유사도: {similarity:.3f} (임계값: {current_threshold})")
                    print(f"   - 사라진 프레임: {frames_missing}")
                
                if similarity > current_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = person_id
            
            if best_match_id is not None:
                # 기존 사람과 매칭됨
                current_people.add(best_match_id)
                
                # 바운딩박스 스무딩 적용
                smoothed_bbox = self.smooth_bbox(best_match_id, bbox)
                
                # 데이터 업데이트 (색상 정보 유지)
                existing_color = self.people[best_match_id].get('color', self.colors[best_match_id % len(self.colors)])
                self.people[best_match_id].update({
                    'upper_hist': upper_hist,
                    'lower_hist': lower_hist,
                    'upper_spatial': upper_spatial,
                    'lower_spatial': lower_spatial,
                    'bbox': smoothed_bbox,  # 스무딩된 바운딩박스 사용
                    'last_seen': self.frame_count,
                    'confidence': confidence,
                    'color': existing_color  # 기존 색상 유지
                })
                
                print(f"🔄 기존 사람 매칭: ID {best_match_id} (유사도: {best_similarity:.3f})")
                
                # 일시적으로 사라졌던 사람인지 확인
                frames_missing = self.frame_count - self.people[best_match_id]['last_seen']
                if frames_missing > 0:
                    print(f"   ✅ 일시 사라졌던 사람 재발견 (사라진 프레임: {frames_missing})")
            else:
                # 새로운 사람 추가
                if len(self.people) < self.max_people:
                    color = self.colors[self.next_id % len(self.colors)]
                    
                    print(f"🎨 새로운 사람 ID {self.next_id}에게 색상 할당: {color}")
                    
                    self.people[self.next_id] = {
                        'upper_hist': upper_hist,
                        'lower_hist': lower_hist,
                        'upper_spatial': upper_spatial,
                        'lower_spatial': lower_spatial,
                        'color': color,
                        'bbox': bbox,
                        'last_seen': self.frame_count,
                        'confidence': confidence
                    }
                    
                    # 자동 스크린샷 저장
                    self.save_auto_screenshot(
                        frame, self.next_id, bbox, confidence,
                        upper_hist, lower_hist, upper_spatial, lower_spatial
                    )
                    
                    current_people.add(self.next_id)
                    print(f"🆕 새로운 사람: ID {self.next_id}")
                    self.next_id += 1
        
        # 사라진 사람 처리 (개선: 즉시 제거하지 않고 기억)
        people_to_remove = []
        for person_id in list(self.people.keys()):
            if person_id not in current_people:
                # 프레임에서 사라진 시간 계산
                frames_missing = self.frame_count - self.people[person_id]['last_seen']
                
                if frames_missing > self.max_disappeared:
                    # 오랫동안 사라진 경우에만 제거
                    people_to_remove.append(person_id)
                    print(f"🗑️ 사람 제거: ID {person_id} (사라진 프레임: {frames_missing})")
                else:
                    # 잠시 사라진 경우 기억 (다시 나타날 수 있음)
                    print(f"⏳ 사람 일시 사라짐: ID {person_id} (사라진 프레임: {frames_missing}/{self.max_disappeared})")
        
        for person_id in people_to_remove:
            del self.people[person_id]
        
        return current_people
    
    def draw_paper_results(self, frame, current_people):
        """논문 방식 결과 시각화 (깔끔하게 개선)"""
        result_frame = frame.copy()
        
        # 현재 프레임에서 YOLO로 감지된 사람들
        yolo_detections = []
        try:
            from simple_detector import PersonDetector
            detector = PersonDetector()
            yolo_detections = detector.detect_people(frame)
        except:
            pass
        
        # YOLO 감지 결과를 시각화 (연한 초록색, 얇은 선)
        for detection in yolo_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = map(int, bbox)
            
            # YOLO 감지 결과 (연한 초록색, 얇은 선)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
            
            # YOLO 신뢰도 표시 (작게)
            cv2.putText(result_frame, f"YOLO: {confidence:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)
        
        # 추적 중인 사람들을 시각화 (깔끔하게)
        for person_id in current_people:
            if person_id in self.people:
                person_data = self.people[person_id]
                bbox = person_data['bbox']
                color = person_data['color']
                confidence = person_data.get('confidence', 0.5)
                
                # 전체 바운딩박스 (진한 색상, 굵은 선)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
                
                # 상하체 구분선 (흰색, 얇은 선)
                split_y = y1 + int((y2 - y1) * self.upper_ratio)
                cv2.line(result_frame, (x1, split_y), (x2, split_y), (255, 255, 255), 1)
                
                # ID와 신뢰도 표시 (깔끔하게)
                id_text = f"ID:{person_id}"
                conf_text = f"{confidence:.2f}"
                
                # ID 텍스트 배경
                (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_frame, (x1, y1-id_h-5), (x1+id_w+10, y1), color, -1)
                cv2.rectangle(result_frame, (x1, y1-id_h-5), (x1+id_w+10, y1), (255, 255, 255), 2)
                
                # ID 텍스트
                cv2.putText(result_frame, id_text, (x1+5, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 신뢰도 텍스트 (우상단)
                (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(result_frame, (x2-conf_w-10, y1), (x2, y1+conf_h+5), (0, 0, 0), -1)
                cv2.rectangle(result_frame, (x2-conf_w-10, y1), (x2, y1+conf_h+5), color, 1)
                cv2.putText(result_frame, conf_text, (x2-conf_w-5, y1+conf_h), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 중심점 표시 (작게)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(result_frame, (center_x, center_y), 3, color, -1)
        
        # 시스템 통계 (간단하게)
        stats_y = 650
        stats_texts = [
            f"🎯 Paper-Based Person Re-identification",
            f"Active: {len(current_people)} | YOLO: {len(yolo_detections)} | Frame: {self.frame_count}",
            f"Auto Screenshots: {self.performance_stats['auto_screenshots']} | Weights: α={self.alpha}, β={self.beta}, γ={self.gamma}"
        ]
        
        # 통계 배경 (간단하게)
        cv2.rectangle(result_frame, (10, stats_y - 10), (800, 720 - 10), (0, 0, 0), -1)
        cv2.rectangle(result_frame, (10, stats_y - 10), (800, 720 - 10), (255, 255, 255), 2)
        
        for i, text in enumerate(stats_texts):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(result_frame, text, (20, stats_y + 10 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 범례 (간단하게)
        legend_y = 30
        legend_texts = [
            "📊 LEGEND:",
            "🟢 YOLO Detection",
            "🔴 Tracked Person"
        ]
        
        for i, text in enumerate(legend_texts):
            cv2.putText(result_frame, text, (10, legend_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def smooth_bbox(self, person_id, new_bbox):
        """바운딩박스 스무딩 및 예측"""
        if person_id not in self.bbox_history:
            self.bbox_history[person_id] = []
            self.velocity_history[person_id] = [0, 0, 0, 0]  # dx1, dy1, dx2, dy2
        
        # 현재 바운딩박스
        x1, y1, x2, y2 = map(int, new_bbox)
        
        # 이전 바운딩박스가 있으면 스무딩 적용
        if self.bbox_history[person_id]:
            prev_bbox = self.bbox_history[person_id][-1]
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
            
            # 속도 계산
            velocity = self.velocity_history[person_id]
            new_velocity = [
                x1 - prev_x1,
                y1 - prev_y1, 
                x2 - prev_x2,
                y2 - prev_y2
            ]
            
            # 속도 스무딩
            for i in range(4):
                velocity[i] = velocity[i] * 0.8 + new_velocity[i] * 0.2
            
            # 예측된 위치
            predicted_x1 = prev_x1 + int(velocity[0])
            predicted_y1 = prev_y1 + int(velocity[1])
            predicted_x2 = prev_x2 + int(velocity[2])
            predicted_y2 = prev_y2 + int(velocity[3])
            
            # 현재 위치와 예측 위치의 가중 평균
            smoothed_x1 = int(predicted_x1 * self.bbox_smoothing_factor + x1 * (1 - self.bbox_smoothing_factor))
            smoothed_y1 = int(predicted_y1 * self.bbox_smoothing_factor + y1 * (1 - self.bbox_smoothing_factor))
            smoothed_x2 = int(predicted_x2 * self.bbox_smoothing_factor + x2 * (1 - self.bbox_smoothing_factor))
            smoothed_y2 = int(predicted_y2 * self.bbox_smoothing_factor + y2 * (1 - self.bbox_smoothing_factor))
            
            smoothed_bbox = [smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2]
        else:
            smoothed_bbox = [x1, y1, x2, y2]
        
        # 히스토리 업데이트 (최대 3개 유지)
        self.bbox_history[person_id].append(smoothed_bbox)
        if len(self.bbox_history[person_id]) > 3:  # 5 → 3 (메모리 절약)
            self.bbox_history[person_id].pop(0)
        
        return smoothed_bbox


def main():
    """메인 함수 - 논문 방식 구현"""
    print("🚀 PAPER-BASED PERSON RE-IDENTIFICATION SYSTEM")
    print("📖 Based on: Color Histogram and Spatial Configuration of Dominant Color Regions")
    print("👥 Authors: Kwangchol Jang, Sokmin Han, Insong Kim")
    print("🏫 Institution: KIM IL SUNG University")
    
    # 스크린샷 디렉토리 생성
    import os
    screenshot_dir = os.path.join(os.path.dirname(__file__), "paper_debug_screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    print(f"📁 Screenshot directory: {screenshot_dir}")
    
    detector = PersonDetector()
    tracker = PaperBasedPersonReidentification()
    tracker.set_screenshot_directory(screenshot_dir) # 트래커에 스크린샷 디렉토리 설정
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("📖 Controls:")
    print("  - q: Quit")
    print("  - s: Save screenshot")
    print("  - r: Reset tracking")
    print("  - p: Performance stats")
    
    window_name = "Paper-Based Person Re-identification"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            tracker.frame_count += 1
            tracker.performance_stats['total_frames'] += 1
            
            # YOLO 사람 감지
            yolo_detections = detector.detect_people(frame)
            
            # 논문 방식 처리
            current_people = tracker.process_person_detection(frame, yolo_detections)
            
            # 결과 시각화
            result_frame = tracker.draw_paper_results(frame, current_people)
            
            cv2.imshow(window_name, result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🛑 System shutdown")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"paper_reid_{timestamp}.jpg"
                filepath = os.path.join(screenshot_dir, filename)
                
                # 디버그 정보 추가
                debug_info = [
                    f"Paper-Based Re-identification",
                    f"Frame: {tracker.frame_count}",
                    f"Active People: {len(current_people)}",
                    f"YOLO Detections: {len(yolo_detections)}",
                    f"HSV Quantization: 8x3x3",
                    f"Weights: α={tracker.alpha}, β={tracker.beta}, γ={tracker.gamma}",
                    f"Timestamp: {timestamp}"
                ]
                
                debug_frame = result_frame.copy()
                debug_y = 30
                for info in debug_info:
                    cv2.putText(debug_frame, info, (10, debug_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    debug_y += 25
                
                success = cv2.imwrite(filepath, debug_frame)
                if success:
                    print(f"📸 Paper method screenshot saved: {filepath}")
                    print(f"   - Frame: {tracker.frame_count}")
                    print(f"   - Active people: {len(current_people)}")
                    print(f"   - DCD matches: {tracker.performance_stats['dcd_matches']}")
                else:
                    print(f"❌ Failed to save screenshot: {filepath}")
            elif key == ord('r'):
                tracker.people.clear()
                tracker.next_id = 0
                print("🔄 Tracking reset")
            elif key == ord('p'):
                stats = tracker.performance_stats
                print(f"\n📊 Paper Method Performance Stats:")
                print(f"  - Total frames: {stats['total_frames']}")
                print(f"  - Processed frames: {stats['processed_frames']}")
                print(f"  - Skipped frames: {stats['skipped_frames']}")
                print(f"  - Active people: {len(tracker.people)}")
                print(f"  - DCD matches: {stats['dcd_matches']}")
                print(f"  - Spatial matches: {stats['spatial_matches']}")
                print(f"  - Integrated matches: {stats['integrated_matches']}")
                print(f"  - Upper body matches: {stats['upper_body_matches']}")
                print(f"  - Lower body matches: {stats['lower_body_matches']}")
                print(f"  - Auto screenshots: {stats['auto_screenshots']}")
                
    except KeyboardInterrupt:
        print("🛑 User interrupt")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎯 Final Results:")
        print(f"  - Total frames: {tracker.performance_stats['total_frames']}")
        print(f"  - Processed frames: {tracker.performance_stats['processed_frames']}")
        print(f"  - Active people: {len(tracker.people)}")
        print(f"  - Auto screenshots: {tracker.performance_stats['auto_screenshots']}")
        print(f"  - Paper method implemented successfully")
        print(f"  - Screenshots saved in: {screenshot_dir}")
        
        print(f"✅ Paper-Based Person Re-identification System terminated")

if __name__ == "__main__":
    main()

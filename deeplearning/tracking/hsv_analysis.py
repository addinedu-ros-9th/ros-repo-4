import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from datetime import datetime

# Qt 플랫폼 플러그인 오류 방지
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_DEBUG_PLUGINS'] = '0'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'

# matplotlib 백엔드를 Agg로 설정
plt.switch_backend('Agg')

def estimate_distance(bbox_height, ref_height=300, ref_distance=1.0):
    distance = ref_height / (bbox_height + 1e-6) * ref_distance
    return round(distance, 2)

def estimate_distance_from_mask(mask, ref_height=300, ref_distance=1.0):
    """세그멘테이션 마스크를 사용한 더 정확한 거리 계산"""
    # 마스크에서 실제 사람 영역의 높이 계산
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 컨투어 (사람) 선택
        cnt = max(contours, key=cv2.contourArea)
        
        # 컨투어의 바운딩 박스
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 실제 사람 높이 (픽셀 단위)
        person_height = h
        
        # 거리 계산 (역비례 관계)
        distance = ref_height / (person_height + 1e-6) * ref_distance
        return round(distance, 2)
    
    # 컨투어를 찾을 수 없는 경우 기본값 반환
    return 2.0

def estimate_distance_advanced(mask, ref_height=300, ref_distance=1.0):
    """고급 거리 계산 - 마스크의 실제 픽셀 수와 형태 고려"""
    # 마스크에서 실제 사람 영역 분석
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        
        # 컨투어의 바운딩 박스
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 실제 사람 영역의 픽셀 수
        person_pixels = cv2.contourArea(cnt)
        
        # 바운딩 박스 영역
        bbox_area = w * h
        
        # 사람이 바운딩 박스를 얼마나 채우는지 (밀도)
        density = person_pixels / (bbox_area + 1e-6)
        
        # 밀도를 고려한 조정된 높이
        adjusted_height = h * density
        
        # 거리 계산
        distance = ref_height / (adjusted_height + 1e-6) * ref_distance
        
        return round(distance, 2), {
            'person_height': h,
            'person_pixels': person_pixels,
            'bbox_area': bbox_area,
            'density': density,
            'adjusted_height': adjusted_height
        }
    
    return 2.0, {}

class HSVAnalyzer:
    def __init__(self):
        self.model = YOLO('yolov8s-seg.pt')
        self.people_data = {}  # ID: {histograms: [], timestamps: [], images: []}
        self.next_id = 0
        
        # 성능 최적화 설정
        self.process_every_n_frames = 3  # 3프레임마다 처리 (성능 향상)
        self.frame_skip_counter = 0
        
        # 매칭 관련 설정
        self.match_threshold = 0.35  # 매칭 임계값
        self.reentry_threshold = 0.30  # 재진입 임계값
        self.min_detection_confidence = 0.6  # 최소 감지 신뢰도
        self.min_person_area = 5000  # 최소 사람 영역
        self.max_frames_without_seen = 300  # 10초 후에도 기억
        
        # 히스토그램 기억 설정
        self.max_histograms_per_person = 10  # 사람당 최대 히스토그램 저장 수
        self.histogram_memory_duration = 30  # 30초간 히스토그램 기억
        
        # 분석 결과 저장 디렉토리
        self.analysis_dir = "./hsv_analysis"
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
            print(f"📁 분석 디렉토리 생성: {self.analysis_dir}")
    
    def extract_histogram(self, img, mask, bins=16):
        """HSV의 모든 채널(H, S, V)을 고려한 히스토그램 추출"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 각 채널별 히스토그램 계산
        h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])
        
        # 정규화
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # 모든 채널을 하나의 벡터로 결합
        combined_hist = np.concatenate([h_hist, s_hist, v_hist])
        
        return combined_hist, h_hist, s_hist, v_hist
    
    def calculate_similarity_metrics(self, hist1, hist2):
        """다양한 유사도 메트릭 계산"""
        # Bhattacharyya 거리
        bhatt_dist = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        
        # 코사인 유사도
        dot_product = np.dot(hist1, hist2)
        norm1 = np.linalg.norm(hist1)
        norm2 = np.linalg.norm(hist2)
        cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        # 상관계수
        corr = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
        
        # Chi-Square 거리
        chi_square = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CHISQR)
        
        # Intersection
        intersection = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_INTERSECT)
        
        return {
            'bhattacharyya': bhatt_dist,
            'cosine_similarity': cosine_sim,
            'correlation': corr,
            'chi_square': chi_square,
            'intersection': intersection
        }
    
    def find_best_match(self, current_hist, current_bbox, used_ids):
        """가장 유사한 사람 찾기 (개선된 버전)"""
        best_match_id = None
        best_score = 0.0
        best_metrics = {}
        
        x1, y1, x2, y2 = current_bbox
        current_area = (x2 - x1) * (y2 - y1)
        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # 현재 프레임에서 감지된 모든 사람과 비교
        for pid, pdata in self.people_data.items():
            # 이미 사용된 ID는 제외
            if pid in used_ids:
                continue
                
            if len(pdata['histograms']) == 0:
                continue
            
            # 모든 히스토그램과 비교 (최근 10개)
            hist_scores = []
            for i, stored_hist in enumerate(pdata['histograms'][-self.max_histograms_per_person:]):
                metrics = self.calculate_similarity_metrics(current_hist, stored_hist)
                hist_score = max(1.0 - metrics['bhattacharyya'], metrics['cosine_similarity'])
                hist_scores.append(hist_score)
            
            # 최고 히스토그램 점수 사용
            best_hist_score = max(hist_scores) if hist_scores else 0.0
            
            # 공간적 유사도 (가중치 대폭 감소: 30% → 10%)
            latest_bbox = pdata['bboxes'][-1]
            stored_area = (latest_bbox[2] - latest_bbox[0]) * (latest_bbox[3] - latest_bbox[1])
            stored_center = ((latest_bbox[0] + latest_bbox[2]) // 2, (latest_bbox[1] + latest_bbox[3]) // 2)
            
            # 중심점 거리 계산
            center_distance = np.sqrt((current_center[0] - stored_center[0])**2 + 
                                     (current_center[1] - stored_center[1])**2)
            max_distance = np.sqrt(640**2 + 480**2)
            spatial_score = 1.0 - (center_distance / max_distance)
            
            # 종합 점수 (히스토그램 90%, 공간적 위치 10%) - 위치 가중치 대폭 감소
            total_score = 0.9 * best_hist_score + 0.1 * spatial_score
            
            # 더 높은 점수를 가진 매칭 선택
            if total_score > best_score:
                best_score = total_score
                best_match_id = pid
                best_metrics = {
                    'hist_score': best_hist_score,
                    'spatial_score': spatial_score,
                    'hist_scores': hist_scores
                }
        
        return best_match_id, best_score, best_metrics
    
    def visualize_histogram_comparison(self, hist1, hist2, person_id1, person_id2, frame_count, save_path):
        """히스토그램 비교 시각화"""
        bins = 16
        h_hist1 = hist1[:bins]
        s_hist1 = hist1[bins:2*bins]
        v_hist1 = hist1[2*bins:]
        h_hist2 = hist2[:bins]
        s_hist2 = hist2[bins:2*bins]
        v_hist2 = hist2[2*bins:]
        
        # 유사도 메트릭 계산
        metrics = self.calculate_similarity_metrics(hist1, hist2)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'HSV Histogram Comparison: {person_id1} vs {person_id2} (Frame {frame_count})', fontsize=16)
        
        # H 채널 비교
        axes[0, 0].bar(np.arange(bins) - 0.2, h_hist1, 0.4, label=person_id1, alpha=0.7, color='blue')
        axes[0, 0].bar(np.arange(bins) + 0.2, h_hist2, 0.4, label=person_id2, alpha=0.7, color='red')
        axes[0, 0].set_title('Hue Channel')
        axes[0, 0].set_xlabel('Hue Bins')
        axes[0, 0].set_ylabel('Normalized Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # S 채널 비교
        axes[0, 1].bar(np.arange(bins) - 0.2, s_hist1, 0.4, label=person_id1, alpha=0.7, color='blue')
        axes[0, 1].bar(np.arange(bins) + 0.2, s_hist2, 0.4, label=person_id2, alpha=0.7, color='red')
        axes[0, 1].set_title('Saturation Channel')
        axes[0, 1].set_xlabel('Saturation Bins')
        axes[0, 1].set_ylabel('Normalized Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # V 채널 비교
        axes[0, 2].bar(np.arange(bins) - 0.2, v_hist1, 0.4, label=person_id1, alpha=0.7, color='blue')
        axes[0, 2].bar(np.arange(bins) + 0.2, v_hist2, 0.4, label=person_id2, alpha=0.7, color='red')
        axes[0, 2].set_title('Value Channel')
        axes[0, 2].set_xlabel('Value Bins')
        axes[0, 2].set_ylabel('Normalized Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 차이 분석
        axes[1, 0].bar(np.arange(bins), np.abs(h_hist1 - h_hist2), color='orange', alpha=0.7)
        axes[1, 0].set_title(f'Hue Difference (Sum: {np.sum(np.abs(h_hist1 - h_hist2)):.4f})')
        axes[1, 0].set_xlabel('Hue Bins')
        axes[1, 0].set_ylabel('Absolute Difference')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(np.arange(bins), np.abs(s_hist1 - s_hist2), color='orange', alpha=0.7)
        axes[1, 1].set_title(f'Saturation Difference (Sum: {np.sum(np.abs(s_hist1 - s_hist2)):.4f})')
        axes[1, 1].set_xlabel('Saturation Bins')
        axes[1, 1].set_ylabel('Absolute Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].bar(np.arange(bins), np.abs(v_hist1 - v_hist2), color='orange', alpha=0.7)
        axes[1, 2].set_title(f'Value Difference (Sum: {np.sum(np.abs(v_hist1 - v_hist2)):.4f})')
        axes[1, 2].set_xlabel('Value Bins')
        axes[1, 2].set_ylabel('Absolute Difference')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return metrics
    
    def create_similarity_matrix(self, save_path):
        """모든 사람 간의 유사도 매트릭스 생성"""
        if len(self.people_data) < 2:
            print("⚠️ 유사도 매트릭스를 생성하려면 최소 2명의 사람이 필요합니다.")
            return
        
        people_ids = list(self.people_data.keys())
        n_people = len(people_ids)
        
        # 유사도 매트릭스 초기화
        similarity_matrix = np.zeros((n_people, n_people))
        bhatt_matrix = np.zeros((n_people, n_people))
        cosine_matrix = np.zeros((n_people, n_people))
        
        print(f"\n📊 {n_people}명의 사람 간 유사도 매트릭스 생성 중...")
        
        for i, pid1 in enumerate(people_ids):
            for j, pid2 in enumerate(people_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    bhatt_matrix[i, j] = 0.0
                    cosine_matrix[i, j] = 1.0
                else:
                    # 각 사람의 최신 히스토그램 사용
                    hist1 = self.people_data[pid1]['histograms'][-1]
                    hist2 = self.people_data[pid2]['histograms'][-1]
                    
                    metrics = self.calculate_similarity_metrics(hist1, hist2)
                    
                    # 종합 유사도 점수
                    hist_score = max(1.0 - metrics['bhattacharyya'], metrics['cosine_similarity'])
                    similarity_matrix[i, j] = hist_score
                    bhatt_matrix[i, j] = metrics['bhattacharyya']
                    cosine_matrix[i, j] = metrics['cosine_similarity']
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 종합 유사도 매트릭스
        im1 = axes[0].imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0].set_title('Overall Similarity Matrix')
        axes[0].set_xticks(range(n_people))
        axes[0].set_yticks(range(n_people))
        axes[0].set_xticklabels(people_ids, rotation=45)
        axes[0].set_yticklabels(people_ids)
        plt.colorbar(im1, ax=axes[0])
        
        # 값 표시
        for i in range(n_people):
            for j in range(n_people):
                axes[0].text(j, i, f'{similarity_matrix[i, j]:.3f}', 
                           ha='center', va='center', fontsize=8)
        
        # Bhattacharyya 거리 매트릭스
        im2 = axes[1].imshow(bhatt_matrix, cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title('Bhattacharyya Distance Matrix')
        axes[1].set_xticks(range(n_people))
        axes[1].set_yticks(range(n_people))
        axes[1].set_xticklabels(people_ids, rotation=45)
        axes[1].set_yticklabels(people_ids)
        plt.colorbar(im2, ax=axes[1])
        
        # 값 표시
        for i in range(n_people):
            for j in range(n_people):
                axes[1].text(j, i, f'{bhatt_matrix[i, j]:.3f}', 
                           ha='center', va='center', fontsize=8)
        
        # 코사인 유사도 매트릭스
        im3 = axes[2].imshow(cosine_matrix, cmap='Blues', vmin=0, vmax=1)
        axes[2].set_title('Cosine Similarity Matrix')
        axes[2].set_xticks(range(n_people))
        axes[2].set_yticks(range(n_people))
        axes[2].set_xticklabels(people_ids, rotation=45)
        axes[2].set_yticklabels(people_ids)
        plt.colorbar(im3, ax=axes[2])
        
        # 값 표시
        for i in range(n_people):
            for j in range(n_people):
                axes[2].text(j, i, f'{cosine_matrix[i, j]:.3f}', 
                           ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 유사도 매트릭스 저장: {save_path}")
        
        # 수치적 분석 결과 출력
        print(f"\n📈 수치적 분석 결과:")
        print(f"   - 평균 유사도: {np.mean(similarity_matrix):.3f}")
        print(f"   - 최대 유사도: {np.max(similarity_matrix):.3f}")
        print(f"   - 최소 유사도: {np.min(similarity_matrix):.3f}")
        print(f"   - 표준편차: {np.std(similarity_matrix):.3f}")
        
        # 가장 유사한 쌍과 가장 다른 쌍 찾기
        max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        min_sim_idx = np.unravel_index(np.argmin(similarity_matrix), similarity_matrix.shape)
        
        if max_sim_idx[0] != max_sim_idx[1]:
            print(f"   - 가장 유사한 쌍: {people_ids[max_sim_idx[0]]} vs {people_ids[max_sim_idx[1]]} (유사도: {similarity_matrix[max_sim_idx]:.3f})")
        if min_sim_idx[0] != min_sim_idx[1]:
            print(f"   - 가장 다른 쌍: {people_ids[min_sim_idx[0]]} vs {people_ids[min_sim_idx[1]]} (유사도: {similarity_matrix[min_sim_idx]:.3f})")
    
    def run_analysis(self, duration_seconds=30):
        """HSV 히스토그램 분석 실행"""
        cap = cv2.VideoCapture(0)  # 2에서 0으로 변경
        
        if not cap.isOpened():
            print(f"❌ 카메라연결 실패")
            return
        
       
        frame_count = 0
        start_time = datetime.now()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            if elapsed_time > duration_seconds:
                break
            
            # 프레임 처리 간격 조절
            if self.frame_skip_counter < self.process_every_n_frames - 1:
                self.frame_skip_counter += 1
                continue
            
            self.frame_skip_counter = 0 # 카운터 초기화
            
            # 사람 감지
            results = self.model(frame, classes=[0])  # class 0 = person
            annotated = frame.copy()
            
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
                    
                    # 거리 추정 (사용자가 추가한 함수 사용)
                    person_height = y2 - y1
                    distance = estimate_distance(person_height, ref_height=300, ref_distance=1.0)  # m 단위
                    
                    # 세그멘테이션 마스크를 사용한 더 정확한 거리 계산
                    est_dist, dist_info = estimate_distance_advanced(mask_cleaned, ref_height=300, ref_distance=1.0)
                    
                    current_detections.append({
                        'hist': combined_hist,
                        'bbox': bbox,
                        'mask': mask_cleaned,
                        'confidence': confidence,
                        'area': area,
                        'distance': est_dist,  # 더 정확한 거리 사용
                        'dist_info': dist_info  # 거리 계산 정보도 저장
                    })
            
            # 감지된 사람들을 영역 크기 순으로 정렬 (큰 사람부터 처리)
            current_detections.sort(key=lambda x: x['area'], reverse=True)
            
            # 이미 매칭된 ID들을 추적
            used_ids = set()
            
            # 각 감지된 사람에 대해 매칭 수행
            for detection in current_detections:
                combined_hist = detection['hist']
                bbox = detection['bbox']
                
                # 매칭 시도
                matched_id, match_score, metrics = self.find_best_match(combined_hist, bbox, used_ids)
                
                print(f"🎯 매칭 결과: {matched_id}, 점수: {match_score:.3f}, 거리: {detection['distance']:.2f}m, 임계값: {self.match_threshold:.3f}")
                
                # 거리 계산 상세 정보 출력
                if detection['dist_info']:
                    print(f"   📏 거리 계산 상세: 높이={detection['dist_info']['person_height']}px, 밀도={detection['dist_info']['density']:.3f}")
                
                # 매칭 성공 여부에 따른 처리
                if matched_id is not None and match_score > self.match_threshold:
                    # 기존 사람 업데이트
                    self.people_data[matched_id]['histograms'].append(combined_hist)
                    self.people_data[matched_id]['bboxes'].append(bbox)
                    self.people_data[matched_id]['timestamps'].append(elapsed_time)
                    used_ids.add(matched_id)  # ID 사용됨 표시
                    
                    # 히스토그램 메모리 관리 (최대 개수 제한)
                    if len(self.people_data[matched_id]['histograms']) > self.max_histograms_per_person:
                        # 가장 오래된 히스토그램 제거
                        self.people_data[matched_id]['histograms'].pop(0)
                        self.people_data[matched_id]['bboxes'].pop(0)
                        self.people_data[matched_id]['timestamps'].pop(0)
                    
                    color = (0, 255, 0)  # 초록색 (기존 사람)
                    print(f"🔄 기존 사람 재식별: {matched_id} (점수: {match_score:.3f})")
                    print(f"   - 히스토그램 점수: {metrics['hist_score']:.3f}")
                    print(f"   - 공간적 점수: {metrics['spatial_score']:.3f}")
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
                    used_ids.add(new_id)  # ID 사용됨 표시
                    
                    color = (0, 0, 255)  # 빨간색 (새로운 사람)
                    print(f"🆕 새로운 사람 감지: {new_id} (최고 점수: {match_score:.3f}, 임계값 미달)")
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
                
                # 거리 표시 (m 단위) - 더 정확한 거리 계산 사용
                distance_text = f"Dist: {detection['distance']:.2f}m"
                cv2.putText(annotated, distance_text, (x1, y2+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 밀도 정보 표시 (디버깅용)
                if detection['dist_info']:
                    density_text = f"Density: {detection['dist_info']['density']:.2f}"
                    cv2.putText(annotated, density_text, (x1, y2+60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 유사도 점수 표시
                score_text = f"{match_score:.3f}"
                cv2.putText(annotated, score_text, (x1, y2+80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 시스템 정보 표시
            info_text = f"People: {len(self.people_data)} | Frame: {frame_count} | Time: {elapsed_time:.1f}s"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 성능 최적화: 5프레임마다만 화면 업데이트
            if frame_count % 5 == 0:
                cv2.imshow("HSV Histogram Analysis", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 분석 결과 생성
        print(f"\n📊 분석 완료!")
        print(f"   - 총 프레임: {frame_count}")
        print(f"   - 감지된 사람 수: {len(self.people_data)}")
        
        # 유사도 매트릭스 생성
        if len(self.people_data) > 1:
            matrix_path = os.path.join(self.analysis_dir, "similarity_matrix.png")
            self.create_similarity_matrix(matrix_path)
        
        # 각 사람의 히스토그램 변화 분석
        for pid, pdata in self.people_data.items():
            if len(pdata['histograms']) > 1:
                print(f"\n👤 {pid} HSV 히스토그램 변화 분석:")
                histograms = np.array(pdata['histograms'])
                
                # 시간에 따른 히스토그램 변화
                hist_variance = np.var(histograms, axis=0)
                hist_mean = np.mean(histograms, axis=0)
                hist_max = np.max(histograms, axis=0)
                hist_min = np.min(histograms, axis=0)
                
                # HSV 채널별 분석
                bins = 16
                h_variance = hist_variance[:bins]
                s_variance = hist_variance[bins:2*bins]
                v_variance = hist_variance[2*bins:]
                
                h_mean = hist_mean[:bins]
                s_mean = hist_mean[bins:2*bins]
                v_mean = hist_mean[2*bins:]
                
                h_max = hist_max[:bins]
                s_max = hist_max[bins:2*bins]
                v_max = hist_max[2*bins:]
                
                print(f"   📊 전체 히스토그램 통계:")
                print(f"      - 분산 (평균): {np.mean(hist_variance):.6f}")
                print(f"      - 분산 (최대): {np.max(hist_variance):.6f}")
                print(f"      - 분산 (최소): {np.min(hist_variance):.6f}")
                print(f"      - 분산 (표준편차): {np.std(hist_variance):.6f}")
                
                print(f"   🎨 HSV 채널별 분산 분석:")
                print(f"      - Hue 분산 (평균): {np.mean(h_variance):.6f}")
                print(f"      - Saturation 분산 (평균): {np.mean(s_variance):.6f}")
                print(f"      - Value 분산 (평균): {np.mean(v_variance):.6f}")
                
                print(f"   📈 HSV 채널별 평균값:")
                print(f"      - Hue 평균: {np.mean(h_mean):.4f}")
                print(f"      - Saturation 평균: {np.mean(s_mean):.4f}")
                print(f"      - Value 평균: {np.mean(v_mean):.4f}")
                
                print(f"   🔥 HSV 채널별 최대값:")
                print(f"      - Hue 최대: {np.max(h_max):.4f}")
                print(f"      - Saturation 최대: {np.max(s_max):.4f}")
                print(f"      - Value 최대: {np.max(v_max):.4f}")
                
                # 안정성 평가
                stability_score = 1.0 - np.mean(hist_variance)
                print(f"   🎯 안정성 평가:")
                print(f"      - 전체 안정성 점수: {stability_score:.4f}")
                print(f"      - Hue 안정성: {1.0 - np.mean(h_variance):.4f}")
                print(f"      - Saturation 안정성: {1.0 - np.mean(s_variance):.4f}")
                print(f"      - Value 안정성: {1.0 - np.mean(v_variance):.4f}")
                
                # 첫 번째와 마지막 히스토그램 비교
                first_hist = histograms[0]
                last_hist = histograms[-1]
                metrics = self.calculate_similarity_metrics(first_hist, last_hist)
                print(f"   🔄 첫-마지막 프레임 비교:")
                print(f"      - Bhattacharyya 거리: {metrics['bhattacharyya']:.4f}")
                print(f"      - 코사인 유사도: {metrics['cosine_similarity']:.4f}")
                print(f"      - 상관계수: {metrics['correlation']:.4f}")
                print(f"      - 종합 유사도: {max(1.0 - metrics['bhattacharyya'], metrics['cosine_similarity']):.4f}")
                
                # 트래킹 지속 가능성 평가
                if stability_score > 0.8:
                    tracking_assessment = "🟢 매우 안정적 - 장기 트래킹 가능"
                elif stability_score > 0.6:
                    tracking_assessment = "🟡 보통 안정적 - 중기 트래킹 가능"
                elif stability_score > 0.4:
                    tracking_assessment = "🟠 불안정 - 단기 트래킹만 가능"
                else:
                    tracking_assessment = "🔴 매우 불안정 - 트래킹 어려움"
                
                print(f"   🎯 트래킹 지속 가능성: {tracking_assessment}")
                
                # 연속성 분석
                consecutive_similarities = []
                for i in range(1, len(histograms)):
                    prev_hist = histograms[i-1]
                    curr_hist = histograms[i]
                    metrics = self.calculate_similarity_metrics(prev_hist, curr_hist)
                    similarity = max(1.0 - metrics['bhattacharyya'], metrics['cosine_similarity'])
                    consecutive_similarities.append(similarity)
                
                if consecutive_similarities:
                    print(f"   📊 연속 프레임 유사도:")
                    print(f"      - 평균: {np.mean(consecutive_similarities):.4f}")
                    print(f"      - 최소: {np.min(consecutive_similarities):.4f}")
                    print(f"      - 최대: {np.max(consecutive_similarities):.4f}")
                    print(f"      - 표준편차: {np.std(consecutive_similarities):.4f}")

if __name__ == "__main__":
    analyzer = HSVAnalyzer()
    
    analyzer.run_analysis(duration_seconds=60)
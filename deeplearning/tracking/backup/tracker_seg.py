import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Qt 플랫폼 플러그인 오류 방지 - 더 강력한 설정
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # offscreen 대신 xcb 사용
# os.environ['DISPLAY'] = ':0'  # 디스플레이 설정 - 주석 처리
os.environ['QT_DEBUG_PLUGINS'] = '0'  # 디버그 메시지 비활성화
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'  # 자동 스케일링 비활성화
os.environ['QT_SCALE_FACTOR'] = '1'  # 스케일 팩터 고정

# matplotlib 백엔드를 Agg로 설정 (GUI 없이)
plt.switch_backend('Agg')

# OpenCV GUI 비활성화 (필요시)
try:
    cv2.setUseOptimized(True)
except:
    pass

model = YOLO('yolov8s-seg.pt')  # YOLOv8 segmentation 모델

tracked_people = {}  # ID: {hist, body_props, last_seen_frame, box, color}
frame_count = 0
next_person_id = 0

# 디버깅 설정
debug_mode = True

# 스크립트 위치 기준으로 디렉토리 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
debug_dir = os.path.join(script_dir, "debug_histograms")
output_dir = os.path.join(script_dir, "output_frames")

# 디렉토리 생성
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
    print(f"📁 디버그 디렉토리 생성: {debug_dir}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 출력 디렉토리 생성: {output_dir}")

print(f"🔧 디버그 모드: {debug_mode}")
print(f"📊 히스토그램 저장 경로: {debug_dir}")
print(f"📸 이미지 저장 경로: {output_dir}")

# 고유 색상 팔레트
colors = [
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

def extract_histogram(img, mask, bins=16):
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
    
    return combined_hist

def calculate_hw_ratio(contour):
    """높이-너비 비율 계산"""
    x, y, w, h = cv2.boundingRect(contour)
    if w > 0:
        return h / w
    return 1.0

def estimate_shoulder_width(contour):
    """어깨 너비 추정 (상단 1/3 영역에서의 최대 너비)"""
    x, y, w, h = cv2.boundingRect(contour)
    if h < 3:
        return w
    
    # 상단 1/3 영역에서의 너비 측정
    top_third = h // 3
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_roi, [contour], -1, 255, -1, offset=(-x, -y))
    
    # 상단 1/3 영역에서의 최대 너비
    top_region = mask_roi[:top_third, :]
    if np.sum(top_region) > 0:
        # 각 행에서의 픽셀 수 계산
        row_widths = np.sum(top_region > 0, axis=1)
        return np.max(row_widths)
    return w

def estimate_torso_leg_ratio(contour):
    """상체-하체 비율 추정"""
    x, y, w, h = cv2.boundingRect(contour)
    if h < 6:
        return 1.0
    
    # 상체(상단 1/2)와 하체(하단 1/2) 영역 분리
    mid_point = h // 2
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_roi, [contour], -1, 255, -1, offset=(-x, -y))
    
    torso_region = mask_roi[:mid_point, :]
    leg_region = mask_roi[mid_point:, :]
    
    torso_area = np.sum(torso_region > 0)
    leg_area = np.sum(leg_region > 0)
    
    if leg_area > 0:
        return torso_area / leg_area
    return 1.0

def extract_body_proportions(mask):
    """세그멘테이션 마스크에서 신체 비율 추출"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        
        # 신체 비율 특징
        height_width_ratio = calculate_hw_ratio(cnt)
        shoulder_width = estimate_shoulder_width(cnt)
        torso_leg_ratio = estimate_torso_leg_ratio(cnt)
        
        return [height_width_ratio, shoulder_width, torso_leg_ratio]
    
    return [1.0, 1.0, 1.0]  # 기본값

def bhattacharyya_distance(hist1, hist2):
    return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)

def cosine_similarity(hist1, hist2):
    """코사인 유사도 계산"""
    dot_product = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def body_proportion_similarity(props1, props2):
    """신체 비율 유사도 계산"""
    if len(props1) != len(props2):
        return 0.0
    
    # 각 비율의 차이를 계산하고 유사도로 변환
    differences = []
    for p1, p2 in zip(props1, props2):
        if max(p1, p2) > 0:
            diff = abs(p1 - p2) / max(p1, p2)
            differences.append(1.0 - diff)
        else:
            differences.append(1.0)
    
    return np.mean(differences)

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

def get_next_color():
    """다음 사용할 색상 반환"""
    global next_person_id
    color = colors[next_person_id % len(colors)]
    return color

def plot_histogram_comparison(current_hist, matched_hist, current_props, matched_props, matched_id, frame_count, save_path):
    """히스토그램 및 신체 비율 비교 그래프 생성"""
    plt.figure(figsize=(20, 12))
    
    # HSV 채널 분리
    bins = 16
    h_hist1 = current_hist[:bins]
    s_hist1 = current_hist[bins:2*bins]
    v_hist1 = current_hist[2*bins:]
    h_hist2 = matched_hist[:bins]
    s_hist2 = matched_hist[bins:2*bins]
    v_hist2 = matched_hist[2*bins:]
    
    # 서브플롯 1: H 채널 히스토그램 비교
    plt.subplot(3, 4, 1)
    x = np.arange(bins)
    width = 0.35
    plt.bar(x - width/2, h_hist1, width, label='Current Person', alpha=0.7, color='blue')
    plt.bar(x + width/2, h_hist2, width, label=f'Matched {matched_id}', alpha=0.7, color='red')
    plt.xlabel('Hue Bins')
    plt.ylabel('Normalized Frequency')
    plt.title('Hue Channel Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: S 채널 히스토그램 비교
    plt.subplot(3, 4, 2)
    plt.bar(x - width/2, s_hist1, width, label='Current Person', alpha=0.7, color='blue')
    plt.bar(x + width/2, s_hist2, width, label=f'Matched {matched_id}', alpha=0.7, color='red')
    plt.xlabel('Saturation Bins')
    plt.ylabel('Normalized Frequency')
    plt.title('Saturation Channel Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: V 채널 히스토그램 비교
    plt.subplot(3, 4, 3)
    plt.bar(x - width/2, v_hist1, width, label='Current Person', alpha=0.7, color='blue')
    plt.bar(x + width/2, v_hist2, width, label=f'Matched {matched_id}', alpha=0.7, color='red')
    plt.xlabel('Value Bins')
    plt.ylabel('Normalized Frequency')
    plt.title('Value Channel Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 4: 신체 비율 비교
    plt.subplot(3, 4, 4)
    prop_labels = ['Height/Width', 'Shoulder Width', 'Torso/Leg']
    plt.bar(np.arange(len(prop_labels)) - width/2, current_props, width, label='Current Person', alpha=0.7, color='blue')
    plt.bar(np.arange(len(prop_labels)) + width/2, matched_props, width, label=f'Matched {matched_id}', alpha=0.7, color='red')
    plt.xlabel('Body Proportions')
    plt.ylabel('Ratio Value')
    plt.title('Body Proportion Comparison')
    plt.xticks(np.arange(len(prop_labels)), prop_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 5-7: 각 채널별 차이 분석
    for i, (hist1, hist2, channel) in enumerate([(h_hist1, h_hist2, 'Hue'), (s_hist1, s_hist2, 'Saturation'), (v_hist1, v_hist2, 'Value')]):
        plt.subplot(3, 4, 5 + i)
        diff = np.abs(hist1 - hist2)
        plt.bar(x, diff, color='orange', alpha=0.7)
        plt.xlabel(f'{channel} Bins')
        plt.ylabel('Absolute Difference')
        plt.title(f'{channel} Difference (Sum: {np.sum(diff):.4f})')
        plt.grid(True, alpha=0.3)
    
    # 서브플롯 8: 유사도 메트릭
    plt.subplot(3, 4, 8)
    bhatt_dist = bhattacharyya_distance(current_hist, matched_hist)
    cosine_sim = cosine_similarity(current_hist, matched_hist)
    body_sim = body_proportion_similarity(current_props, matched_props)
    
    metrics = ['Bhattacharyya\nDistance', 'Cosine\nSimilarity', 'Body\nProportion']
    values = [bhatt_dist, cosine_sim, body_sim]
    colors_metric = ['red' if bhatt_dist > 0.3 else 'green', 
                    'green' if cosine_sim > 0.7 else 'red',
                    'green' if body_sim > 0.7 else 'red']
    
    bars = plt.bar(metrics, values, color=colors_metric, alpha=0.7)
    plt.ylabel('Value')
    plt.title('Similarity Metrics')
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 서브플롯 9-12: 통계 정보
    plt.subplot(3, 4, 9)
    stats_data = [np.mean(current_hist), np.std(current_hist), np.mean(matched_hist), np.std(matched_hist)]
    stats_labels = ['Cur_Mean', 'Cur_Std', 'Match_Mean', 'Match_Std']
    plt.bar(stats_labels, stats_data, color=['blue', 'blue', 'red', 'red'], alpha=0.7)
    plt.ylabel('Value')
    plt.title('Histogram Statistics')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 10)
    diff = np.abs(current_hist - matched_hist)
    plt.hist(diff, bins=15, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Difference Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Difference Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 11)
    prop_diff = np.abs(np.array(current_props) - np.array(matched_props))
    plt.bar(prop_labels, prop_diff, color='purple', alpha=0.7)
    plt.xlabel('Body Proportions')
    plt.ylabel('Absolute Difference')
    plt.title('Body Proportion Differences')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 12)
    # 종합 점수 계산
    hist_score = max(1.0 - bhatt_dist, cosine_sim)
    total_score = 0.6 * hist_score + 0.4 * body_sim
    scores = [hist_score, body_sim, total_score]
    score_labels = ['Histogram\nScore', 'Body\nScore', 'Total\nScore']
    plt.bar(score_labels, scores, color=['blue', 'green', 'red'], alpha=0.7)
    plt.ylabel('Score')
    plt.title('Final Matching Scores')
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 개선된 히스토그램 및 신체 비율 비교 그래프 저장: {save_path}")
    print(f"   - Bhattacharyya Distance: {bhatt_dist:.3f}")
    print(f"   - Cosine Similarity: {cosine_sim:.3f}")
    print(f"   - Body Proportion Similarity: {body_sim:.3f}")
    print(f"   - Total Score: {total_score:.3f}")

def find_best_match(current_hist, current_props, current_bbox, matched_ids_this_frame):
    """개선된 매칭 알고리즘 - HSV 전체 채널 + 신체 비율 고려"""
    best_match_id = None
    best_score = 0.0
    best_metric = "none"
    
    # 현재 바운딩박스 정보
    x1, y1, x2, y2 = current_bbox
    current_area = (x2 - x1) * (y2 - y1)
    current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    for pid, pdata in tracked_people.items():
        # 이미 이 프레임에서 매칭된 ID는 제외
        if pid in matched_ids_this_frame:
            continue
            
        # 1. 히스토그램 유사도 (HSV 전체 채널)
        bhatt_dist = bhattacharyya_distance(current_hist, pdata['hist'])
        bhatt_score = 1.0 - bhatt_dist
        cosine_sim = cosine_similarity(current_hist, pdata['hist'])
        hist_score = max(bhatt_score, cosine_sim)
        
        # 2. 신체 비율 유사도
        body_sim = body_proportion_similarity(current_props, pdata['body_props'])
        
        # 3. 공간적 유사도 (가장 중요하게 변경)
        stored_bbox = pdata['box']
        stored_area = (stored_bbox[2] - stored_bbox[0]) * (stored_bbox[3] - stored_bbox[1])
        stored_center = ((stored_bbox[0] + stored_bbox[2]) // 2, (stored_bbox[1] + stored_bbox[3]) // 2)
        
        area_ratio = min(current_area, stored_area) / max(current_area, stored_area)
        center_distance = np.sqrt((current_center[0] - stored_center[0])**2 + 
                                 (current_center[1] - stored_center[1])**2)
        max_distance = np.sqrt(640**2 + 480**2)
        spatial_score = 1.0 - (center_distance / max_distance)
        
        # 공간적 위치를 가장 중요하게 고려 (80% 가중치)
        spatial_score_combined = 0.8 * spatial_score + 0.2 * area_ratio
        
        # 4. 종합 점수 계산 (공간적 위치 50%, 히스토그램 30%, 신체 비율 20%)
        if spatial_score_combined > 0.5:  # 위치가 가까우면 우선 매칭
            total_score = 0.5 * spatial_score_combined + 0.3 * hist_score + 0.2 * body_sim
        elif hist_score > 0.3 and body_sim > 0.3 and spatial_score_combined > 0.3:  # 조건 대폭 완화
            total_score = 0.4 * spatial_score_combined + 0.4 * hist_score + 0.2 * body_sim
        elif hist_score > 0.2 and body_sim > 0.2 and spatial_score_combined > 0.2:  # 조건 더 대폭 완화
            total_score = 0.3 * spatial_score_combined + 0.4 * hist_score + 0.3 * body_sim
        else:
            total_score = 0.2 * spatial_score_combined + 0.4 * hist_score + 0.4 * body_sim
        
        # 디버깅 정보 출력
        if debug_mode:
            print(f"🔍 매칭 후보 {pid}:")
            print(f"   - Histogram Score: {hist_score:.3f} (Bhatt: {bhatt_score:.3f}, Cos: {cosine_sim:.3f})")
            print(f"   - Body Proportion Score: {body_sim:.3f}")
            print(f"   - Spatial Score: {spatial_score_combined:.3f}")
            print(f"   - Total Score: {total_score:.3f}")
        
        if total_score > best_score:
            best_score = total_score
            best_match_id = pid
            best_metric = f"Hist:{hist_score:.3f}, Body:{body_sim:.3f}, Spatial:{spatial_score_combined:.3f}"
    
    return best_match_id, best_score, best_metric

# 매칭 임계값을 전역 변수로 정의
match_threshold = 0.15  # 0.35에서 0.15로 대폭 낮춤 (매우 관대한 매칭)

# 추적 안정성을 위한 설정
max_frames_without_seen = 999999  # 거의 무제한으로 보존 (10초 후에도 기억)
reentry_threshold = 0.10  # 재진입 임계값도 0.25에서 0.10으로 대폭 낮춤

def apply_nms(boxes, scores, iou_threshold=0.5):
    """중복 박스 제거를 위한 NMS 적용"""
    if len(boxes) == 0:
        return [], []
    
    # 박스를 numpy 배열로 변환
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 박스 좌표를 [x1, y1, x2, y2] 형식으로 변환
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 박스 면적 계산
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 점수 순으로 정렬
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # 가장 높은 점수의 박스 선택
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # 나머지 박스들과의 IoU 계산
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # IoU 임계값보다 낮은 박스들만 유지
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep], scores[keep]

def filter_overlapping_detections(detections, iou_threshold=0.5):
    """YOLOv8 결과에서 중복 감지를 필터링"""
    if len(detections) == 0:
        return []
    
    all_boxes = []
    all_scores = []
    
    # 모든 감지 결과 수집
    for detection in detections:
        all_boxes.append(detection['box'])
        all_scores.append(detection['confidence'])
    
    # NMS 적용
    filtered_boxes, filtered_scores = apply_nms(all_boxes, all_scores, iou_threshold)
    
    # 필터링된 결과를 원래 형식으로 변환
    filtered_detections = []
    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        # 원본 detection에서 해당하는 seg 데이터 찾기
        for detection in detections:
            if np.array_equal(detection['box'], box) and detection['confidence'] == score:
                filtered_detections.append({
                    'seg': detection['seg'],
                    'box': box,
                    'confidence': score
                })
                break
    
    return filtered_detections

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 오래된 추적 데이터 정리
        # 이 부분은 더 이상 사용되지 않으므로 제거
        
        # 사람만 감지하도록 클래스 필터링
        results = model(frame, classes=[0])  # class 0 = person만 감지
        annotated = frame.copy()
        
        # 현재 프레임에서 이미 매칭된 ID들을 추적
        matched_ids_this_frame = set()

        # YOLOv8 결과에서 중복 감지 필터링
        all_detections = []
        for result in results:
            for i in range(len(result.boxes)):
                seg = result.masks.data[i]
                box = result.boxes[i]
                confidence = box.conf[0].item()
                all_detections.append({
                    'seg': seg,
                    'box': box.xyxy[0].cpu().numpy(),
                    'confidence': confidence
                })
        
        # 중복 박스 제거
        filtered_detections = filter_overlapping_detections(all_detections)

        for detection in filtered_detections:
            seg = detection['seg']
            box = detection['box']
            confidence = detection['confidence']
                
            mask = seg.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            hist = extract_histogram(frame, mask_resized)
            body_props = extract_body_proportions(mask_resized)

            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            # 세그멘테이션 마스크를 사용한 고급 거리 계산
            est_dist, dist_info = estimate_distance_advanced(mask_resized)
            
            current_bbox = (x1, y1, x2, y2)

            # 개선된 매칭 알고리즘 사용 (이미 매칭된 ID 제외)
            matched_id, match_score, match_metric = find_best_match(hist, body_props, current_bbox, matched_ids_this_frame)
            
            # 프레임 재진입 감지 및 특별 처리
            reentry_detected = False
            if matched_id is None or match_score <= match_threshold:
                # 일반 매칭이 실패한 경우, 모든 추적 데이터와 재매칭 시도 (이미 매칭된 ID 제외)
                for pid, pdata in tracked_people.items():
                    # 이미 이 프레임에서 매칭된 ID는 제외
                    if pid in matched_ids_this_frame:
                        continue
                        
                    # 재진입 시에도 히스토그램과 신체 비율 모두 고려
                    bhatt_dist = bhattacharyya_distance(hist, pdata['hist'])
                    bhatt_score = 1.0 - bhatt_dist
                    cosine_sim = cosine_similarity(hist, pdata['hist'])
                    body_sim = body_proportion_similarity(body_props, pdata['body_props'])
                    
                    # 재진입 시 종합 점수 계산 (히스토그램 60%, 신체 비율 40%)
                    hist_score = max(bhatt_score, cosine_sim)
                    reentry_score = 0.6 * hist_score + 0.4 * body_sim
                    
                    # 재진입 시에도 높은 유사도 요구 (조건 완화)
                    if reentry_score > reentry_threshold and reentry_score > match_score:
                        matched_id = pid
                        match_score = reentry_score
                        match_metric = f"Reentry:Hist:{hist_score:.3f},Body:{body_sim:.3f}"
                        reentry_detected = True
                        frames_missing = frame_count - pdata['last_seen']
                        print(f"🔄 프레임 재진입 감지: {pid} (점수: {reentry_score:.3f}, {frames_missing}프레임 후)")
                        break
            
            if matched_id is not None and match_score > match_threshold:
                # 기존 사람 업데이트
                tracked_people[matched_id]['hist'] = hist
                tracked_people[matched_id]['body_props'] = body_props
                tracked_people[matched_id]['box'] = current_bbox
                tracked_people[matched_id]['last_seen'] = frame_count
                color = tracked_people[matched_id]['color']
                
                # 현재 프레임에서 매칭된 ID로 기록
                matched_ids_this_frame.add(matched_id)
                
                if reentry_detected:
                    print(f"🔄 프레임 재진입: {matched_id}")
                    print(f"   - 매칭 점수: {match_score:.3f}")
                    print(f"   - 매칭 메트릭: {match_metric}")
                    print(f"   - 프레임: {frame_count}")
                    print(f"   - 보이지 않았던 시간: {frame_count - tracked_people[matched_id]['last_seen']} 프레임")
                    
                    # 재진입 시 스크린샷 저장
                    # output_path = os.path.join(output_dir, f"reentry_{matched_id}_frame_{frame_count:04d}.jpg")
                    # cv2.imwrite(output_path, annotated)
                    # print(f"📸 재진입 스크린샷 저장: {output_path}")
                    
                    # 재진입 시 히스토그램 저장
                    # save_path = os.path.join(debug_dir, f"reentry_{matched_id}_frame_{frame_count}.png")
                    # plot_histogram_comparison(hist, tracked_people[matched_id]['hist'], body_props, tracked_people[matched_id]['body_props'], matched_id, frame_count, save_path)
                else:
                    print(f"🔄 기존 사람 재식별: {matched_id}")
                    print(f"   - 매칭 점수: {match_score:.3f}")
                    print(f"   - 매칭 메트릭: {match_metric}")
                    print(f"   - 프레임: {frame_count}")
                
                # 디버깅 그래프 생성 (더 자주 저장)
                # if frame_count % 30 == 0:  # 30프레임마다 (1초마다)
                #     save_path = os.path.join(debug_dir, f"hist_comparison_{matched_id}_frame_{frame_count}.png")
                #     plot_histogram_comparison(hist, tracked_people[matched_id]['hist'], body_props, tracked_people[matched_id]['body_props'], matched_id, frame_count, save_path)
                        
            else:
                # 새로운 사람
                matched_id = f"Person_{next_person_id}"
                color = get_next_color()
                tracked_people[matched_id] = {
                    'hist': hist,
                    'body_props': body_props,
                    'box': current_bbox,
                    'last_seen': frame_count,
                    'color': color
                }
                next_person_id += 1
                
                # 현재 프레임에서 매칭된 ID로 기록
                matched_ids_this_frame.add(matched_id)
                
                print(f"🆕 새로운 사람 감지: {matched_id}")
                print(f"   - 최고 매칭 점수: {match_score:.3f} (임계값: {match_threshold})")
                print(f"   - 프레임: {frame_count}")
                print(f"   - 현재 추적 중인 사람 수: {len(tracked_people)}")
                
                # 새로운 사람 감지 시 스크린샷 저장
                output_path = os.path.join(output_dir, f"new_person_{matched_id}_frame_{frame_count:04d}.jpg")
                cv2.imwrite(output_path, annotated)
                print(f"📸 새로운 사람 스크린샷 저장: {output_path}")
                
                # 새로운 사람 감지 시 히스토그램 저장 (항상 저장)
                # save_path = os.path.join(debug_dir, f"new_person_{matched_id}_frame_{frame_count}.png")
                # plot_histogram_comparison(hist, hist, body_props, body_props, matched_id, frame_count, save_path)  # 자기 자신과 비교

            # 시각화 (개선된 버전)
            # 바운딩박스 (두께 증가)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # ID 표시 (더 큰 폰트, 더 명확하게)
            id_text = matched_id
            (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # ID 배경 (더 큰 패딩)
            cv2.rectangle(annotated, (x1, y1-id_h-10), (x1+id_w+15, y1), color, -1)
            cv2.rectangle(annotated, (x1, y1-id_h-10), (x1+id_w+15, y1), (255, 255, 255), 2)
            
            # ID 텍스트
            cv2.putText(annotated, id_text, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 거리 정보 표시 (개선된 버전)
            if dist_info:  # 고급 거리 계산 정보가 있는 경우
                dist_text = f"{est_dist}m"
                density_text = f"D:{dist_info['density']:.2f}"
                
                # 거리 텍스트
                (dist_w, dist_h), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x2-dist_w-10, y1), (x2, y1+dist_h+5), (0, 0, 0), -1)
                cv2.rectangle(annotated, (x2-dist_w-10, y1), (x2, y1+dist_h+5), color, 1)
                cv2.putText(annotated, dist_text, (x2-dist_w-5, y1+dist_h), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 밀도 정보 (선택적)
                if debug_mode:
                    (density_w, density_h), _ = cv2.getTextSize(density_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated, (x2-density_w-10, y1+dist_h+10), (x2, y1+dist_h+density_h+15), (0, 0, 0), -1)
                    cv2.rectangle(annotated, (x2-density_w-10, y1+dist_h+10), (x2, y1+dist_h+density_h+15), color, 1)
                    cv2.putText(annotated, density_text, (x2-density_w-5, y1+dist_h+density_h+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # 기본 거리 표시
                dist_text = f"{est_dist}m"
                (dist_w, dist_h), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x2-dist_w-10, y1), (x2, y1+dist_h+5), (0, 0, 0), -1)
                cv2.rectangle(annotated, (x2-dist_w-10, y1), (x2, y1+dist_h+5), color, 1)
                cv2.putText(annotated, dist_text, (x2-dist_w-5, y1+dist_h), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            # 중심점 (더 큰 원)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated, (center_x, center_y), 5, color, -1)
            cv2.circle(annotated, (center_x, center_y), 5, (255, 255, 255), 1)

        # 시스템 정보 표시
        info_text = f"People: {len(tracked_people)} | Frame: {frame_count} | Threshold: {match_threshold} | Reentry: {reentry_threshold} | Memory: ∞"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 실시간 웹캠 화면 표시
        cv2.imshow("YOLOv8-Seg + HSV Histogram + Body Proportion Tracking", annotated)
        
        # q 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 대신 키보드 인터럽트로 종료
        try:
            pass
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 중단됨")
            break

    cap.release()
    cv2.destroyAllWindows()

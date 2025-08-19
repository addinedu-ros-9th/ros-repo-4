#!/usr/bin/env python3
"""
YOLO로 사람 감지하고 세그멘테이션으로 색상 분석
"""

import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime

# YOLO 모델 경로 추가
sys.path.append('/home/ckim/ros-repo-4/deeplearning/src')
from shared_models import get_shared_seg_model, SEG_MODEL_LOCK

class SimplePersonColorAnalyzer:
    def __init__(self):
        self.output_dir = "simple_person_analysis"
        self.seg_model = None
        
    def load_model(self):
        """YOLO 세그멘테이션 모델 로드"""
        if self.seg_model is None:
            print("Loading YOLO segmentation model...")
            self.seg_model = get_shared_seg_model()
            print("✅ YOLO model loaded")
        return self.seg_model
    
    def extract_color_histogram(self, img, mask, bins=16):
        """HSV 색상 히스토그램 추출"""
        # HSV로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 마스크 적용
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        # 히스토그램 계산
        h_hist = cv2.calcHist([masked_hsv], [0], mask, [bins], [0, 180])
        s_hist = cv2.calcHist([masked_hsv], [1], mask, [bins], [0, 256])
        v_hist = cv2.calcHist([masked_hsv], [2], mask, [bins], [0, 256])
        
        # 정규화
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        # 결합된 히스토그램
        combined = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
        
        return {
            'h': h_hist.flatten(),
            's': s_hist.flatten(),
            'v': v_hist.flatten(),
            'combined': combined
        }
    
    def calculate_similarity(self, hist1, hist2):
        """코사인 유사도 계산"""
        dot_product = np.dot(hist1, hist2)
        norm1 = np.linalg.norm(hist1)
        norm2 = np.linalg.norm(hist2)
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    
    def analyze_image(self, image_path):
        """이미지에서 사람 감지하고 색상 분석"""
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 이미지 로드
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Cannot load image: {image_path}")
            return None
        
        print(f"Image size: {image.shape}")
        
        # YOLO로 사람 감지 및 세그멘테이션 (높은 confidence)
        model = self.load_model()
        
        with SEG_MODEL_LOCK:
            results = model(image, conf=0.5, verbose=False)  # confidence threshold 높임
        
        if not results or len(results) == 0 or results[0].masks is None:
            print("❌ No people detected")
            return None
        
        masks = results[0].masks
        boxes = results[0].boxes
        confidences = results[0].boxes.conf
        
        print(f"✅ Initially detected {len(masks)} people")
        
        # 자동 필터링: 실제 사람만 선택
        filtered_persons = self._filter_real_people(masks, boxes, confidences, image.shape)
        
        if not filtered_persons:
            print("❌ No valid people after filtering")
            return None
        
        print(f"✅ Filtered to {len(filtered_persons)} real people")
        
        # 각 사람별 색상 분석
        person_data = {}
        
        for i, (mask, box, conf) in enumerate(filtered_persons):
            # 마스크를 이미지 크기로 리사이즈
            mask_resized = cv2.resize(mask.cpu().numpy(), (image.shape[1], image.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # 색상 히스토그램 추출
            histograms = self.extract_color_histogram(image, mask_binary)
            
            person_id = f"Person_{i+1}"
            person_data[person_id] = {
                'id': person_id,
                'name': person_id,
                'bbox': box.cpu().numpy().astype(int).tolist(),
                'confidence': float(conf),
                'histograms': histograms,
                'mask': mask_binary
            }
            
            print(f"✅ {person_id} analysis completed (conf: {conf:.3f})")
        
        if person_data:
            # 시각화 생성
            self._create_visualization(person_data, image)
            
            # 유사도 분석
            self._analyze_similarities(person_data)
            
            print(f"✅ Analysis completed! Results saved to '{self.output_dir}' folder.")
            return person_data
        else:
            print("❌ No person data to analyze.")
            return None
    
    def _filter_real_people(self, masks, boxes, confidences, image_shape):
        """실제 사람만 필터링"""
        height, width = image_shape[:2]
        filtered = []
        
        for i, (mask, box, conf) in enumerate(zip(masks.data, boxes.xyxy, confidences)):
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # 1. Confidence threshold (이미 0.5로 설정됨)
            if conf < 0.5:
                continue
            
            # 2. 바운딩박스 크기 필터링
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            # 너무 작거나 큰 바운딩박스 제거
            min_area = (width * height) * 0.01  # 이미지의 1% 이상
            max_area = (width * height) * 0.8   # 이미지의 80% 이하
            
            if bbox_area < min_area or bbox_area > max_area:
                continue
            
            # 3. 종횡비 필터링 (사람은 세로로 긴 형태)
            aspect_ratio = bbox_height / bbox_width
            if aspect_ratio < 1.2:  # 높이가 너비보다 1.2배 이상
                continue
            
            # 4. 이미지 경계 내에 있는지 확인
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue
            
            # 5. 중복 제거 (IoU 기반)
            is_duplicate = False
            for existing_mask, existing_box, _ in filtered:
                existing_x1, existing_y1, existing_x2, existing_y2 = existing_box.cpu().numpy()
                
                # IoU 계산
                intersection_x1 = max(x1, existing_x1)
                intersection_y1 = max(y1, existing_y1)
                intersection_x2 = min(x2, existing_x2)
                intersection_y2 = min(y2, existing_y2)
                
                if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    union_area = bbox_area + (existing_x2 - existing_x1) * (existing_y2 - existing_y1) - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > 0.3:  # IoU가 0.3 이상이면 중복으로 간주
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append((mask, box, conf))
                print(f"  - Kept detection {i+1}: bbox={[int(x1), int(y1), int(x2), int(y2)]}, "
                      f"area={bbox_area:.0f}, aspect_ratio={aspect_ratio:.2f}, conf={conf:.3f}")
        
        # 6. 최대 3명까지만 선택 (confidence 순으로 정렬)
        filtered.sort(key=lambda x: x[2], reverse=True)
        filtered = filtered[:3]
        
        return filtered
    
    def _create_visualization(self, person_data, original_image):
        """발표용 시각화 생성"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. 원본 이미지에 바운딩박스 표시
        vis_image = original_image.copy()
        # 실제 옷 색상에 맞게 조정: 초록색, 핑크색, 파란색
        colors = [(0, 255, 0), (255, 192, 203), (0, 0, 255)]  # 초록, 핑크, 파란색
        
        for i, (person_id, person) in enumerate(person_data.items()):
            x1, y1, x2, y2 = person['bbox']
            color = colors[i % len(colors)]
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, person['name'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(f'{self.output_dir}/annotated_image.jpg', vis_image)
        
        # 2. 발표용 핵심 그래프: 색상 분포 + 유사도 매트릭스
        fig = plt.figure(figsize=(16, 8))
        
        # 왼쪽: 색상 분포 비교
        ax1 = plt.subplot(1, 2, 1)
        
        for i, (person_id, person) in enumerate(person_data.items()):
            hist = person['histograms']['h']  # Hue 채널
            color = colors[i % len(colors)]
            ax1.plot(hist, label=person['name'], color=tuple(c/255 for c in color), 
                    linewidth=3, marker='o', markersize=4)
        
        ax1.set_title('Color Distribution Differences (Hue Channel)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Color Bins', fontsize=12)
        ax1.set_ylabel('Normalized Frequency', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 오른쪽: 유사도 매트릭스
        ax2 = plt.subplot(1, 2, 2)
        
        person_ids = list(person_data.keys())
        similarity_matrix = np.zeros((len(person_ids), len(person_ids)))
        
        for i, id1 in enumerate(person_ids):
            for j, id2 in enumerate(person_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    hist1 = person_data[id1]['histograms']['combined']
                    hist2 = person_data[id2]['histograms']['combined']
                    similarity_matrix[i, j] = self.calculate_similarity(hist1, hist2)
        
        # 히트맵 생성
        im = sns.heatmap(similarity_matrix, 
                        xticklabels=[person_data[pid]['name'] for pid in person_ids],
                        yticklabels=[person_data[pid]['name'] for pid in person_ids],
                        annot=True, cmap='RdYlBu_r', vmin=0, vmax=1, fmt='.3f',
                        cbar_kws={'label': 'Similarity Score'}, ax=ax2)
        
        ax2.set_title('Person Re-identification (Color Similarity Matrix)', fontsize=16, fontweight='bold')
        
        # 전체 제목 (한 줄로 줄임)
        fig.suptitle('Person Re-identification using Color Distribution Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 설명 텍스트 추가
        fig.text(0.5, 0.02, 
                'Left: Different color distributions enable person distinction\nRight: Low similarity scores confirm different persons', 
                ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        plt.savefig(f'{self.output_dir}/presentation_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Presentation chart saved: {self.output_dir}/presentation_chart.png")
    
    def _analyze_similarities(self, person_data):
        """유사도 분석"""
        person_ids = list(person_data.keys())
        
        print("\n=== Color Similarity Analysis ===")
        
        for i, id1 in enumerate(person_ids):
            for j, id2 in enumerate(person_ids):
                if i < j:  # 중복 제거
                    hist1 = person_data[id1]['histograms']['combined']
                    hist2 = person_data[id2]['histograms']['combined']
                    
                    similarity = self.calculate_similarity(hist1, hist2)
                    
                    print(f"\n{person_data[id1]['name']} vs {person_data[id2]['name']}:")
                    print(f"  - Similarity: {similarity:.4f}")

def main():
    """메인 실행 함수"""
    print("🖼️ Simple Person Color Analysis")
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            return
        
        analyzer = SimplePersonColorAnalyzer()
        results = analyzer.analyze_image(image_path)
        
        if results:
            print(f"✅ Analysis completed! Analyzed {len(results)} persons.")
        else:
            print("❌ Analysis failed.")
    else:
        print("Usage: python3 simple_person_color_analysis.py <image_path>")
        print("Example: python3 simple_person_color_analysis.py /path/to/image.jpg")

if __name__ == "__main__":
    main() 
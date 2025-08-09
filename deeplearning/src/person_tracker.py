import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
import threading
import queue
import time
import gc
import os

from shared_models import get_shared_seg_model, SEG_MODEL_LOCK

# 캐시 비우기 옵션
CLEAR_CACHE_AFTER_INFERENCE = os.environ.get("CLEAR_CACHE_AFTER_INFERENCE", "0").lower() == "1"

def free_cuda_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        except Exception:
            pass

# GPU 사용 가능 여부 확인 및 메모리 제한 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# GPU 메모리 제한 설정 (PersonTracker용 40%)
if torch.cuda.is_available():
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        person_tracker_memory = int(total_memory * 0.4)
        torch.cuda.set_per_process_memory_fraction(0.4, 0)  # 40% 제한
        print(f"🎮 PersonTracker GPU 메모리 제한: {person_tracker_memory / 1024**3:.1f}GB")
    except Exception as e:
        print(f"⚠️ GPU 메모리 제한 설정 실패: {e}")


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


class PersonTracker:
    def __init__(self):
        # YOLO 모델 초기화 → 공유 모델 사용
        self.model = get_shared_seg_model()
        self.model_lock = SEG_MODEL_LOCK
        
        # 사람 재식별 데이터
        self.people_data = {}
        self.next_id = 0
        
        # 성능 최적화 설정
        self.process_every_n_frames = 1  # 3 → 1로 변경 (매 프레임 처리)
        self.frame_skip_counter = 0
        
        # 매칭 관련 설정 (베이지색/검정색 구분 강화)
        self.match_threshold = 0.60  # 0.65 → 0.60로 완화
        self.reentry_threshold = 0.55  # 0.60 → 0.55로 완화
        self.min_detection_confidence = 0.45  # 0.6 → 0.45로 완화
        self.min_person_area = 3000  # 5000 → 3000으로 완화
        self.max_frames_without_seen = 300
        
        # 히스토그램 기억 설정 (더 많은 히스토그램 저장)
        self.max_histograms_per_person = 25  # 20 → 25로 증가 (더 많은 샘플)
        self.histogram_memory_duration = 30
        
        # 스레드 설정 (gesture_recognizer와 일관성)
        self.frame_queue = queue.Queue(maxsize=1)  # 3 → 1로 줄여서 지연 최소화
        self.running = True
        self.lock = threading.Lock()
        
        # 결과 저장
        self.latest_detections = []
        
        # 메모리 관리
        self.last_memory_cleanup = time.time()
        self.memory_cleanup_interval = 30.0  # 10 → 30초로 늘림 (정리 빈도 감소)
        
        # 워커 스레드 자동 시작
        self.worker_thread = threading.Thread(target=self.tracking_worker)
        self.worker_thread.start()
        
        print("✅ Person Tracker 초기화 완료")
    
    def cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # 메모리 상태 출력 (디버깅용)
                if hasattr(self, '_cleanup_count'):
                    self._cleanup_count += 1
                else:
                    self._cleanup_count = 0
                
                # 5번마다 한 번만 로그 출력 (빈도 감소)
                if self._cleanup_count % 5 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"🧹 PersonTracker GPU 메모리 정리: {allocated:.1f}MB 할당, {reserved:.1f}MB 예약")
            except Exception as e:
                print(f"GPU 메모리 정리 중 오류: {e}")
    
    def tracking_worker(self):
        """사람 추적 워커"""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                
                frame, frame_id, elapsed_time = frame_data
                
                # 주기적 메모리 정리
                current_time = time.time()
                if current_time - self.last_memory_cleanup > self.memory_cleanup_interval:
                    self.cleanup_gpu_memory()
                    self.last_memory_cleanup = current_time
                
                # YOLO 감지 (공유 모델 락으로 직렬화)
                with self.model_lock:
                    infer_device = 0 if torch.cuda.is_available() else 'cpu'
                    results = self.model(frame, imgsz=640, device=infer_device, verbose=False)
                if CLEAR_CACHE_AFTER_INFERENCE:
                    free_cuda_cache()
                
                current_detections = []
                current_detection_ids = set()  # 현재 프레임에서 감지된 ID들
                
                if results[0].masks is not None:
                    for i, (mask, box) in enumerate(zip(results[0].masks.data, results[0].boxes.data)):
                        if results[0].names[int(box[5])] == 'person':
                            confidence = box[4].item()
                            if confidence >= self.min_detection_confidence:
                                x1, y1, x2, y2 = map(int, box[:4])
                                area = (x2 - x1) * (y2 - y1)
                                
                                if area >= self.min_person_area:
                                    # 히스토그램 추출
                                    mask_np = mask.cpu().numpy().astype(np.uint8)
                                    combined_hist, _, _, _ = self.extract_histogram(frame, mask_np)
                                    
                                    # 매칭 시도
                                    bbox = [x1, y1, x2, y2]
                                    best_match_id, best_score, metrics = self.find_best_match(
                                        combined_hist, bbox, [], elapsed_time
                                    )
                                    
                                    # 공간적 제약 추가 (바운딩 박스가 너무 멀리 떨어져 있으면 새로운 사람)
                                    spatial_constraint_passed = True
                                    if best_match_id is not None and best_match_id in self.people_data:
                                        latest_bbox = self.people_data[best_match_id]['bboxes'][-1]
                                        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                        stored_center = ((latest_bbox[0] + latest_bbox[2]) // 2, (latest_bbox[1] + latest_bbox[3]) // 2)
                                        center_distance = np.sqrt((current_center[0] - stored_center[0])**2 + 
                                                                 (current_center[1] - stored_center[1])**2)
                                        
                                        # 100픽셀 이상 떨어져 있으면 새로운 사람으로 처리
                                        if center_distance > 100:
                                            spatial_constraint_passed = False
                                            if frame_id % 30 == 0:
                                                print(f"📍 공간적 제약 실패: {best_match_id} (거리: {center_distance:.1f}픽셀)")
                                    
                                    # 매칭 조건을 더 엄격하게 설정
                                    if (best_match_id is not None and 
                                        best_score > self.match_threshold and
                                        metrics.get('hist_score', 0) > 0.4 and
                                        spatial_constraint_passed):  # 공간적 제약도 체크
                                        
                                        # 기존 사람 매칭
                                        person_id = best_match_id
                                        
                                        # 매칭 디버깅 (30프레임마다)
                                        if frame_id % 180 == 0:  # 30 → 180으로 늘림
                                            print(f"🔄 기존 사람 매칭: {person_id} (점수: {best_score:.3f})")
                                            print(f"   - 히스토그램 점수: {metrics.get('hist_score', 0):.3f}")
                                            print(f"   - 매칭 임계값: {self.match_threshold}")
                                    else:
                                        # 새로운 사람 (매칭 실패 또는 임계값 미달)
                                        person_id = f"Person_{self.next_id}"
                                        self.next_id += 1
                                        self.people_data[person_id] = {
                                            'histograms': [],
                                            'bboxes': [],
                                            'timestamps': []
                                        }
                                        
                                        # 매칭 실패 디버깅 (30프레임마다)
                                        if frame_id % 180 == 0:  # 30 → 180으로 늘림
                                            if best_match_id is not None:
                                                print(f"❌ 매칭 실패: {best_match_id} (점수: {best_score:.3f} < 임계값: {self.match_threshold})")
                                                print(f"   - 히스토그램 점수: {metrics.get('hist_score', 0):.3f}")
                                                if not spatial_constraint_passed:
                                                    print(f"   - 공간적 제약 실패")
                                            else:
                                                print(f"🆕 새로운 사람 등록: {person_id} (매칭 가능한 사람 없음)")
                                    
                                    # 데이터 업데이트
                                    self.people_data[person_id]['histograms'].append(combined_hist)
                                    self.people_data[person_id]['bboxes'].append(bbox)
                                    self.people_data[person_id]['timestamps'].append(elapsed_time)
                                    
                                    # 히스토그램 개수 제한
                                    if len(self.people_data[person_id]['histograms']) > self.max_histograms_per_person:
                                        self.people_data[person_id]['histograms'].pop(0)
                                        self.people_data[person_id]['bboxes'].pop(0)
                                        self.people_data[person_id]['timestamps'].pop(0)
                                    
                                    current_detections.append({
                                        'id': person_id,
                                        'bbox': bbox,
                                        'confidence': confidence,
                                        'score': best_score if best_match_id else 0.0
                                    })
                                    
                                    current_detection_ids.add(person_id)
                
                # 사라진 사람 정리 (현재 프레임에서 감지되지 않은 사람들)
                people_to_remove = []
                for person_id in list(self.people_data.keys()):
                    if person_id not in current_detection_ids:
                        # 마지막 감지 시간 확인
                        if person_id in self.people_data and self.people_data[person_id]['timestamps']:
                            last_seen = self.people_data[person_id]['timestamps'][-1]
                            time_since_last_seen = elapsed_time - last_seen
                            
                            # 일정 시간 이상 사라진 사람 제거
                            if time_since_last_seen > 10.0:  # 10초 이상 사라지면 제거
                                people_to_remove.append(person_id)
                                if frame_id % 30 == 0:
                                    print(f"🗑️ 사라진 사람 제거: {person_id} (마지막 감지: {time_since_last_seen:.1f}초 전)")
                
                # 사라진 사람 데이터 정리
                for person_id in people_to_remove:
                    del self.people_data[person_id]
                
                # 결과 저장
                with self.lock:
                    self.latest_detections = current_detections
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"PersonTracker 워커 오류: {e}")
                continue
    
    def get_latest_detections(self):
        """최신 감지 결과 반환"""
        with self.lock:
            return self.latest_detections.copy()
    
    def add_frame(self, frame, frame_id, elapsed_time):
        """프레임 추가 (비동기)"""
        try:
            self.frame_queue.put_nowait((frame, frame_id, elapsed_time))
        except queue.Full:
            pass
    
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
            if best_hist_score < 0.40:  # 0.35 → 0.40로 증가 (베이지색/검정색 구분 강화)
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
    
    def start(self):
        """추적기 시작"""
        print("🚀 Person Tracker 시작")
    
    def stop(self):
        """추적기 중지"""
        self.running = False
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join()
        print("🛑 Person Tracker 중지")
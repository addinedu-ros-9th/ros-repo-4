#!/usr/bin/env python3
"""
강의실 환경용 객체인식 시스템 (사람만 인식)
- 사람 인식만 수행
- 트래킹을 위한 기본 감지 기능
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        사람 인식기 초기화
        
        Args:
            model_path: YOLO 모델 경로 (기본: yolov8n.pt)
        """
        print("🚀 사람 인식 시스템 초기화 중...")
        
        # YOLOv8 모델 로드
        self.model = YOLO(model_path)
        print(f"✅ YOLOv8 모델 로드 완료: {model_path}")
        
        # 사람 클래스만 (COCO 데이터셋 기준)
        self.person_class_id = 0  # COCO에서 person은 0번
        
        print(f"🎯 타겟 클래스: person")
        
        # 성능 통계
        self.frame_count = 0
        self.start_time = time.time()
    
    def detect_people(self, image):
        """
        이미지에서 사람들을 인식
        
        Args:
            image: OpenCV 이미지 (BGR)
            
        Returns:
            list: 감지된 사람들의 정보 [{confidence, bbox, center}, ...]
        """
        # YOLO 추론
        results = self.model(image, verbose=False)
        
        detected_people = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 클래스 ID와 신뢰도
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 사람만 감지 (class_id == 0)
                    if class_id == self.person_class_id and confidence > 0.5:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        person_info = {
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [int((x1+x2)/2), int((y1+y2)/2)],
                            'area': (int(x2)-int(x1)) * (int(y2)-int(y1))
                        }
                        detected_people.append(person_info)
        
        return detected_people
    
    def draw_detections(self, image, detections):
        """
        감지된 사람들을 이미지에 그리기
        
        Args:
            image: 원본 이미지
            detections: detect_people()에서 반환된 사람 리스트
            
        Returns:
            image: 바운딩박스가 그려진 이미지
        """
        result_image = image.copy()
        
        # 사람 색상 (초록색)
        person_color = (0, 255, 0)
        
        for person in detections:
            confidence = person['confidence']
            x1, y1, x2, y2 = person['bbox']
            area = person['area']
            
            # 바운딩 박스 그리기
            cv2.rectangle(result_image, (x1, y1), (x2, y2), person_color, 2)
            
            # 라벨 텍스트
            label = f"Person: {confidence:.2f}"
            
            # 텍스트 배경
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_image, (x1, y1-label_h-10), (x1+label_w, y1), person_color, -1)
            
            # 텍스트
            cv2.putText(result_image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 면적 정보
            area_text = f"Area: {area}"
            cv2.putText(result_image, area_text, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 1)
        
        # 통계 정보
        stats_text = f"People: {len(detections)}"
        cv2.putText(result_image, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS 정보 표시
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(result_image, fps_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image

def main():
    """
    웹캠을 이용한 실시간 사람 인식 테스트
    """
    print("🎥 웹캠 실시간 사람 인식 시작...")
    
    # 사람 인식기 초기화
    detector = PersonDetector()
    
    # 웹캠 연결 (여러 인덱스 시도)
    cap = None
    for cam_id in [0, 1]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            print(f"📹 웹캠 {cam_id} 연결 성공!")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다!")
        return
    
    # 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("📖 사용법:")
    print("  - q: 종료")
    print("  - s: 스크린샷 저장")
    print("  - d: 사람 감지 결과 상세 출력 토글")
    
    # 상태 변수
    detailed_output = False
    screenshot_count = 0
    
    # 윈도우 생성 및 설정
    window_name = "사람 인식 시스템"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다!")
                break
            
            # 사람 인식
            detections = detector.detect_people(frame)
            
            # 결과 시각화
            result_frame = detector.draw_detections(frame, detections)
            
            # 감지된 사람 정보 출력 (옵션)
            if detections and detailed_output:
                print(f"👥 감지된 사람: {len(detections)}명")
                for i, person in enumerate(detections):
                    print(f"  - Person {i+1}: {person['confidence']:.2f} at {person['center']}, Area: {person['area']}")
            
            # 화면에 표시
            cv2.imshow(window_name, result_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🛑 사용자가 종료를 요청했습니다")
                break
            elif key == ord('s'):
                # 스크린샷 저장
                filename = f"person_detection_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"📸 스크린샷 저장: {filename}")
                screenshot_count += 1
            elif key == ord('d'):
                # 상세 출력 토글
                detailed_output = not detailed_output
                status = "켜짐" if detailed_output else "꺼짐"
                print(f"🔍 상세 출력: {status}")
                
    except KeyboardInterrupt:
        print("🛑 사용자가 중단했습니다")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 최종 통계
        elapsed_time = time.time() - detector.start_time
        avg_fps = detector.frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"✅ 프로그램 종료")
        print(f"📊 통계: {detector.frame_count}프레임, 평균 FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    main() 
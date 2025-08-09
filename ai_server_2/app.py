#!/usr/bin/env python3
import os
import json
import time
import threading
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import cv2
from flask import Flask, request, jsonify

# YOLO (COCO) - 자동 다운로드 사용
from ultralytics import YOLO

# 중앙 송신 유틸 (GUI 알림용)
from sender import send_gesture_come, send_user_disappear, send_user_appear

app = Flask(__name__)

# 환경 설정
AI_HTTP_HOST = os.environ.get("AI_HTTP_HOST", "0.0.0.0")
AI_HTTP_PORT = int(os.environ.get("AI_HTTP_PORT", "5007"))  # 다른 포트 사용
ROBOT_ID = int(os.environ.get("ROBOT_ID", "3"))

# YOLO 모델
yolo_det = YOLO('yolov8n.pt')   # COCO 탐지용

def ok():
    return jsonify({"status_code": 200})


@app.errorhandler(404)
def not_found(_):
    return jsonify({"status_code": 404, "error": "not_found"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"status_code": 405, "error": "method_not_allowed"}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"status_code": 500, "error": "internal_error"}), 500


def estimate_obstacle_bbox_by_angles(left_angle: float, right_angle: float, frame_shape, camera_fov=60.0):
    """
    left_angle, right_angle을 기반으로 이미지에서 장애물의 예상 바운딩박스 영역을 추정 (라이다 360도 고려)
    """
    # 각도를 0-360도 범위로 정규화
    def normalize_angle(angle):
        return angle % 360.0
    
    left_norm = normalize_angle(left_angle)
    right_norm = normalize_angle(right_angle)
    
    # 각도 범위 계산
    if left_norm <= right_norm:
        angle_range = right_norm - left_norm
    else:
        # 360도를 넘어가는 경우 (예: 350도 ~ 10도)
        angle_range = (360.0 - left_norm) + right_norm
    
    # 중심 각도 계산
    center_angle = (left_norm + right_norm) / 2.0
    if left_norm > right_norm:
        center_angle = (left_norm + right_norm + 360.0) / 2.0
        center_angle = normalize_angle(center_angle)
    
    # 각도 범위를 기반으로 크기 추정
    if angle_range < 10.0:  # 좁은 각도
        bbox_size = 0.2  # 화면의 20%
    elif angle_range < 30.0:  # 중간 각도
        bbox_size = 0.3  # 화면의 30%
    else:  # 넓은 각도
        bbox_size = 0.4  # 화면의 40%
    
    # 라이다 각도를 카메라 이미지 좌표로 변환
    frame_width = frame_shape[1]
    frame_height = frame_shape[0]
    
    # 카메라 선택 및 좌표 변환
    if 0.0 <= center_angle <= 180.0:
        # 전면 카메라
        camera = 'front'
        # 라이다 0~180도를 카메라 이미지 x좌표로 변환
        # 0도(정면) → 이미지 중앙, 90도(우측) → 이미지 우측, 180도(후면) → 이미지 좌측
        relative_angle = center_angle - 90.0  # -90 ~ +90도
    else:
        # 후면 카메라
        camera = 'back'
        # 라이다 180~360도를 카메라 이미지 x좌표로 변환
        relative_angle = center_angle - 270.0  # -90 ~ +90도
    
    # 각도를 픽셀 좌표로 변환
    pixels_per_degree = frame_width / camera_fov
    x_offset = relative_angle * pixels_per_degree
    estimated_center_x = frame_width / 2.0 + x_offset
    
    # 바운딩박스 크기 계산
    bbox_width = int(frame_width * bbox_size)
    bbox_height = int(frame_height * bbox_size)
    
    # 바운딩박스 좌표 계산
    x1 = max(0, int(estimated_center_x - bbox_width // 2))
    y1 = max(0, int(frame_height // 2 - bbox_height // 2))
    x2 = min(frame_width, x1 + bbox_width)
    y2 = min(frame_height, y1 + bbox_height)
    
    return [x1, y1, x2, y2]


def debug_angle_mapping(left_angle: float, right_angle: float, frame_shape):
    """
    라이다 각도 → 카메라 이미지 좌표 변환 디버깅 함수
    """
    def normalize_angle(angle):
        return angle % 360.0
    
    left_norm = normalize_angle(left_angle)
    right_norm = normalize_angle(right_angle)
    
    # 중심 각도 계산
    center_angle = (left_norm + right_norm) / 2.0
    if left_norm > right_norm:
        center_angle = (left_norm + right_norm + 360.0) / 2.0
        center_angle = normalize_angle(center_angle)
    
    frame_width = frame_shape[1]
    frame_height = frame_shape[0]
    
    # 카메라 선택
    if 0.0 <= center_angle <= 180.0:
        camera = 'front'
        relative_angle = center_angle - 90.0
    else:
        camera = 'back'
        relative_angle = center_angle - 270.0
    
    # 픽셀 좌표 계산
    pixels_per_degree = frame_width / 60.0  # 60도 FOV 가정
    x_offset = relative_angle * pixels_per_degree
    pixel_x = frame_width / 2.0 + x_offset
    
    return {
        'camera': camera,
        'center_angle': center_angle,
        'relative_angle': relative_angle,
        'pixel_x': pixel_x,
        'frame_width': frame_width,
        'frame_height': frame_height
    }


def detect_and_visualize_obstacle(frame: np.ndarray, left_angle: float, right_angle: float) -> tuple:
    """
    left_angle, right_angle을 기반으로 특정 장애물만 탐지하고 시각화
    """
    # 예상 장애물 영역 추정
    estimated_bbox = estimate_obstacle_bbox_by_angles(left_angle, right_angle, frame.shape)
    
    # 전체 이미지에서 YOLO 실행
    results = yolo_det(frame, conf=0.25, verbose=False)
    
    # 가장 적합한 장애물 찾기 (예상 영역과 겹치는 정도로 판단)
    best_match = None
    best_iou = 0.0
    
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        names = results[0].names
        
        for c, s, b in zip(clss, confs, xyxy):
            # IoU 계산
            iou = calculate_iou(estimated_bbox, b.tolist())
            
            # IoU가 높고 confidence가 높은 객체 선택
            if iou > best_iou and s > 0.3:
                best_iou = iou
                best_match = {
                    'label': names.get(c, str(c)),
                    'confidence': float(s),
                    'bbox': b.tolist(),
                    'iou': iou
                }
    
    # 시각화된 이미지 생성
    visualized_frame = frame.copy()
    
    # 예상 영역 표시 (파란색 점선)
    cv2.rectangle(visualized_frame, 
                  (estimated_bbox[0], estimated_bbox[1]), 
                  (estimated_bbox[2], estimated_bbox[3]), 
                  (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(visualized_frame, "Expected Area", 
                (estimated_bbox[0], estimated_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 발견된 장애물 표시 (빨간색 실선)
    if best_match:
        bbox = best_match['bbox']
        cv2.rectangle(visualized_frame, 
                      (bbox[0], bbox[1]), 
                      (bbox[2], bbox[3]), 
                      (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(visualized_frame, 
                    f"{best_match['label']} ({best_match['confidence']:.2f})", 
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 각도 정보 표시
        cv2.putText(visualized_frame, 
                    f"Angles: left={left_angle:.1f}, right={right_angle:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(visualized_frame, 
                    f"IoU: {best_match['iou']:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return visualized_frame, best_match


def calculate_iou(box1, box2):
    """두 바운딩박스의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def save_jpg(frame: np.ndarray, prefix: str) -> str:
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_dir = '/home/ckim/ros-repo-4/ai_server_2/image'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{prefix}_{ts}.jpg')
    cv2.imwrite(path, frame)
    return path


@app.route("/obstacle/detected", methods=["POST"])
def obstacle_detected():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    # AI Server 1로부터 받는 데이터
    robot_id = data.get("robot_id")
    left_angle = data.get("left_angle")
    right_angle = data.get("right_angle")
    image_data = data.get("image_data")  # base64 인코딩된 이미지
    ts = data.get("timestamp")

    if robot_id is None or left_angle is None or right_angle is None:
        return jsonify({"status_code": 400, "error": "invalid_fields"}), 400

    # 각도를 float로 변환
    try:
        left_angle_float = float(left_angle)
        right_angle_float = float(right_angle)
    except ValueError:
        return jsonify({"status_code": 400, "error": "invalid_angle_format"}), 400

    # base64 이미지 디코딩
    import base64
    try:
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status_code": 400, "error": "invalid_image"}), 400
            
    except Exception as e:
        return jsonify({"status_code": 400, "error": f"image_decode_error: {str(e)}"}), 400

    # 각도 기반 장애물 탐지 및 시각화
    visualized_frame, detected_obstacle = detect_and_visualize_obstacle(frame, left_angle_float, right_angle_float)
    
    # 시각화된 이미지 저장
    img_path = save_jpg(visualized_frame, f"obstacle_detected")
    
    app.logger.info(f"[IF-01] YOLO COCO 처리 완료, saved={img_path}, detected_obstacle={detected_obstacle}")

    # TODO: GUI 알림 전송 (GUI_BASE/notify 등)
    # 여기서 GUI에 장애물 감지 알림을 보낼 수 있습니다
    
    return jsonify({
        "status_code": 200,
        "detected_obstacle": detected_obstacle,
        "image_path": img_path
    })


@app.route("/infer/coco", methods=["POST"])
def infer_coco():
    """Base64 JPEG 이미지를 받아 COCO 추론 결과(JSON) 반환"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"status_code": 400, "error": "image_required"}), 400

    import base64
    try:
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status_code": 400, "error": "invalid_image"}), 400
    except Exception as e:
        return jsonify({"status_code": 400, "error": f"image_decode_error: {str(e)}"}), 400

    # COCO 추론
    results = yolo_det(frame, conf=0.25, verbose=False)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        names = results[0].names
        
        for c, s, b in zip(clss, confs, xyxy):
            detections.append({
                'label': names.get(c, str(c)),
                'confidence': float(s),
                'bbox': b.tolist(),
            })

    return jsonify({"status_code": 200, "detections": detections})


@app.route("/debug/angle_mapping", methods=["POST"])
def debug_angle_mapping_endpoint():
    """라이다 각도 → 카메라 좌표 변환 디버깅"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    left_angle = data.get("left_angle")
    right_angle = data.get("right_angle")
    image_data = data.get("image_data")  # base64 인코딩된 이미지

    if left_angle is None or right_angle is None:
        return jsonify({"status_code": 400, "error": "missing_angles"}), 400

    try:
        left_angle_float = float(left_angle)
        right_angle_float = float(right_angle)
    except ValueError:
        return jsonify({"status_code": 400, "error": "invalid_angle_format"}), 400

    # 이미지 디코딩 (선택사항)
    frame_shape = None
    if image_data:
        try:
            import base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame_shape = frame.shape
        except Exception as e:
            app.logger.warning(f"이미지 디코딩 실패: {e}")

    # 기본 이미지 크기 (640x480) 사용
    if frame_shape is None:
        frame_shape = (480, 640, 3)  # height, width, channels

    # 각도 매핑 디버깅
    mapping_info = debug_angle_mapping(left_angle_float, right_angle_float, frame_shape)

    return jsonify({
        "status_code": 200,
        "input_angles": {
            "left_angle": left_angle_float,
            "right_angle": right_angle_float
        },
        "mapping_result": mapping_info,
        "explanation": {
            "camera_selection": f"라이다 중심각 {mapping_info['center_angle']:.1f}도 → {mapping_info['camera']} 카메라",
            "coordinate_transform": f"상대각도 {mapping_info['relative_angle']:.1f}도 → 픽셀 x좌표 {mapping_info['pixel_x']:.1f}",
            "frame_info": f"이미지 크기: {mapping_info['frame_width']}x{mapping_info['frame_height']}"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "server_type": "ai_server_2 (YOLO COCO obstacle detection only)",
    })


if __name__ == "__main__":
    app.run(host=AI_HTTP_HOST, port=AI_HTTP_PORT, debug=False) 
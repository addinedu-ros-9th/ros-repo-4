#!/usr/bin/env python3
import os
import json
import time
import threading
import base64
import requests
from datetime import datetime
from typing import Optional
import socket

import numpy as np
import cv2
from flask import Flask, request, jsonify

# 외부 모듈 경로 추가 (공유메모리 리더)
import sys
sys.path.append('/home/ckim/ros-repo-4/deeplearning/src')
from shared_memory_reader import DualCameraSharedMemoryReader

# 중앙 송신 유틸
from sender import send_gesture_come, send_user_disappear, send_user_appear

# 중앙 WebSocket 클라이언트
try:
    import websocket  # pip install websocket-client
except Exception:
    websocket = None

app = Flask(__name__)

# 환경 설정
AI_HTTP_HOST = os.environ.get("AI_HTTP_HOST", "0.0.0.0")
AI_HTTP_PORT = int(os.environ.get("AI_HTTP_PORT", "5006"))
ROBOT_ID = int(os.environ.get("ROBOT_ID", "3"))
# 추가: 로컬 타임아웃 워커 사용 여부(기본 비활성)
USE_TIMEOUT_WORKER = os.environ.get("USE_TIMEOUT_WORKER", "0") == "1"
# 중앙 WebSocket URL
CENTRAL_WS_URL = os.environ.get("CENTRAL_WS_URL", "ws://192.168.0.36:3000")

# AI Server 2 설정
AI_SERVER_2_URL = os.environ.get("AI_SERVER_2_URL", "http://192.168.0.74:5007")
# 중앙 HTTP BASE (stop_tracking 전송용)
CENTRAL_HTTP_BASE = os.environ.get("CENTRAL_HTTP_BASE", "http://192.168.0.36:8080")

# 상태 저장
STATE = {
    "tracking": False,
    "last_obstacle": None,
    "worker_running": False,
    "last_tracking_update": {"visible": False, "ts": 0.0}, # 추적 정보 수신용
    "disappear_sent": False,
    "come_sent_ids": {},
    "come_gesture_active": False,  # 기존 게이트(남겨두되 사용 안함)
    "last_come_person_id": None,
    "last_come_time": 0.0,
    "last_coco_infer": None,
    # 타겟 기반 추적 상태(가장 큰 사람)
    "target_person_id": None,
    "target_visible": False,
    # 중앙 WS 기반 게이트: alert_idle → True, alert_occupied → False
    "can_send_come": True,
    # 트래킹 사라짐 타이머
    "disappear_deadline_ts": 0.0,
    "central_stop_sent": False,
}

# 리소스
reader = DualCameraSharedMemoryReader()

worker_thread: Optional[threading.Thread] = None
worker_lock = threading.Lock()
ws_thread: Optional[threading.Thread] = None


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


def choose_camera_by_angles(left_angle: float, right_angle: float) -> str:
    """left_angle, right_angle을 기반으로 카메라 선택 (라이다 360도 고려)"""
    # 각도를 0-360도 범위로 정규화
    def normalize_angle(angle: float) -> float:
        return angle % 360.0
    left_norm = normalize_angle(left_angle)
    right_norm = normalize_angle(right_angle)
    center = (left_norm + right_norm) / 2.0
    if left_norm > right_norm:
        center = (left_norm + right_norm + 360.0) / 2.0
        center = normalize_angle(center)
    # 0~180 전면, 180~360 후면
    return 'front' if 0.0 <= center <= 180.0 else 'back'


def get_latest_frame(camera: str) -> Optional[np.ndarray]:
    frame = reader.read_frame(camera)
    return frame


def request_with_retry(url: str, payload: dict, timeout: float = 5.0, retries: int = 2, backoff: float = 0.5):
    """단순 재시도 유틸 (지수 백오프)"""
    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            return resp
        except Exception as e:
            app.logger.warning(f"request failed (attempt {attempt+1}/{retries+1}): {e}")
            if attempt == retries:
                return None
            time.sleep(backoff * (2 ** attempt))
            attempt += 1


def send_image_to_ai_server2(frame: np.ndarray, left_angle: float, right_angle: float, robot_id: int, ts: int) -> bool:
    """AI Server 2로 이미지와 각도 정보 전송 (재시도 포함)"""
    try:
        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # AI Server 2로 전송할 데이터 (기존 형식 유지)
        payload = {
            "robot_id": robot_id,
            "left_angle": left_angle,
            "right_angle": right_angle,
            "image_data": image_base64,
            "timestamp": ts
        }
        
        # 재시도 포함 POST 요청
        resp = request_with_retry(f"{AI_SERVER_2_URL}/obstacle/detected", payload, timeout=5.0, retries=2, backoff=0.5)
        if resp is not None and resp.status_code == 200:
            app.logger.info(f"[IF-01] AI Server 2로 이미지 전송 성공")
            return True
        else:
            code = resp.status_code if resp is not None else 'no_response'
            app.logger.error(f"[IF-01] AI Server 2 전송 실패: {code}")
            return False
            
    except Exception as e:
        app.logger.error(f"[IF-01] AI Server 2 전송 오류: {e}")
        return False


def send_image_to_ai_server2_coco(frame: np.ndarray) -> Optional[dict]:
    """AI Server 2의 /infer/coco 호출. 성공 시 JSON 반환, 실패 시 None"""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        payload = {"image": image_base64}
        resp = request_with_retry(f"{AI_SERVER_2_URL}/infer/coco", payload, timeout=5.0, retries=2, backoff=0.5)
        if resp is not None and resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        app.logger.error(f"/infer/coco 프록시 오류: {e}")
        return None


# 중앙 WebSocket 수신 루프: alert_idle / alert_occupied 제어
_defunct = False

# 로컬 트래킹 중단/초기화
def _local_stop_tracking():
    with worker_lock:
        STATE["tracking"] = False
        STATE["target_person_id"] = None
        STATE["target_visible"] = False
        STATE["disappear_sent"] = False
        STATE["disappear_deadline_ts"] = 0.0
        STATE["central_stop_sent"] = False
    app.logger.info("[TRACK] local stop & reset (by navigating_complete or timeout)")

# 중앙에 stop_tracking 통지 (있으면 사용)
def _notify_central_stop_tracking():
    url = f"{CENTRAL_HTTP_BASE}/stop_tracking"
    payload = {"robot_id": ROBOT_ID}
    try:
        resp = requests.post(url, json=payload, timeout=1.5)
        app.logger.info(f"[IF-STOP] central /stop_tracking -> {resp.status_code}")
    except Exception as e:
        app.logger.warning(f"[IF-STOP] central /stop_tracking failed: {e}")


def central_ws_loop():
    if websocket is None:
        app.logger.warning("websocket-client 미설치: CENTRAL_WS 비활성")
        return
    global _defunct
    # 프록시 환경변수 무시 (직접 접속)
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        if os.environ.get(k):
            os.environ.pop(k, None)
    while not _defunct:
        try:
            app.logger.info(f"[WS] connecting to central: {CENTRAL_WS_URL}")
            app.logger.info(f"[WS] robot_id: {ROBOT_ID}")
            
            ws = websocket.create_connection(
                CENTRAL_WS_URL,
                timeout=5,
                http_proxy_host=None,
                http_proxy_port=None,
                enable_multithread=True,
                sockopt=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
            )
            app.logger.info("[WS] connected to central")
            
            # 연결 후 초기 메시지 전송 (중앙 서버가 기대할 수 있음)
            try:
                init_msg = {
                    "type": "ai_connect",
                    "robot_id": ROBOT_ID,
                    "timestamp": int(time.time())
                }
                ws.send(json.dumps(init_msg))
                app.logger.info(f"[WS] sent init message: {init_msg}")
            except Exception as e:
                app.logger.warning(f"[WS] init message send failed: {e}")
            
            while True:
                msg = ws.recv()
                if not msg:
                    break
                app.logger.info(f"[WS] received: {msg}")
                try:
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    if msg_type == "alert_idle":
                        with worker_lock:
                            STATE["can_send_come"] = True
                        app.logger.info("[WS] alert_idle → can_send_come=True")
                    elif msg_type == "alert_occupied":
                        with worker_lock:
                            STATE["can_send_come"] = False
                        app.logger.info("[WS] alert_occupied → can_send_come=False")
                    elif msg_type == "navigating_complete":
                        # 중앙에서 길안내 완료 알림 → 트래킹 중단 및 초기화
                        _local_stop_tracking()
                    else:
                        app.logger.info(f"[WS] unknown message type: {msg_type}")
                except Exception as e:
                    app.logger.warning(f"[WS] parse error: {e}")
        except Exception as e:
            app.logger.warning(f"[WS] connection failed: {e}")
            app.logger.info(f"[WS] 재연결 시도 3초 후...")
            time.sleep(3.0)


def ensure_ws_client():
    global ws_thread
    if ws_thread is None:
        ws_thread = threading.Thread(target=central_ws_loop, daemon=True)
        ws_thread.start()


# (Flask 3) before_first_request 제거됨. __main__에서 ensure_ws_client()를 호출합니다.


def tracking_worker():
    # 로컬 타임아웃 워커는 기본 비활성. 중앙이 10초 정책을 관리.
    if not USE_TIMEOUT_WORKER:
        return
    app.logger.info('[worker] tracking worker started')
    while STATE["worker_running"]:
        time.sleep(1.0)
        if not STATE["tracking"]:
            continue
        now = time.time()
        with worker_lock:
            last_update = STATE["last_tracking_update"]
            if (now - last_update["ts"]) > 10.0 and not STATE["disappear_sent"]:
                code, _ = send_user_disappear()
                app.logger.info(f"[IF-04] user_disappear sent (no update for 10s) -> {code}")
                STATE["disappear_sent"] = True
    app.logger.info('[worker] tracking worker stopped')


def ensure_worker():
    global worker_thread
    if not STATE["worker_running"] and USE_TIMEOUT_WORKER:
        STATE["worker_running"] = True
        worker_thread = threading.Thread(target=tracking_worker, daemon=True)
        worker_thread.start()


@app.route("/obstacle/detected", methods=["POST"])
def obstacle_detected():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    # 새로운 스키마: { robot_id:int, left_angle:float, right_angle:float, timestamp:int }
    robot_id = data.get("robot_id")
    left_angle = data.get("left_angle")
    right_angle = data.get("right_angle") 
    ts = data.get("timestamp")

    if robot_id is None or left_angle is None or right_angle is None:
        return jsonify({"status_code": 400, "error": "invalid_fields"}), 400

    # 각도가 이미 float인지 확인하고 변환
    try:
        left_angle_float = float(left_angle) if isinstance(left_angle, str) else left_angle
        right_angle_float = float(right_angle) if isinstance(right_angle, str) else right_angle
    except (ValueError, TypeError):
        return jsonify({"status_code": 400, "error": "invalid_angle_format"}), 400

    # 각도를 기반으로 카메라 선택
    camera = choose_camera_by_angles(left_angle_float, right_angle_float)
    frame = get_latest_frame(camera)
    if frame is None:
        return jsonify({"status_code": 503, "error": "no_frame"}), 503

    # AI Server 2로 이미지 전송
    success = send_image_to_ai_server2(frame, left_angle_float, right_angle_float, robot_id, ts)
    
    app.logger.info(f"[IF-01] camera={camera}, left_angle={left_angle_float}, right_angle={right_angle_float}, ai_server2_sent={success}")

    STATE["last_obstacle"] = {
        "robot_id": robot_id,
        "left_angle": left_angle_float,
        "right_angle": right_angle_float,
        "timestamp": ts or int(datetime.utcnow().timestamp()),
        "camera": camera,
        "ai_server2_sent": success,
    }

    return ok()


@app.route("/infer/coco", methods=["POST"])
def infer_coco_proxy():
    """현재 프레임을 캡처하여 AI Server 2의 /infer/coco에 프록시 호출"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    camera = data.get("camera", "front")  # 'front' | 'back'
    frame = get_latest_frame(camera)
    if frame is None:
        return jsonify({"status_code": 503, "error": "no_frame"}), 503

    result = send_image_to_ai_server2_coco(frame)
    if result is None:
        return jsonify({"status_code": 502, "error": "upstream_failed"}), 502

    with worker_lock:
        STATE["last_coco_infer"] = {"camera": camera, "result": result, "ts": int(time.time())}
    return jsonify({"status_code": 200, "result": result})


@app.route("/start_tracking", methods=["POST"])  # 기존 엔드포인트 유지
def start_tracking():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    robot_id = data.get("robot_id")
    if robot_id is None:
        return jsonify({"status_code": 400, "error": "robot_id_required"}), 400

    with worker_lock:
        STATE["tracking"] = True
        # last_tracking_update 리셋
        STATE["last_tracking_update"] = {"visible": False, "ts": time.time()}
        STATE["disappear_sent"] = False
        # 타겟 상태 초기화
        STATE["target_person_id"] = None
        STATE["target_visible"] = False
    
    ensure_worker()
    app.logger.info(f"[/start_tracking] tracking started for robot_id={robot_id}")
    return ok()


@app.route("/stop_tracking", methods=["POST"])  # 옵션: 중앙에서 종료 명령
def stop_tracking():
    STATE["tracking"] = False
    STATE["target_bbox"] = None
    # 타겟 상태 초기화
    with worker_lock:
        STATE["target_person_id"] = None
        STATE["target_visible"] = False
        STATE["disappear_sent"] = False
    return ok()


@app.route("/gesture/come_local", methods=["POST"])  # dual가 로컬로 전달하면 중앙으로 포워드
def gesture_come_local():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    person_id = str(data.get("person_id", "unknown"))
    left_angle = float(data.get("left_angle", 0.0))
    right_angle = float(data.get("right_angle", data.get("right_angle", 0.0)))
    ts = int(data.get("timestamp", time.time()))

    now = time.time()
    
    # 중앙 WS 게이트: alert_idle → 허용, alert_occupied → 차단
    with worker_lock:
        can_send = STATE.get("can_send_come", True)
    if not can_send:
        app.logger.info(f"[IF-03] gate=occupied: come 전송 차단 (person_id={person_id})")
        return ok()

    # 같은 사람 중복 전송 방지 (5초 이내 중복 금지)
    last = STATE["come_sent_ids"].get(person_id, 0.0)
    if now - last < 5.0:
        app.logger.info(f"[IF-03] 같은 사람 중복 전송 방지 (person_id={person_id})")
        return ok()

    # come 제스처 전송
    code, _ = send_gesture_come(left_angle, right_angle, ts)
    app.logger.info(f"[IF-03] gesture/come forwarded person_id={person_id} -> {code}")
    
    # 로컬 게이트도 즉시 닫아 과다 전송 방지 (중앙에서 alert_occupied 도착 전까지)
    with worker_lock:
        STATE["can_send_come"] = False
        STATE["come_sent_ids"][person_id] = now
        STATE["last_come_person_id"] = person_id
        STATE["last_come_time"] = now
    
    return ok()


@app.route("/return_command", methods=["POST"])  # 기존 유지
def gesture_return_command():
    """중앙 서버로부터 return_command를 받아 come 제스처 상태를 리셋"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    robot_id = data.get("robot_id")
    if robot_id is None:
        return jsonify({"status_code": 400, "error": "robot_id_required"}), 400

    # 상태 리셋
    with worker_lock:
        was_active = STATE["target_visible"]
        STATE["target_person_id"] = None
        STATE["target_visible"] = False
        STATE["disappear_sent"] = False
        # 중앙에서 alert_idle이 올 때까지는 can_send_come=False 유지 권장
        # 여기서는 변경하지 않음
    
    app.logger.info(f"[IF-06] return_command received, tracking state reset (was_visible={was_active})")
    
    # dual_camera_system_shared에도 return_command 전달
    try:
        dual_camera_url = "http://localhost:5008"
        response = requests.post(f"{dual_camera_url}/gesture/return_command", json=data, timeout=1.0)
        if response.status_code == 200:
            app.logger.info(f"[IF-06] dual_camera_system_shared에 return_command 전달 성공")
        else:
            app.logger.warning(f"[IF-06] dual_camera_system_shared 전달 실패: {response.status_code}")
    except Exception as e:
        app.logger.warning(f"[IF-06] dual_camera_system_shared 전달 오류: {e}")
    
    return ok()


@app.route("/tracking/update", methods=["POST"])  # 기존 유지 + 타겟 전환 이벤트 즉시 보고
def tracking_update():
    """dual_camera_system_shared.py 로부터 주기적인 추적 상태를 받음
    - 기존 last_tracking_update 동작 유지
    - person_id(가장 큰 사람) 기준 타겟 사라짐/재등장 즉시 보고
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400
    
    is_visible = bool(data.get("person_visible", False))
    person_id = data.get("person_id")  # dual 측에서 가장 큰 사람의 통합 ID
    now = time.time()

    with worker_lock:
        if not STATE["tracking"]:
            return ok()
        # 기존: 보이면 마지막 업데이트 갱신(중앙 10초 정책 유지를 위해)
        if is_visible:
            STATE["last_tracking_update"] = {"visible": True, "ts": now}
            # 마지막으로 본 시각 업데이트
            STATE["disappear_deadline_ts"] = 0.0
            STATE["central_stop_sent"] = False
        
        # 타겟 미선정 → 보이는 ID를 타겟으로 지정(보고는 안함)
        if STATE["target_person_id"] is None:
            if is_visible and person_id:
                STATE["target_person_id"] = person_id
                STATE["target_visible"] = True
                STATE["disappear_sent"] = False
                STATE["disappear_deadline_ts"] = 0.0
                STATE["central_stop_sent"] = False
            return ok()
        
        # 타겟 선정 이후
        target_id = STATE["target_person_id"]
        if is_visible and person_id == target_id:
            # 재등장(이전이 사라짐 상태였다면 보고)
            if STATE["disappear_sent"]:
                code, _ = send_user_appear()
                app.logger.info(f"[IF-05] target user_appear sent -> {code}")
            STATE["disappear_sent"] = False
            STATE["target_visible"] = True
            STATE["disappear_deadline_ts"] = 0.0
            STATE["central_stop_sent"] = False
        else:
            # 즉시 사라짐(이전에 visible이었다면 한번만 전송)
            if STATE["target_visible"] and not STATE["disappear_sent"]:
                code, _ = send_user_disappear()
                app.logger.info(f"[IF-04] target user_disappear sent -> {code}")
                STATE["disappear_sent"] = True
                # 10초 타이머 시작
                STATE["disappear_deadline_ts"] = now + 10.0
                STATE["central_stop_sent"] = False
            STATE["target_visible"] = False

            # 10초 경과 확인 → 중앙에 stop_tracking 통지 후 로컬 중단
            deadline = STATE.get("disappear_deadline_ts", 0.0)
            if deadline > 0.0 and now >= deadline and not STATE.get("central_stop_sent", False):
                _notify_central_stop_tracking()
                STATE["central_stop_sent"] = True
                # 로컬 트래킹 중단 및 초기화
                # (재입장 시 central alert_idle 필요)
                # 이 함수는 lock 내부이므로 즉시 상태 갱신만 수행
                STATE["tracking"] = False
                STATE["target_person_id"] = None
                STATE["target_visible"] = False
                STATE["disappear_deadline_ts"] = 0.0
 
    return ok()


@app.route("/health", methods=["GET"])  # 간단 헬스체크
def health():
    return jsonify({
        "status": "ok",
        "tracking": STATE["tracking"],
        "last_obstacle": STATE["last_obstacle"],
        "worker_running": STATE["worker_running"],
        "come_gesture_active": STATE["come_gesture_active"],
        "can_send_come": STATE["can_send_come"],
        "last_come_person_id": STATE["last_come_person_id"],
        "last_come_time": STATE["last_come_time"],
        "last_coco_infer": STATE["last_coco_infer"],
        "server_type": "ai_server_1 (communication + image forwarding)",
        "ai_server_2_url": AI_SERVER_2_URL,
        "central_ws_url": CENTRAL_WS_URL,
    })


if __name__ == "__main__":
    ensure_ws_client()
    app.run(host=AI_HTTP_HOST, port=AI_HTTP_PORT, debug=False) 
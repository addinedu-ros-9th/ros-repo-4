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
import yaml

import numpy as np
import cv2
from flask import Flask, request, jsonify

# 외부 모듈 경로 추가 (공유메모리 리더)
import sys
sys.path.append('/home/ckim/ros-repo-4/deeplearning/src')
from shared_memory_reader import DualCameraSharedMemoryReader

# 듀얼 카메라 시스템 전역 변수 import
try:
    from dual_camera_system_shared import BACK_CAMERA_CONTROL, GESTURE_RESET_FLAG
    DUAL_CAMERA_AVAILABLE = True
    print("✅ 듀얼 카메라 제어 모듈 로드 성공")
except ImportError:
    DUAL_CAMERA_AVAILABLE = False
    print("⚠️ 듀얼 카메라 제어 모듈 로드 실패")

# 중앙 송신 유틸
from sender import send_gesture_come, send_user_disappear, send_user_appear

# 중앙 WebSocket 클라이언트
try:
    import websocket  # pip install websocket-client
except Exception:
    websocket = None

app = Flask(__name__)

# config.yaml에서 설정 로드
def load_config():
    try:
        config_path = "/home/ckim/ros-repo-4/config.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        central_ip = config["central_server"]["ip"]
        websocket_port = config["central_server"]["websocket_port"]
        http_port = config["central_server"]["http_port"]
        
        return central_ip, websocket_port, http_port
    except Exception as e:
        print(f"config.yaml 로드 실패: {e}")
        # 기본값 반환
        return "192.168.0.36", 3000, 8080

# 환경 설정
AI_HTTP_HOST = os.environ.get("AI_HTTP_HOST", "0.0.0.0")
AI_HTTP_PORT = int(os.environ.get("AI_HTTP_PORT", "8000"))
ROBOT_ID = int(os.environ.get("ROBOT_ID", "3"))
# 추가: 로컬 타임아웃 워커 사용 여부(기본 비활성)
USE_TIMEOUT_WORKER = os.environ.get("USE_TIMEOUT_WORKER", "0") == "1"

# config.yaml에서 중앙 서버 설정 로드
CENTRAL_IP, WEBSOCKET_PORT, HTTP_PORT = load_config()
CENTRAL_WS_URL = os.environ.get("CENTRAL_WS_URL", f"ws://{CENTRAL_IP}:{WEBSOCKET_PORT}/?client_type=ai")

# AI Server 2 설정
AI_SERVER_2_URL = os.environ.get("AI_SERVER_2_URL", "http://192.168.0.74:5007")
# 중앙 HTTP BASE (stop_tracking 전송용)
CENTRAL_HTTP_BASE = os.environ.get("CENTRAL_HTTP_BASE", f"http://{CENTRAL_IP}:{HTTP_PORT}")

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
    front_frame, back_frame = reader.read_frames()
    if camera == 'front':
        return front_frame
    elif camera == 'back':
        return back_frame
    else:
        return None


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
        print("❌ websocket-client 미설치!")
        return
    
    print(f"🚀 WebSocket 루프 시작됨! URL: {CENTRAL_WS_URL}")
    app.logger.info("=== WebSocket 연결 시작 ===")
    app.logger.info(f"[WS] 설정된 URL: {CENTRAL_WS_URL}")
    app.logger.info(f"[WS] ROBOT_ID: {ROBOT_ID}")
    
    global _defunct
    # 프록시 환경변수 무시 (직접 접속)
    app.logger.info("[WS] 프록시 환경변수 제거 중...")
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        if os.environ.get(k):
            app.logger.info(f"[WS] 프록시 환경변수 제거: {k}")
            os.environ.pop(k, None)
    
    while not _defunct:
        try:
            print(f"🔗 WebSocket 연결 시도: {CENTRAL_WS_URL}")
            app.logger.info(f"[WS] WebSocket 연결 시도: {CENTRAL_WS_URL}")
            
            # WebSocket 연결 시도
            ws = websocket.create_connection(
                CENTRAL_WS_URL,
                http_proxy_host=None,
                http_proxy_port=None,
                enable_multithread=True,
                sockopt=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
            )
            print("✅ WebSocket 연결 성공!")
            app.logger.info("[WS] ✅ WebSocket 연결 성공!")
            
            # 연결 후 초기 메시지 전송
            try:
                init_msg = {
                    "type": "ai_connect",
                    "robot_id": ROBOT_ID,
                    "timestamp": int(time.time())
                }
                app.logger.info(f"[WS] 초기 메시지 전송: {init_msg}")
                ws.send(json.dumps(init_msg))
                app.logger.info("[WS] ✅ 초기 메시지 전송 성공")
                print("📤 초기 메시지 전송 완료")
            except Exception as e:
                app.logger.warning(f"[WS] ⚠️ 초기 메시지 전송 실패: {e}")
                print(f"❌ 초기 메시지 전송 실패: {e}")
            
            # 메시지 수신 루프 - 중앙 서버의 alert_idle/alert_occupied 대기
            app.logger.info("[WS] 중앙 서버 메시지 수신 대기 중... (alert_idle/alert_occupied)")
            print("👂 alert_idle/alert_occupied 대기 중...")
            while not _defunct:
                try:
                    # 블로킹 방식으로 메시지 수신 (타임아웃 없음)
                    message = ws.recv()
                    app.logger.info(f"[WS] 메시지 수신: {message}")
                    print(f"📨 WebSocket 메시지: {message}")
                    
                    # JSON 파싱
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "alert_idle":
                            with worker_lock:
                                STATE["can_send_come"] = True
                            app.logger.info("[WS] alert_idle 수신 -> can_send_come = True")
                            print("🟢 alert_idle 수신! COME 제스처 활성화")
                            
                        elif msg_type == "alert_occupied":
                            with worker_lock:
                                STATE["can_send_come"] = False
                            app.logger.info("[WS] alert_occupied 수신 -> can_send_come = False")
                            print("🔴 alert_occupied 수신! COME 제스처 비활성화")
                            
                        else:
                            app.logger.info(f"[WS] 기타 메시지 타입: {msg_type}")
                            print(f"❓ 알 수 없는 메시지: {msg_type}")
                            
                    except json.JSONDecodeError as e:
                        app.logger.warning(f"[WS] JSON 파싱 실패: {e}")
                        print(f"❌ JSON 파싱 실패: {e}")
                        
                except websocket.WebSocketConnectionClosedException:
                    app.logger.warning("[WS] WebSocket 연결이 닫힘 - 재연결 시도")
                    print("🔌 연결 끊어짐 - 재연결 시도")
                    break
                except Exception as e:
                    app.logger.error(f"[WS] 메시지 수신 중 오류: {e}")
                    app.logger.error(f"[WS] 오류 타입: {type(e).__name__}")
                    print(f"❌ 메시지 수신 오류: {e}")
                    break
            
            # 연결 종료
            try:
                ws.close()
                app.logger.info("[WS] WebSocket 연결 종료")
            except Exception as e:
                app.logger.warning(f"[WS] WebSocket 연결 종료 중 오류: {e}")
                
        except Exception as e:
            app.logger.error(f"[WS] WebSocket 연결 실패: {e}")
            app.logger.error(f"[WS] 오류 타입: {type(e).__name__}")
            print(f"❌ WebSocket 연결 실패: {e}")
        
        # 재연결 전 대기 (5초)
        if not _defunct:
            app.logger.info("[WS] 5초 후 재연결 시도...")
            print("⏰ 5초 후 재연결...")
            time.sleep(5)
    
    app.logger.info("=== WebSocket 루프 종료 ===")
    print("�� WebSocket 루프 종료")


def ensure_ws_client():
    global ws_thread, _defunct
    print("[WS] ensure_ws_client() 호출됨")
    print(f"[WS] 현재 ws_thread 상태: {ws_thread}")
    print(f"[WS] 현재 _defunct 상태: {_defunct}")
    
    if ws_thread is None or not ws_thread.is_alive():
        if ws_thread is not None:
            print("[WS] 기존 스레드가 종료됨 - 새로운 스레드 생성")
        
        # _defunct 플래그 리셋
        _defunct = False
        
        print("[WS] WebSocket 스레드 생성 중...")
        ws_thread = threading.Thread(target=central_ws_loop, daemon=True)
        print("[WS] WebSocket 스레드 시작 중...")
        ws_thread.start()
        print("[WS] ✅ WebSocket 스레드 시작 완료")
    else:
        print("[WS] WebSocket 스레드가 이미 실행 중입니다")


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


@app.route("/gesture/come", methods=["POST"])  # dual_camera_system_shared에서 직접 호출
def gesture_come():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400

    person_id = str(data.get("person_id", "unknown"))
    left_angle = float(data.get("left_angle", 0.0))
    right_angle = float(data.get("right_angle", 0.0))
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
    app.logger.info(f"[IF-03] gesture/come sent person_id={person_id} -> {code}")
    
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

    # AI 서버 상태 리셋
    with worker_lock:
        was_active = STATE["target_visible"]
        STATE["target_person_id"] = None
        STATE["target_visible"] = False
        STATE["disappear_sent"] = False
        STATE["disappear_deadline_ts"] = 0.0
        STATE["central_stop_sent"] = False
        # 트래킹도 초기화
        STATE["tracking"] = False
    
    app.logger.info(f"[IF-06] return_command received, all state reset (was_visible={was_active})")
    
    # dual_camera_system_shared에 제스처 리셋 요청 (전역 변수 사용)
    if DUAL_CAMERA_AVAILABLE:
        global GESTURE_RESET_FLAG
        GESTURE_RESET_FLAG["reset_requested"] = True
        app.logger.info(f"[IF-06] 듀얼 카메라 시스템에 제스처 리셋 요청 (전역 변수)")
    else:
        app.logger.warning(f"[IF-06] 듀얼 카메라 시스템 제어 불가능")
    
    return ok()


@app.route("/tracking/update", methods=["POST"])  # 기존 유지 + 타겟 전환 이벤트 즉시 보고
def tracking_update():
    """dual_camera_system_shared.py 후면카메라로부터 추적 상태를 받음
    - 사람 사라짐 즉시 user_disappear 전송
    - AI 서버에서 10초 카운트다운 시작
    - 10초 내 재등장 시 user_appear 전송
    - 10초 초과 시 return_command를 중앙에 전송하고 트래킹 초기화
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status_code": 400, "error": "invalid_json"}), 400
    
    is_visible = bool(data.get("person_visible", False))
    person_id = data.get("person_id")  # 후면카메라에서 가장 큰 사람의 통합 ID
    now = time.time()

    with worker_lock:
        if not STATE["tracking"]:
            return ok()
            
        # 기존: 보이면 마지막 업데이트 갱신
        if is_visible:
            STATE["last_tracking_update"] = {"visible": True, "ts": now}
            # 타이머 리셋
            STATE["disappear_deadline_ts"] = 0.0
            STATE["central_stop_sent"] = False
        
        # 타겟 미선정 → 보이는 ID를 타겟으로 지정
        if STATE["target_person_id"] is None:
            if is_visible and person_id:
                STATE["target_person_id"] = person_id
                STATE["target_visible"] = True
                STATE["disappear_sent"] = False
                STATE["disappear_deadline_ts"] = 0.0
                STATE["central_stop_sent"] = False
                app.logger.info(f"[TRACK] 새로운 타겟 설정: {person_id}")
            return ok()
        
        # 타겟 선정 이후
        target_id = STATE["target_person_id"]
        if is_visible and person_id == target_id:
            # 재등장 (이전이 사라짐 상태였다면 user_appear 전송)
            if STATE["disappear_sent"]:
                code, _ = send_user_appear()
                app.logger.info(f"[IF-05] target user_appear sent -> {code}")
                STATE["disappear_sent"] = False
                STATE["disappear_deadline_ts"] = 0.0
                STATE["central_stop_sent"] = False
            STATE["target_visible"] = True
        else:
            # 사라짐 (즉시 user_disappear 전송 + 10초 카운트다운 시작)
            if STATE["target_visible"] and not STATE["disappear_sent"]:
                code, _ = send_user_disappear()
                app.logger.info(f"[IF-04] target user_disappear sent -> {code}")
                STATE["disappear_sent"] = True
                # 10초 카운트다운 시작
                STATE["disappear_deadline_ts"] = now + 10.0
                STATE["central_stop_sent"] = False
                app.logger.info(f"[TRACK] 10초 카운트다운 시작 (deadline: {STATE['disappear_deadline_ts']})")
            STATE["target_visible"] = False

            # 10초 경과 확인 → stop_tracking을 중앙에 전송하고 트래킹 초기화
            deadline = STATE.get("disappear_deadline_ts", 0.0)
            if deadline > 0.0 and now >= deadline and not STATE.get("central_stop_sent", False):
                # stop_tracking을 중앙에 전송
                try:
                    stop_payload = {"robot_id": ROBOT_ID}
                    resp = requests.post(f"{CENTRAL_HTTP_BASE}/stop_tracking", json=stop_payload, timeout=2.0)
                    app.logger.info(f"[IF-07] stop_tracking sent to central -> {resp.status_code}")
                except Exception as e:
                    app.logger.warning(f"[IF-07] stop_tracking 전송 실패: {e}")
                
                STATE["central_stop_sent"] = True
                # AI 서버에서 트래킹 초기화
                STATE["tracking"] = False
                STATE["target_person_id"] = None
                STATE["target_visible"] = False
                STATE["disappear_sent"] = False
                STATE["disappear_deadline_ts"] = 0.0
                app.logger.info(f"[TRACK] 10초 초과로 트래킹 초기화 및 stop_tracking 전송")

    return ok()


@app.route("/health", methods=["GET"])
def health():
    """AI 서버 상태 및 웹소켓 상태 반환"""
    with worker_lock:
        return jsonify({
            "status": "ok",
            "can_send_come": STATE.get("can_send_come", True),
            "tracking": STATE.get("tracking", False),
            "robot_id": ROBOT_ID,
            "timestamp": int(time.time())
        })


if __name__ == "__main__":
    print("=== AI Server HTTP 시작 ===")
    print(f"중앙 서버 IP: {CENTRAL_IP}")
    print(f"WebSocket 포트: {WEBSOCKET_PORT}")
    print(f"WebSocket URL: {CENTRAL_WS_URL}")
    
    print("ensure_ws_client() 호출 중...")
    ensure_ws_client()
    print("ensure_ws_client() 호출 완료")
    
    print("Flask 앱 시작 중...")
    app.run(host=AI_HTTP_HOST, port=AI_HTTP_PORT, debug=False) 
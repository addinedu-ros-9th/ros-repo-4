#!/usr/bin/env python3
import os
import time
import requests
import yaml
import json

# config.yaml에서 중앙 서버 설정 로드
def load_central_config():
    try:
        config_path = "/home/ckim/ros-repo-4/config.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        central_ip = config["central_server"]["ip"]
        http_port = config["central_server"]["http_port"]
        
        return f"http://{central_ip}:{http_port}"
    except Exception as e:
        print(f"config.yaml 로드 실패: {e}")
        # 기본값 반환
        return "http://192.168.0.36:8080"

CENTRAL_BASE = os.environ.get("CENTRAL_BASE", load_central_config())
ROBOT_ID = int(os.environ.get("ROBOT_ID", "3"))


def send_gesture_come(left_angle: float, right_angle: float, ts: int | None = None):
    url = f"{CENTRAL_BASE}/gesture/come"
    payload = {
        "robot_id": ROBOT_ID,
        "left_angle": str(left_angle),  # 문자열로 변환
        "right_angle": str(right_angle),  # 문자열로 변환
        "timestamp": ts or int(time.time()),
    }
    
    print(f"🚀 중앙 서버로 전송: {url}")
    print(f"📦 Payload: {payload}")
    print(f"📦 Payload 타입: {type(payload)}")
    print(f"📦 JSON 직렬화: {json.dumps(payload)}")
    
    try:
        headers = {'Content-Type': 'application/json'}
        print(f"📦 Headers: {headers}")
        
        r = requests.post(url, json=payload, headers=headers, timeout=0.5)
        print(f"✅ 중앙 서버 응답: {r.status_code} - {r.text}")
        return r.status_code, r.text
    except Exception as e:
        print(f"❌ 중앙 서버 전송 실패: {e}")
        return None, str(e)


def send_user_disappear():
    url = f"{CENTRAL_BASE}/user_disappear"
    payload = {"robot_id": ROBOT_ID}
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)


def send_user_appear():
    url = f"{CENTRAL_BASE}/user_appear"
    payload = {"robot_id": ROBOT_ID}
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)


if __name__ == "__main__":
    print(send_gesture_come(10.0, 30.0))
    print(send_user_disappear())
    print(send_user_appear()) 
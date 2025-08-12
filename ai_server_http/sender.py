#!/usr/bin/env python3
import os
import time
import requests

CENTRAL_BASE = os.environ.get("CENTRAL_BASE", "http://192.168.0.10:8080")
ROBOT_ID = int(os.environ.get("ROBOT_ID", "3"))


def send_gesture_come(left_angle: float, right_angle: float, ts: int | None = None):
    url = f"{CENTRAL_BASE}/gesture/come"
    payload = {
        "robot_id": ROBOT_ID,
        "left_angle": left_angle,  # float 그대로 전송
        "right_angle": right_angle,  # float 그대로 전송
        "timestamp": ts or int(time.time()),
    }
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        return r.status_code, r.text
    except Exception as e:
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
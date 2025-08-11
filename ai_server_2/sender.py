#!/usr/bin/env python3
import os
import time
import requests

ROBOT_ID = int(os.environ.get("ROBOT_ID", "3"))
GUI_BASE = os.environ.get("GUI_BASE", "http://192.168.0.74:3000")


def send_gui_obstacle_alert(image_b64: str, label: str, left_angle: float | None = None, right_angle: float | None = None):
    """GUI(0.74)로 직접 장애물 알림(이미지+라벨) 전송"""
    url = f"{GUI_BASE}/obstacle/alert"
    payload = {
        "robot_id": ROBOT_ID,
        "label": label,
        "image_b64": image_b64,
        "timestamp": int(time.time()),
    }
    if left_angle is not None:
        payload["left_angle"] = left_angle
    if right_angle is not None:
        payload["right_angle"] = right_angle
    try:
        r = requests.post(url, json=payload, timeout=1.5)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e) 
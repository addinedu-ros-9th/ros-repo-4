#!/usr/bin/env python3
import cv2
import glob
import os
import time
import subprocess

def list_video_devices():
    devs = sorted(glob.glob('/dev/video*'),
                  key=lambda p: int(''.join(ch for ch in p if ch.isdigit()) or -1))
    # 인덱스만 반환 (videoN → N)
    return [int(os.path.basename(d).replace('video', '')) for d in devs]

def get_v4l2_card_name(dev_path: str) -> str:
    # v4l2-ctl 설치되어 있으면 카드 이름 추출
    try:
        out = subprocess.run(
            ['v4l2-ctl', '-d', dev_path, '--all'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1.0
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                # 예: "Card type   : HD USB Camera"
                if 'Card type' in line:
                    return line.split(':', 1)[-1].strip()
                if 'Card:' in line:
                    return line.split(':', 1)[-1].strip()
        return ''
    except Exception:
        return ''

def open_capture(idx: int):
    # CAP_V4L2 우선, 실패 시 기본 백엔드
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    return cap

def draw_overlay(frame, lines, y0=28, dy=22):
    x, y = 12, y0
    for text in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y += dy

def probe_camera(idx: int):
    dev_path = f'/dev/video{idx}'
    cap = open_capture(idx)
    opened = cap.isOpened()
    info_lines = [f'[{idx}] {dev_path}']
    card_name = get_v4l2_card_name(dev_path)
    if card_name:
        info_lines.append(f'Card: {card_name}')

    if not opened:
        print(f'열기 실패: {dev_path}')
        return None, info_lines

    # 속성 읽기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    info_lines.append(f'Resolution: {w}x{h}, FPS: {fps:.2f}')

    return cap, info_lines

def main():
    devices = list_video_devices()
    if not devices:
        print('연결된 비디오 장치를 찾지 못했습니다. (/dev/video*)')
        return

    print('검출된 장치:', ', '.join(f'/dev/video{d}' for d in devices))
    cur = 0

    cv2.namedWindow('Camera Probe', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Probe', 960, 540)

    while True:
        idx = devices[cur]
        cap, info = probe_camera(idx)
        start = time.time()
        last_saved = None

        # 미리보기 루프
        while True:
            if cap is None:
                # 장치 열기 실패 시 안내 프레임 생성
                frame = 255 * (0 * (0,0,0))
                frame = cv2.cvtColor(cv2.UMat(480, 640, cv2.CV_8UC1).get(), cv2.COLOR_GRAY2BGR)
                draw_overlay(frame, info + ['열기 실패: n키로 다음 장치 이동'])
                cv2.imshow('Camera Probe', frame)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # 읽기 실패 상태 표시
                    frame = 255 * (0 * (0,0,0))
                    frame = cv2.cvtColor(cv2.UMat(480, 640, cv2.CV_8UC1).get(), cv2.COLOR_GRAY2BGR)
                    draw_overlay(frame, info + ['프레임 읽기 실패'])
                    cv2.imshow('Camera Probe', frame)
                else:
                    # 현재 해상도/FPS 재확인
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    lines = info[:-1] + [f'Resolution: {w}x{h}, FPS: {fps:.2f}']
                    draw_overlay(frame, lines + ['n:다음  p:이전  s:스냅샷  q:종료'])
                    cv2.imshow('Camera Probe', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if cap is not None: cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                if cap is not None: cap.release()
                cur = (cur + 1) % len(devices)
                break
            elif key == ord('p'):
                if cap is not None: cap.release()
                cur = (cur - 1) % len(devices)
                break
            elif key == ord('s') and frame is not None:
                ts = int(time.time())
                out = f'camera_{idx}_{ts}.jpg'
                try:
                    cv2.imwrite(out, frame)
                    last_saved = out
                    print(f'스냅샷 저장: {out}')
                except Exception as e:
                    print(f'저장 실패: {e}')

            # 5초 자동 순환(원하면 해제)
            if time.time() - start > 5.0:
                if cap is not None: cap.release()
                cur = (cur + 1) % len(devices)
                break

if __name__ == '__main__':
    main()
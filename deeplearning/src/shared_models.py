import os
import threading
from typing import Optional

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None  # 런타임 임포트 오류 대비

# GCN 관련 임포트 준비
import sys
try:
    import torch
except Exception:
    torch = None

# 새로운: 경량 워밍업을 위한 numpy 임포트
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # numpy가 없으면 워밍업을 건너뜀

# SlidingShiftGCN 경로 추가
_GCN_PATH_APPENDED = False

def _ensure_gcn_path():
    global _GCN_PATH_APPENDED
    if not _GCN_PATH_APPENDED:
        sys.path.append('/home/ckim/ros-repo-4/deeplearning/gesture_recognition/src')
        _GCN_PATH_APPENDED = True

try:
    _ensure_gcn_path()
    from train_sliding_shift_gcn import SlidingShiftGCN  # type: ignore
except Exception:
    SlidingShiftGCN = None  # 지연 임포트로 대체

# 전역 모델 인스턴스와 락 (스레드 안전)
_seg_model = None
_pose_model = None
_gcn_model = None

_init_lock = threading.Lock()

# 추론 시 동시 호출을 직렬화하기 위한 락
SEG_MODEL_LOCK = threading.Lock()
POSE_MODEL_LOCK = threading.Lock()
GCN_MODEL_LOCK = threading.Lock()


def _resolve_model_path(preferred_path: str, fallback_path: str) -> str:
    if os.path.exists(preferred_path):
        return preferred_path
    return fallback_path


# 새로운: 공통 워밍업/최적화 유틸리티
_def_imgsz = 512  # 성능/정확도 균형

def _maybe_fuse(model_obj):
    try:
        # Ultralytics YOLO는 fuse() 지원 (Conv+BN 결합)
        model_obj.fuse()
    except Exception:
        pass


def _warmup_yolo(model_obj):
    if np is None:
        return
    try:
        device = 0 if (torch is not None and torch.cuda.is_available()) else 'cpu'
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        # 한 번의 짧은 추론으로 커널/그래프 워밍업
        _ = model_obj(dummy, imgsz=_def_imgsz, device=device, verbose=False)
    except Exception:
        pass


def get_shared_seg_model() -> Optional["YOLO"]:
    global _seg_model
    if _seg_model is None:
        with _init_lock:
            if _seg_model is None:
                if YOLO is None:
                    raise RuntimeError("Ultralytics YOLO가 설치되어 있지 않습니다.")
                seg_path = _resolve_model_path(
                    "/home/ckim/ros-repo-4/deeplearning/yolov8s-seg.pt",
                    "./deeplearning/yolov8s-seg.pt",
                )
                _seg_model = YOLO(seg_path)
                # 최적화 및 워밍업
                _maybe_fuse(_seg_model)
                _warmup_yolo(_seg_model)
    return _seg_model


def get_shared_pose_model() -> Optional["YOLO"]:
    global _pose_model
    if _pose_model is None:
        with _init_lock:
            if _pose_model is None:
                if YOLO is None:
                    raise RuntimeError("Ultralytics YOLO가 설치되어 있지 않습니다.")
                pose_path = _resolve_model_path(
                    "/home/ckim/ros-repo-4/deeplearning/yolov8n-pose.pt",
                    "./yolov8n-pose.pt",
                )
                _pose_model = YOLO(pose_path)
                # 최적화 및 워밍업
                _maybe_fuse(_pose_model)
                _warmup_yolo(_pose_model)
    return _pose_model


def get_shared_gcn_model() -> Optional["SlidingShiftGCN"]:
    global _gcn_model
    if _gcn_model is None:
        with _init_lock:
            if _gcn_model is None:
                if torch is None:
                    raise RuntimeError("PyTorch가 설치되어 있지 않습니다.")
                _ensure_gcn_path()
                global SlidingShiftGCN
                if SlidingShiftGCN is None:
                    from train_sliding_shift_gcn import SlidingShiftGCN as _SlidingShiftGCN  # type: ignore
                    SlidingShiftGCN = _SlidingShiftGCN
                model_path = _resolve_model_path(
                    "/home/ckim/ros-repo-4/deeplearning/gesture_recognition/models/sliding_shift_gcn_model.pth",
                    "./deeplearning/gesture_recognition/models/sliding_shift_gcn_model.pth",
                )
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = SlidingShiftGCN(num_classes=2, num_joints=9, dropout=0.3)
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location=device)
                    model.load_state_dict(state)
                model = model.to(device)
                model.eval()
                _gcn_model = model
    return _gcn_model 
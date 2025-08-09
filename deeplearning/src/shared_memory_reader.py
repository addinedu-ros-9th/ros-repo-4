"""
공유 메모리에서 프레임을 읽는 Python 클래스
AI 서버에서 공유 메모리에 저장한 프레임을 읽어옵니다.
"""
import cv2
import numpy as np
import mmap
import os
import time
from typing import Optional, Tuple

class SharedMemoryReader:
    def __init__(self, shm_name: str, width: int, height: int):
        self.shm_name = shm_name
        self.width = width
        self.height = height
        self.frame_size = width * height * 3  # BGR format
        self.shm_fd = None
        self.shm_mmap = None
        self.available = False
        
        # 프레임 신선도 추적
        self._last_checksum: Optional[int] = None
        self._last_change_ts: float = 0.0
        self._last_reopen_ts: float = 0.0
        
        self._open_shared_memory()
    
    def _open_shared_memory(self):
        """공유 메모리 열기"""
        try:
            # 공유 메모리 열기
            self.shm_fd = os.open(self.shm_name, os.O_RDONLY)
            if self.shm_fd == -1:
                print(f"공유 메모리를 열 수 없습니다: {self.shm_name}")
                return
            
            # 메모리 매핑
            self.shm_mmap = mmap.mmap(self.shm_fd, self.frame_size, 
                                    access=mmap.ACCESS_READ)
            self.available = True
            print(f"공유 메모리 열기 성공: {self.shm_name}")
            
        except Exception as e:
            print(f"공유 메모리 열기 실패: {e}")
            self.available = False
    
    def reopen(self) -> bool:
        """공유 메모리를 재오픈"""
        try:
            self.close()
        except Exception:
            pass
        self._last_checksum = None
        self._last_change_ts = 0.0
        self._open_shared_memory()
        self._last_reopen_ts = time.time()
        return self.available

    def maybe_reopen(self, min_interval_sec: float = 1.0) -> bool:
        now = time.time()
        if now - self._last_reopen_ts >= min_interval_sec:
            return self.reopen()
        return False
    
    def _compute_checksum(self, frame_array: np.ndarray) -> int:
        """프레임 전반에서 샘플링하여 경량 체크섬 계산"""
        n = frame_array.size
        if n == 0:
            return 0
        # 4개 지점에서 4096 바이트 샘플링
        chunk = 4096
        idxs = [0, max(0, n//3 - chunk//2), max(0, (2*n)//3 - chunk//2), max(0, n - chunk)]
        total = 0
        for i in idxs:
            total += int(frame_array[i:i+chunk].sum())
        return total & 0xFFFFFFFF
    
    def read_frame(self) -> Optional[np.ndarray]:
        """프레임 읽기 및 신선도 갱신"""
        if not self.available or self.shm_mmap is None:
            return None
        
        try:
            # 공유 메모리에서 데이터 읽기
            self.shm_mmap.seek(0)
            frame_data = self.shm_mmap.read(self.frame_size)
            
            if len(frame_data) != self.frame_size:
                return None
            
            # 바이트 데이터를 numpy 배열로 변환
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame_array.reshape(self.height, self.width, 3)
            
            # 신선도 체크섬(샘플링) 계산
            checksum = self._compute_checksum(frame_array)
            if self._last_checksum is None or checksum != self._last_checksum:
                self._last_checksum = checksum
                self._last_change_ts = time.time()
            
            return frame
            
        except Exception as e:
            print(f"프레임 읽기 실패: {e}")
            return None
    
    def is_available(self) -> bool:
        """공유 메모리 사용 가능 여부"""
        return self.available
    
    def stale_seconds(self) -> float:
        """마지막 변경 이후 경과 시간(초). 변경 이력이 없으면 매우 큰 값 반환"""
        if self._last_change_ts == 0.0:
            return 1e9
        return max(0.0, time.time() - self._last_change_ts)
    
    def close(self):
        """공유 메모리 닫기"""
        if self.shm_mmap:
            self.shm_mmap.close()
            self.shm_mmap = None
        
        if self.shm_fd and self.shm_fd != -1:
            os.close(self.shm_fd)
            self.shm_fd = None
        
        self.available = False
        print(f"공유 메모리 닫기: {self.shm_name}")


class DualCameraSharedMemoryReader:
    """전면/후면 카메라 공유 메모리 읽기"""
    def __init__(self):
        self.front_reader = SharedMemoryReader("/dev/shm/front_camera_frame", 640, 480)
        self.back_reader = SharedMemoryReader("/dev/shm/back_camera_frame", 640, 480)
        
        # 폴백 웹캠 핸들
        self._front_cap = None
        self._back_cap = None
        self._use_fallback = os.environ.get('FALLBACK_WEBCAM', '0') == '1'
    
    def _ensure_fallback(self):
        if not self._use_fallback:
            return
        if self._front_cap is None:
            try:
                self._front_cap = cv2.VideoCapture('/dev/video0')
            except Exception:
                self._front_cap = None
        if self._back_cap is None:
            try:
                self._back_cap = cv2.VideoCapture('/dev/video2')
            except Exception:
                self._back_cap = None
    
    def read_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """전면/후면 프레임 읽기 (+ 스테일 시 폴백 지원)"""
        front_frame = self.front_reader.read_frame()
        back_frame = self.back_reader.read_frame()
        
        front_stale = self.front_reader.stale_seconds()
        back_stale = self.back_reader.stale_seconds()
        
        # 스테일 감지: 1.0초 넘게 변경 없으면 재오픈 시도 및 폴백 사용
        if front_stale > 3.0:  # 1.0 → 3.0초로 늘림 (재오픈 빈도 감소)
            if front_stale > 5.0:  # 5초 이상일 때만 로그 출력
                print("⚠️ front 공유메모리 프레임이 3초 이상 갱신되지 않음 - 재오픈 시도")
            self.front_reader.maybe_reopen(min_interval_sec=3.0)
        if back_stale > 3.0:  # 1.0 → 3.0초로 늘림 (재오픈 빈도 감소)
            if back_stale > 5.0:  # 5초 이상일 때만 로그 출력
                print("⚠️ back 공유메모리 프레임이 3초 이상 갱신되지 않음 - 재오픈 시도")
            self.back_reader.maybe_reopen(min_interval_sec=3.0)
        
        if self._use_fallback:
            self._ensure_fallback()
            if (front_frame is None) or (front_stale > 3.0):
                if self._front_cap is not None:
                    ok, fr = self._front_cap.read()
                    if ok:
                        front_frame = fr
            if (back_frame is None) or (back_stale > 3.0):
                if self._back_cap is not None:
                    ok, fr = self._back_cap.read()
                    if ok:
                        back_frame = fr
        
        return front_frame, back_frame
    
    def is_available(self) -> bool:
        """두 카메라 모두 사용 가능한지 확인"""
        return self.front_reader.is_available() and self.back_reader.is_available()
    
    def close(self):
        """모든 공유 메모리 닫기"""
        self.front_reader.close()
        self.back_reader.close()
        if self._front_cap is not None:
            self._front_cap.release()
            self._front_cap = None
        if self._back_cap is not None:
            self._back_cap.release()
            self._back_cap = None


if __name__ == "__main__":
    # 테스트 코드
    reader = DualCameraSharedMemoryReader()
    
    print("공유 메모리에서 프레임 읽기 테스트...")
    print("Ctrl+C로 종료")
    
    try:
        while True:
            front_frame, back_frame = reader.read_frames()
            
            if front_frame is not None:
                cv2.imshow("Front Camera (Shared Memory)", front_frame)
            
            if back_frame is not None:
                cv2.imshow("Back Camera (Shared Memory)", back_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("테스트 종료")
    finally:
        reader.close()
        cv2.destroyAllWindows() 
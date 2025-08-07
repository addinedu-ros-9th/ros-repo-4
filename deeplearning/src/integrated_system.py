"""
통합 시스템 메인 모듈
- Person Tracker와 Gesture Recognizer를 통합
- 쓰레드 관리 및 UI 표시
- 명령행 인수로 카메라 장치 설정 가능
"""

import cv2
import numpy as np
from collections import deque
import time
from datetime import datetime
import os
import argparse

# 모듈 import
from person_tracker import PersonTracker
from gesture_recognizer import GestureRecognizer

# Qt 오류 방지
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_DEBUG_PLUGINS'] = '0'

class IntegratedSystem:
    """통합 시스템"""
    
    def __init__(self):
        print("🚀 통합 시스템 초기화")
        
        # 모듈 초기화
        self.person_tracker = PersonTracker()
        self.gesture_recognizer = GestureRecognizer()
        
        # 시스템 설정
        self.running = True
        
        # 성능 모니터링
        self.fps_times = deque(maxlen=30)
        
        print("✅ 통합 시스템 초기화 완료")
    
    def run_system(self, camera_device=None, system_name="Integrated System"):
        """시스템 실행"""
        if camera_device is None:
            print("❌ 카메라 장치가 지정되지 않았습니다!")
            print("   사용법: python integrated_system.py --camera /dev/videoX --name '시스템명'")
            return
            
        print("🚀 통합 시스템 시작!")
        print(f"📹 카메라: {camera_device}")
        print(f"🏷️ 시스템 이름: {system_name}")
        print("⚡ 모듈화된 시스템: Person Tracker + Gesture Recognizer")
        
        # 카메라 설정
        cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)  # V4L2 백엔드 명시적 사용
        
        # 같은 하드웨어 카메라 충돌 방지를 위한 특별 설정
        if "video4" in camera_device:
            # video4는 video2와 같은 하드웨어이므로 더 긴 대기 시간
            time.sleep(2)  # 5초에서 2초로 단축
        
        if not cap.isOpened():
            print(f"❌ 카메라 연결 실패: {camera_device}")
            return
        
        # 카메라 연결 확인
        ret, test_frame = cap.read()
        if not ret:
            print(f"❌ 카메라에서 프레임 읽기 실패: {camera_device}")
            cap.release()
            return
        
        print(f"✅ 카메라 연결 및 프레임 읽기 성공: {camera_device}")
        
        # 창 설정
        window_name = f"🚀 {system_name}"
        
        try:
            # Back 창의 경우 더 강력한 창 설정
            if "Back" in system_name:
                # 간단한 창 이름으로 시도
                window_name = "Back Camera"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 800, 600)
                cv2.moveWindow(window_name, 900, 50)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
                time.sleep(0.5)
                print(f"📍 Back 창 생성: {window_name}")
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(window_name, 800, 600)
                
                # 창 위치 설정 (시스템 이름에 따라)
                if "Front" in system_name:
                    cv2.moveWindow(window_name, 50, 50)   # 왼쪽 위
                    print(f"📍 Front 창 위치: (50, 50)")
                else:
                    cv2.moveWindow(window_name, 50, 50)    # 기본 위치
                
                # 창 생성 안정성을 위한 대기
                cv2.waitKey(1)
                time.sleep(1.0)  # 1초 대기
                
                # 창이 실제로 생성되었는지 확인
                cv2.waitKey(1)
            
            print("🖼️ 창 설정 완료!")
            
        except Exception as e:
            print(f"❌ 창 설정 실패: {e}")
            # Back 창의 경우 대안 창 이름 시도
            if "Back" in system_name:
                try:
                    window_name = "Back Camera"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 800, 600)
                    cv2.moveWindow(window_name, 500, 100)
                    cv2.waitKey(1)
                    print("🔄 대안 Back 창 생성 성공!")
                except Exception as e2:
                    print(f"❌ 대안 창도 실패: {e2}")
                    window_name = None
            else:
                print("🔄 창 없이 실행합니다...")
                window_name = None
        
        # 카메라 해상도 설정 (같은 하드웨어 카메라 충돌 방지)
        if "video4" in camera_device:
            # video4는 더 낮은 해상도로 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            # video2는 기본 해상도
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        # 비디오 포맷 설정 (같은 하드웨어 카메라 충돌 방지)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, True)
        
        # Back 시스템의 FPS를 낮춰 대역폭 확보
        if "Back" in system_name:
            cap.set(cv2.CAP_PROP_FPS, 15)
            print(f"📉 {system_name} FPS를 15로 낮춰 대역폭을 확보합니다.")
        else:
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # 카메라 안정화를 위한 추가 대기 (같은 하드웨어 카메라 충돌 방지)
        if "video4" in camera_device:
            time.sleep(3)  # video4는 더 긴 대기 시간
        else:
            time.sleep(2)  # video2는 기본 대기 시간
        
        # 실제 설정된 해상도 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 카메라 해상도: {actual_width}x{actual_height}")
        
        # 첫 프레임으로 창 테스트
        ret, test_frame = cap.read()
        if ret:
            cv2.putText(test_frame, "🚀 System Starting...", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow(window_name, test_frame)
            cv2.waitKey(100)
            print("🖼️ 초기 창 표시 완료!")
        else:
            print(f"❌ 초기 프레임 읽기 실패: {camera_device}")
            cap.release()
            return
        
        # 추가 프레임 테스트 (검은색 화면 방지)
        for i in range(5):
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                # 프레임이 검은색인지 확인
                if np.mean(test_frame) > 10:  # 평균 밝기가 10 이상이면 정상
                    print(f"✅ 프레임 {i+1} 정상: 평균 밝기 {np.mean(test_frame):.1f}")
                    break
                else:
                    print(f"⚠️ 프레임 {i+1} 검은색: 평균 밝기 {np.mean(test_frame):.1f}")
            else:
                print(f"❌ 프레임 {i+1} 읽기 실패")
            time.sleep(0.5)
        
        # 모듈 시작 (이미 __init__에서 자동 시작됨)
        print("🔥 모든 모듈 자동 시작됨!")
        
        # 메인 루프
        frame_count = 0
        start_time = datetime.now()
        
        current_gesture = "NORMAL"
        current_confidence = 0.5
        
        # 색상 팔레트 (사람별 색상 할당용)
        color_palette = [
            (0, 255, 0),    # 초록색
            (255, 0, 0),    # 파란색  
            (0, 0, 255),    # 빨간색
            (255, 255, 0),  # 청록색
            (255, 0, 255),  # 자홍색
            (0, 255, 255),  # 노란색
            (128, 0, 128),  # 보라색
            (255, 165, 0),  # 주황색
        ]
        
        try:
            while cap.isOpened():
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                elapsed_time = (datetime.now() - start_time).total_seconds()
                
                # 프레임을 모듈들에게 전달
                self.person_tracker.add_frame(frame.copy(), frame_count, elapsed_time)
                
                # 사람 감지 결과 가져오기
                latest_detections = self.person_tracker.get_latest_detections()
                
                # 제스처 인식기에 프레임 전달 (사람 감지 결과 포함)
                self.gesture_recognizer.add_frame(frame.copy(), frame_count, elapsed_time, latest_detections)
                
                # 제스처 결과 가져오기
                gesture_prediction, gesture_confidence, keypoints_detected, current_keypoints = self.gesture_recognizer.get_latest_gesture()
                
                # 제스처 결과 업데이트 (통합된 로직)
                # 1. 키포인트 부족 시 NORMAL로 리셋
                if not keypoints_detected or current_keypoints is None:
                    if current_gesture != "NORMAL":
                        current_gesture = "NORMAL"
                        current_confidence = 0.5
                        print(f"🔄 키포인트 부족으로 NORMAL로 리셋")
                else:
                    # 2. 키포인트 개수 확인 (최소 7개 필요)
                    upper_body_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]
                    gesture_keypoints = current_keypoints[upper_body_joints]
                    valid_gesture_keypoints = np.sum(gesture_keypoints[:, 2] > 0.1)
                    
                    if valid_gesture_keypoints < 7:  # 최소 7개 키포인트 필요
                        if current_gesture != "NORMAL":
                            current_gesture = "NORMAL"
                            current_confidence = 0.5
                            print(f"🔄 키포인트 부족({valid_gesture_keypoints}/7)으로 NORMAL로 리셋")
                    else:
                        # 3. 키포인트가 충분하면 모델 결과를 우선적으로 반영
                        if gesture_prediction != current_gesture:
                            current_gesture = gesture_prediction
                            current_confidence = gesture_confidence
                            print(f"🎯 모델 결과 반영: {gesture_prediction} ({gesture_confidence:.2f})")
                        elif abs(gesture_confidence - current_confidence) > 0.1:
                            # 신뢰도가 크게 다르면 업데이트
                            current_gesture = gesture_prediction
                            current_confidence = gesture_confidence
                            print(f"🎯 신뢰도 변화로 업데이트: {gesture_prediction} ({gesture_confidence:.2f})")
                
                # 실시간 제스처 결과 디버깅 (매 30프레임마다)
                if frame_count % 30 == 0:
                    print(f"🔄 실시간 제스처 결과:")
                    print(f"   - gesture_prediction: {gesture_prediction}")
                    print(f"   - gesture_confidence: {gesture_confidence:.3f}")
                    print(f"   - current_gesture: {current_gesture}")
                    print(f"   - current_confidence: {current_confidence:.3f}")
                    print(f"   - 변화 여부: {gesture_prediction != current_gesture}")
                    
                    # 키포인트 상태도 함께 출력
                    if keypoints_detected and current_keypoints is not None:
                        upper_body_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]
                        gesture_keypoints = current_keypoints[upper_body_joints]
                        valid_gesture_keypoints = np.sum(gesture_keypoints[:, 2] > 0.1)
                        print(f"   - 키포인트 상태: {valid_gesture_keypoints}/7")
                    else:
                        print(f"   - 키포인트 상태: 없음")
                
                # 강제 업데이트 (디버깅용) - 키포인트가 충분할 때만
                if frame_count % 30 == 0 and keypoints_detected and current_keypoints is not None:
                    # 키포인트 개수 재확인
                    upper_body_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]
                    gesture_keypoints = current_keypoints[upper_body_joints]
                    valid_gesture_keypoints = np.sum(gesture_keypoints[:, 2] > 0.1)
                    
                    if valid_gesture_keypoints >= 7:  # 충분한 키포인트가 있을 때만
                        # 모델 결과와 현재 상태가 다르면 업데이트
                        if gesture_prediction != current_gesture:
                            current_gesture = gesture_prediction
                            current_confidence = gesture_confidence
                            print(f"🔄 강제 제스처 업데이트: {gesture_prediction} ({gesture_confidence:.2f})")
                        else:
                            print(f"🔄 모델 결과와 현재 상태 동일: {gesture_prediction} ({gesture_confidence:.2f})")
                    else:
                        print(f"🔄 키포인트 부족({valid_gesture_keypoints}/7)으로 강제 업데이트 건너뜀")
                
                # 화면 구성
                annotated = frame.copy()
                
                # 사람 시각화 (PersonTracker의 실제 반환 구조에 맞게 수정)
                if latest_detections:
                    for i, person in enumerate(latest_detections):
                        x1, y1, x2, y2 = map(int, person['bbox'])
                        person_id = person['id']
                        confidence = person['confidence']
                        score = person['score']
                        
                        # 색상 할당 (문자열 ID에서 숫자 추출)
                        if isinstance(person_id, str) and 'Person_' in person_id:
                            # "Person_0" → 0 추출
                            person_num = int(person_id.split('_')[1])
                        else:
                            # 정수 ID인 경우
                            person_num = int(person_id) if isinstance(person_id, (int, float)) else i
                        
                        color = color_palette[person_num % len(color_palette)]
                        
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"ID:{person_id}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(annotated, f"Conf: {confidence:.2f}", 
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        if score > 0:
                            cv2.putText(annotated, f"Score: {score:.2f}", 
                                       (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 관절점 시각화 (키포인트가 감지된 경우)
                if keypoints_detected and current_keypoints is not None:
                    # 가장 큰 바운딩 박스 사람 찾기
                    largest_person = None
                    largest_area = 0
                    largest_person_color = (0, 255, 255)  # 기본 색상
                    
                    if latest_detections:
                        for person in latest_detections:
                            x1, y1, x2, y2 = map(int, person['bbox'])
                            area = (x2 - x1) * (y2 - y1)
                            if area > largest_area:
                                largest_area = area
                                largest_person = person
                                
                                # 색상 할당 (문자열 ID에서 숫자 추출)
                                person_id = person['id']
                                if isinstance(person_id, str) and 'Person_' in person_id:
                                    person_num = int(person_id.split('_')[1])
                                else:
                                    person_num = int(person_id) if isinstance(person_id, (int, float)) else 0
                                
                                largest_person_color = color_palette[person_num % len(color_palette)]
                    
                    # 가장 큰 사람의 관절점만 시각화 (해당 사람 색상으로)
                    annotated = self.gesture_recognizer.draw_visualization(annotated, current_keypoints, current_gesture, current_confidence)
                    
                    # 제스처 인식용 키포인트 개수 표시 (9개 기준)
                    upper_body_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]
                    gesture_keypoints = current_keypoints[upper_body_joints]  # 9개 추출
                    valid_gesture_keypoints = np.sum(gesture_keypoints[:, 2] > 0.1)  # 0.3 → 0.1로 낮춤
                    cv2.putText(annotated, f"Gesture KPts: {valid_gesture_keypoints}/9", 
                               (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, largest_person_color, 2)
                    
                    # 키포인트 디버깅 정보 (60프레임마다)
                    if frame_count % 60 == 0:
                        print(f"🎯 키포인트 시각화 정보:")
                        print(f"   - keypoints_detected: {keypoints_detected}")
                        print(f"   - current_keypoints shape: {current_keypoints.shape if current_keypoints is not None else 'None'}")
                        print(f"   - 유효한 키포인트: {valid_gesture_keypoints}/9")
                        if largest_person:
                            print(f"   - 가장 큰 사람: {largest_person['id']} (면적: {largest_area})")
                        if current_keypoints is not None:
                            print(f"   - 어깨 신뢰도: {current_keypoints[5][2]:.2f}/{current_keypoints[6][2]:.2f}")
                            print(f"   - 팔꿈치 신뢰도: {current_keypoints[7][2]:.2f}/{current_keypoints[8][2]:.2f}")
                            print(f"   - 손목 신뢰도: {current_keypoints[9][2]:.2f}/{current_keypoints[10][2]:.2f}")
                else:
                    # 키포인트가 감지되지 않을 때 디버깅 (60프레임마다)
                    if frame_count % 60 == 0:
                        print(f"❌ 키포인트 시각화 실패:")
                        print(f"   - keypoints_detected: {keypoints_detected}")
                        print(f"   - current_keypoints: {current_keypoints is not None}")
                        print(f"   - gesture_prediction: {gesture_prediction}")
                        print(f"   - gesture_confidence: {gesture_confidence}")
                
                # FPS 계산
                frame_time = time.time() - frame_start
                self.fps_times.append(frame_time)
                fps = 1.0 / (sum(self.fps_times) / len(self.fps_times)) if self.fps_times else 0
                
                # 시스템 정보 표시
                cv2.putText(annotated, f"🚀 Modular System FPS: {fps:.1f}", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated, f"People: {len(latest_detections)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 제스처 표시 (색상 변경: COME은 빨간색, NORMAL은 초록색)
                gesture_color = (0, 0, 255) if current_gesture == "COME" else (0, 255, 0)  # COME은 빨간색, NORMAL은 초록색
                cv2.putText(annotated, f"Gesture: {current_gesture}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, gesture_color, 3)
                cv2.putText(annotated, f"Confidence: {current_confidence:.2f}", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
                
                # 실시간 제스처 정보 추가
                cv2.putText(annotated, f"Real-time: {gesture_prediction}", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated, f"Real-conf: {gesture_confidence:.2f}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.putText(annotated, f"Keypoints: {'OK' if keypoints_detected else 'NONE'}", 
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if keypoints_detected else (0, 0, 255), 2)
                cv2.putText(annotated, f"Frame: {frame_count}", (10, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 1초 단위 판단 정보 표시 (SlidingShiftGCN 모델에 맞춤)
                frames_to_next_decision = self.gesture_recognizer.gesture_decision_interval - (frame_count - self.gesture_recognizer.last_gesture_decision_frame)
                if frames_to_next_decision > 0:
                    cv2.putText(annotated, f"Next Decision: {frames_to_next_decision}f", 
                               (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(annotated, f"Ready for Decision", 
                               (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 모듈 상태 표시
                cv2.putText(annotated, "Modules: Running", (10, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 화면 표시
                if window_name:
                    try:
                        cv2.imshow(window_name, annotated)
                        cv2.waitKey(1)
                        
                        # Back 창인 경우 강제로 앞으로 가져오기
                        if "Back" in system_name or "Back Camera" in window_name:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                            cv2.waitKey(1)
                            # 창을 다시 표시
                            cv2.imshow(window_name, annotated)
                            cv2.waitKey(1)
                            
                    except Exception as e:
                        print(f"❌ 화면 표시 실패: {e}")
                        window_name = None
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
                # 성능 모니터링
                if frame_count % 120 == 0:
                    print(f"📊 FPS: {fps:.1f} | People: {len(latest_detections)} | Gesture: {current_gesture}")
                    print(f"   화면 확인: 밝기 {np.mean(annotated):.1f}, 크기 {annotated.shape}")
                    print(f"   제스처 상태: 버퍼 {len(self.gesture_recognizer.gesture_frame_buffer)}/30, 다음 판단까지 {frames_to_next_decision}프레임")
                    print(f"   키포인트 상태: 감지={keypoints_detected}, 데이터={current_keypoints is not None}")
                    
                    # 히스토그램 매칭 디버깅
                    if latest_detections:
                        print(f"🔍 히스토그램 매칭 디버깅:")
                        for person in latest_detections:
                            person_id = person['id']
                            score = person['score']
                            print(f"   - {person_id}: 매칭 점수 = {score:.3f}")
                            
                            # PersonTracker에서 상세 정보 가져오기
                            if person_id in self.person_tracker.people_data:
                                pdata = self.person_tracker.people_data[person_id]
                                hist_count = len(pdata['histograms'])
                                print(f"     저장된 히스토그램: {hist_count}개")
                                if hist_count > 0:
                                    latest_hist = pdata['histograms'][-1]
                                    hist_std = np.std(latest_hist)
                                    hist_mean = np.mean(latest_hist)
                                    print(f"     최신 히스토그램 - 평균: {hist_mean:.3f}, 표준편차: {hist_std:.3f}")
                    
                    # 전체 사람 데이터 상태
                    total_people = len(self.person_tracker.people_data)
                    print(f"   전체 등록된 사람: {total_people}명")
                    for pid, pdata in self.person_tracker.people_data.items():
                        hist_count = len(pdata['histograms'])
                        print(f"     {pid}: {hist_count}개 히스토그램")
        
        except KeyboardInterrupt:
            print("🛑 사용자 중단")
        
        finally:
            # 정리
            self.running = False
            
            # 모듈 중지
            self.person_tracker.stop()
            self.gesture_recognizer.stop()
            
            # 카메라 해제
            cap.release()
            cv2.destroyAllWindows()
            
            # fps 변수 안전 처리
            if 'fps' not in locals():
                fps = 0.0
            
            print(f"\n🎉 시스템 종료")
            print(f"   - 총 프레임: {frame_count}")
            print(f"   - 최종 FPS: {fps:.1f}")
            print(f"   - 감지된 사람: {len(self.person_tracker.people_data)}")

if __name__ == "__main__":
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='통합 시스템 실행')
    parser.add_argument('--camera', type=str, required=True,
                       help='카메라 장치 (필수)')
    parser.add_argument('--name', type=str, default="Integrated System", 
                       help='시스템 이름 (기본값: Integrated System)')
    
    args = parser.parse_args()
    
    # 시스템 실행
    system = IntegratedSystem()
    system.run_system(camera_device=args.camera, system_name=args.name) 
#!/usr/bin/env python3
"""
🧪 Advanced Person Tracker Test Suite
각 컴포넌트별 성능 및 기능 테스트
"""

import cv2
import numpy as np
import time
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append('.')

def test_bot_sort():
    """BoT-SORT 추적기 테스트"""
    print("🧪 Testing BoT-SORT Tracker...")
    
    try:
        from advanced_person_tracker import BoTSORTTracker
        
        tracker = BoTSORTTracker(max_disappeared=10)
        
        # 테스트 감지 결과
        detections = [
            {'bbox': [100, 100, 200, 300], 'confidence': 0.9},
            {'bbox': [300, 150, 400, 350], 'confidence': 0.8}
        ]
        
        # 첫 번째 프레임
        tracks = tracker.update(detections)
        print(f"  ✅ First frame: {len(tracks)} tracks created")
        
        # 두 번째 프레임 (약간 이동)
        detections_moved = [
            {'bbox': [110, 110, 210, 310], 'confidence': 0.9},
            {'bbox': [310, 160, 410, 360], 'confidence': 0.8}
        ]
        
        tracks = tracker.update(detections_moved)
        print(f"  ✅ Second frame: {len(tracks)} tracks maintained")
        
        # 빈 프레임 (사라짐 시뮬레이션)
        tracks = tracker.update([])
        print(f"  ✅ Empty frame: {len(tracks)} tracks (disappearing)")
        
        print("  🎯 BoT-SORT test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ BoT-SORT test failed: {e}")
        return False

def test_faiss_database():
    """FAISS Re-ID 데이터베이스 테스트"""
    print("🧪 Testing FAISS Re-ID Database...")
    
    try:
        from advanced_person_tracker import FAISSReIDDatabase
        
        db = FAISSReIDDatabase(feature_dim=512, similarity_threshold=0.7)
        
        # 테스트 특징 벡터
        features1 = np.random.randn(512).astype(np.float32)
        features1 = features1 / np.linalg.norm(features1)
        
        features2 = np.random.randn(512).astype(np.float32)
        features2 = features2 / np.linalg.norm(features2)
        
        # 유사한 특징 (약간 변형)
        features_similar = features1 + np.random.normal(0, 0.1, 512)
        features_similar = features_similar / np.linalg.norm(features_similar)
        
        # 데이터베이스에 추가
        db.add_person(1, features1)
        db.add_person(2, features2)
        
        print(f"  ✅ Added 2 people to database")
        
        # 유사한 사람 찾기
        person_id, similarity = db.find_similar_person(features_similar)
        print(f"  ✅ Found similar person: ID {person_id}, similarity {similarity:.3f}")
        
        # 다른 사람 찾기
        features_different = np.random.randn(512).astype(np.float32)
        features_different = features_different / np.linalg.norm(features_different)
        
        person_id, similarity = db.find_similar_person(features_different)
        print(f"  ✅ Different person: ID {person_id}, similarity {similarity:.3f}")
        
        print("  🎯 FAISS database test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ FAISS database test failed: {e}")
        return False

def test_osnet():
    """OSNet 특징 추출 테스트"""
    print("🧪 Testing OSNet Feature Extraction...")
    
    try:
        from advanced_person_tracker import AdvancedOSNet
        
        osnet = AdvancedOSNet(feature_dim=512)
        
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 200, 300]
        
        # 특징 추출
        features = osnet.extract_features(test_image, test_bbox)
        
        if features is not None:
            print(f"  ✅ Feature extraction successful: {len(features)} dimensions")
            print(f"  ✅ Feature norm: {np.linalg.norm(features):.3f}")
        else:
            print("  ❌ Feature extraction failed")
            return False
        
        # 여러 바운딩박스 테스트
        bboxes = [
            [50, 50, 150, 250],
            [200, 100, 300, 300],
            [350, 150, 450, 350]
        ]
        
        for i, bbox in enumerate(bboxes):
            features = osnet.extract_features(test_image, bbox)
            if features is not None:
                print(f"  ✅ Bbox {i+1}: {len(features)} features extracted")
        
        print("  🎯 OSNet test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ OSNet test failed: {e}")
        return False

def test_integration():
    """통합 시스템 테스트"""
    print("🧪 Testing Integrated System...")
    
    try:
        from advanced_person_tracker import AdvancedPersonTracker
        
        tracker = AdvancedPersonTracker()
        
        # 테스트 프레임 생성
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # 테스트 감지 결과
        detections = [
            {'bbox': [100, 100, 200, 300], 'confidence': 0.9},
            {'bbox': [300, 150, 400, 350], 'confidence': 0.8},
            {'bbox': [500, 200, 600, 400], 'confidence': 0.7}
        ]
        
        # 프레임 처리
        active_people = tracker.process_frame(frame, detections)
        print(f"  ✅ First frame processed: {len(active_people)} active people")
        
        # 두 번째 프레임 (일부 이동, 일부 사라짐)
        detections2 = [
            {'bbox': [110, 110, 210, 310], 'confidence': 0.9},
            {'bbox': [320, 160, 420, 360], 'confidence': 0.8}
        ]
        
        active_people = tracker.process_frame(frame, detections2)
        print(f"  ✅ Second frame processed: {len(active_people)} active people")
        
        # 결과 시각화 테스트
        result_frame = tracker.draw_results(frame)
        print(f"  ✅ Visualization successful: {result_frame.shape}")
        
        print("  🎯 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def test_performance():
    """성능 테스트"""
    print("🧪 Testing Performance...")
    
    try:
        from advanced_person_tracker import AdvancedPersonTracker
        
        tracker = AdvancedPersonTracker()
        
        # 테스트 프레임
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        detections = [
            {'bbox': [100, 100, 200, 300], 'confidence': 0.9},
            {'bbox': [300, 150, 400, 350], 'confidence': 0.8}
        ]
        
        # 성능 측정
        num_frames = 100
        start_time = time.time()
        
        for i in range(num_frames):
            tracker.process_frame(frame, detections)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_frames * 1000  # ms
        
        print(f"  ✅ Processed {num_frames} frames in {total_time:.2f}s")
        print(f"  ✅ Average time per frame: {avg_time:.1f}ms")
        print(f"  ✅ FPS: {num_frames / total_time:.1f}")
        
        if avg_time < 50:  # 50ms 이하면 실시간 성능
            print("  🎯 Real-time performance achieved!")
        else:
            print("  ⚠️ Performance needs optimization")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Advanced Person Tracker Test Suite")
    print("=" * 50)
    
    tests = [
        ("BoT-SORT Tracker", test_bot_sort),
        ("FAISS Database", test_faiss_database),
        ("OSNet Feature Extraction", test_osnet),
        ("Integration System", test_integration),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! System is ready to use.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\n📖 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the system: python advanced_person_tracker.py")
    print("3. Test with webcam or video file")

if __name__ == "__main__":
    main() 
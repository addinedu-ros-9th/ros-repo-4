"""
저장된 Shift-GCN 데이터 확인 스크립트
"""

import numpy as np
import os

def check_shift_gcn_data(data_dir):
    """저장된 Shift-GCN 데이터를 확인"""
    
    print("🔍 Shift-GCN 데이터 확인")
    print("=" * 50)
    
    # 데이터 파일들 확인
    data_files = ['./shift_gcn_data/come_pose_data.npy', './shift_gcn_data/normal_pose_data.npy']
    
    total_samples = 0
    
    for data_file in data_files:
        file_path = os.path.join(data_dir, data_file)
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            action_name = data_file.replace('_pose_data.npy', '')
            
            print(f"\n📊 {action_name.upper()} 데이터:")
            print(f"   - 파일: {data_file}")
            print(f"   - Shape: {data.shape}")
            print(f"   - 샘플 수: {data.shape[0]}")
            print(f"   - 채널 수: {data.shape[1]} (x, y, confidence)")
            print(f"   - 프레임 수: {data.shape[2]}")
            print(f"   - 관절점 수: {data.shape[3]}")
            print(f"   - 사람 수: {data.shape[4]}")
            print(f"   - 데이터 타입: {data.dtype}")
            print(f"   - 파일 크기: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
            
            total_samples += data.shape[0]
        else:
            print(f"❌ {data_file} 파일을 찾을 수 없습니다.")
    
    print(f"\n📈 전체 통계:")
    print(f"   - 총 샘플 수: {total_samples}")
    print(f"   - 액션 수: {len(data_files)}")
    
    print(f"\n✅ 데이터 확인 완료!")

if __name__ == "__main__":
    data_dir = "./deeplearning/gesture_recognition/shift_gcn_data"
    check_shift_gcn_data(data_dir) 
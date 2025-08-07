import torch
import numpy as np
from train_shift_gcn_simple import SimpleShiftGCN

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleShiftGCN(num_classes=2, num_joints=9)
    model.load_state_dict(torch.load('models/simple_shift_gcn_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    come_data = np.load('shift_gcn_data/come_pose_data.npy')
    normal_data = np.load('shift_gcn_data/normal_pose_data.npy')
    
    print('=== 모델 테스트 ===')
    
    with torch.no_grad():
        for i in range(5):
            come_input = torch.FloatTensor(come_data[i:i+1]).to(device)
            normal_input = torch.FloatTensor(normal_data[i:i+1]).to(device)
            
            come_output = model(come_input)
            normal_output = model(normal_input)
            
            come_prob = torch.softmax(come_output, dim=1)
            normal_prob = torch.softmax(normal_output, dim=1)
            
            print(f'샘플 {i+1}:')
            print(f'  COME 입력 -> 출력: {come_output[0].cpu().numpy()}, 확률: {come_prob[0].cpu().numpy()}')
            print(f'  NORMAL 입력 -> 출력: {normal_output[0].cpu().numpy()}, 확률: {normal_prob[0].cpu().numpy()}')
            print()

if __name__ == "__main__":
    test_model() 
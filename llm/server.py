#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, Response, stream_template
from flask_cors import CORS
import json
import sys
import time
import io
import torch
from robot_system import RobotSystem

app = Flask(__name__)
CORS(app)  # CORS 활성화

print("🚀 PyTorch 기반 RobotSystem 서버 초기화 중...")

# GPU 환경 확인
if torch.cuda.is_available():
    print(f"✅ GPU 모드: {torch.cuda.get_device_name(0)}")
    print(f"📊 CUDA 버전: {torch.version.cuda}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✅ Apple Silicon MPS 모드")
else:
    print("💻 CPU 모드")

print(f"🔧 PyTorch 버전: {torch.__version__}")

# RobotSystem 인스턴스 생성 (PyTorch 최적화 설정)
robot = RobotSystem(
    use_real_model=True, 
    use_reasoning=False, 
    fast_mode=True, 
    debug_mode=False
)

@app.route('/api/chat', methods=['POST'])
def chat():
    """일반 채팅 API - 최종 응답만 반환"""
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': '메시지가 없습니다'}), 400
        
        print(f"📱 Android에서 메시지 수신: {user_input}")
        
        # RobotSystem 처리 (최종 응답만 반환)
        response = robot.process_user_input(user_input)
        print(f"🔍 일반 채팅 서버에서 받은 응답: '{response}'")
        print(f"🔍 일반 채팅 응답 길이: {len(response) if response else 0}")
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"❌ 처리 오류: {e}")
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    # GPU 상태 정보 포함
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_available': True,
            'gpu_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'memory_allocated': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
            'memory_reserved': f"{torch.cuda.memory_reserved() / 1024**3:.1f}GB"
        }
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_info = {
            'gpu_available': True,
            'gpu_name': 'Apple Silicon MPS',
            'device_type': 'mps'
        }
    else:
        gpu_info = {
            'gpu_available': False,
            'device_type': 'cpu'
        }
    
    return jsonify({
        'status': 'ok', 
        'message': 'PyTorch 기반 RobotSystem 서버가 정상 동작 중입니다',
        'pytorch_version': torch.__version__,
        'gpu_info': gpu_info
    })

@app.route('/api/stream', methods=['POST'])
def stream_chat():
    """실시간 스트리밍 채팅 (Server-Sent Events) - PyTorch 최적화"""
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': '메시지가 없습니다'}), 400
        
        print(f"📱 Android에서 스트리밍 요청: {user_input}")
        
        def generate_stream():
            try:
                # RobotSystem 처리 (PyTorch 기반 스트리밍)
                response = robot.process_user_input(user_input)
                print(f"🔍 서버에서 받은 응답: '{response}'")
                print(f"🔍 응답 길이: {len(response) if response else 0}")
                print(f"🔍 응답이 비어있나? {not response}")
                print(f"🔍 응답이 공백만 있나? {not response.strip() if response else True}")
                
                if response and response.strip():
                    # 최종 응답을 단어 단위로 스트리밍 (공백 보존)
                    # 공백을 기준으로 분할하되 공백도 포함
                    import re
                    words_with_spaces = re.split(r'(\s+)', response)
                    
                    for i, word in enumerate(words_with_spaces):
                        if word:  # 빈 문자열이 아닌 경우만
                            yield f"data: {json.dumps({'type': 'stream', 'content': word, 'index': i})}\n\n"
                            time.sleep(0.05)  # 자연스러운 스트리밍 속도
                    
                    # 완료 신호
                    yield f"data: {json.dumps({'type': 'complete', 'content': response})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'content': '응답을 생성할 수 없습니다'})}\n\n"
                    
            except Exception as e:
                error_msg = f"스트리밍 오류: {str(e)}"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        
        return Response(generate_stream(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': f'스트리밍 서버 오류: {str(e)}'}), 500

@app.route('/api/token_stream', methods=['POST'])
def token_stream_chat():
    """토큰 단위 실시간 스트리밍 - PyTorch 최적화"""
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': '메시지가 없습니다'}), 400
        
        print(f"📱 Android에서 토큰 스트리밍 요청: {user_input}")
        
        def generate_token_stream():
            try:
                # RobotSystem 처리 (PyTorch 기반 토큰 스트리밍)
                response = robot.process_user_input(user_input)
                print(f"🔍 토큰 스트리밍 서버에서 받은 응답: '{response}'")
                print(f"🔍 토큰 스트리밍 응답 길이: {len(response) if response else 0}")
                
                if response and response.strip():
                    # 최종 응답을 문자 단위로 스트리밍 (공백 보존)
                    for i, char in enumerate(response):
                        # 공백 문자도 포함하여 전송
                        yield f"data: {json.dumps({'type': 'token', 'content': char, 'index': i})}\n\n"
                        time.sleep(0.01)  # 빠른 토큰 스트리밍
                    
                    # 완료 신호
                    yield f"data: {json.dumps({'type': 'complete', 'content': response})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'content': '응답을 생성할 수 없습니다'})}\n\n"
                    
            except Exception as e:
                error_msg = f"토큰 스트리밍 오류: {str(e)}"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        
        return Response(generate_token_stream(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': f'토큰 스트리밍 서버 오류: {str(e)}'}), 500

@app.route('/api/gpu_status', methods=['GET'])
def gpu_status():
    """GPU 상태 상세 정보"""
    try:
        status = {}
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            status = {
                'cuda_available': True,
                'device_count': device_count,
                'current_device': current_device,
                'device_name': torch.cuda.get_device_name(current_device),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'total_memory': f"{torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f}GB",
                'allocated_memory': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
                'reserved_memory': f"{torch.cuda.memory_reserved() / 1024**3:.1f}GB",
                'free_memory': f"{(torch.cuda.get_device_properties(current_device).total_memory - torch.cuda.memory_reserved()) / 1024**3:.1f}GB"
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            status = {
                'mps_available': True,
                'device_name': 'Apple Silicon MPS',
                'pytorch_version': torch.__version__,
                'device_type': 'mps'
            }
        else:
            status = {
                'gpu_available': False,
                'device_type': 'cpu',
                'pytorch_version': torch.__version__
            }
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': f'GPU 상태 확인 오류: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n🚀 PyTorch 기반 RobotSystem HTTP 서버 시작...")
    print("📱 Android 앱에서 다음 엔드포인트로 접속 가능:")
    print("  - 일반 채팅: http://localhost:5000/api/chat")
    print("  - 실시간 스트리밍: http://localhost:5000/api/stream")
    print("  - 토큰 스트리밍: http://localhost:5000/api/token_stream")
    print("🔍 서버 상태 확인:")
    print("  - 기본 상태: http://localhost:5000/api/health")
    print("  - GPU 상태: http://localhost:5000/api/gpu_status")
    print("⚡ PyTorch 기반 최적화된 스트리밍!")
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"🎮 GPU 가속: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple Silicon MPS 가속")
    else:
        print("💻 CPU 모드 (GPU 가속 없음)")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 
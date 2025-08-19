#!/bin/bash

# AI Server와 Central Server 빌드 및 실행 스크립트

echo "=== ROS2 Hospital Robot System 빌드 및 실행 ==="

# 현재 디렉토리 저장
CURRENT_DIR=$(pwd)
WORKSPACE_ROOT="/home/wonho/ros-repo-4"

cd $WORKSPACE_ROOT

echo "1. 인터페이스 빌드중..."
colcon build --packages-select control_interfaces
if [ $? -ne 0 ]; then
    echo "❌ 인터페이스 빌드 실패!"
    exit 1
fi

# echo "2. AI Server 빌드중..."
# colcon build --packages-select ai_server
# if [ $? -ne 0 ]; then
#     echo "❌ AI Server 빌드 실패!"
#     exit 1
# fi

echo "3. Central Server 빌드중..."
colcon build --packages-select central_server  
if [ $? -ne 0 ]; then
    echo "❌ Central Server 빌드 실패!"
    exit 1
fi

echo "✅ 모든 패키지 빌드 완료!"

# 환경 변수 설정
source install/setup.bash

echo ""
echo "=== 실행 옵션 ==="
echo "1. AI Server만 실행: ros2 run ai_server ai_server_node"
echo "2. Central Server만 실행: ros2 run central_server central_server_node"  
echo "3. 모두 실행 (새 터미널 필요):"
echo "   터미널 1: ros2 run central_server central_server_node"
echo "   터미널 2: ros2 run ai_server ai_server_node"
echo ""

# 원래 디렉토리로 복원
cd $CURRENT_DIR

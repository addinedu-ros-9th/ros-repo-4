#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Any

class RobotConfig:
    """로봇 시스템 설정"""
    
    # DB 설정
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', 'heR@491!'),
        'database': os.getenv('DB_NAME', 'HeroDB'),
        'charset': 'utf8mb4'
    }
    
    # HTTP 상태 코드 매핑
    HTTP_STATUS_CODES = {
        # 성공
        'SUCCESS': 200,
        
        # 클라이언트 에러
        'BAD_REQUEST': 400,
        'UNAUTHORIZED': 401,
        'NOT_FOUND': 404,
        'METHOD_NOT_ALLOWED': 405,
        
        # 서버 에러
        'INTERNAL_SERVER_ERROR': 500,
        'SERVICE_UNAVAILABLE': 503
    }
    
    # 에러 메시지
    ERROR_MESSAGES = {
        400: "잘못된 요청입니다.",
        401: "요청한 정보를 찾을 수 없거나 응답에 실패했습니다.",
        404: "요청한 리소스를 찾을 수 없습니다.",
        405: "허용되지 않는 메소드입니다.",
        500: "서버 내부 오류가 발생했습니다.",
        503: "서비스를 사용할 수 없습니다."
    }
    
    # 로봇 초기 설정
    ROBOT_INITIAL_STATUS = {
        "current_location": "중앙 대기 공간",
        "is_moving": False,
        "speed": "normal"
    }
    
    # 시설 매칭 설정
    FACILITY_MATCHING = {
        "max_similarity_score": 0.8,
        "min_common_chars": 2
    }
    
    # 네비게이션 설정
    NAVIGATION = {
        "simulation_delay": 1.0,  # 초
        "safety_check_delay": 0.5  # 초
    }
    
    # 센트럴 서버 설정
    CENTRAL_SERVER_URL = os.getenv('CENTRAL_SERVER_URL', 'http://localhost:8080') 
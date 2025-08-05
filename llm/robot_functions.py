#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import mysql.connector
import requests
from mysql.connector import Error
from typing import Dict, Any, Optional, Tuple
from config import RobotConfig

class RobotFunctions:
    """병원 안내 로봇의 실제 기능 구현"""
    
    def __init__(self, db_config: Dict[str, Any] = None):
        # DB 연결 설정 (config 파일에서 가져오거나 매개변수로 받음)
        self.db_config = db_config or RobotConfig.DB_CONFIG
        
        # 로봇 상태 (config에서 초기값 가져옴)
        self.robot_status = RobotConfig.ROBOT_INITIAL_STATUS.copy()
    
    def _get_current_position(self) -> Optional[Dict[str, float]]:
        """현재 로봇 위치 가져오기 - 내부 헬퍼 함수"""
        try:
            response = requests.post(
                f"{RobotConfig.CENTRAL_SERVER_URL}/get/robot_location",
                json={"robot_id": 3},
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.RequestException:
            return None
    
    def _get_facilities_from_db(self) -> Tuple[Dict[str, Dict[str, Any]], int]:
        """DB에서 시설 정보 조회 - (facilities, status_code) 반환"""
        connection = None
        cursor = None
        
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            query = "SELECT department_id, department_name, location_x, location_y, yaw FROM department ORDER BY department_id"
            cursor.execute(query)
            results = cursor.fetchall()
            
            facilities = {}
            for row in results:
                department_name = row['department_name']
                location_x = row['location_x']
                location_y = row['location_y']
                
                # 위치 정보 생성 (개선된 로직)
                if location_x < 0:
                    if location_y > 0:
                        location = "왼쪽 상단"
                    elif location_y < 0:
                        location = "왼쪽 하단"
                    else:  # location_y == 0
                        location = "왼쪽 중앙"
                else:  # location_x >= 0
                    if location_y > 0:
                        location = "오른쪽 상단"
                    elif location_y < 0:
                        location = "오른쪽 하단"
                    else:  # location_y == 0
                        location = "오른쪽 중앙"
                
                facilities[department_name] = {
                    "exists": True,
                    "area": department_name,
                    "location": location,
                    "description": f"{department_name}",
                    "department_id": row['department_id'],
                    "x": location_x,
                    "y": location_y,
                    "yaw": row['yaw']
                }
            
            return facilities, RobotConfig.HTTP_STATUS_CODES['SUCCESS']
            
        except Error as e:
            print(f"❌ DB 연결 실패: {e}")
            # 하드코딩된 기본 데이터 반환 (503 에러와 함께)
            return {
                "X-ray": {"exists": True, "area": "X-ray", "location": "왼쪽 상단", "description": "X-ray 검사실"},
                "CT": {"exists": True, "area": "CT", "location": "왼쪽 중앙", "description": "CT 검사실"},
                "초음파": {"exists": True, "area": "초음파", "location": "왼쪽 하단", "description": "초음파 검사실"},
                "뇌종양": {"exists": True, "area": "뇌종양", "location": "오른쪽 상단", "description": "뇌종양 전문 치료"},
                "유방암": {"exists": True, "area": "유방암", "location": "오른쪽 상단", "description": "유방암 전문 치료"},
                "대장암": {"exists": True, "area": "대장암", "location": "오른쪽 하단", "description": "대장암 전문 치료"},
                "위암": {"exists": True, "area": "위암", "location": "오른쪽 하단", "description": "위암 전문 치료"},
                "폐암": {"exists": True, "area": "폐암", "location": "오른쪽 하단", "description": "폐암 전문 치료"},
            }, RobotConfig.HTTP_STATUS_CODES['SERVICE_UNAVAILABLE']
        finally:
            # 리소스 정리
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def _normalize_facility_name(self, name: str) -> str:
        """시설명 정규화 (띄어쓰기, 동의어 처리) - 개선된 버전"""
        # 앞뒤 공백과 특수문자 제거
        normalized = name.strip().replace(" ", "").replace("\t", "").replace("?", "").replace("!", "")
        
        # 소문자로 변환해서 매칭 (한글은 그대로)
        normalized_lower = normalized.lower()
        
        # 확장된 동의어 매핑
        synonyms = {
            # 영상의학과 - 다양한 표현
            "엑스레이": "X-ray",
            "엑스선": "X-ray", 
            "x레이": "X-ray",
            "xray": "X-ray",
            "엑스레이실": "X-ray",
            "씨티": "CT",
            "시티": "CT",
            "ct실": "CT",
            "씨티실": "CT",
            "시티실": "CT",
            "초음파실": "초음파",
            
            # 암센터 - 다양한 표현
            "대장암과": "대장암",
            "위암과": "위암", 
            "폐암과": "폐암",
            "유방암과": "유방암",
            "뇌종양과": "뇌종양",
            
            # 센터 접미사 처리
            "대장암센터": "대장암",
            "위암센터": "위암",
            "폐암센터": "폐암",
            "유방암센터": "유방암",
            "뇌종양센터": "뇌종양",
            
            # 실 접미사 처리
            "대장암실": "대장암",
            "위암실": "위암",
            "폐암실": "폐암",
            "유방암실": "유방암",
            "뇌종양실": "뇌종양",
        }
        
        # 동의어 변환 (대소문자 무관)
        for synonym, standard in synonyms.items():
            if normalized_lower == synonym.lower():
                normalized = standard
                break
                
        return normalized

    def _smart_facility_matching(self, target: str, facilities: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """똑똑한 시설 매칭 - 다단계 매칭"""
        # 1단계: 정규화된 정확한 매칭
        normalized_target = self._normalize_facility_name(target)
        if normalized_target in facilities:
            return normalized_target
            
        # 2단계: DB의 모든 키를 정규화해서 비교
        for facility_name in facilities.keys():
            if self._normalize_facility_name(facility_name) == normalized_target:
                return facility_name
                
        # 3단계: 부분 매칭 (포함 관계)
        for facility_name in facilities.keys():
            normalized_facility = self._normalize_facility_name(facility_name)
            # 타겟이 시설명에 포함되거나, 시설명이 타겟에 포함
            if (normalized_target in normalized_facility or 
                normalized_facility in normalized_target):
                return facility_name
                
        # 4단계: 유사도 기반 매칭 (레벤슈타인 거리 등은 나중에 추가)
        # 현재는 간단한 키워드 매칭
        target_keywords = set(normalized_target.lower())
        best_match = None
        best_score = 0
        
        for facility_name in facilities.keys():
            facility_keywords = set(self._normalize_facility_name(facility_name).lower())
            # 공통 글자 개수로 유사도 계산
            common = len(target_keywords & facility_keywords)
            if common > best_score and common > 0:
                best_score = common
                best_match = facility_name
                
        return best_match

    def query_facility(self, target: str) -> Dict[str, Any]:
        """시설 정보 조회 - DB 기반 매칭 로직"""
        print(f"🔍 시설 조회: {target}")
        
        # 입력 검증
        if not target or not target.strip():
            return {
                "function": "query_facility",
                "facility": target,
                "status_code": RobotConfig.HTTP_STATUS_CODES['BAD_REQUEST'],
                "result": {"error": "bad_request", "message": RobotConfig.ERROR_MESSAGES[400]}
            }
        
        # DB에서 시설 정보 조회
        facilities, db_status = self._get_facilities_from_db()
        
        # 스마트 매칭 사용
        matched_facility = self._smart_facility_matching(target, facilities)
        
        if matched_facility:
            location = facilities[matched_facility]["location"]
            area = facilities[matched_facility]["area"]
            result = f"{location} ({area})"
            print(f"✅ 매칭됨: {target} → {matched_facility} - {result}")
            return {
                "function": "query_facility",
                "facility": target,
                "status_code": RobotConfig.HTTP_STATUS_CODES['SUCCESS'],
                "result": result
            }
        
        # 매칭 실패 시
        print(f"❌ 없음: {target}")
        return {
            "function": "query_facility",
            "facility": target,
            "status_code": RobotConfig.HTTP_STATUS_CODES['NOT_FOUND'],
            "result": {"error": "not_found", "target": target, "message": RobotConfig.ERROR_MESSAGES[404]}
        }
    
    def navigate(self, target: str) -> Dict[str, Any]:
        """네비게이션 시작 - 데이터셋 포맷에 맞춤"""
        print(f"🚶 네비게이션 시작: {target}")
        
        # 입력 검증
        if not target or not target.strip():
            return {
                "function": "navigate",
                "target": target,
                "status_code": RobotConfig.HTTP_STATUS_CODES['BAD_REQUEST'],
                "result": {"error": "bad_request", "message": RobotConfig.ERROR_MESSAGES[400]}
            }
        
        # 목적지 확인 (스마트 매칭 포함)
        facility_result = self.query_facility(target)
        
        # 에러 체크
        if facility_result.get("status_code") != RobotConfig.HTTP_STATUS_CODES['SUCCESS']:
            return {
                "function": "navigate",
                "target": target,
                "status_code": facility_result.get("status_code", RobotConfig.HTTP_STATUS_CODES['NOT_FOUND']),
                "result": facility_result.get("result", {"error": "not_found", "message": RobotConfig.ERROR_MESSAGES[404]})
            }
        
        # 이동 시작 (실제로는 로봇 모터 제어)
        self.robot_status["is_moving"] = True
        destination_location = facility_result["result"]
        
        print(f"🎯 목적지: {target} ({destination_location})")
        print("🤖 이동 중...")
        
        # 시뮬레이션 (실제로는 실제 이동)
        time.sleep(RobotConfig.NAVIGATION["simulation_delay"])
        
        # 도착
        self.robot_status["current_location"] = f"{target} 앞"
        self.robot_status["is_moving"] = False
        
        result_message = f"{target}({destination_location})로 안내해드리겠습니다. 따라오세요!"
        return {
            "function": "navigate",
            "target": target,
            "status_code": RobotConfig.HTTP_STATUS_CODES['SUCCESS'],
            "result": result_message
        }
    
    def get_position(self) -> Dict[str, Any]:
        """현재 위치 조회 - 가장 가까운 department 기준으로 설명"""
        print(f"📍 현재 위치 조회")
        
        try:
            # DB에서 시설 정보 조회 (거리 정보 포함)
            facilities, db_status = self._get_facilities_from_db()
            
            if db_status != RobotConfig.HTTP_STATUS_CODES['SUCCESS']:
                return {
                    "function": "get_position",
                    "status_code": db_status,
                    "result": {"error": "service_unavailable", "message": RobotConfig.ERROR_MESSAGES[503]}
                }
            
            # 현재 로봇 위치 가져오기
            current_position = self._get_current_position()
            
            if not current_position:
                return {
                    "function": "get_position",
                    "status_code": RobotConfig.HTTP_STATUS_CODES['SERVICE_UNAVAILABLE'],
                    "result": {"error": "service_unavailable", "message": RobotConfig.ERROR_MESSAGES[503]}
                }
            
            # 가장 가까운 시설 찾기
            closest_facility = None
            min_distance = float('inf')
            
            robot_x = current_position.get("x", 0)
            robot_y = current_position.get("y", 0)
            
            for facility_name, facility_info in facilities.items():
                fx = facility_info["x"]
                fy = facility_info["y"]
                distance = ((robot_x - fx) ** 2 + (robot_y - fy) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_facility = facility_name
            
            if closest_facility:
                result = f"현재 위치: {closest_facility} 근처"
            else:
                result = "현재 위치: 알 수 없는 위치"
            
            return {
                "function": "get_position",
                "status_code": RobotConfig.HTTP_STATUS_CODES['SUCCESS'],
                "result": result
            }
                
        except Exception as e:
            print(f"❌ 위치 조회 실패: {e}")
            return {
                "function": "get_position",
                "status_code": RobotConfig.HTTP_STATUS_CODES['INTERNAL_SERVER_ERROR'],
                "result": {"error": "internal_error", "message": RobotConfig.ERROR_MESSAGES[500]}
            }
    
    def list_facilities(self) -> Dict[str, Any]:
        """병원 내 모든 시설 목록 조회 - DB 기반"""
        print(f"🏥 전체 시설 목록 조회")
        
        # DB에서 시설 정보 조회
        facilities, db_status = self._get_facilities_from_db()
        
        # 모든 시설 목록
        all_facilities = list(facilities.keys())
        result = f"전체 시설: {', '.join(all_facilities)}"
        print(f"✅ {result}")
        
        return {
            "function": "list_facilities",
            "status_code": RobotConfig.HTTP_STATUS_CODES['SUCCESS'],
            "result": result
        }
    


# 함수 매핑 딕셔너리
FUNCTION_MAP = {
    "query_facility": "query_facility",
    "navigate": "navigate", 
    "get_position": "get_position",
    "list_facilities": "list_facilities"
}

def execute_function(robot: RobotFunctions, function_call: Dict[str, Any]) -> Dict[str, Any]:
    """함수 호출 실행"""
    action = function_call.get("action")
    
    if action not in FUNCTION_MAP:
        return {
            "error": f"알 수 없는 함수: {action}",
            "status_code": RobotConfig.HTTP_STATUS_CODES['METHOD_NOT_ALLOWED']
        }
    
    func_name = FUNCTION_MAP[action]
    func = getattr(robot, func_name)
    
    # 매개변수 추출
    kwargs = {}
    if "target" in function_call:
        kwargs["target"] = function_call["target"]
    
    # 함수 실행
    try:
        result = func(**kwargs)
        return result
    except Exception as e:
        return {
            "error": f"함수 실행 오류: {str(e)}",
            "status_code": RobotConfig.HTTP_STATUS_CODES['INTERNAL_SERVER_ERROR']
        }

if __name__ == "__main__":
    # 테스트
    robot = RobotFunctions()
    
    print("=== 로봇 함수 테스트 ===")
    
    # 시설 조회 테스트
    result1 = robot.query_facility("대장암센터")
    print(f"결과: {result1}\n")
    
    # 네비게이션 테스트
    result2 = robot.navigate("대장암센터") 
    print(f"결과: {result2}\n")
    
    # 상태 조회 테스트
    result3 = robot.get_position()
    print(f"결과: {result3}\n")
    
    # 함수 호출 실행 테스트
    function_call = {"action": "query_facility", "target": "내과"}
    result4 = execute_function(robot, function_call)
    print(f"함수 호출 결과: {result4}")
    
    # 에러 테스트
    print("\n=== 에러 테스트 ===")
    error_result1 = robot.query_facility("")  # 빈 문자열
    print(f"빈 문자열 테스트: {error_result1}")
    
    error_result2 = robot.query_facility("존재하지않는시설")
    print(f"존재하지 않는 시설 테스트: {error_result2}")
    
    error_result3 = execute_function(robot, {"action": "unknown_function"})
    print(f"알 수 없는 함수 테스트: {error_result3}") 
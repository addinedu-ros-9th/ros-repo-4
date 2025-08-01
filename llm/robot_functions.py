#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from typing import Dict, Any, Optional

class RobotFunctions:
    """병원 안내 로봇의 실제 기능 구현"""
    
    def __init__(self, db_path: str = "hospital.db"):
        # 새로운 병원 시설 데이터베이스 (층 개념 제거)
        self.hospital_db = {
            # 영상의학과 (왼쪽 영역)
            "X-ray": {"exists": True, "area": "영상의학과", "location": "왼쪽 상단", "description": "X-ray 검사실"},
            "CT": {"exists": True, "area": "영상의학과", "location": "왼쪽 중앙", "description": "CT 검사실"},
            "초음파": {"exists": True, "area": "영상의학과", "location": "왼쪽 하단", "description": "초음파 검사실"},
            
            # 암센터 (오른쪽 영역)
            "뇌종양": {"exists": True, "area": "암센터", "location": "오른쪽 상단", "description": "뇌종양 전문 치료"},
            "유방암": {"exists": True, "area": "암센터", "location": "오른쪽 상단", "description": "유방암 전문 치료"},
            "대장암": {"exists": True, "area": "암센터", "location": "오른쪽 하단", "description": "대장암 전문 치료"},
            "위암": {"exists": True, "area": "암센터", "location": "오른쪽 하단", "description": "위암 전문 치료"},
            "폐암": {"exists": True, "area": "암센터", "location": "오른쪽 하단", "description": "폐암 전문 치료"},
            
            # 기존 호환성을 위한 별칭
            "대장암센터": {"exists": True, "area": "암센터", "location": "오른쪽 하단", "description": "대장암 전문 치료"},
            "위암센터": {"exists": True, "area": "암센터", "location": "오른쪽 하단", "description": "위암 전문 치료"},
            "폐암센터": {"exists": True, "area": "암센터", "location": "오른쪽 하단", "description": "폐암 전문 치료"},
            "유방암센터": {"exists": True, "area": "암센터", "location": "오른쪽 상단", "description": "유방암 전문 치료"},
            "뇌종양센터": {"exists": True, "area": "암센터", "location": "오른쪽 상단", "description": "뇌종양 전문 치료"},
        }
        
        # 로봇 상태 (중앙 대기 공간에서 시작)
        self.robot_status = {
            "current_location": "중앙 대기 공간",
            "is_moving": False,
            "speed": "normal"
        }
    
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

    def _smart_facility_matching(self, target: str) -> Optional[str]:
        """똑똑한 시설 매칭 - 다단계 매칭"""
        # 1단계: 정규화된 정확한 매칭
        normalized_target = self._normalize_facility_name(target)
        if normalized_target in self.hospital_db:
            return normalized_target
            
        # 2단계: DB의 모든 키를 정규화해서 비교
        for facility_name in self.hospital_db.keys():
            if self._normalize_facility_name(facility_name) == normalized_target:
                return facility_name
                
        # 3단계: 부분 매칭 (포함 관계)
        for facility_name in self.hospital_db.keys():
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
        
        for facility_name in self.hospital_db.keys():
            facility_keywords = set(self._normalize_facility_name(facility_name).lower())
            # 공통 글자 개수로 유사도 계산
            common = len(target_keywords & facility_keywords)
            if common > best_score and common > 0:
                best_score = common
                best_match = facility_name
                
        return best_match

    def query_facility(self, target: str) -> Dict[str, Any]:
        """시설 정보 조회 - 개선된 매칭 로직"""
        print(f"🔍 시설 조회: {target}")
        
        # 스마트 매칭 사용
        matched_facility = self._smart_facility_matching(target)
        
        if matched_facility:
            location = self.hospital_db[matched_facility]["location"]
            area = self.hospital_db[matched_facility]["area"]
            result = f"{location} ({area})"
            print(f"✅ 매칭됨: {target} → {matched_facility} - {result}")
            return {
                "function": "query_facility",
                "facility": target,
                "result": result
            }
        
        # 매칭 실패 시
        print(f"❌ 없음: {target}")
        return {
            "function": "query_facility",
            "facility": target,
            "result": {"error": "not_found", "target": target, "message": f"{target}는 이 병원에 없는 시설입니다."}
        }
    
    def navigate(self, target: str) -> Dict[str, Any]:
        """네비게이션 시작 - 데이터셋 포맷에 맞춤"""
        print(f"🚶 네비게이션 시작: {target}")
        
        # 목적지 확인 (스마트 매칭 포함)
        facility_result = self.query_facility(target)
        
        # 에러 체크
        if isinstance(facility_result.get("result"), dict) and "error" in facility_result["result"]:
            return {
                "function": "navigate",
                "target": target,
                "result": {"error": "not_found", "target": target, "message": f"{target}를 찾을 수 없습니다."}
            }
        
        # 이동 시작 (실제로는 로봇 모터 제어)
        self.robot_status["is_moving"] = True
        destination_location = facility_result["result"]
        
        print(f"🎯 목적지: {target} ({destination_location})")
        print("🤖 이동 중...")
        
        # 시뮬레이션 (실제로는 실제 이동)
        time.sleep(1)  # 이동 시뮬레이션
        
        # 도착
        self.robot_status["current_location"] = f"{target} 앞"
        self.robot_status["is_moving"] = False
        
        result_message = f"{target}({destination_location})로 안내해드리겠습니다. 따라오세요!"
        return {
            "function": "navigate",
            "target": target,
            "result": result_message
        }
    
    def get_position(self) -> Dict[str, Any]:
        """현재 위치 조회 - 데이터셋 포맷에 맞춤"""
        print(f"📍 현재 위치 조회")
        return {
            "function": "get_position",
            "result": f"현재 위치: {self.robot_status['current_location']}"
        }
    
    def system_check(self) -> Dict[str, Any]:
        """시스템 상태 확인 - 데이터셋 포맷에 맞춤"""
        print(f"⚙️ 시스템 상태 확인")
        return {
            "function": "system_check",
            "result": "시스템 상태: 정상"
        }
    
    def stop(self, safety_check: bool = True) -> Dict[str, Any]:
        """로봇 정지 - 데이터셋 포맷에 맞춤"""
        print(f"🛑 로봇 정지 (안전체크: {safety_check})")
        
        if safety_check:
            print("🔒 안전 체크 수행 중...")
            time.sleep(0.5)  # 안전 체크 시뮬레이션
        
        # 로봇 정지 (실제로는 모터 제어)
        self.robot_status["is_moving"] = False
        self.robot_status["speed"] = "stop"
        
        return {
            "function": "stop",
            "result": "로봇이 안전하게 정지했습니다"
        }
    
    def start(self) -> Dict[str, Any]:
        """로봇 시작/재개 - 데이터셋 포맷에 맞춤"""
        print(f"▶️ 로봇 시작/재개")
        
        self.robot_status["speed"] = "normal"
        
        return {
            "function": "start",
            "result": "로봇이 시작되었습니다"
        }
    
    def speed_up(self) -> Dict[str, Any]:
        """속도 증가 - 데이터셋 포맷에 맞춤"""
        print(f"⏩ 속도 증가")
        
        self.robot_status["speed"] = "fast"
        
        return {
            "function": "speed_up",
            "result": "속도가 증가했습니다"
        }
    
    def speed_down(self) -> Dict[str, Any]:
        """속도 감소 - 데이터셋 포맷에 맞춤"""
        print(f"⏪ 속도 감소")
        
        self.robot_status["speed"] = "slow"
        
        return {
            "function": "speed_down",
            "result": "속도가 감소했습니다"
        }
    
    def list_facilities_by_area(self, area: str) -> Dict[str, Any]:
        """구역별 시설 목록 조회 - 데이터셋 포맷에 맞춤"""
        print(f"🏢 {area} 시설 목록 조회")
        
        if area == "영상의학과":
            facilities = ["X-ray", "CT", "초음파"]
            result = f"영상의학과 시설: {', '.join(facilities)}"
        elif area == "암센터":
            facilities = ["뇌종양", "유방암", "대장암", "위암", "폐암"]
            result = f"암센터 시설: {', '.join(facilities)}"
        else:
            result = f"{area}에는 등록된 시설이 없습니다."
        
        print(f"✅ {result}")
        return {
            "function": "list_facilities_by_area",
            "area": area,
            "result": result
        }
    
    def list_facilities_by_floor(self, floor: str) -> Dict[str, Any]:
        """특정 층의 시설 목록 조회 - 층 개념 제거, 데이터셋 포맷에 맞춤"""
        print(f"🏢 {floor} 시설 목록 조회")
        
        # 층 개념 제거 - 모든 시설이 한 공간에 있음
        if floor in ["1층", "1", "1층", "전체", "모든"]:
            # 영상의학과
            imaging_facilities = ["X-ray", "CT", "초음파"]
            # 암센터
            cancer_facilities = ["뇌종양", "유방암", "대장암", "위암", "폐암"]
            
            result = f"영상의학과: {', '.join(imaging_facilities)}\n암센터: {', '.join(cancer_facilities)}"
            print(f"✅ {result}")
        else:
            result = f"{floor}에는 등록된 시설이 없습니다. 모든 시설은 한 공간에 있습니다."
            print(f"❌ {result}")
        
        return {
            "function": "list_facilities_by_floor",
            "floor": floor,
            "result": result
        }
    
    def list_facilities(self) -> Dict[str, Any]:
        """병원 내 모든 시설 목록 조회 - 데이터셋 포맷에 맞춤"""
        print(f"🏥 전체 시설 목록 조회")
        
        # 영상의학과
        imaging_facilities = ["X-ray", "CT", "초음파"]
        # 암센터
        cancer_facilities = ["뇌종양", "유방암", "대장암", "위암", "폐암"]
        
        result = f"영상의학과: {', '.join(imaging_facilities)}\n암센터: {', '.join(cancer_facilities)}"
        print(f"✅ {result}")
        
        return {
            "function": "list_facilities",
            "result": result
        }
    
    def general_response(self, message: str = "", context: str = "", **kwargs) -> str:
        """일반적인 대화 응답 (자연어 텍스트 반환)"""
        print(f"💬 일반 대화 응답 (메시지: {message}, 맥락: {context})")
        
        # 맥락별 응답
        if context == "purpose":
            purpose = kwargs.get("purpose", "")
            if purpose == "검사":
                response_text = "검사라면 영상의학과에 가시는 게 좋겠어요. X-ray, CT, 초음파 중 어떤 검사를 받으시나요?"
            elif purpose == "진료":
                response_text = "진료라면 암센터에 가시는 게 좋겠어요. 뇌종양, 유방암, 대장암, 위암, 폐암 중 어떤 진료를 받으시나요?"
            else:
                response_text = "무엇을 도와드릴까요?"
        
        elif context == "reference":
            reference = kwargs.get("reference", "")
            if reference == "previous":
                response_text = "네, 이전에 말씀하신 곳으로 안내해드리겠습니다."
            elif reference == "same_area":
                response_text = "네, 같은 구역이라서 가까워요. 다른 시설도 안내해드릴까요?"
            else:
                response_text = "무엇을 도와드릴까요?"
        
        else:
            # 메시지 내용에 따른 응답 선택 (데이터셋 패턴 참고)
            message_lower = message.lower().strip()
            
            # 인사말 응답
            greetings = [
                "안녕하세요! 저는 병원 안내 로봇 영웅이입니다. 무엇을 도와드릴까요?",
                "안녕하세요! 어떻게 도와드릴까요?",
                "안녕하세요! 영웅이가 안내해드릴게요. 무엇이 필요하신가요?",
                "안녕하세요! 병원 안내를 도와드리는 영웅이입니다."
            ]
            
            # 감사 응답
            thanks = [
                "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요.",
                "별말씀을요! 언제든 도와드리겠습니다.",
                "고맙다고 하시니 영웅이가 기뻐요! 😊"
            ]
            
            # 로봇 정보 질문 응답
            robot_info = [
                "저는 이 병원의 안내 로봇입니다. 병원 내 시설 안내와 이동을 도와드려요.",
                "병원 시설 안내, 길 찾기, 이동 도움을 제공할 수 있어요. 무엇이 필요하신가요?",
                "영웅이는 병원 안내 로봇입니다. 시설 위치 안내와 이동을 도와드려요."
            ]
            
            # 병원 정보 응답
            hospital_info = [
                "이 병원은 영상의학과와 암센터로 구성되어 있습니다. 영상의학과는 왼쪽에, 암센터는 오른쪽에 있어요.",
                "영상의학과(왼쪽)와 암센터(오른쪽)로 나뉘어 있어요. 어디로 안내해드릴까요?",
                "병원은 영상의학과와 암센터로 구성되어 있습니다. 어느 구역에 관심이 있으신가요?"
            ]
            
            # 화장실 관련 응답
            bathroom = [
                "화장실은 각 구역마다 있습니다. 현재 계신 구역의 화장실로 안내해드릴까요?",
                "각 구역에 화장실이 있어요. 지금 계신 구역 화장실로 갈까요?"
            ]
            
            # 에러 처리 응답
            error_handling = [
                "죄송하지만 질문을 이해하지 못했습니다. 좀 더 자세히 말씀해주시겠어요?",
                "네, 무엇을 도와드릴까요? 편하게 말씀해주세요.",
                "네, 천천히 말씀해주세요. 무엇이 필요하신가요?"
            ]
            
            # 기본 응답
            general = [
                "네, 무엇을 도와드릴까요?",
                "영웅이가 도와드릴게요!",
                "무엇이 필요하신가요?",
                "어떤 도움이 필요하신가요?"
            ]
            
            import random
            
            # 메시지 내용에 따른 응답 선택
            if any(word in message_lower for word in ["안녕", "hello", "hi", "헬로", "하이"]):
                response_text = random.choice(greetings)
            elif any(word in message_lower for word in ["고마워", "고맙", "감사", "thank"]):
                response_text = random.choice(thanks)
            elif any(word in message_lower for word in ["누구야", "뭘 할 수 있어", "무엇을 할 수 있어", "기능이 뭐야", "할 수 있는 일"]):
                response_text = random.choice(robot_info)
            elif any(word in message_lower for word in ["병원", "구조", "어떤 구조", "구성"]):
                response_text = random.choice(hospital_info)
            elif any(word in message_lower for word in ["화장실", "화장실이 어디", "화장실 어디야"]):
                response_text = random.choice(bathroom)
            elif any(word in message_lower for word in ["음", "저기", "아노", "음...", "저기...", "아노..."]):
                response_text = random.choice(error_handling)
            else:
                response_text = random.choice(general)
        
        # 자연어 응답 직접 반환 (general_response는 실제 텍스트 반환)
        return response_text

# 함수 매핑 딕셔너리
FUNCTION_MAP = {
    "query_facility": "query_facility",
    "navigate": "navigate", 
    "get_position": "get_position",
    "system_check": "system_check",
    "stop": "stop",
    "start": "start",
    "speed_up": "speed_up",
    "speed_down": "speed_down",
    "list_facilities": "list_facilities",
    "list_facilities_by_floor": "list_facilities_by_floor",
    "general_response": "general_response"
}

def execute_function(robot: RobotFunctions, function_call: Dict[str, Any]) -> Dict[str, Any]:
    """함수 호출 실행"""
    action = function_call.get("action")
    
    if action not in FUNCTION_MAP:
        return {"error": f"알 수 없는 함수: {action}"}
    
    func_name = FUNCTION_MAP[action]
    func = getattr(robot, func_name)
    
    # 매개변수 추출
    kwargs = {}
    if "target" in function_call:
        kwargs["target"] = function_call["target"]
    if "safety_check" in function_call:
        kwargs["safety_check"] = function_call["safety_check"]
    
    # 함수 실행
    try:
        result = func(**kwargs)
        return result
    except Exception as e:
        return {"error": f"함수 실행 오류: {str(e)}"}

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
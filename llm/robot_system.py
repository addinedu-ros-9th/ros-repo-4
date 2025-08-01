#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sqlite3
import sys
import time
import torch
from typing import Dict, List, Any, Optional
from robot_functions import RobotFunctions

# GPU 감지 및 최적화 함수
def check_gpu_availability():
    """GPU 사용 가능성을 자세히 확인"""
    print("🔍 GPU 환경 확인 중...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        
        # 안전한 메모리 사용량 확인
        try:
            reserved_memory = torch.cuda.memory_reserved(current_device)
            free_memory = total_memory - reserved_memory
        except Exception as e:
            print(f"⚠️ 메모리 정보 확인 실패: {e}")
            reserved_memory = 0
            free_memory = total_memory
        
        print(f"✅ CUDA 사용 가능!")
        print(f"  🎮 GPU 개수: {device_count}")
        print(f"  🎮 현재 GPU: {device_name}")
        print(f"  💾 총 메모리: {total_memory / 1024**3:.1f}GB")
        print(f"  💾 사용 가능: {free_memory / 1024**3:.1f}GB")
        
        # CUDA 버전 확인
        print(f"  🔧 CUDA 버전: {torch.version.cuda}")
        print(f"  🔧 PyTorch 버전: {torch.__version__}")
        
        return True, current_device
    else:
        print("❌ CUDA 사용 불가")
        print("  💻 CPU 모드로 실행됩니다")
        
        # CPU 정보
        print(f"  🔧 PyTorch 버전: {torch.__version__}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  🍎 MPS (Apple Silicon) 감지됨")
            return True, torch.device("mps")
        
        return False, torch.device("cpu")

class CustomStreamer:
    """실시간 토큰 스트리밍을 위한 커스텀 스트리머"""
    
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True, 
                 decode_kwargs=None, fast_mode=False, debug_mode=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.decode_kwargs = decode_kwargs or {}
        self.token_cache = []
        self.print_len = 0
        self.current_length = 0
        self.fast_mode = fast_mode  # 빠른 모드 설정
        self.debug_mode = debug_mode  # 디버그 모드 설정

    def put(self, value):
        """새로운 토큰을 받을 때 호출됨"""
        try:
            # 디버그 정보
            if self.debug_mode:
                print(f"🔍 스트리머 입력: shape={value.shape}, dim={value.dim()}")
            
            # 텐서 차원 안전하게 처리
            if value.dim() == 0:
                # 스칼라를 1차원 텐서로 변환
                value = value.unsqueeze(0)
            elif value.dim() == 1:
                # 1차원 텐서를 2차원으로 변환 (배치 차원 추가)
                value = value.unsqueeze(0)
            
            # 배치 크기 확인
            if value.shape[0] > 1:
                if self.debug_mode:
                    print(f"⚠️ 배치 크기 {value.shape[0]} 무시됨")
                return

            # 빈 텐서 체크
            if value.numel() == 0:
                if self.debug_mode:
                    print("⚠️ 빈 텐서 무시됨")
                return
            
            # 텐서 유효성 체크
            if not torch.is_tensor(value):
                if self.debug_mode:
                    print("⚠️ 텐서가 아닌 객체 무시됨")
                return
                
            # 토큰 시퀀스 추출
            token_sequence = value[0].tolist()
            
            if self.debug_mode:
                print(f"🔍 토큰 시퀀스 길이: {len(token_sequence)}, 현재 길이: {self.current_length}")
            
            # 프롬프트 건너뛰기 로직 (간소화)
            if self.skip_prompt and len(token_sequence) <= 1:
                # 첫 번째 토큰만 건너뛰기 (프롬프트 시작 토큰)
                if self.debug_mode:
                    print(f"🔍 첫 번째 토큰 건너뛰기: {token_sequence}")
                return
                
            # 새로운 토큰 처리 (간소화)
            if len(token_sequence) > 0:
                # 모든 토큰을 캐시에 추가
                self.token_cache.extend(token_sequence)
                
                if self.debug_mode:
                    print(f"🔍 토큰 캐시에 추가: {len(token_sequence)}개 토큰")
                    print(f"🔍 총 캐시 길이: {len(self.token_cache)}")
                
                # 전체 토큰 캐시를 디코딩
                try:
                    text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
                    
                    # 출력 가능한 새로운 부분만 추출
                    if len(text) > self.print_len:
                        new_text = text[self.print_len:]
                        self.print_len = len(text)
                        
                        if self.debug_mode:
                            print(f"🔍 출력할 텍스트: '{new_text}'")
                        
                        # 즉시 출력 (버퍼링 없이)
                        print(new_text, end='', flush=True)
                        
                        # 빠른 모드에 따른 지연시간 조정
                        if self.fast_mode:
                            time.sleep(0.001)  # 더 빠른 출력
                        else:
                            time.sleep(0.002)  # 일반 속도
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ 디코딩 실패: {e}")
                    pass
                    
        except Exception as e:
            # 전체 처리 실패 시 무시하고 계속 진행
            if self.debug_mode:
                print(f"⚠️ 스트리머 오류 (무시됨): {e}")
            pass

    def end(self):
        """생성이 끝났을 때 호출됨"""
        print()  # 마지막에 개행 추가
        sys.stdout.flush()

class RobotSystem:
    def __init__(self, db_path: str = "hospital.db", use_real_model: bool = False, 
                 model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B", use_reasoning: bool = False, 
                 debug_mode: bool = False, fast_mode: bool = False):
        self.robot_functions = RobotFunctions(db_path)
        self.conversation_history: List[Dict[str, str]] = []
        self.use_real_model = use_real_model
        self.use_reasoning = use_reasoning  # Reasoning 모드 사용 여부
        self.fast_mode = fast_mode  # 빠른 응답 모드
        
        # 맥락 파악을 위한 이전 질문 추적
        self.last_user_question = ""
        self.last_response = ""
        
        # 디버깅 모드 설정
        self.debug_mode = debug_mode
        
        # 모델 설정 저장
        self.model_name = model_name
        
        # GPU 환경 확인
        self.gpu_available, self.device = check_gpu_availability()
        
        # 빠른 응답 모드 설정
        if fast_mode:
            print("⚡ 빠른 응답 모드 활성화!")
            print("  - 더 짧은 응답 생성 (1024 vs 2048 토큰)")
            print("  - EXAONE 4.0 공식 권장값 유지")
            print("  - 최소 스트리밍 지연시간 (0.001초)")
        
        # 실제 EXAONE 모델 사용 시
        if use_real_model:
            self._init_exaone_model()
    
    def debug_print(self, message: str):
        """디버깅 모드일 때만 출력"""
        if self.debug_mode:
            print(message)
    
    def _init_exaone_model(self):
        """간단한 EXAONE 모델 초기화"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"🔄 {self.model_name} 모델 로딩 중...")
            
            # 토크나이저 로드
            print("📝 토크나이저 로딩...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 모델 로드 (간단한 방식)
            print("🤖 모델 로딩...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # GPU로 이동 (가능한 경우)
            if self.gpu_available:
                self.model = self.model.to('cuda')
                print("✅ GPU로 모델 이동 완료")
            else:
                self.model = self.model.to('cpu')
                print("✅ CPU로 모델 이동 완료")
            
            # 토크나이저 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ 모델 로딩 완료!")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("🔄 시뮬레이션 모드로 전환...")
            self.use_real_model = False
    
    def add_to_history(self, role: str, content: str):
        """대화 히스토리에 추가"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # 히스토리가 너무 길어지면 정리 (최근 12개 대화만 유지 - 맥락 강화)
        if len(self.conversation_history) > 24:
            self.conversation_history = self.conversation_history[-24:]
    
    def get_function_prompt(self, user_input: str) -> str:
        """단순화된 프롬프트 - 더 이상 사용하지 않음"""
        return user_input
    
    def get_contextual_function_prompt(self, contextual_input: str) -> str:
        """단순화된 맥락 프롬프트 - 더 이상 사용하지 않음"""
        return contextual_input
    
    def get_response_prompt(self, user_input: str, function_result: str) -> str:
        """단순화된 응답 프롬프트 - 더 이상 사용하지 않음"""
        return function_result
    
    def _should_use_reasoning(self, user_input: str) -> bool:
        """복잡한 질문인지 판단하여 Reasoning 모드 사용 여부 결정"""
        user_lower = user_input.lower()
        
        # 복잡한 질문 패턴들
        reasoning_patterns = [
            # "왜", "어떻게", "무엇 때문에" 등의 질문
            "왜", "어떻게", "무엇 때문에", "어떤 이유로",
            # 비교 질문
            "차이점", "다른 점", "비교", "어떤 것이",
            # 설명 요청
            "설명해", "알려줘", "이유가", "원인이",
            # 복합적인 질문
            "가장 중요한", "가장 좋은", "어떤 것이 더",
            # 추론이 필요한 질문
            "만약", "만약에", "가정해보면", "생각해보면",
            # 분석 요청
            "분석", "검토", "고려", "생각해보면",
            # 숫자 + 문제 (예: "3번 문제")
            "번 문제", "번째 문제", "문제 같아"
        ]
        
        # 복잡한 질문 패턴이 포함되어 있는지 확인
        for pattern in reasoning_patterns:
            if pattern in user_lower:
                return True
        
        # 질문 길이가 길면 복잡할 가능성
        if len(user_input) > 30:
            return True
            
        return False
    
    def _is_context_related(self, user_input: str) -> bool:
        """현재 질문이 이전 질문과 연관성이 있는지 판단"""
        if not self.last_user_question:
            return False
            
        user_lower = user_input.lower()
        last_question_lower = self.last_user_question.lower()
        
        # 맥락 연관성 패턴들
        context_patterns = [
            # 숫자 + 문제 (예: "3번 문제")
            "번 문제", "번째 문제", "문제 같아",
            # 단계별 질문
            "단계", "단계가", "단계는",
            # 추가 질문
            "그리고", "또한", "추가로", "더",
            # 구체화
            "구체적으로", "자세히", "예를 들어",
            # 확인
            "맞나", "맞아", "그래", "네",
            # 반대
            "아니", "그런데", "하지만",
            # 이해 관련
            "이해가", "이해가 안", "잘 이해", "모르겠어",
            # 단어 관련
            "단어", "용어", "말이"
        ]
        
        # 패턴 매칭
        for pattern in context_patterns:
            if pattern in user_lower:
                return True
        
        # 키워드 연관성 확인
        common_keywords = ["ct", "x-ray", "xray", "영상", "검사", "장비", "문제", "단계", "이해", "설명", "모르겠어", "알려줘"]
        current_keywords = [kw for kw in common_keywords if kw in user_lower]
        last_keywords = [kw for kw in common_keywords if kw in last_question_lower]
        
        if current_keywords and last_keywords:
            print(f"🔍 키워드 연관성 발견: {current_keywords} ↔ {last_keywords}")
            return True
        
        # 추가: 이전 질문에 대한 구체적인 추가 질문인지 확인
        follow_up_patterns = [
            "자세히", "구체적으로", "더", "추가로", "그리고", "또한",
            "설명해", "알려줘", "모르겠어", "이해가 안", "잘 모르겠어",
            "너가", "당신이", "로봇이", "영웅이가", "말한", "설명한"
        ]
        
        for pattern in follow_up_patterns:
            if pattern in user_lower:
                print(f"🔍 후속 질문 패턴 발견: '{pattern}'")
                return True
            
        return False

    def process_user_input(self, user_input: str) -> str:
        """사용자 입력 처리 - 자동 모드 전환 포함"""
        # 히스토리에 사용자 입력 추가
        self.add_to_history("사용자", user_input)
        
        # 맥락 파악: 이전 질문과 연관성 확인
        context_related = self._is_context_related(user_input)
        self.debug_print(f"🔍 맥락 분석: '{user_input}'")
        self.debug_print(f"🔍 이전 질문: '{self.last_user_question}'")
        self.debug_print(f"🔍 맥락 연관성: {context_related}")
        
        # 첫 번째 질문인 경우 특별 처리
        if not self.last_user_question:
            self.debug_print("🆕 첫 번째 질문으로 인식")
            context_related = False
        
        try:
            if self.use_real_model:
                # 자동으로 복잡한 질문인지 판단
                should_reasoning = self._should_use_reasoning(user_input)
                
                if should_reasoning and not self.use_reasoning:
                    self.debug_print("🧠 복잡한 질문 감지! Reasoning 모드로 자동 전환")
                    self.use_reasoning = True
                elif not should_reasoning and self.use_reasoning:
                    self.debug_print("💬 일반 질문 감지! Non-reasoning 모드로 자동 전환")
                    self.use_reasoning = False
                
                # 맥락 정보를 히스토리에 추가
                if context_related and self.last_user_question:
                    self.debug_print(f"🔗 맥락 연관성 감지: '{self.last_user_question}' → '{user_input}'")
                    # 이전 질문을 참조하는 맥락 프롬프트 추가
                    contextual_input = f"""이전 대화 맥락:
사용자: {self.last_user_question}
영웅이: {self.last_response}

현재 질문: {user_input}

이전 질문에 대한 추가 질문이나 연관된 질문인 것 같습니다. 
이전 대화의 맥락을 고려하여 현재 질문에 적절히 답변해주세요.
만약 사용자가 "너가 설명한", "당신이 말한" 등의 표현을 사용하면,
이전 대화에서 자신이 설명한 내용을 참고해서 답변해주세요."""
                    self.debug_print(f"📝 맥락 프롬프트 생성: {contextual_input[:100]}...")
                else:
                    contextual_input = user_input
                    self.debug_print(f"📝 일반 프롬프트 사용: {contextual_input}")
                
                self.debug_print(f"🎯 최종 입력: {contextual_input[:50]}...")
                
                # 모드에 따라 다른 처리
                if self.use_reasoning:
                    # Reasoning 모드 (<think> 블록 사용)
                    response = self._call_real_exaone_reasoning(contextual_input)
                else:
                    # Non-reasoning 모드 (Agentic tool use)
                    response = self._call_real_exaone_simple(contextual_input)
            else:
                # 시뮬레이션 모드
                response = self._simple_simulation(user_input)
            
            # 히스토리에 응답 추가
            self.add_to_history("영웅이", response)
            
            # 이전 질문과 응답 추적 업데이트
            self.last_user_question = user_input
            self.last_response = response
            
            return response
            
        except Exception as e:
            print(f"❌ 처리 오류: {e}")
            return "어머, 영웅이가 잠시 문제가 생겼어요! 다시 한 번 말씀해주시겠어요?"
    
    def _call_real_exaone_simple(self, user_input: str) -> str:
        """대폭 개선된 Agentic tool use - 맥락 인식과 함수 선택 개선"""
        try:
            # 대화 히스토리를 포함한 맥락 구성
            conversation_context = ""
            if len(self.conversation_history) > 1:
                # 최근 4개 대화만 포함 (너무 길어지지 않게)
                recent_history = self.conversation_history[-8:]  # 4쌍의 대화
                context_items = []
                for entry in recent_history:
                    context_items.append(f"{entry['role']}: {entry['content']}")
                conversation_context = "\n".join(context_items)
            
            # tools 정의 (공식 문서 방식)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "query_facility",
                        "description": "병원 내 시설의 위치를 조회할 때 사용. '어디야', '위치', '찾아' 등의 질문에 사용",
                        "parameters": {
                            "type": "object",
                            "required": ["facility"],
                            "properties": {
                                "facility": {
                                    "type": "string",
                                    "description": "조회할 시설명 (CT, X-ray, 초음파, 폐암, 위암, 대장암, 유방암, 뇌종양 등)"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "navigate",
                        "description": "사용자를 특정 위치로 안내할 때 사용. '안내해줘', '데려다줘', '동행해줘', '가자' 등의 요청에 사용",
                        "parameters": {
                            "type": "object",
                            "required": ["target"],
                            "properties": {
                                "target": {
                                    "type": "string",
                                    "description": "안내할 목적지"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "start_registration",
                        "description": "접수나 예약 확인을 요청할 때 사용. '접수', '접수하려면', '예약 확인', '예약 내역', '예약 정보' 등의 요청에 사용",
                        "parameters": {
                            "type": "object",
                            "required": [],
                            "properties": {}
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "general_response",
                        "description": "일반적인 대화나 인사, 설명이 필요할 때 사용",
                        "parameters": {
                            "type": "object",
                            "required": ["message"],
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "사용자의 메시지"
                                }
                            }
                        }
                    }
                }
            ]
            
            # 개선된 지시사항과 맥락을 포함한 메시지
            system_prompt = f"""당신은 병원 안내 로봇입니다. 이름은 '영웅이'입니다. 친근하고 간결하게 답변하세요.

중요한 규칙:
1. 위치 질문('어디야', '어디있어', '찾아')은 query_facility 사용
2. 이동 요청('안내해줘', '데려다줘', '동행해줘', '가자', '가져다줘')은 navigate 사용  
3. 접수/예약 요청('접수', '접수하려면', '예약 확인', '예약 내역', '예약 정보')은 start_registration 사용
4. 일반 대화('안녕', '고마워', '뭐야')는 general_response 사용
5. 응답은 간결하고 자연스럽게 (길고 현학적인 답변 금지)
6. 대화 맥락을 고려하여 이전 언급된 장소를 기억하세요

{f"이전 대화 맥락:{conversation_context}" if conversation_context else ""}

사용자 질문: {user_input}"""
            
            # 공식 문서 방식으로 메시지 구성
            messages = [{"role": "user", "content": system_prompt}]
            
            # 공식 문서와 동일한 방식으로 호출
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                tools=tools,
            )
            
            # 실시간 스트리밍을 위한 개선된 생성 로직
            print(f"🤖 {'🧠 Reasoning' if self.use_reasoning else '💬'} 모드로 실시간 답변 중:", end=" ", flush=True)
            
            # TextIteratorStreamer 강제 사용 (서버 스트리밍을 위해)
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 생성 파라미터 설정 (EXAONE 4.0 공식 권장값 준수)
            if self.fast_mode:
                # 빠른 응답 모드 (공식 권장값 기반 + 최적화)
                max_tokens = 1024  # 더 짧은 응답
                temperature = 0.6 if self.use_reasoning else 0.1  # 공식 권장값 유지
                top_p = 0.95  # 공식 권장값 고정
            else:
                # 일반 모드 설정 (EXAONE 4.0 공식 권장값)
                max_tokens = 2048
                temperature = 0.6 if self.use_reasoning else 0.1  # 공식 권장값
                top_p = 0.95  # 공식 권장값 고정
            
            generation_kwargs = {
                "input_ids": input_ids.to(self.model.device),
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": None,
                "streamer": streamer,  # 스트리머 사용
            }
            
            # 스트리밍 방식에 따른 처리
            if hasattr(streamer, '__iter__'):  # TextIteratorStreamer인 경우
                # 스레드 기반 스트리밍 (작동하는 방식)
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # 실시간 스트리밍 출력 및 함수 호출 감지
                full_streamed_text = ""
                for text in streamer:
                    print(text, end="", flush=True)
                    full_streamed_text += text
                
                thread.join()  # 스레드 완료 대기
                
                # 스트리밍 완료 후 함수 호출 처리
                if "<tool_call>" in full_streamed_text:
                    print("\n🔧 함수 호출 형식 감지됨")
                    function_result = self._parse_and_execute_tool_call_improved(full_streamed_text, user_input)
                    print(f"🤖 답변: {function_result}")
                    return function_result
                else:
                    # 일반 텍스트 응답 처리
                    if full_streamed_text.strip() and "Available Tools" not in full_streamed_text:
                        return full_streamed_text.strip()
                    else:
                        print("\n❌ 적절한 응답을 생성하지 못했습니다")
                        fallback_result = self._fallback_response(user_input)
                        print(f"🤖 답변: {fallback_result}")
                        return fallback_result
            # TextIteratorStreamer만 사용하므로 else 블록 제거
            
            # 스트리밍이 완료되었으므로 빈 문자열 반환 (이미 출력됨)
            return ""
            
        except Exception as e:
            print(f"❌ 모델 호출 실패: {e}")
            return self._fallback_response(user_input)
    
    def _generate_simple_response(self, user_input: str) -> str:
        """맥락을 고려한 간단한 자연어 응답 생성"""
        user_lower = user_input.lower().strip()
        
        # 맥락에서 최근 언급된 시설 찾기
        recent_facility = self._extract_recent_facility()
        
        # 인사말
        if any(word in user_lower for word in ["안녕", "hello", "hi"]):
            return "안녕하세요! 저는 병원 안내 로봇 영웅이입니다. 무엇을 도와드릴까요?"
        
        # 감사 인사
        elif any(word in user_lower for word in ["고마", "감사", "thank"]):
            return "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요."
        
        # 로봇 정보 질문
        elif any(word in user_lower for word in ["누구", "이름", "뭐야", "뭔가", "정체"]):
            return "저는 병원 안내 로봇 영웅이입니다. 병원 시설 안내와 길찾기를 도와드려요!"
        
        # 맥락 기반 응답
        elif recent_facility and any(word in user_lower for word in ["그곳", "거기", "그 곳", "그거", "그 시설"]):
            return f"네, {recent_facility} 말씀이시죠? 더 자세한 안내가 필요하시나요?"
        
        # 기본 응답
        else:
            return "무엇을 도와드릴까요? 병원 시설 안내나 위치 조회를 도와드릴 수 있어요."
    
    def _extract_recent_facility(self) -> Optional[str]:
        """대화 맥락에서 최근 언급된 시설명 추출"""
        if len(self.conversation_history) == 0:
            return None
            
        # 최근 6개 대화에서 시설명 찾기
        for entry in reversed(self.conversation_history[-6:]):
            content = entry['content'].lower()
            # 시설명 직접 매칭
            facilities = {
                'ct': 'CT', 'x-ray': 'X-ray', '엑스레이': 'X-ray', 'xray': 'X-ray',
                '초음파': '초음파', '폐암': '폐암', '위암': '위암', '대장암': '대장암', 
                '유방암': '유방암', '뇌종양': '뇌종양', '시티': 'CT', '씨티': 'CT'
            }
            
            for key, facility in facilities.items():
                if key in content:
                    return facility
                    
        return None

    def _parse_and_execute_tool_call_improved(self, response: str, user_input: str) -> str:
        """개선된 함수 호출 파싱 및 실행"""
        try:
            import re
            import json
            
            # <tool_call> 태그에서 JSON 추출
            tool_call_pattern = r'<tool_call>\s*({[^<]*})\s*</tool_call>'
            matches = re.findall(tool_call_pattern, response, re.DOTALL)
            
            print(f"🔍 찾은 함수 호출: {len(matches)}개")
            
            for match in matches:
                tool_call_json = match.strip()
                print(f"🔍 검사 중인 JSON: {tool_call_json}")
                
                # 예시 텍스트 무시
                if any(placeholder in tool_call_json for placeholder in [
                    "function_1_name", "argument_1_name", "function_2_name", "argument_2_name"
                ]):
                    print("❌ 예시 텍스트 무시")
                    continue
                
                try:
                    tool_call = json.loads(tool_call_json)
                    function_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})
                    
                    print(f"🔧 실행할 함수: {function_name}({arguments})")
                    
                    if function_name == "query_facility":
                        facility = arguments.get("facility", "")
                        result = self.robot_functions.query_facility(facility)
                        
                        if "error" not in result["result"]:
                            return f"네! {facility}는 {result['result']}에 있어요. 😊"
                        else:
                            return f"죄송해요, {facility}는 이 병원에 없는 시설이에요. 다른 시설을 찾아드릴까요?"
                    
                    elif function_name == "navigate":
                        target = arguments.get("target", "")
                        
                        # 만약 target이 위치 설명이면 맥락에서 실제 시설명 찾기
                        if any(word in target for word in ["왼쪽", "오른쪽", "중앙", "상단", "하단", "영상의학과", "암센터"]):
                            recent_facility = self._extract_recent_facility()
                            if recent_facility:
                                print(f"🎯 맥락에서 추출한 시설: {recent_facility}")
                                target = recent_facility
                            else:
                                return "어떤 시설로 안내해드릴까요? 구체적인 시설명을 말씀해주세요."
                        
                        result = self.robot_functions.navigate(target)
                        
                        if "error" not in result.get("result", ""):
                            return f"좋아요! {target}로 안내해드릴게요. 저를 따라오세요! 🚀"
                        else:
                            return f"죄송해요, {target}를 찾을 수 없어요. 정확한 시설명을 말씀해주시겠어요?"
                    
                    elif function_name == "start_registration":
                        result = self.robot_functions.start_registration()
                        return result["result"]
                    
                    elif function_name == "general_response":
                        message = arguments.get("message", user_input)
                        # 모델이 추출한 메시지를 직접 사용
                        if message and message != user_input:
                            return message
                        else:
                            return self._generate_simple_response(message)
                    
                    else:
                        print(f"❌ 알 수 없는 함수: {function_name}")
                        continue
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 실패: {e}")
                    continue
                except Exception as e:
                    print(f"❌ 함수 실행 실패: {e}")
                    continue
            
            # 모든 함수 호출 실패 시 fallback
            print("❌ 모든 함수 호출 실패")
            return self._fallback_response(user_input)
                
        except Exception as e:
            print(f"❌ 함수 호출 파싱 실패: {e}")
            return self._fallback_response(user_input)
    
    def _fallback_response(self, user_input: str) -> str:
        """개선된 fallback 응답 - 맥락 고려"""
        user_lower = user_input.lower().strip()
        
        # 맥락에서 시설명 추출
        recent_facility = self._extract_recent_facility()
        
        # 이동 요청인데 목적지가 명확하지 않은 경우
        if any(word in user_lower for word in ["안내", "가자", "동행", "데려다", "이동"]):
            if recent_facility:
                return f"{recent_facility}로 안내해드릴까요?"
            else:
                return "어디로 안내해드릴까요?"
        
        # 위치 질문인데 시설명이 명확하지 않은 경우  
        elif any(word in user_lower for word in ["어디", "위치", "찾아"]):
            return "어떤 시설을 찾으시나요? CT, X-ray, 초음파, 각종 암센터 등이 있어요."
        
        # 접수/예약 요청
        elif any(word in user_lower for word in ["접수", "접수하려면", "접수하고 싶어요", "예약", "예약 확인", "예약 내역", "예약 정보"]):
            return "접수 화면으로 이동할게요. 잠시만 기다려주세요."
        
        # 인사
        elif any(word in user_lower for word in ["안녕", "hello", "hi"]):
            return "안녕하세요! 저는 병원 안내 로봇 영웅이입니다. 무엇을 도와드릴까요?"
        
        # 감사
        elif any(word in user_lower for word in ["고마", "감사", "thank"]):
            return "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요."
        
        # 기본
        else:
            return "무엇을 도와드릴까요? 병원 시설 안내, 위치 조회, 예약 확인, 접수 등을 도와드릴 수 있어요."
    
    def _simple_simulation(self, user_input: str) -> str:
        """단순한 시뮬레이션"""
        user_lower = user_input.lower().strip()
        
        # 인사말
        if any(word in user_lower for word in ["안녕", "hello", "hi"]):
            return "안녕하세요! 저는 병원 안내 로봇 영웅이입니다. 무엇을 도와드릴까요?"
        
        # 감사
        elif any(word in user_lower for word in ["고마", "감사", "thank"]):
            return "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요."
        
        # 시설 위치 조회
        elif any(word in user_input for word in ["CT", "X-ray", "초음파", "뇌종양", "유방암", "대장암", "위암", "폐암"]):
            if "CT" in user_input:
                return "CT는 왼쪽 중앙 영상의학과에 있어요."
            elif "X-ray" in user_input:
                return "X-ray는 왼쪽 상단 영상의학과에 있어요."
            elif "초음파" in user_input:
                return "초음파는 왼쪽 하단 영상의학과에 있어요."
            elif "뇌종양" in user_input:
                return "뇌종양은 오른쪽 상단 암센터에 있어요."
            elif "유방암" in user_input:
                return "유방암은 오른쪽 상단 암센터에 있어요."
            elif "대장암" in user_input:
                return "대장암은 오른쪽 하단 암센터에 있어요."
            elif "위암" in user_input:
                return "위암은 오른쪽 하단 암센터에 있어요."
            elif "폐암" in user_input:
                return "폐암은 오른쪽 하단 암센터에 있어요."
        
        # 네비게이션
        elif any(word in user_lower for word in ["가줘", "안내", "이동", "데려다"]):
            return "좋아요! 저를 따라오세요!"
        
        # 기본 응답
        else:
            return "무엇을 도와드릴까요?"
    
    def _advanced_llm_simulation(self, user_input: str, context_prompt: str) -> str:
        """기존 복잡한 시뮬레이션 - 더 이상 사용하지 않음"""
        return self._simple_simulation(user_input)
    
    def _call_real_exaone(self, user_input: str, context_prompt: str, is_function_call: bool = False) -> str:
        """기존 복잡한 모델 호출 - 더 이상 사용하지 않음"""
        return self._call_real_exaone_simple(user_input)
    
    def _execute_function_raw(self, function_call: Dict[str, Any], user_input: str = "") -> str:
        """기존 복잡한 함수 실행 - 더 이상 사용하지 않음"""
        return "함수 실행이 단순화되었습니다."

    def _generate_natural_response(self, user_input: str, function_result: str) -> str:
        """기존 복잡한 자연어 응답 생성 - 더 이상 사용하지 않음"""
        return self._simple_simulation(user_input)
    
    def _simple_fallback(self, user_input: str) -> str:
        """기존 복잡한 fallback - 더 이상 사용하지 않음"""
        return self._simple_simulation(user_input)
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []
    
    def _clean_reasoning_response(self, response: str) -> str:
        """Reasoning 모드 응답에서 <think> 블록을 제거하고 정리된 답변만 반환"""
        try:
            print(f"🔍 원본 응답 길이: {len(response)}")
            print(f"🔍 원본 응답 미리보기: {response[:300]}...")
            
            # <think> 블록 제거
            if "<think>" in response and "</think>" in response:
                think_start = response.find("<think>")
                think_end = response.find("</think>") + 8
                response = response[:think_start] + response[think_end:]
                print(f"🔍 <think> 블록 제거 후 길이: {len(response)}")
            else:
                print("🔍 <think> 블록을 찾을 수 없음")
            
            # 시스템 프롬프트나 불필요한 텍스트 제거
            unwanted_patterns = [
                "당신은 병원 안내 로봇입니다",
                "사용자 질문:",
                "중요한 규칙:",
                "이전 대화 맥락:",
                "Available Tools",
                "사용자 질문:"
            ]
            
            for pattern in unwanted_patterns:
                if pattern in response:
                    response = response.replace(pattern, "")
                    print(f"🔍 '{pattern}' 제거됨")
            
            # 불필요한 공백 제거 및 정리
            response = response.strip()
            print(f"🔍 정리 후 길이: {len(response)}")
            
            # 빈 응답이거나 너무 짧은 경우 처리
            if not response or len(response) < 10:
                print("❌ 응답이 너무 짧음")
                return ""
            
            # 응답이 너무 긴 경우 적절히 자르기 (최대 1000자)
            if len(response) > 1000:
                response = response[:1000] + "..."
                print("🔍 응답이 1000자로 잘림")
            
            print(f"✅ 최종 정리된 응답: {response[:100]}...")
            return response
            
        except Exception as e:
            print(f"❌ Reasoning 응답 정리 실패: {e}")
            return response.strip() 

    def _call_real_exaone_reasoning(self, user_input: str) -> str:
        """Reasoning 모드용 EXAONE 호출 (<think> 블록 사용) - 실시간 스트리밍"""
        try:
            # 대화 히스토리를 포함한 맥락 구성
            conversation_context = ""
            if len(self.conversation_history) > 1:
                # 최근 4개 대화만 포함 (너무 길어지지 않게)
                recent_history = self.conversation_history[-8:]  # 4쌍의 대화
                context_items = []
                for entry in recent_history:
                    context_items.append(f"{entry['role']}: {entry['content']}")
                conversation_context = "\n".join(context_items)
            
            # Reasoning 모드용 지시사항 - 함수 호출 방식 사용
            system_prompt = f"""당신은 병원 안내 로봇입니다. 이름은 '영웅이'입니다. 
복잡한 질문에 대해서는 단계별로 생각한 후 적절한 함수를 호출하세요.

중요한 규칙:
1. 위치 질문('어디야', '어디있어', '찾아')은 query_facility 사용
2. 이동 요청('안내해줘', '데려다줘', '동행해줘', '가자', '가져다줘')은 navigate 사용  
3. 예약 확인 요청('예약 확인', '예약 내역', '예약 정보', '예약돼 있는지')은 check_reservation 사용
4. 접수 요청('접수', '접수하려면', '접수하고 싶어요', '접수 좀 도와주세요')은 start_registration 사용
5. 일반 대화('안녕', '고마워', '뭐야')는 general_response 사용
6. 복잡한 설명이 필요한 질문도 general_response로 친근하게 답변
7. 답변은 간결하고 자연스럽게 (길고 현학적인 답변 금지)

{f"이전 대화 맥락:{conversation_context}" if conversation_context else ""}

사용자 질문: {user_input}

중요: 사용자가 "너가 설명한", "당신이 말한" 등의 표현을 사용하면, 
이전 대화에서 자신이 설명한 내용을 참고해서 답변해주세요."""
            
            # tools 정의 (non-reasoning 모드와 동일)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "query_facility",
                        "description": "병원 내 시설의 위치를 조회할 때 사용. '어디야', '위치', '찾아' 등의 질문에 사용",
                        "parameters": {
                            "type": "object",
                            "required": ["facility"],
                            "properties": {
                                "facility": {
                                    "type": "string",
                                    "description": "조회할 시설명 (CT, X-ray, 초음파, 폐암, 위암, 대장암, 유방암, 뇌종양 등)"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "navigate",
                        "description": "사용자를 특정 위치로 안내할 때 사용. '안내해줘', '데려다줘', '동행해줘', '가자' 등의 요청에 사용",
                        "parameters": {
                            "type": "object",
                            "required": ["target"],
                            "properties": {
                                "target": {
                                    "type": "string",
                                    "description": "안내할 목적지"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "general_response",
                        "description": "일반적인 대화나 인사, 설명이 필요할 때 사용",
                        "parameters": {
                            "type": "object",
                            "required": ["message"],
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "사용자의 메시지"
                                }
                            }
                        }
                    }
                }
            ]
            
            # 메시지 구성
            messages = [{"role": "user", "content": system_prompt}]
            
            # 토크나이저 적용 (함수 호출 방식 사용)
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                tools=tools,  # 함수 호출 활성화
            )
            print("✅ 함수 호출 파라미터 적용됨")
            
            print("🧠 Reasoning 모드로 실시간 답변 중:", end=" ", flush=True)
            
            # Reasoning 모드용 스트리머 (TextIteratorStreamer 강제 사용)
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            reasoning_streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Reasoning 모드 생성 파라미터 (EXAONE 4.0 공식 권장값)
            max_tokens = 1024 if self.fast_mode else 2048
            
            # Reasoning 모드 스레드 기반 스트리밍
            reasoning_kwargs = {
                "input_ids": input_ids.to(self.model.device),
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": 0.6,  # 공식 권장값 (Reasoning 모드)
                "top_p": 0.95,  # 공식 권장값 고정
                "repetition_penalty": 1.1,  # transformers 지원 파라미터
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": None,  # Attention mask 자동 생성
                "streamer": reasoning_streamer,  # TextIteratorStreamer 사용
            }
            
            # Reasoning 모드 스트리밍 방식에 따른 처리
            if hasattr(reasoning_streamer, '__iter__'):  # TextIteratorStreamer인 경우
                print("🔍 TextIteratorStreamer 사용")
                reasoning_thread = Thread(target=self.model.generate, kwargs=reasoning_kwargs)
                reasoning_thread.start()
                
                # Reasoning 모드 실시간 스트리밍 출력 및 함수 호출 감지
                full_reasoning_text = ""
                token_count = 0
                for text in reasoning_streamer:
                    print(text, end="", flush=True)
                    full_reasoning_text += text
                    token_count += 1
                    if token_count % 10 == 0:  # 10개 토큰마다 진행상황 출력
                        print(f" [토큰 {token_count}]", end="", flush=True)
                
                reasoning_thread.join()  # 스레드 완료 대기
                print(f"\n🔍 총 생성된 토큰 수: {token_count}")
                
                # Reasoning 모드 스트리밍 완료 후 응답 처리
                print(f"\n🔍 전체 스트리밍 텍스트 길이: {len(full_reasoning_text)}")
                print(f"🔍 스트리밍 텍스트 미리보기: {full_reasoning_text[:200]}...")
                
                if "<tool_call>" in full_reasoning_text:
                    print("\n🔧 함수 호출 형식 감지됨")
                    function_result = self._parse_and_execute_tool_call_improved(full_reasoning_text, user_input)
                    print(f"🤖 답변: {function_result}")
                    return function_result
                else:
                    # 일반 텍스트 응답 처리
                    if full_reasoning_text.strip() and "Available Tools" not in full_reasoning_text:
                        return full_reasoning_text.strip()
                    else:
                        print("\n❌ 적절한 응답을 생성하지 못했습니다")
                        fallback_result = self._fallback_response(user_input)
                        print(f"🤖 답변: {fallback_result}")
                        return fallback_result
            # TextIteratorStreamer만 사용하므로 else 블록 제거
            
        except Exception as e:
            print(f"\n❌ Reasoning 모드 호출 실패: {e}")
            return self._fallback_response(user_input)

# 테스트용 메인 함수
if __name__ == "__main__":
    # 시뮬레이션 모드로 테스트 (빠른 모드 활성화, 디버그 모드 비활성화)
    robot = RobotSystem(use_real_model=False, use_reasoning=False, fast_mode=True, debug_mode=False)
    
    print("🤖 병원 안내 로봇 영웅이 테스트")
    print("📝 모드: 자동 전환 (복잡한 질문 감지)")
    print("⚡ 빠른 응답 모드 활성화")
    print("💡 EXAONE 4.0 공식 권장값 적용 중")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("수동 모드 전환:")
    print("  - 'reasoning' → Reasoning 모드")
    print("  - 'non-reasoning' → Non-reasoning 모드")
    print("  - 'fast' → 빠른 응답 모드 켜기")
    print("  - 'normal' → 일반 응답 모드")
    print("예시:")
    print("  - 'CT 어디야?' → Non-reasoning 모드")
    print("  - '왜 CT와 X-ray가 다른가요?' → Reasoning 모드")
    
    while True:
        try:
            user_input = input("\n👤 질문: ")
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("👋 안녕히 가세요!")
                break
            elif user_input.lower() == 'reasoning':
                # Reasoning 모드로 강제 전환
                robot.use_reasoning = True
                print("🧠 수동 전환: Reasoning 모드")
                continue
            elif user_input.lower() == 'non-reasoning':
                # Non-reasoning 모드로 강제 전환
                robot.use_reasoning = False
                print("💬 수동 전환: Non-reasoning 모드")
                continue
            elif user_input.lower() == 'fast':
                # 빠른 응답 모드 활성화
                robot.fast_mode = True
                print("⚡ 빠른 응답 모드 활성화!")
                continue
            elif user_input.lower() == 'normal':
                # 일반 응답 모드로 전환
                robot.fast_mode = False
                print("🐌 일반 응답 모드로 전환")
                continue
            
            response = robot.process_user_input(user_input)
            
            # 스트리밍 응답이 이미 출력되었으므로 추가 텍스트가 있을 때만 출력
            if response and response.strip():
                print(f"🤖 영웅이: {response}")
            
            print()  # 다음 질문을 위한 개행
            sys.stdout.flush()  # 출력 버퍼 즉시 플러시
            
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            break 
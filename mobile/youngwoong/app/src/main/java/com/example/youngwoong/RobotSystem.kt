package com.example.youngwoong

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * 안드로이드용 병원 안내 로봇 시스템
 * Python의 RobotSystem을 완전히 포팅한 통합 클래스
 */
class RobotSystem(
    private val context: Context,
    private val useRealModel: Boolean = true,
    private val useReasoning: Boolean = false,
    private val debugMode: Boolean = false,
    private val fastMode: Boolean = true
) {
    
    private lateinit var exaoneModel: EXAONEModel
    private var streamingManager: StreamingManager? = null
    private var isInitialized = false
    
    // 실시간 응답 Flow
    private val _responseFlow = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 1000
    )
    val responseFlow: SharedFlow<String> = _responseFlow.asSharedFlow()
    
    // 완료된 응답 Flow
    private val _completedResponseFlow = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 1
    )
    val completedResponseFlow: SharedFlow<String> = _completedResponseFlow.asSharedFlow()
    
    // 시스템 상태 Flow
    private val _systemStatusFlow = MutableStateFlow("초기화 중...")
    val systemStatusFlow: StateFlow<String> = _systemStatusFlow.asStateFlow()
    
    companion object {
        private const val TAG = "RobotSystem"
    }
    
    /**
     * 시스템 초기화
     */
    suspend fun initialize() = withContext(Dispatchers.Default) {
        try {
            _systemStatusFlow.value = "🔄 EXAONE 모델 로딩 중..."
            
            // EXAONE 모델 초기화
            exaoneModel = EXAONEModel(context).apply {
                setDebugMode(debugMode)
                setFastMode(fastMode)
                setReasoningMode(useReasoning)
            }
            
            if (useRealModel) {
                exaoneModel.loadModel()
                
                // 스트리밍 매니저 초기화 (실제 모델 사용 시에만)
                streamingManager = StreamingManager(
                    tokenizer = exaoneModel.tokenizer,
                    debugMode = debugMode
                )
                
                _systemStatusFlow.value = "✅ 실제 EXAONE 모델 로드 완료"
            } else {
                _systemStatusFlow.value = "✅ 시뮬레이션 모드 준비 완료"
            }
            
            isInitialized = true
            debugPrint("🤖 로봇 시스템 초기화 완료")
            debugPrint("📊 모드: ${if (useRealModel) "실제 모델" else "시뮬레이션"}")
            debugPrint("🧠 Reasoning: ${if (useReasoning) "활성화" else "비활성화"}")
            debugPrint("⚡ 빠른 모드: $fastMode")
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ 시스템 초기화 실패: ${e.message}")
            _systemStatusFlow.value = "❌ 초기화 실패: ${e.message}"
            throw e
        }
    }
    
    /**
     * 사용자 입력 처리 (스트리밍 지원)
     */
    suspend fun processUserInputWithStreaming(userInput: String) = withContext(Dispatchers.Default) {
        if (!isInitialized) {
            throw IllegalStateException("시스템이 초기화되지 않았습니다.")
        }
        
        try {
            debugPrint("👤 사용자 입력: $userInput")
            
            if (useRealModel && streamingManager != null) {
                // 실시간 스트리밍 모드
                processWithStreaming(userInput)
            } else {
                // 일반 모드 (시뮬레이션 또는 스트리밍 비지원)
                val response = exaoneModel.processUserInput(userInput)
                _completedResponseFlow.emit(response)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ 입력 처리 실패: ${e.message}")
            val errorResponse = "어머, 영웅이가 잠시 문제가 생겼어요! 다시 한 번 말씀해주시겠어요?"
            _completedResponseFlow.emit(errorResponse)
        }
    }
    
    /**
     * 실시간 스트리밍으로 처리
     */
    private suspend fun processWithStreaming(userInput: String) = withContext(Dispatchers.Default) {
        streamingManager?.let { manager ->
            // 새로운 스트리밍 세션 시작
            val streamer = manager.startNewSession(
                fastMode = fastMode,
                onTextUpdate = { newText ->
                    // 실시간 텍스트 업데이트를 Flow로 전송
                    CoroutineScope(Dispatchers.Default).launch {
                        _responseFlow.emit(newText)
                    }
                },
                onComplete = { finalText ->
                    // 완료된 응답을 Flow로 전송
                    CoroutineScope(Dispatchers.Default).launch {
                        _completedResponseFlow.emit(finalText)
                    }
                }
            )
            
            // 백그라운드에서 모델 실행 및 토큰 스트리밍
            launch {
                try {
                    // 실제 모델 처리 로직 (간소화된 버전)
                    simulateStreamingGeneration(userInput, streamer)
                } catch (e: Exception) {
                    debugPrint("❌ 스트리밍 생성 중 오류: ${e.message}")
                    streamer.end()
                }
            }
        }
    }
    
    /**
     * 스트리밍 생성 시뮬레이션 (실제 TensorFlow Lite 연동 시 대체 필요)
     */
    private suspend fun simulateStreamingGeneration(userInput: String, streamer: AndroidStreamer) {
        try {
            // 일반적인 응답 생성
            val fullResponse = exaoneModel.processUserInput(userInput)
            
            // 응답을 토큰 단위로 분할하여 스트리밍 시뮬레이션
            val words = fullResponse.split(" ")
            for ((index, word) in words.withIndex()) {
                val tokenToEmit = if (index == words.size - 1) word else "$word "
                
                // 단어를 토큰으로 인코딩 (간소화)
                val tokens = intArrayOf(index) // 실제로는 tokenizer.encode 사용
                
                streamer.put(tokens)
                
                // 적절한 지연시간
                delay(if (fastMode) 50 else 100)
            }
            
            // 스트리밍 완료
            streamer.end()
            
        } catch (e: Exception) {
            debugPrint("❌ 스트리밍 시뮬레이션 실패: ${e.message}")
            streamer.end()
        }
    }
    
    /**
     * 일반 사용자 입력 처리 (스트리밍 없음)
     */
    suspend fun processUserInput(userInput: String): String {
        if (!isInitialized) {
            throw IllegalStateException("시스템이 초기화되지 않았습니다.")
        }
        
        return try {
            debugPrint("👤 사용자 입력: $userInput")
            exaoneModel.processUserInput(userInput)
        } catch (e: Exception) {
            Log.e(TAG, "❌ 입력 처리 실패: ${e.message}")
            "어머, 영웅이가 잠시 문제가 생겼어요! 다시 한 번 말씀해주시겠어요?"
        }
    }
    
    /**
     * 시스템 설정 변경
     */
    fun updateSettings(
        useReasoning: Boolean? = null,
        fastMode: Boolean? = null,
        debugMode: Boolean? = null
    ) {
        if (!isInitialized) return
        
        useReasoning?.let { 
            exaoneModel.setReasoningMode(it)
            debugPrint("🧠 Reasoning 모드: ${if (it) "활성화" else "비활성화"}")
        }
        
        fastMode?.let { 
            exaoneModel.setFastMode(it)
            debugPrint("⚡ 빠른 모드: $it")
        }
        
        debugMode?.let {
            exaoneModel.setDebugMode(it)
            debugPrint("🔍 디버그 모드: $it")
        }
    }
    
    /**
     * 대화 히스토리 초기화
     */
    fun clearHistory() {
        if (!isInitialized) return
        
        exaoneModel.clearHistory()
        debugPrint("🗑️ 대화 히스토리 초기화됨")
    }
    
    /**
     * 시스템 상태 정보
     */
    fun getSystemInfo(): Map<String, Any> {
        return if (isInitialized) {
            mapOf(
                "initialized" to true,
                "useRealModel" to useRealModel,
                "modelLoaded" to exaoneModel.isModelLoaded(),
                "historySize" to exaoneModel.getHistorySize(),
                "isStreaming" to (streamingManager?.isStreaming() ?: false),
                "fastMode" to fastMode,
                "debugMode" to debugMode
            )
        } else {
            mapOf(
                "initialized" to false,
                "error" to "시스템이 초기화되지 않았습니다."
            )
        }
    }
    
    /**
     * 특정 질문 유형별 처리 함수들
     */
    
    // 병원 시설 안내
    suspend fun handleFacilityQuery(facility: String): String {
        val query = "${facility}가 어디에 있나요?"
        return processUserInput(query)
    }
    
    // 네비게이션 요청
    suspend fun handleNavigationRequest(destination: String): String {
        val query = "${destination}로 안내해주세요"
        return processUserInput(query)
    }
    
    // 일반 대화
    suspend fun handleGeneralChat(message: String): String {
        return processUserInput(message)
    }
    
    /**
     * 리소스 정리
     */
    fun close() {
        streamingManager?.stopCurrentSession()
        
        if (isInitialized) {
            exaoneModel.close()
        }
        
        isInitialized = false
        debugPrint("🔒 로봇 시스템 종료됨")
    }
    
    private fun debugPrint(message: String) {
        if (debugMode) {
            Log.d(TAG, message)
        }
        println(message)
    }
}

/**
 * 로봇 시스템 빌더 클래스
 */
class RobotSystemBuilder(private val context: Context) {
    private var useRealModel = true
    private var useReasoning = false
    private var debugMode = false
    private var fastMode = true
    
    fun useRealModel(enabled: Boolean) = apply { useRealModel = enabled }
    fun useReasoning(enabled: Boolean) = apply { useReasoning = enabled }
    fun enableDebugMode(enabled: Boolean) = apply { debugMode = enabled }
    fun enableFastMode(enabled: Boolean) = apply { fastMode = enabled }
    
    fun build(): RobotSystem {
        return RobotSystem(
            context = context,
            useRealModel = useRealModel,
            useReasoning = useReasoning,
            debugMode = debugMode,
            fastMode = fastMode
        )
    }
}

/**
 * 사용 편의를 위한 확장 함수들
 */

// 간단한 질문 처리
suspend fun RobotSystem.ask(question: String): String {
    return processUserInput(question)
}

// 시설 찾기
suspend fun RobotSystem.findFacility(facilityName: String): String {
    return handleFacilityQuery(facilityName)
}

// 길찾기
suspend fun RobotSystem.navigateTo(destination: String): String {
    return handleNavigationRequest(destination)
} 
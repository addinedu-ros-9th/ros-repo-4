package com.example.youngwoong

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File
import org.json.JSONObject
import org.json.JSONArray
import java.io.BufferedReader
import java.io.InputStreamReader
import android.util.Log
import kotlinx.coroutines.*

// 대화 엔트리 데이터 클래스
data class ConversationEntry(
    val role: String,
    val content: String,
    val timestamp: Long = System.currentTimeMillis()
)

// 함수 호출 데이터 클래스
data class ToolCall(
    val name: String,
    val arguments: Map<String, Any>
)

class EXAONEModel(private val context: Context) {
    private var interpreter: Interpreter? = null
    private lateinit var tokenizer: OriginalTokenizer
    private var config: JSONObject? = null
    
    // 고도화된 기능들
    private val robotFunctions = RobotFunctions(context)
    private val conversationHistory = mutableListOf<ConversationEntry>()
    private var useReasoningMode = false
    private var fastMode = true
    private var debugMode = false
    
    // 맥락 추적
    private var lastUserQuestion = ""
    private var lastResponse = ""
    
    companion object {
        private const val TAG = "EXAONEModel"
        private const val MAX_HISTORY_SIZE = 24
    }
    
    fun loadModel() {
        try {
            // 모델 파일 로드
            val modelFile = context.assets.open("exaone_model.tflite")
            val modelBytes = modelFile.readBytes()
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelBuffer.order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            
            // 인터프리터 설정
            val options = Interpreter.Options()
            interpreter = Interpreter(modelBuffer, options)
            
            // 원본 토크나이저 초기화
            tokenizer = OriginalTokenizer(context)
            
            // 설정 파일 로드
            val configFile = context.assets.open("model_config.json")
            val configString = configFile.bufferedReader().use { it.readText() }
            config = JSONObject(configString)
            
            debugPrint("✅ EXAONE 모델 로드 완료")
            debugPrint("💡 고도화된 프롬프트 엔지니어링 시스템 활성화")
            debugPrint("⚡ 빠른 응답 모드: $fastMode")
            
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e(TAG, "❌ 모델 로드 실패: ${e.message}")
        }
    }
    
    private fun debugPrint(message: String) {
        if (debugMode) {
            Log.d(TAG, message)
        }
        println(message)
    }
    
    /**
     * 메인 사용자 입력 처리 함수 - Python의 process_user_input 포팅
     */
    suspend fun processUserInput(userInput: String): String = withContext(Dispatchers.Default) {
        // 히스토리에 사용자 입력 추가
        addToHistory("사용자", userInput)
        
        // 맥락 분석
        val contextRelated = isContextRelated(userInput)
        debugPrint("🔍 맥락 분석: '$userInput'")
        debugPrint("🔍 이전 질문: '$lastUserQuestion'")
        debugPrint("🔍 맥락 연관성: $contextRelated")
        
        try {
            // 자동으로 복잡한 질문인지 판단
            val shouldReasoning = shouldUseReasoning(userInput)
            
            if (shouldReasoning && !useReasoningMode) {
                debugPrint("🧠 복잡한 질문 감지! Reasoning 모드로 자동 전환")
                useReasoningMode = true
            } else if (!shouldReasoning && useReasoningMode) {
                debugPrint("💬 일반 질문 감지! Non-reasoning 모드로 자동 전환")
                useReasoningMode = false
            }
            
            // 맥락 정보를 히스토리에 추가
            val contextualInput = if (contextRelated && lastUserQuestion.isNotEmpty()) {
                debugPrint("🔗 맥락 연관성 감지: '$lastUserQuestion' → '$userInput'")
                buildContextualPrompt(userInput)
            } else {
                userInput
            }
            
            debugPrint("📝 최종 입력: ${contextualInput.take(50)}...")
            
            // 모드에 따라 다른 처리
            val response = if (useReasoningMode) {
                callEXAONEReasoning(contextualInput)
            } else {
                callEXAONESimple(contextualInput)
            }
            
            // 히스토리에 응답 추가
            addToHistory("영웅이", response)
            
            // 이전 질문과 응답 추적 업데이트
            lastUserQuestion = userInput
            lastResponse = response
            
            response
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ 처리 오류: ${e.message}")
            "어머, 영웅이가 잠시 문제가 생겼어요! 다시 한 번 말씀해주시겠어요?"
        }
    }
    
    /**
     * 대화 히스토리에 추가
     */
    private fun addToHistory(role: String, content: String) {
        conversationHistory.add(ConversationEntry(role, content))
        
        // 히스토리가 너무 길어지면 정리
        if (conversationHistory.size > MAX_HISTORY_SIZE) {
            val toRemove = conversationHistory.size - MAX_HISTORY_SIZE
            repeat(toRemove) {
                conversationHistory.removeAt(0)
            }
        }
    }
    
    /**
     * 복잡한 질문인지 판단하여 Reasoning 모드 사용 여부 결정
     */
    private fun shouldUseReasoning(userInput: String): Boolean {
        val userLower = userInput.lowercase()
        
        val reasoningPatterns = listOf(
            "왜", "어떻게", "무엇 때문에", "어떤 이유로",
            "차이점", "다른 점", "비교", "어떤 것이",
            "설명해", "알려줘", "이유가", "원인이",
            "가장 중요한", "가장 좋은", "어떤 것이 더",
            "만약", "만약에", "가정해보면", "생각해보면",
            "분석", "검토", "고려", "생각해보면",
            "번 문제", "번째 문제", "문제 같아"
        )
        
        // 복잡한 질문 패턴이 포함되어 있는지 확인
        for (pattern in reasoningPatterns) {
            if (userLower.contains(pattern)) {
                return true
            }
        }
        
        // 질문 길이가 길면 복잡할 가능성
        return userInput.length > 30
    }
    
    /**
     * 현재 질문이 이전 질문과 연관성이 있는지 판단
     */
    private fun isContextRelated(userInput: String): Boolean {
        if (lastUserQuestion.isEmpty()) return false
        
        val userLower = userInput.lowercase()
        val lastQuestionLower = lastUserQuestion.lowercase()
        
        val contextPatterns = listOf(
            "번 문제", "번째 문제", "문제 같아",
            "단계", "단계가", "단계는",
            "그리고", "또한", "추가로", "더",
            "구체적으로", "자세히", "예를 들어",
            "맞나", "맞아", "그래", "네",
            "아니", "그런데", "하지만",
            "이해가", "이해가 안", "잘 이해", "모르겠어",
            "단어", "용어", "말이"
        )
        
        // 패턴 매칭
        for (pattern in contextPatterns) {
            if (userLower.contains(pattern)) {
                return true
            }
        }
        
        // 키워드 연관성 확인
        val commonKeywords = listOf("ct", "x-ray", "xray", "영상", "검사", "장비", "문제", "단계", "이해", "설명", "모르겠어", "알려줘")
        val currentKeywords = commonKeywords.filter { userLower.contains(it) }
        val lastKeywords = commonKeywords.filter { lastQuestionLower.contains(it) }
        
        if (currentKeywords.isNotEmpty() && lastKeywords.isNotEmpty()) {
            debugPrint("🔍 키워드 연관성 발견: $currentKeywords ↔ $lastKeywords")
            return true
        }
        
        return false
    }
    
    /**
     * 맥락 프롬프트 구성
     */
    private fun buildContextualPrompt(userInput: String): String {
        return """이전 대화 맥락:
사용자: $lastUserQuestion
영웅이: $lastResponse

현재 질문: $userInput

이전 질문에 대한 추가 질문이나 연관된 질문인 것 같습니다. 
이전 대화의 맥락을 고려하여 현재 질문에 적절히 답변해주세요.
만약 사용자가 "너가 설명한", "당신이 말한" 등의 표현을 사용하면,
이전 대화에서 자신이 설명한 내용을 참고해서 답변해주세요."""
    }
    
    /**
     * Non-reasoning 모드 EXAONE 호출 (Agentic tool use)
     */
    private suspend fun callEXAONESimple(userInput: String): String = withContext(Dispatchers.Default) {
        try {
            // 대화 히스토리를 포함한 맥락 구성
            val conversationContext = buildConversationContext()
            
            // tools 정의
            val tools = buildToolsDefinition()
            
            // 개선된 지시사항과 맥락을 포함한 시스템 프롬프트
            val systemPrompt = buildSystemPrompt(userInput, conversationContext)
            
            debugPrint("🤖 💬 모드로 답변 중...")
            
            // TensorFlow Lite에서의 처리 (단순화된 버전)
            val response = generateWithTensorFlowLite(systemPrompt)
            
            // 함수 호출 형식 감지 및 처리
            if (response.contains("<tool_call>")) {
                debugPrint("🔧 함수 호출 형식 감지됨")
                return@withContext parseAndExecuteToolCall(response, userInput)
            } else {
                return@withContext response.takeIf { it.isNotBlank() } ?: fallbackResponse(userInput)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ 모델 호출 실패: ${e.message}")
            return@withContext fallbackResponse(userInput)
        }
    }
    
    /**
     * Reasoning 모드 EXAONE 호출
     */
    private suspend fun callEXAONEReasoning(userInput: String): String = withContext(Dispatchers.Default) {
        try {
            val conversationContext = buildConversationContext()
            
            val systemPrompt = """당신은 병원 안내 로봇입니다. 이름은 '영웅이'입니다. 
복잡한 질문에 대해서는 단계별로 생각해보세요.

${if (conversationContext.isNotEmpty()) "이전 대화 맥락:$conversationContext" else ""}

사용자 질문: $userInput

중요: 사용자가 "너가 설명한", "당신이 말한" 등의 표현을 사용하면, 
이전 대화에서 자신이 설명한 내용을 참고해서 답변해주세요."""
            
            debugPrint("🧠 Reasoning 모드로 답변 중...")
            
            val response = generateWithTensorFlowLite(systemPrompt)
            
            if (response.contains("<tool_call>")) {
                debugPrint("🔧 함수 호출 형식 감지됨")
                return@withContext parseAndExecuteToolCall(response, userInput)
            } else {
                return@withContext response.takeIf { it.isNotBlank() } ?: fallbackResponse(userInput)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Reasoning 모드 호출 실패: ${e.message}")
            return@withContext fallbackResponse(userInput)
        }
    }
    
    /**
     * 대화 맥락 구성
     */
    private fun buildConversationContext(): String {
        if (conversationHistory.size <= 1) return ""
        
        val recentHistory = conversationHistory.takeLast(8) // 최근 4쌍의 대화
        return recentHistory.joinToString("\n") { "${it.role}: ${it.content}" }
    }
    
    /**
     * 도구 정의 구성
     */
    private fun buildToolsDefinition(): List<Map<String, Any>> {
        return listOf(
            mapOf(
                "type" to "function",
                "function" to mapOf(
                    "name" to "query_facility",
                    "description" to "병원 내 시설의 위치를 조회할 때 사용. '어디야', '위치', '찾아' 등의 질문에 사용",
                    "parameters" to mapOf(
                        "type" to "object",
                        "required" to listOf("facility"),
                        "properties" to mapOf(
                            "facility" to mapOf(
                                "type" to "string",
                                "description" to "조회할 시설명 (CT, X-ray, 초음파, 폐암, 위암, 대장암, 유방암, 뇌종양 등)"
                            )
                        )
                    )
                )
            ),
            mapOf(
                "type" to "function",
                "function" to mapOf(
                    "name" to "navigate",
                    "description" to "사용자를 특정 위치로 안내할 때 사용. '안내해줘', '데려다줘', '동행해줘', '가자' 등의 요청에 사용",
                    "parameters" to mapOf(
                        "type" to "object",
                        "required" to listOf("target"),
                        "properties" to mapOf(
                            "target" to mapOf(
                                "type" to "string",
                                "description" to "안내할 목적지"
                            )
                        )
                    )
                )
            ),
            mapOf(
                "type" to "function",
                "function" to mapOf(
                    "name" to "general_response",
                    "description" to "일반적인 대화나 인사, 설명이 필요할 때 사용",
                    "parameters" to mapOf(
                        "type" to "object",
                        "required" to listOf("message"),
                        "properties" to mapOf(
                            "message" to mapOf(
                                "type" to "string",
                                "description" to "사용자의 메시지"
                            )
                        )
                    )
                )
            )
        )
    }
    
    /**
     * 시스템 프롬프트 구성
     */
    private fun buildSystemPrompt(userInput: String, conversationContext: String): String {
        return """당신은 병원 안내 로봇입니다. 이름은 '영웅이'입니다. 친근하고 간결하게 답변하세요.

중요한 규칙:
1. 위치 질문('어디야', '어디있어', '찾아')은 query_facility 사용
2. 이동 요청('안내해줘', '데려다줘', '동행해줘', '가자', '가져다줘')은 navigate 사용  
3. 일반 대화('안녕', '고마워', '뭐야')는 general_response 사용
4. 응답은 간결하고 자연스럽게 (길고 현학적인 답변 금지)
5. 대화 맥락을 고려하여 이전 언급된 장소를 기억하세요

${if (conversationContext.isNotEmpty()) "이전 대화 맥락:$conversationContext" else ""}

사용자 질문: $userInput"""
    }
    
    /**
     * TensorFlow Lite를 사용한 생성 (기존 로직 활용)
     */
    private fun generateWithTensorFlowLite(prompt: String): String {
        return try {
            // 기존 generateResponseWithContext 로직 활용
            generateResponseWithContext("", prompt)
        } catch (e: Exception) {
            Log.e(TAG, "TensorFlow Lite 생성 실패: ${e.message}")
            ""
        }
    }
    
    /**
     * 함수 호출 파싱 및 실행
     */
    private fun parseAndExecuteToolCall(response: String, userInput: String): String {
        try {
            // <tool_call> 태그에서 JSON 추출 (간단한 패턴 매칭)
            val toolCallRegex = "<tool_call>\\s*([^<]*)\\s*</tool_call>".toRegex()
            val matches = toolCallRegex.findAll(response)
            
            debugPrint("🔍 찾은 함수 호출: ${matches.count()}개")
            
            for (match in matches) {
                val toolCallJson = match.groupValues[1].trim()
                debugPrint("🔍 검사 중인 JSON: $toolCallJson")
                
                try {
                    val toolCall = JSONObject(toolCallJson)
                    val functionName = toolCall.getString("name")
                    val arguments = toolCall.getJSONObject("arguments")
                    
                    debugPrint("🔧 실행할 함수: $functionName($arguments)")
                    
                    return when (functionName) {
                        "query_facility" -> {
                            val facility = arguments.getString("facility")
                            val result = robotFunctions.queryFacility(facility)
                            
                            if (result["success"] as Boolean) {
                                "네! ${facility}는 ${result["result"]}에 있어요. 😊"
                            } else {
                                "죄송해요, ${facility}는 이 병원에 없는 시설이에요. 다른 시설을 찾아드릴까요?"
                            }
                        }
                        "navigate" -> {
                            val target = arguments.getString("target")
                            val result = robotFunctions.navigate(target)
                            
                            if (result["success"] as Boolean) {
                                "좋아요! ${target}로 안내해드릴게요. 저를 따라오세요! 🚀"
                            } else {
                                "죄송해요, ${target}를 찾을 수 없어요. 정확한 시설명을 말씀해주시겠어요?"
                            }
                        }
                        "general_response" -> {
                            val message = arguments.getString("message")
                            val result = robotFunctions.generalResponse(message)
                            result["result"] as String
                        }
                        else -> {
                            debugPrint("❌ 알 수 없는 함수: $functionName")
                            continue
                        }
                    }
                } catch (e: Exception) {
                    debugPrint("❌ JSON 파싱 실패: ${e.message}")
                    continue
                }
            }
            
            // 모든 함수 호출 실패 시 fallback
            debugPrint("❌ 모든 함수 호출 실패")
            return fallbackResponse(userInput)
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ 함수 호출 파싱 실패: ${e.message}")
            return fallbackResponse(userInput)
        }
    }
    
    /**
     * 맥락에서 최근 언급된 시설명 추출
     */
    private fun extractRecentFacility(): String? {
        if (conversationHistory.isEmpty()) return null
        
        val recentEntries = conversationHistory.takeLast(6)
        val facilities = mapOf(
            "ct" to "CT", "x-ray" to "X-ray", "엑스레이" to "X-ray", "xray" to "X-ray",
            "초음파" to "초음파", "폐암" to "폐암", "위암" to "위암", "대장암" to "대장암",
            "유방암" to "유방암", "뇌종양" to "뇌종양", "시티" to "CT", "씨티" to "CT"
        )
        
        for (entry in recentEntries.reversed()) {
            val content = entry.content.lowercase()
            for ((key, facility) in facilities) {
                if (content.contains(key)) {
                    return facility
                }
            }
        }
        
        return null
    }
    
    /**
     * 개선된 fallback 응답
     */
    private fun fallbackResponse(userInput: String): String {
        val userLower = userInput.lowercase().trim()
        val recentFacility = extractRecentFacility()
        
        return when {
            // 이동 요청인데 목적지가 명확하지 않은 경우
            userLower.any { it in listOf("안내", "가자", "동행", "데려다", "이동") } -> {
                if (recentFacility != null) {
                    "${recentFacility}로 안내해드릴까요?"
                } else {
                    "어디로 안내해드릴까요?"
                }
            }
            // 위치 질문인데 시설명이 명확하지 않은 경우
            userLower.any { it in listOf("어디", "위치", "찾아") } -> {
                "어떤 시설을 찾으시나요? CT, X-ray, 초음파, 각종 암센터 등이 있어요."
            }
            // 인사
            userLower.any { it in listOf("안녕", "hello", "hi") } -> {
                "안녕하세요! 저는 병원 안내 로봇 영웅이입니다. 무엇을 도와드릴까요?"
            }
            // 감사
            userLower.any { it in listOf("고마", "감사", "thank") } -> {
                "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요."
            }
            // 기본
            else -> {
                "무엇을 도와드릴까요? 병원 시설 안내나 위치 조회를 도와드릴 수 있어요."
            }
        }
    }
    
    fun generateResponse(inputText: String): String {
        try {
            // 텍스트 토크나이징
            val inputIds = tokenizer.encode(inputText)
            
            // 입력 버퍼 생성 (모델이 기대하는 형태: [1, 1])
            val inputBuffer = ByteBuffer.allocateDirect(4) // 1 * 1 * 4 bytes
            inputBuffer.order(ByteOrder.nativeOrder())
            inputBuffer.putInt(inputIds.firstOrNull() ?: 0)
            
            // 출력 버퍼 생성 (vocab_size = 102400)
            val vocabSize = 102400
            val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4)
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // 추론 실행
            interpreter?.run(inputBuffer, outputBuffer)
            
            // 결과 디코딩
            return tokenizer.decodeFromBuffer(outputBuffer)
            
        } catch (e: Exception) {
            e.printStackTrace()
            return "오류가 발생했습니다: ${e.message}"
        }
    }
    
    fun generateResponseWithContext(context: String, userInput: String): String {
        try {
            // 컨텍스트와 사용자 입력을 결합
            val fullInput = "$context\n사용자: $userInput\n어시스턴트:"
            
            // 텍스트 토크나이징
            val inputIds = tokenizer.encode(fullInput)
            
            // 입력 버퍼 생성
            val inputBuffer = ByteBuffer.allocateDirect(4)
            inputBuffer.order(ByteOrder.nativeOrder())
            inputBuffer.putInt(inputIds.firstOrNull() ?: 0)
            
            // 출력 버퍼 생성
            val vocabSize = 102400
            val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4)
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // 추론 실행
            interpreter?.run(inputBuffer, outputBuffer)
            
            // 결과 디코딩
            return tokenizer.decodeFromBuffer(outputBuffer)
            
        } catch (e: Exception) {
            e.printStackTrace()
            return "오류가 발생했습니다: ${e.message}"
        }
    }
    
    // 설정 함수들
    fun setReasoningMode(enabled: Boolean) {
        useReasoningMode = enabled
        debugPrint("${if (enabled) "🧠 Reasoning" else "💬 Non-reasoning"} 모드로 설정됨")
    }
    
    fun setFastMode(enabled: Boolean) {
        fastMode = enabled
        debugPrint("⚡ 빠른 응답 모드: $enabled")
    }
    
    fun setDebugMode(enabled: Boolean) {
        debugMode = enabled
        debugPrint("🔍 디버그 모드: $enabled")
    }
    
    fun clearHistory() {
        conversationHistory.clear()
        lastUserQuestion = ""
        lastResponse = ""
        debugPrint("🗑️ 대화 히스토리 초기화됨")
    }
    
    fun getHistorySize(): Int = conversationHistory.size
    
    fun isModelLoaded(): Boolean {
        return interpreter != null
    }
    
    fun getModelInfo(): String {
        return config?.toString() ?: "모델 정보를 불러올 수 없습니다."
    }
    
    fun close() {
        interpreter?.close()
    }
}

// 사용 예시를 위한 확장 함수들
suspend fun EXAONEModel.generateHospitalResponse(userQuery: String): String {
    return processUserInput(userQuery)
}

suspend fun EXAONEModel.generateSimpleResponse(userInput: String): String {
    return processUserInput(userInput)
} 
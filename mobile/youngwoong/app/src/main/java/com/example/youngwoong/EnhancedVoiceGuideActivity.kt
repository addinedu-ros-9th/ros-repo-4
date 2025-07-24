package com.example.youngwoong

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect

/**
 * 고도화된 EXAONE 병원 안내 로봇 데모 액티비티
 * Python의 프롬프트 엔지니어링 시스템을 완전히 포팅한 예제
 */
class EnhancedVoiceGuideActivity : AppCompatActivity() {
    
    private lateinit var robotSystem: RobotSystem
    private lateinit var inputText: EditText
    private lateinit var responseText: TextView
    private lateinit var statusText: TextView
    private lateinit var sendButton: Button
    private lateinit var clearButton: Button
    private lateinit var settingsButton: Button
    
    // 설정 변수들
    private var useRealModel = false // 시뮬레이션 모드로 시작
    private var useReasoning = false
    private var fastMode = true
    private var debugMode = true
    
    companion object {
        private const val TAG = "EnhancedVoiceGuide"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_guide)
        
        initializeViews()
        initializeRobotSystem()
        setupEventListeners()
        setupStreamingObservers()
    }
    
    private fun initializeViews() {
        // 기존 레이아웃의 뷰들을 재활용
        inputText = findViewById(R.id.input_text) ?: EditText(this).apply {
            hint = "영웅이에게 무엇이든 물어보세요!"
        }
        
        responseText = findViewById(R.id.response_text) ?: TextView(this).apply {
            text = "안녕하세요! 저는 병원 안내 로봇 영웅이입니다. 🤖"
        }
        
        statusText = findViewById(R.id.status_text) ?: TextView(this).apply {
            text = "시스템 초기화 중..."
        }
        
        sendButton = findViewById(R.id.send_button) ?: Button(this).apply {
            text = "질문하기"
        }
        
        clearButton = findViewById(R.id.clear_button) ?: Button(this).apply {
            text = "대화 초기화"
        }
        
        settingsButton = findViewById(R.id.settings_button) ?: Button(this).apply {
            text = "설정"
        }
        
        // 초기 상태 설정
        sendButton.isEnabled = false
        responseText.text = "🔄 시스템을 초기화하고 있습니다...\n잠시만 기다려주세요."
    }
    
    private fun initializeRobotSystem() {
        lifecycleScope.launch {
            try {
                // 로봇 시스템 빌더를 사용하여 생성
                robotSystem = RobotSystemBuilder(this@EnhancedVoiceGuideActivity)
                    .useRealModel(useRealModel)
                    .useReasoning(useReasoning)
                    .enableFastMode(fastMode)
                    .enableDebugMode(debugMode)
                    .build()
                
                // 시스템 초기화
                robotSystem.initialize()
                
                // UI 업데이트
                runOnUiThread {
                    sendButton.isEnabled = true
                    statusText.text = "✅ 시스템 준비 완료!"
                    responseText.text = """
                        🤖 안녕하세요! 저는 병원 안내 로봇 영웅이입니다.
                        
                        💡 이렇게 물어보세요:
                        • "CT 어디야?" - 시설 위치 조회
                        • "초음파실로 안내해줘" - 네비게이션  
                        • "왜 CT와 X-ray가 다른가요?" - 복잡한 질문
                        • "안녕하세요" - 일반 대화
                        
                        🧠 Reasoning 모드: ${if (useReasoning) "활성화" else "비활성화"}
                        ⚡ 빠른 모드: $fastMode
                        📊 모델: ${if (useRealModel) "실제 EXAONE" else "시뮬레이션"}
                    """.trimIndent()
                }
                
                Log.d(TAG, "✅ 로봇 시스템 초기화 완료")
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ 로봇 시스템 초기화 실패: ${e.message}")
                runOnUiThread {
                    statusText.text = "❌ 초기화 실패"
                    responseText.text = "시스템 초기화에 실패했습니다.\n다시 시도해주세요."
                    Toast.makeText(this@EnhancedVoiceGuideActivity, 
                        "시스템 초기화 실패: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
    
    private fun setupEventListeners() {
        // 질문 전송 버튼
        sendButton.setOnClickListener {
            val userInput = inputText.text.toString().trim()
            if (userInput.isNotEmpty()) {
                processUserInput(userInput)
                inputText.text.clear()
            } else {
                Toast.makeText(this, "질문을 입력해주세요!", Toast.LENGTH_SHORT).show()
            }
        }
        
        // 대화 초기화 버튼
        clearButton.setOnClickListener {
            robotSystem.clearHistory()
            responseText.text = "🗑️ 대화 히스토리가 초기화되었습니다.\n새로운 대화를 시작해보세요!"
            Toast.makeText(this, "대화 히스토리가 초기화되었습니다.", Toast.LENGTH_SHORT).show()
        }
        
        // 설정 버튼
        settingsButton.setOnClickListener {
            showSettingsDialog()
        }
        
        // Enter 키로 질문 전송
        inputText.setOnEditorActionListener { _, _, _ ->
            sendButton.performClick()
            true
        }
    }
    
    private fun setupStreamingObservers() {
        // 실시간 스트리밍 응답 관찰
        lifecycleScope.launch {
            robotSystem.responseFlow.collect { newText ->
                runOnUiThread {
                    responseText.append(newText)
                }
            }
        }
        
        // 완료된 응답 관찰
        lifecycleScope.launch {
            robotSystem.completedResponseFlow.collect { finalResponse ->
                runOnUiThread {
                    // 스트리밍이 없었다면 전체 응답 표시
                    if (!finalResponse.isBlank()) {
                        responseText.text = "🤖 영웅이: $finalResponse"
                    }
                    
                    sendButton.isEnabled = true
                    statusText.text = "✅ 응답 완료"
                }
            }
        }
        
        // 시스템 상태 관찰
        lifecycleScope.launch {
            robotSystem.systemStatusFlow.collect { status ->
                runOnUiThread {
                    statusText.text = status
                }
            }
        }
    }
    
    private fun processUserInput(userInput: String) {
        // UI 상태 업데이트
        sendButton.isEnabled = false
        statusText.text = "🤖 영웅이가 생각하고 있어요..."
        responseText.text = "🤖 영웅이: "
        
        // 백그라운드에서 처리
        lifecycleScope.launch {
            try {
                Log.d(TAG, "👤 사용자 질문: $userInput")
                
                // 스트리밍 지원 모드로 처리
                robotSystem.processUserInputWithStreaming(userInput)
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ 질문 처리 실패: ${e.message}")
                runOnUiThread {
                    responseText.text = "🤖 영웅이: 죄송해요, 잠시 문제가 생겼어요. 다시 한 번 말씀해주시겠어요?"
                    sendButton.isEnabled = true
                    statusText.text = "❌ 오류 발생"
                    Toast.makeText(this@EnhancedVoiceGuideActivity, 
                        "오류: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    private fun showSettingsDialog() {
        val options = arrayOf(
            "${if (useRealModel) "✅" else "❌"} 실제 EXAONE 모델 사용",
            "${if (useReasoning) "✅" else "❌"} Reasoning 모드",
            "${if (fastMode) "✅" else "❌"} 빠른 응답 모드",
            "${if (debugMode) "✅" else "❌"} 디버그 모드"
        )
        
        val builder = androidx.appcompat.app.AlertDialog.Builder(this)
        builder.setTitle("🔧 로봇 설정")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> {
                        useRealModel = !useRealModel
                        Toast.makeText(this, 
                            "실제 모델 사용: ${if (useRealModel) "활성화" else "비활성화"}", 
                            Toast.LENGTH_SHORT).show()
                        // 재시작 필요 알림
                        Toast.makeText(this, "설정 적용을 위해 앱을 다시 시작해주세요.", Toast.LENGTH_LONG).show()
                    }
                    1 -> {
                        useReasoning = !useReasoning
                        robotSystem.updateSettings(useReasoning = useReasoning)
                        Toast.makeText(this, 
                            "Reasoning 모드: ${if (useReasoning) "활성화" else "비활성화"}", 
                            Toast.LENGTH_SHORT).show()
                    }
                    2 -> {
                        fastMode = !fastMode
                        robotSystem.updateSettings(fastMode = fastMode)
                        Toast.makeText(this, 
                            "빠른 모드: ${if (fastMode) "활성화" else "비활성화"}", 
                            Toast.LENGTH_SHORT).show()
                    }
                    3 -> {
                        debugMode = !debugMode
                        robotSystem.updateSettings(debugMode = debugMode)
                        Toast.makeText(this, 
                            "디버그 모드: ${if (debugMode) "활성화" else "비활성화"}", 
                            Toast.LENGTH_SHORT).show()
                    }
                }
            }
            .setNegativeButton("닫기", null)
            .show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        if (::robotSystem.isInitialized) {
            robotSystem.close()
        }
        Log.d(TAG, "🔒 액티비티 종료됨")
    }
    
    override fun onPause() {
        super.onPause()
        // 필요시 스트리밍 일시정지
    }
    
    override fun onResume() {
        super.onResume()
        // 필요시 스트리밍 재개
    }
}

/**
 * 사용 예제를 위한 헬퍼 클래스
 */
class RobotUsageExamples {
    
    companion object {
        
        /**
         * 간단한 사용 예제
         */
        suspend fun simpleUsageExample(context: android.content.Context) {
            // 1. 로봇 시스템 생성 및 초기화
            val robot = RobotSystemBuilder(context)
                .useRealModel(false) // 시뮬레이션 모드
                .enableFastMode(true)
                .build()
            
            robot.initialize()
            
            // 2. 간단한 질문들
            val greeting = robot.ask("안녕하세요")
            println("🤖 $greeting")
            
            val facilityQuery = robot.findFacility("CT")
            println("🤖 $facilityQuery")
            
            val navigation = robot.navigateTo("초음파실")
            println("🤖 $navigation")
            
            // 3. 리소스 정리
            robot.close()
        }
        
        /**
         * 고급 사용 예제 (스트리밍 포함)
         */
        suspend fun advancedUsageExample(context: android.content.Context) {
            val robot = RobotSystemBuilder(context)
                .useRealModel(true) // 실제 모델 사용
                .useReasoning(true) // Reasoning 모드
                .enableDebugMode(true)
                .build()
            
            robot.initialize()
            
            // 스트리밍 응답 관찰
            kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.Main).launch {
                robot.responseFlow.collect { text ->
                    print(text) // 실시간 출력
                }
            }
            
            // 복잡한 질문 처리
            robot.processUserInputWithStreaming("왜 CT와 X-ray의 원리가 다른가요? 자세히 설명해주세요.")
            
            robot.close()
        }
        
        /**
         * 특정 시나리오별 테스트
         */
        suspend fun scenarioTests(context: android.content.Context) {
            val robot = RobotSystemBuilder(context)
                .useRealModel(false)
                .build()
            
            robot.initialize()
            
            // 시나리오 1: 시설 찾기
            println("=== 시설 찾기 테스트 ===")
            listOf("CT", "X-ray", "초음파", "뇌종양").forEach { facility ->
                val response = robot.findFacility(facility)
                println("Q: ${facility} 어디야? → A: $response")
            }
            
            // 시나리오 2: 네비게이션
            println("\n=== 네비게이션 테스트 ===")
            listOf("CT실", "초음파실", "암센터").forEach { destination ->
                val response = robot.navigateTo(destination)
                println("Q: ${destination}로 가자 → A: $response")
            }
            
            // 시나리오 3: 일반 대화
            println("\n=== 일반 대화 테스트 ===")
            listOf("안녕하세요", "고마워요", "영웅이는 누구야?").forEach { message ->
                val response = robot.handleGeneralChat(message)
                println("Q: $message → A: $response")
            }
            
            robot.close()
        }
    }
} 
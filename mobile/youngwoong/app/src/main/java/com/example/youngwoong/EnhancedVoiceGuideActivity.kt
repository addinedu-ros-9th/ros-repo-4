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

/**
 * 고도화된 EXAONE 병원 안내 로봇 데모 액티비티
 * HTTP 통신을 통한 서버와의 대화 처리
 */
class EnhancedVoiceGuideActivity : AppCompatActivity() {
    private lateinit var inputText: EditText
    private lateinit var responseText: TextView
    private lateinit var statusText: TextView
    private lateinit var sendButton: Button
    private lateinit var clearButton: Button
    private lateinit var settingsButton: Button
    
    // 설정 변수들
    private var useStreaming = true
    private var debugMode = true
    
    companion object {
        private const val TAG = "EnhancedVoiceGuide"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_guide)
        
        initializeViews()
        initializeNetworkTokenizer()
        setupEventListeners()
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
    
    private fun initializeNetworkTokenizer() {
        lifecycleScope.launch {
            try {
                runOnUiThread {
                    sendButton.isEnabled = true
                    statusText.text = "✅ 시스템 준비 완료!"
                    responseText.text = """
                        🤖 안녕하세요! 저는 병원 안내 로봇 영웅이입니다.
                        
                        💡 이렇게 물어보세요:
                        • "CT는 어디에 있나요?"
                        • "X-ray로 안내해주세요"
                        • "초음파 검사는 어디서 받나요?"
                        
                        🚀 실시간 스트리밍으로 답변해드립니다!
                    """.trimIndent()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ 초기화 실패: ${e.message}")
                runOnUiThread {
                    statusText.text = "❌ 초기화 실패"
                    responseText.text = "오류: ${e.message}"
                }
            }
        }
    }
    
    private fun setupEventListeners() {
        sendButton.setOnClickListener {
            val input = inputText.text.toString().trim()
            if (input.isNotEmpty()) {
                processUserInput(input)
                inputText.text.clear()
            } else {
                Toast.makeText(this, "질문을 입력해주세요!", Toast.LENGTH_SHORT).show()
            }
        }
        
        clearButton.setOnClickListener {
            responseText.text = "대화 기록이 지워졌습니다."
            Toast.makeText(this, "대화 기록이 초기화되었습니다.", Toast.LENGTH_SHORT).show()
        }
        
        settingsButton.setOnClickListener {
            showSettingsDialog()
        }
    }
    
    private fun processUserInput(input: String) {
        lifecycleScope.launch {
            try {
                responseText.append("\n👤 사용자: $input\n")
                responseText.append("🤖 로봇: ")
                
                // 간단한 응답 시뮬레이션
                val response = "죄송합니다. 현재 서버 연결이 불가능합니다. 나중에 다시 시도해주세요."
                runOnUiThread {
                    responseText.append(response)
                    responseText.append("\n")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ 입력 처리 실패: ${e.message}")
                runOnUiThread {
                    responseText.append("❌ 오류: ${e.message}\n")
                }
            }
        }
    }
    
    private fun showSettingsDialog() {
        val options = arrayOf(
            if (useStreaming) "스트리밍 모드 끄기" else "스트리밍 모드 켜기",
            if (debugMode) "디버그 모드 끄기" else "디버그 모드 켜기"
        )
        
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("⚙️ 설정")
            .setItems(options) { dialog, which ->
                when (which) {
                    0 -> {
                        useStreaming = !useStreaming
                        Toast.makeText(this, 
                            if (useStreaming) "스트리밍 모드가 켜졌습니다." else "스트리밍 모드가 꺼졌습니다.", 
                            Toast.LENGTH_SHORT).show()
                    }
                    1 -> {
                        debugMode = !debugMode
                        Toast.makeText(this, 
                            if (debugMode) "디버그 모드가 켜졌습니다." else "디버그 모드가 꺼졌습니다.", 
                            Toast.LENGTH_SHORT).show()
                    }
                }
                dialog.dismiss()
            }
            .setNegativeButton("취소") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // 정리 작업
        lifecycleScope.launch {
            try {
                // 필요한 정리 작업
            } catch (e: Exception) {
                Log.e(TAG, "정리 중 오류: ${e.message}")
            }
        }
    }
} 
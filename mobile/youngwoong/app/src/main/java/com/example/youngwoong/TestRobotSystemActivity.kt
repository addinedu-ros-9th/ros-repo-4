package com.example.youngwoong

import android.os.Bundle
import android.util.Log
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * 로봇 시스템 테스트 액티비티
 * HTTP 통신을 통한 LLM 서버와의 대화 테스트
 */
class TestRobotSystemActivity : AppCompatActivity() {
    private lateinit var responseTextView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var inputEditText: EditText
    private lateinit var sendButton: Button
    private lateinit var clearButton: Button
    private lateinit var streamingToggleButton: Button
    
    private var isStreamingMode = true
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()
    
    companion object {
        private const val TAG = "TestRobotSystem"
        private val BASE_URL = NetworkConfig.getLlmServerUrl() // LLM 서버 URL
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test_robot_system)
        
        initializeViews()
        setupButtons()
        initializeNetworkTokenizer()
    }
    
    private fun initializeViews() {
        responseTextView = findViewById(R.id.responseTextView)
        statusTextView = findViewById(R.id.statusTextView)
        inputEditText = findViewById(R.id.inputEditText)
        sendButton = findViewById(R.id.sendButton)
        clearButton = findViewById(R.id.clearButton)
        streamingToggleButton = findViewById(R.id.reasoningToggleButton) // 기존 버튼 재사용
    }
    
    private fun setupButtons() {
        sendButton.setOnClickListener {
            val input = inputEditText.text.toString().trim()
            if (input.isNotEmpty()) {
                processUserInput(input)
                inputEditText.text.clear()
            }
        }
        
        clearButton.setOnClickListener {
            responseTextView.text = ""
            statusTextView.text = "대화 기록이 지워졌습니다."
        }
        
        streamingToggleButton.setOnClickListener {
            isStreamingMode = !isStreamingMode
            updateStreamingButton()
        }
        
        updateStreamingButton()
    }
    
    private fun updateStreamingButton() {
        streamingToggleButton.text = if (isStreamingMode) "스트리밍 ON" else "스트리밍 OFF"
        streamingToggleButton.setBackgroundColor(
            if (isStreamingMode) 0xFF2196F3.toInt() else 0xFF9E9E9E.toInt()
        )
    }
    
    private fun initializeNetworkTokenizer() {
        lifecycleScope.launch {
            try {
                statusTextView.text = "🔄 서버 연결 확인 중..."
                
                // 서버 상태 확인
                val isServerHealthy = checkServerHealth()
                if (isServerHealthy) {
                    statusTextView.text = "✅ 서버 연결 성공! 대화를 시작하세요."
                    Log.d(TAG, "✅ 서버 연결 성공")
                } else {
                    statusTextView.text = "❌ 서버 연결 실패. PC 서버를 확인해주세요."
                    Log.e(TAG, "❌ 서버 연결 실패")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ 초기화 실패: ${e.message}")
                statusTextView.text = "❌ 초기화 실패: ${e.message}"
                responseTextView.text = "오류: ${e.message}"
            }
        }
    }
    
    private suspend fun checkServerHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$BASE_URL/api/health")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            val isHealthy = response.isSuccessful
            Log.d(TAG, "서버 상태 확인: ${if (isHealthy) "정상" else "비정상"}")
            return@withContext isHealthy
        } catch (e: Exception) {
            Log.e(TAG, "서버 상태 확인 실패: ${e.message}")
            return@withContext false
        }
    }
    
    private fun processUserInput(input: String) {
        lifecycleScope.launch {
            try {
                responseTextView.append("\n👤 사용자: $input\n")
                responseTextView.append("🤖 로봇: ")
                
                if (isStreamingMode) {
                    // 실시간 스트리밍 모드
                    sendStreamingMessage(input)
                } else {
                    // 일반 모드
                    val response = sendMessage(input)
                    runOnUiThread {
                        responseTextView.append(response)
                        responseTextView.append("\n")
                    }
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ 입력 처리 실패: ${e.message}")
                runOnUiThread {
                    responseTextView.append("❌ 오류: ${e.message}\n")
                }
            }
        }
    }
    
    private suspend fun sendMessage(message: String): String = withContext(Dispatchers.IO) {
        try {
            val jsonBody = JSONObject().apply {
                put("message", message)
            }
            
            val requestBody = jsonBody.toString().toRequestBody(jsonMediaType)
            val request = Request.Builder()
                .url("$BASE_URL/api/chat")
                .post(requestBody)
                .build()
            
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                val responseBody = response.body?.string()
                val jsonResponse = JSONObject(responseBody ?: "{}")
                val reply = jsonResponse.optString("content", "응답이 없습니다.")
                Log.d(TAG, "메시지 전송 성공: ${reply.length}자")
                return@withContext reply
            } else {
                Log.e(TAG, "메시지 전송 실패: ${response.code}")
                return@withContext "서버 오류가 발생했습니다."
            }
        } catch (e: Exception) {
            Log.e(TAG, "메시지 전송 오류: ${e.message}")
            return@withContext "네트워크 오류가 발생했습니다."
        }
    }
    
    private suspend fun sendStreamingMessage(message: String) = withContext(Dispatchers.IO) {
        try {
            val jsonBody = JSONObject().apply {
                put("message", message)
            }
            
            val requestBody = jsonBody.toString().toRequestBody(jsonMediaType)
            val request = Request.Builder()
                .url("$BASE_URL/api/stream")
                .post(requestBody)
                .build()
            
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    response.body?.source()?.let { source ->
                        while (!source.exhausted()) {
                            val line = source.readUtf8LineStrict()
                            if (line.startsWith("data: ")) {
                                val data = line.substring(6)
                                try {
                                    val jsonData = JSONObject(data)
                                    val type = jsonData.optString("type", "")
                                    val content = jsonData.optString("content", "")
                                    
                                    when (type) {
                                        "stream", "token" -> {
                                            runOnUiThread {
                                                responseTextView.append(content)
                                            }
                                        }
                                        "complete" -> {
                                            runOnUiThread {
                                                responseTextView.append("\n")
                                            }
                                            break
                                        }
                                        "error" -> {
                                            runOnUiThread {
                                                responseTextView.append("❌ 오류: $content\n")
                                            }
                                            break
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e(TAG, "스트림 데이터 파싱 오류: ${e.message}")
                                }
                            }
                        }
                    }
                } else {
                    Log.e(TAG, "스트리밍 요청 실패: ${response.code}")
                    runOnUiThread {
                        responseTextView.append("❌ 스트리밍 요청 실패\n")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "스트리밍 오류: ${e.message}")
            runOnUiThread {
                responseTextView.append("❌ 스트리밍 오류: ${e.message}\n")
            }
        }
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
    
    /**
     * 테스트 질문 버튼 클릭 처리
     */
    fun onTestQuestionsClick(view: android.view.View) {
        showTestQuestionsDialog()
    }
    
    private fun showTestQuestionsDialog() {
        val questions = arrayOf(
            "안녕하세요, 병원 안내 로봇입니다.",
            "CT는 어디에 있나요?",
            "X-ray는 어디에 있나요?",
            "초음파는 어디에 있나요?",
            "폐암센터로 안내해주세요.",
            "위암센터는 어디에 있나요?",
            "대장암 검사는 어디서 받나요?",
            "유방암 검사는 어디서 받나요?",
            "뇌종양 치료는 어디서 받나요?",
            "응급실은 어디에 있나요?"
        )
        
        AlertDialog.Builder(this)
            .setTitle("🧪 테스트 질문 선택")
            .setItems(questions) { dialog, which ->
                val selectedQuestion = questions[which]
                inputEditText.setText(selectedQuestion)
                processUserInput(selectedQuestion)
                dialog.dismiss()
            }
            .setNegativeButton("취소") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }
} 
package com.example.youngwoong

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.View
import android.view.animation.AlphaAnimation
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.airbnb.lottie.LottieAnimationView
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.*

class VoiceGuideActivity : AppCompatActivity() {

    private var isListening = false
    private lateinit var voiceAnimation: LottieAnimationView
    private lateinit var dimView: View
    private lateinit var textPrompt: TextView
    private lateinit var textUserMessage: TextView
    private lateinit var textBotMessage: TextView
    private var blinkAnimation: AlphaAnimation? = null

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechIntent: Intent
    private lateinit var textToSpeech: TextToSpeech
    
    // LLM 서버 통신을 위한 변수들
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()
    
    companion object {
        private const val TAG = "VoiceGuideActivity"
        private const val BASE_URL = "http://192.168.0.31:5000" // Flask 서버 URL
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_guide)

        checkPermissions()
        setupSTT()
        setupTTS()

        voiceAnimation = findViewById(R.id.voice_animation)
        dimView = findViewById(R.id.dim_view)
        textPrompt = findViewById(R.id.text_prompt)
        textUserMessage = findViewById(R.id.text_user_message)
        textBotMessage = findViewById(R.id.text_bot_message)

        findViewById<ImageView>(R.id.btn_back).setOnClickListener {
            applyAlphaEffect(it)
            returnToMainMenu()
        }

        findViewById<ImageView>(R.id.btn_voice).setOnClickListener {
            applyAlphaEffect(it)
            toggleListening(it as ImageView)
        }
    }

    private fun checkPermissions() {
        val permissions = listOf(Manifest.permission.RECORD_AUDIO)
        val notGranted = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (notGranted.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, notGranted.toTypedArray(), 1000)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1000 && grantResults.any { it != PackageManager.PERMISSION_GRANTED }) {
            Toast.makeText(this, "🎙️ 음성 인식 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }

    private fun returnToMainMenu() {
        val intent = Intent(this, MainMenuActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    private fun toggleListening(button: ImageView) {
        isListening = !isListening

        if (isListening) {
            dimView.alpha = 0f
            dimView.visibility = View.VISIBLE
            dimView.animate().alpha(0.15f).setDuration(300).start()

            voiceAnimation.apply {
                alpha = 0f
                visibility = View.VISIBLE
                animate().alpha(1f).setDuration(300).withStartAction {
                    playAnimation()
                }.start()
            }

            textPrompt.apply {
                visibility = View.VISIBLE
                text = "듣고 있어요..."
                setTextColor(Color.parseColor("#FFA000"))
                startBlinking(this)
            }

            speechRecognizer.startListening(speechIntent)
            button.setImageResource(R.drawable.btn_voice_stop)

        } else {
            dimView.animate().alpha(0f).setDuration(300).withEndAction {
                dimView.visibility = View.GONE
            }.start()

            voiceAnimation.animate().alpha(0f).setDuration(300).withEndAction {
                voiceAnimation.pauseAnimation()
                voiceAnimation.visibility = View.GONE
            }.start()

            textPrompt.apply {
                text = "터치로 대화를 시작합니다"
                setTextColor(Color.parseColor("#000000"))
                stopBlinking(this)
            }

            speechRecognizer.stopListening()
            speechRecognizer.cancel()
            button.setImageResource(R.drawable.btn_voice_start)
        }
    }

    private fun startBlinking(view: View) {
        blinkAnimation = AlphaAnimation(1.0f, 0.0f).apply {
            duration = 600
            repeatMode = AlphaAnimation.REVERSE
            repeatCount = AlphaAnimation.INFINITE
        }
        view.startAnimation(blinkAnimation)
    }

    private fun stopBlinking(view: View) {
        blinkAnimation?.cancel()
        view.clearAnimation()
    }

    private fun setupSTT() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)

        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}

            override fun onError(error: Int) {
                val errorMessage = when (error) {
                    SpeechRecognizer.ERROR_AUDIO -> "오디오 오류"
                    SpeechRecognizer.ERROR_CLIENT -> "클라이언트 오류"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "권한 부족"
                    SpeechRecognizer.ERROR_NETWORK -> "네트워크 오류"
                    SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "네트워크 시간 초과"
                    SpeechRecognizer.ERROR_NO_MATCH -> "음성 인식 실패"
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "인식기 사용 중"
                    SpeechRecognizer.ERROR_SERVER -> "서버 오류"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "음성 입력 시간 초과"
                    else -> "알 수 없는 오류"
                }
                
                Log.e("STT", "❌ 음성 인식 오류: $error ($errorMessage)")

                if (isListening) {
                    // 간단한 재시도 로직
                    speechRecognizer.cancel()
                    speechRecognizer.startListening(speechIntent)
                } else {
                    runOnUiThread {
                        textPrompt.text = "터치로 대화를 시작합니다"
                    }
                }
            }

            override fun onResults(results: Bundle?) {
                val result = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.getOrNull(0)

                result?.let { userMessage ->
                    runOnUiThread {
                        textUserMessage.text = userMessage
                        textPrompt.text = "🤖 로봇이 응답하고 있습니다..."
                        textBotMessage.text = ""
                    }
                    
                    // LLM 서버로 메시지 전송
                    sendMessageToLLM(userMessage)
                }

                if (isListening) {
                    speechRecognizer.startListening(speechIntent)
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        speechIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1) // 최대 결과 수만 추가
        }
    }
    
    private fun setupTTS() {
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                Log.d(TAG, "TTS 초기화 성공")
                
                // 사용 가능한 언어 목록 확인
                val availableLocales = textToSpeech.availableLanguages
                Log.d(TAG, "사용 가능한 언어: $availableLocales")
                
                // 한국어 설정 시도 (여러 방법)
                var languageSet = false
                
                // 1. Locale.KOREA 시도 (Java 예제와 동일)
                val koreaResult = textToSpeech.setLanguage(Locale.KOREA)
                if (koreaResult != TextToSpeech.LANG_MISSING_DATA && koreaResult != TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.d(TAG, "한국어 설정 성공 (Locale.KOREA)")
                    languageSet = true
                } else {
                    Log.d(TAG, "Locale.KOREA 설정 실패: $koreaResult")
                    
                    // 2. Locale.KOREAN 시도
                    val koreanResult = textToSpeech.setLanguage(Locale.KOREAN)
                    if (koreanResult != TextToSpeech.LANG_MISSING_DATA && koreanResult != TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.d(TAG, "한국어 설정 성공 (Locale.KOREAN)")
                        languageSet = true
                    } else {
                        Log.d(TAG, "Locale.KOREAN 설정 실패: $koreanResult")
                        
                        // 3. Locale("ko", "KR") 시도
                        val koKRResult = textToSpeech.setLanguage(Locale("ko", "KR"))
                        if (koKRResult != TextToSpeech.LANG_MISSING_DATA && koKRResult != TextToSpeech.LANG_NOT_SUPPORTED) {
                            Log.d(TAG, "한국어 설정 성공 (ko-KR)")
                            languageSet = true
                        } else {
                            Log.d(TAG, "ko-KR 설정 실패: $koKRResult")
                            
                            // 4. 기본 언어 사용
                            val defaultResult = textToSpeech.setLanguage(Locale.getDefault())
                            if (defaultResult != TextToSpeech.LANG_MISSING_DATA && defaultResult != TextToSpeech.LANG_NOT_SUPPORTED) {
                                Log.d(TAG, "기본 언어 설정 성공: ${Locale.getDefault()}")
                                languageSet = true
                            }
                        }
                    }
                }
                
                if (languageSet) {
                    // TTS 속도와 피치 설정
                    textToSpeech.setSpeechRate(0.9f) // 약간 느리게
                    textToSpeech.setPitch(1.0f) // 기본 피치
                    Log.d(TAG, "TTS 설정 완료 - 속도: 0.9, 피치: 1.0")
                } else {
                    Log.e(TAG, "모든 언어 설정 시도 실패")
                    Toast.makeText(this, "음성 합성 언어 설정에 실패했습니다.", Toast.LENGTH_SHORT).show()
                }
            } else {
                Log.e(TAG, "TTS 초기화 실패: $status")
                Toast.makeText(this, "음성 합성 초기화에 실패했습니다.", Toast.LENGTH_SHORT).show()
            }
        }
        
        // TTS 진행 상태 리스너 설정
        textToSpeech.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                Log.d(TAG, "TTS 시작: $utteranceId")
            }
            
            override fun onDone(utteranceId: String?) {
                Log.d(TAG, "TTS 완료: $utteranceId")
            }
            
            override fun onError(utteranceId: String?) {
                Log.e(TAG, "TTS 오류: $utteranceId")
            }
        })
    }
    
    private fun sendMessageToLLM(message: String) {
        lifecycleScope.launch {
            try {
                Log.d(TAG, "LLM 서버로 메시지 전송: $message")
                
                val response = sendMessageToServer(message)
                
                runOnUiThread {
                    textBotMessage.text = response
                    textPrompt.text = "터치로 대화를 시작합니다"
                    // TTS로 응답 음성 출력
                    speakResponse(response)
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "LLM 서버 통신 오류: ${e.message}")
                runOnUiThread {
                    textBotMessage.text = "죄송합니다. 서버 연결에 문제가 있습니다."
                    textPrompt.text = "터치로 대화를 시작합니다"
                    speakResponse("죄송합니다. 서버 연결에 문제가 있습니다.")
                }
            }
        }
    }
    
    private suspend fun sendMessageToServer(message: String): String = withContext(Dispatchers.IO) {
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
                Log.d(TAG, "서버 응답 원본: $responseBody")
                
                val jsonResponse = JSONObject(responseBody ?: "{}")
                Log.d(TAG, "JSON 응답: $jsonResponse")
                
                val reply = jsonResponse.optString("response", "응답이 없습니다.")
                Log.d(TAG, "파싱된 응답: $reply")
                Log.d(TAG, "LLM 응답 수신: ${reply.length}자")
                return@withContext reply
            } else {
                Log.e(TAG, "LLM 서버 응답 실패: ${response.code}")
                return@withContext "서버 오류가 발생했습니다."
            }
        } catch (e: Exception) {
            Log.e(TAG, "LLM 서버 통신 오류: ${e.message}")
            return@withContext "네트워크 오류가 발생했습니다."
        }
    }
    
    private fun speakResponse(text: String) {
        try {
            if (::textToSpeech.isInitialized) {
                Log.d(TAG, "TTS로 음성 출력: $text")
                val result = textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "response_utterance")
                if (result == TextToSpeech.ERROR) {
                    Log.e(TAG, "TTS 음성 출력 실패")
                } else {
                    Log.d(TAG, "TTS 음성 출력 성공")
                }
            } else {
                Log.e(TAG, "TTS가 초기화되지 않았습니다.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "TTS 오류: ${e.message}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
        if (::textToSpeech.isInitialized) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
    }
}

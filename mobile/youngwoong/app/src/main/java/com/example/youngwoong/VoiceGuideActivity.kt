package com.example.youngwoong

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.*
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
import androidx.activity.OnBackPressedCallback
import com.airbnb.lottie.LottieAnimationView
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.*

data class LLMResult(
    val reply: String,
    val functionName: String,
    val selectedText: String?,   // ← 여기 담아줄 것
    val statusCode: Int          // ← 200 체크용
)

class VoiceGuideActivity : AppCompatActivity() {

    private var isListening = false
    private var pendingFunctionName: String? = null
    private lateinit var voiceButton: ImageView
    private lateinit var voiceAnimation: LottieAnimationView
    private lateinit var dimView: View
    private lateinit var textPrompt: TextView
    private lateinit var textUserMessage: TextView
    private lateinit var textBotMessage: TextView
    private var blinkAnimation: AlphaAnimation? = null
    private var loadingJob: Job? = null
    private lateinit var streamer: AndroidStreamer
    private var pendingSelectedText: String? = null
    private var pendingStatusCode: Int? = null

    private val timeoutHandler = Handler(Looper.getMainLooper())
    private val timeoutRunnable = Runnable {
        Log.d(TAG, "🕒 VoiceGuideActivity 30초 타임아웃 발생")
        disableInteraction()
        sendTimeoutAlert()
        navigateToMain()
    }

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechIntent: Intent
    private lateinit var textToSpeech: TextToSpeech

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    companion object {
        private const val TAG = "VoiceGuideActivity"
        private val BASE_URL = NetworkConfig.getLlmServerUrl()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_guide)

        streamer = AndroidStreamer(debugMode = true)

        checkPermissions()
        setupSTT()
        setupTTS()

        voiceAnimation = findViewById(R.id.voice_animation)
        dimView = findViewById(R.id.dim_view)
        textPrompt = findViewById(R.id.text_prompt)
        textUserMessage = findViewById(R.id.text_user_message)
        textBotMessage = findViewById(R.id.text_bot_message)

        val backButton = findViewById<ImageView>(R.id.btn_back)
        val voiceButton = findViewById<ImageView>(R.id.btn_voice)

        // 🔐 나중에 제어를 위해 멤버 변수로 저장
        this.voiceButton = voiceButton

        // ✅ 뒤로가기 버튼 클릭 시 MainMenu로 이동
        backButton.setOnClickListener {
            applyAlphaEffect(it)
            val intent = Intent(this, MainMenuActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()
        }

        // ✅ 휴대폰 물리 뒤로가기 버튼 처리
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                val intent = Intent(this@VoiceGuideActivity, MainMenuActivity::class.java)
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }
        })

        // ✅ 음성 버튼
        voiceButton.setOnClickListener {
            applyAlphaEffect(it)
            toggleListening(voiceButton)
        }

        resetTimeoutTimer()
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

    override fun onUserInteraction() {
        super.onUserInteraction()
        resetTimeoutTimer()
    }

    override fun onResume() {
        super.onResume()
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_FULLSCREEN
        resetTimeoutTimer()
    }

    override fun onPause() {
        super.onPause()
        timeoutHandler.removeCallbacks(timeoutRunnable)
    }

    private fun resetTimeoutTimer() {
        timeoutHandler.removeCallbacks(timeoutRunnable)
        timeoutHandler.postDelayed(timeoutRunnable, 30_000) // 30초
    }

    private fun disableInteraction() {
        runOnUiThread {
            voiceButton.isEnabled = false
            findViewById<ImageView>(R.id.btn_back).isEnabled = false
            dimView.isClickable = true
        }
    }

    private fun sendTimeoutAlert() {
        val url = NetworkConfig.getTimeoutAlertUrl()
        val json = JSONObject().apply { put("robot_id", 3) }
        val body = json.toString().toRequestBody(jsonMediaType)
        val request = Request.Builder().url(url).post(body).build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "❌ Timeout 전송 실패: ${e.message}")
            }
            override fun onResponse(call: Call, response: Response) {
                Log.d(TAG, "✅ Timeout 전송 성공: ${response.code}")
            }
        })
    }

    private fun navigateToMain() {
        val intent = Intent(this, MainActivity::class.java).apply {
            putExtra("from_timeout", true)
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
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
                animate().alpha(1f).setDuration(300).withStartAction { playAnimation() }.start()
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
            override fun onBeginningOfSpeech() {
                resetTimeoutTimer()
            }
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
                    speechRecognizer.cancel()
                    speechRecognizer.startListening(speechIntent)
                } else {
                    runOnUiThread {
                        textPrompt.text = "터치로 대화를 시작합니다"
                    }
                }
            }

            override fun onResults(results: Bundle?) {
                val result = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.getOrNull(0)

                result?.let { userMessage ->
                    runOnUiThread {
                        isListening = false

                        dimView.animate().alpha(0f).setDuration(300).withEndAction {
                            dimView.visibility = View.GONE
                        }.start()

                        voiceAnimation.animate().alpha(0f).setDuration(300).withEndAction {
                            voiceAnimation.pauseAnimation()
                            voiceAnimation.visibility = View.GONE
                        }.start()

                        textPrompt.apply {
                            visibility = View.VISIBLE
                            setTextColor(Color.parseColor("#000000"))
                            stopBlinking(this)
                        }

                        findViewById<ImageView>(R.id.btn_voice).setImageResource(R.drawable.btn_voice_start)

                        textUserMessage.text = userMessage
                        startLoadingDots("🤖 로봇이 응답하고 있습니다", textPrompt)
                        textBotMessage.text = ""

                        voiceButton.isEnabled = false  // ✅ 버튼 비활성화
                    }
                    resetTimeoutTimer()
                    sendMessageToLLM(userMessage)
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        speechIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
    }

    private fun setupTTS() {
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech.setLanguage(Locale.KOREA)
                textToSpeech.setSpeechRate(0.9f)
                textToSpeech.setPitch(1.0f)

                // ✅ TTS 초기화 후 호출 여부 확인
                if (intent.getBooleanExtra("voice_triggered", false)) {
                    speakResponse("병원 안내 로봇 영웅이 입니다. 무엇을 도와드릴까요?")
                }

            } else {
                Toast.makeText(this, "TTS 초기화 실패", Toast.LENGTH_SHORT).show()
            }
        }
        textToSpeech.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) {
                runOnUiThread {
                    voiceButton.isEnabled = true
                    val patientId = intent.getStringExtra("patient_id") ?: "unknown"

                    when (pendingFunctionName) {
                        "appointment_service" -> {
                            startActivity(Intent(this@VoiceGuideActivity, AuthenticationActivity::class.java))
                            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                            finish()
                        }
                        "navigate" -> {
                            if (pendingStatusCode == 200) {
                                val dest = (pendingSelectedText?.takeIf { it.isNotBlank() }) ?: "병원 로비"
                                val i = Intent(this@VoiceGuideActivity, GuidanceWaitingActivity::class.java).apply {
                                    putExtra("selected_text", dest)
                                    putExtra("isFromCheckin", false)
                                    putExtra("patient_id", patientId)
                                }
                                startActivity(i)
                                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                                finish()
                            } else {
                                Log.w(TAG, "navigate 실패(status=$pendingStatusCode) → 화면 이동 안 함")
                                // 필요하면 Toast 표시:
                                // Toast.makeText(this@VoiceGuideActivity, "목적지를 다시 말씀해 주세요.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                    pendingFunctionName = null
                    pendingSelectedText = null
                    pendingStatusCode = null
                }
            }
            override fun onError(utteranceId: String?) {}
        })
    }

    private fun sendMessageToLLM(message: String) {
        lifecycleScope.launch {
            try {
                val result = sendMessageToServer(message)

                withContext(Dispatchers.Main) {
                    stopLoadingDots()

                    // 빈 응답 대비
                    val replyText = result.reply.ifBlank { "응답이 없습니다." }
                    textBotMessage.text = replyText
                    textPrompt.text = "터치로 대화를 시작합니다"

                    // 다음 화면 분기를 위한 상태 보관
                    pendingFunctionName = result.functionName.ifBlank { null }
                    pendingSelectedText = result.selectedText
                        ?.takeIf { it.isNotBlank() }
                        ?: textUserMessage.text?.toString()?.takeIf { it.isNotBlank() }

                    // (옵션) 상태코드 보관 → onDone에서 200 체크용
                    pendingStatusCode = result.statusCode

                    // 화면 전환은 오직 TTS onDone에서 처리
                    speakResponse(replyText)
                }
            } catch (e: Exception) {
                Log.e(TAG, "LLM 오류: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    stopLoadingDots()
                    val err = "죄송합니다. 서버 연결에 문제가 있습니다."
                    textBotMessage.text = err
                    textPrompt.text = "터치로 대화를 시작합니다"

                    // 실패 시 네비게이션 차단
                    pendingFunctionName = null
                    pendingSelectedText = null
                    pendingStatusCode = null

                    speakResponse(err)
                }
            }
        }
    }



    private suspend fun sendMessageToServer(message: String): LLMResult = withContext(Dispatchers.IO) {
        val jsonBody = JSONObject().apply { put("message", message) }
        val request = Request.Builder()
            .url("$BASE_URL/api/chat")
            .post(jsonBody.toString().toRequestBody(jsonMediaType))
            .build()

        val response = client.newCall(request).execute()
        val bodyString = response.body?.string()
        Log.d(TAG, "📦 LLM 원시 응답: $bodyString")

        if (response.isSuccessful && !bodyString.isNullOrBlank()) {
            try {
                val json = JSONObject(bodyString)

                val functionName = json.optString("function_name",
                    json.optString("function", ""))

                // 기본값(루트 기준)
                var statusCode = json.optInt("status_code", response.code)
                var selectedText: String? = json.optString("target", null)
                var reply = json.optString("response", "")

                // ✅ function_result 우선 반영
                val fnRes = json.opt("function_result")
                if (fnRes is JSONObject) {
                    statusCode = fnRes.optInt("status_code", statusCode)
                    selectedText = fnRes.optString("selected_text", null)
                        ?: fnRes.optString("target", null)
                                ?: fnRes.optString("destination", null)
                                ?: selectedText

                    val resultField = fnRes.opt("result")
                    when (resultField) {
                        is String -> if (statusCode != 200 || reply.isBlank()) reply = resultField
                        is JSONObject -> {
                            val msg = resultField.optString("message", "")
                            if (msg.isNotBlank() && (statusCode != 200 || reply.isBlank())) reply = msg
                        }
                    }
                }

                if (reply.isBlank()) reply = "응답이 없습니다."
                return@withContext LLMResult(reply, functionName, selectedText, statusCode)

            } catch (e: Exception) {
                Log.e(TAG, "❌ JSON 파싱 오류: ${e.message}")
                return@withContext LLMResult("응답 파싱 오류", "", null, response.code)
            }
        } else {
            Log.e(TAG, "❌ 서버 응답 오류: ${response.code}")
            return@withContext LLMResult("서버 오류가 발생했습니다.", "", null, response.code)
        }
    }

    private fun speakResponse(text: String) {
        if (!::textToSpeech.isInitialized) return
        resetTimeoutTimer()
        // 리스너는 setupTTS()에서만 설정합니다.
        // (원한다면 여기서 버튼만 잠깐 비활성화)
        // voiceButton.isEnabled = false
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "response_utterance")
    }


    private fun startLoadingDots(baseText: String, textView: TextView) {
        loadingJob?.cancel()
        loadingJob = lifecycleScope.launch {
            var dotCount = 0
            while (isActive) {
                val dots = ".".repeat(dotCount % 4)
                textView.text = baseText + dots
                dotCount++
                delay(500)
            }
        }
    }

    private fun stopLoadingDots() {
        loadingJob?.cancel()
        loadingJob = null
    }

    override fun onDestroy() {
        super.onDestroy()
        timeoutHandler.removeCallbacks(timeoutRunnable)
        speechRecognizer.destroy()
        if (::textToSpeech.isInitialized) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
        stopLoadingDots()
    }
}

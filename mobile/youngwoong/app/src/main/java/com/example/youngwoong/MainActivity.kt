package com.example.youngwoong

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.*
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import android.util.Log
import android.view.View
import androidx.activity.OnBackPressedCallback


class MainActivity : AppCompatActivity() {

    private lateinit var leftEye: ImageView
    private lateinit var rightEye: ImageView
    private lateinit var tapPrompt: ImageView
    private var testButton: Button? = null

    private val handler = Handler(Looper.getMainLooper())
    private var angle = 0.0
    private var currentOffsetAngle = 0.0
    private var targetOffsetAngle = 0.0
    private var currentVerticalOffset = 0.0
    private var targetVerticalOffset = 0.0

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechIntent: Intent
    private val wakeWords = listOf("영웅아", "영화", "영아", "영우아", "영웅이")

    private var isBlocked = false
    private var hasNavigatedToMenu = false
    private var blinkHandler: Handler? = null
    private var blinkRunnable: Runnable? = null

    private lateinit var webSocketClient: RobotStatusWebSocketClient

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        leftEye = findViewById(R.id.left_eye)
        rightEye = findViewById(R.id.right_eye)
        tapPrompt = findViewById(R.id.tap_prompt)
        testButton = findViewById(R.id.testButton)

        setupSTT()

        testButton?.apply {
            visibility = View.VISIBLE
            setOnClickListener {
                startActivity(Intent(this@MainActivity, TestRobotSystemActivity::class.java))
            }
        }

        val fromTimeout = intent.getBooleanExtra("from_timeout", false)
        Log.d("MainActivity", "📥 from_timeout 값: $fromTimeout")
        if (fromTimeout) {
            isBlocked = true
            tapPrompt.setImageResource(R.drawable.robot_returning_notice)
            stopPromptBlink()
        } else {
            startPromptBlink()
            startListening()
        }

        checkPermissions()
        startEyeAnimation()
        startPromptBlink()

        // 🔙 뒤로가기 버튼 대응 (AndroidX 방식)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (!isBlocked) {
                    finish()
                } else {
                    Log.d("MainActivity", "🚫 뒤로 가기 차단됨 (isBlocked=true)")
                }
            }
        })

        // WebSocket 연결
        webSocketClient = RobotStatusWebSocketClient(
            url = "ws://192.168.0.10:3000/?client_type=gui",
            targetRobotId = "3"
        ) { status -> handleRobotStatusChange(status) }
        webSocketClient.connect()
    }

    private fun checkPermissions() {
        val permissions = listOf(Manifest.permission.RECORD_AUDIO)
        val notGranted = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (notGranted.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, notGranted.toTypedArray(), 1001)
        } else {
            if (!isBlocked) {
                startListening() // ✅ 로봇 복귀 중이면 STT 시작 안 함
            }
        }
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
                speechRecognizer.cancel()
                startListening()
            }

            override fun onResults(results: Bundle?) {
                val result = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.getOrNull(0)
                    ?: return

                for (keyword in wakeWords) {
                    if (result.contains(keyword)) {
                        sendCallRequest("/call_with_voice") // 🔊 음성 호출 요청
                        val intent = Intent(this@MainActivity, VoiceGuideActivity::class.java)
                        intent.putExtra("voice_triggered", true)
                        startActivity(intent)
                        finish()
                        return
                    }
                }
                startListening()
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

    private fun startListening() {
        speechRecognizer.startListening(speechIntent)
    }

    private fun startEyeAnimation() {
        val radiusX = 30f
        val radiusY = 15f
        val baseYOffset = 20f

        val runnable = object : Runnable {
            override fun run() {
                angle += 0.04
                if (angle > 2 * PI) angle = 0.0

                currentOffsetAngle += (targetOffsetAngle - currentOffsetAngle) * 0.05
                currentVerticalOffset += (targetVerticalOffset - currentVerticalOffset) * 0.05

                val offsetX = (radiusX * cos(angle + currentOffsetAngle)).toFloat()
                val offsetY = baseYOffset + (radiusY * sin(angle + currentVerticalOffset)).toFloat()

                leftEye.translationX = offsetX
                leftEye.translationY = offsetY
                rightEye.translationX = offsetX
                rightEye.translationY = offsetY

                handler.postDelayed(this, 16)
            }
        }

        handler.post(runnable)
    }

    private fun handleRobotStatusChange(status: String) {
        runOnUiThread {
            when (status) {
                "occupied" -> {
                    isBlocked = true
                    tapPrompt.setImageResource(R.drawable.admin_control_notice)
                    stopPromptBlink()
                }
                "idle" -> {
                    isBlocked = false
                    tapPrompt.setImageResource(R.drawable.tap_to_start)
                    startPromptBlink()
                }
                "arrived_to_call" -> {            // ✅ 호출 지점 도착 → 메인메뉴로 이동
                    safeGoToMainMenu()
                }
            }
        }
    }

    private fun startPromptBlink() {
        if (blinkHandler != null) return
        blinkHandler = Handler(Looper.getMainLooper())
        blinkRunnable = object : Runnable {
            var visible = true
            override fun run() {
                tapPrompt.animate().alpha(if (visible) 1f else 0f).setDuration(500).start()
                visible = !visible
                blinkHandler?.postDelayed(this, 700)
            }
        }
        blinkHandler?.post(blinkRunnable!!)
    }

    private fun stopPromptBlink() {
        blinkHandler?.removeCallbacks(blinkRunnable!!)
        blinkHandler = null
        blinkRunnable = null
        tapPrompt.animate().alpha(1f).setDuration(200).start()
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (event?.action == MotionEvent.ACTION_DOWN && !isBlocked) {
            sendCallRequest("/call_with_screen") // 👆 화면 터치 호출 요청
            val intent = Intent(this, MainMenuActivity::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()
            return true
        }
        return super.onTouchEvent(event)
    }

    private fun safeGoToMainMenu() {
        if (hasNavigatedToMenu) return
        hasNavigatedToMenu = true
        isBlocked = true

        try { speechRecognizer.cancel() } catch (_: Exception) {}
        stopPromptBlink()

        val intent = Intent(this, MainMenuActivity::class.java)
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    private fun sendCallRequest(endpoint: String) {
        val json = JSONObject().apply { put("robot_id", 3) }
        val body = json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

        val request = Request.Builder()
            .url(NetworkConfig.getCentralServerUrl() + endpoint)
            .post(body)
            .build()

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val response = OkHttpClient().newCall(request).execute()
                val statusCode = response.code
                Log.d("CallRequest", "📡 $endpoint 호출 결과: $statusCode")
            } catch (e: Exception) {
                Log.e("CallRequest", "❌ 호출 실패: $endpoint - ${e.message}")
            }
        }
    }

    override fun onResume() {
        super.onResume()
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_FULLSCREEN
    }


    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
        webSocketClient.disconnect()
    }
}

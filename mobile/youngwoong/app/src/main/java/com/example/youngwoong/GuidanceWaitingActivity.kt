package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class GuidanceWaitingActivity : AppCompatActivity() {

    private lateinit var leftEye: ImageView
    private lateinit var rightEye: ImageView
    private lateinit var guidingText: TextView
    private lateinit var backButton: ImageView
    private lateinit var touchHintText: TextView

    private var selectedText: String? = null
    private var isFromCheckin: Boolean = false
    private var patientId: String? = null

    private val handler = Handler(Looper.getMainLooper())
    private var angle = 0.0

    private var currentOffsetAngle = 0.0
    private var targetOffsetAngle = 0.0

    private var currentVerticalOffset = 0.0
    private var targetVerticalOffset = 0.0

    private var isCompleted = false   // 중복 전환 방지
    private var webSocketClient: RobotStatusWebSocketClient? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_waiting)

        // ✅ 인텐트에서 값 받기
        selectedText = intent.getStringExtra("selected_text")
        isFromCheckin = intent.getBooleanExtra("isFromCheckin", false)
        patientId = intent.getStringExtra("patient_id")

        Log.d("ResumeIntent", "✅ 초기화됨 selected_text = $selectedText, isFromCheckin = $isFromCheckin, patient_id = $patientId")

        // 🔧 UI 요소 초기화
        leftEye = findViewById(R.id.left_eye)
        rightEye = findViewById(R.id.right_eye)
        guidingText = findViewById(R.id.text_guiding)
        backButton = findViewById(R.id.btn_cancel)
        touchHintText = findViewById(R.id.text_touch_hint)

        startEyeAnimation()
        startTouchHintBlink()

        // 🔙 뒤로가기 버튼 동작
        backButton.setOnClickListener {
            applyAlphaEffect(backButton)
            backButton.postDelayed({
                sendRobotStopStatus()
                navigateToConfirm()
            }, 100)
        }
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

    private fun startTouchHintBlink() {
        val blinkHandler = Handler(Looper.getMainLooper())

        val blinkRunnable = object : Runnable {
            var visible = true
            override fun run() {
                touchHintText.animate()
                    .alpha(if (visible) 1f else 0f)
                    .setDuration(500)
                    .start()

                visible = !visible
                blinkHandler.postDelayed(this, 700)
            }
        }

        blinkHandler.post(blinkRunnable)
    }

    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }

    // ✅ 터치 시 정지 명령 전송 후 확인 화면으로 이동
    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (event?.action == MotionEvent.ACTION_DOWN) {
            sendRobotStopStatus()

            // 디버깅 로그 추가
            Log.d("ResumeIntent", "📤 selected_text = $selectedText, isFromCheckin = $isFromCheckin, patient_id = $patientId")

            // Resume 페이지로 이동
            val intent = Intent(this, GuidanceResumeActivity::class.java).apply {
                putExtra("selected_text", selectedText ?: "")
                putExtra("isFromCheckin", isFromCheckin)
                putExtra("patient_id", patientId ?: "")
            }
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()
        }
        return super.onTouchEvent(event)
    }


    private fun navigateToConfirm() {
        val selectedText = intent.getStringExtra("selected_text")
        val intent = Intent(this, GuidanceConfirmActivity::class.java)

        if (selectedText != null) {
            intent.putExtra("selected_text", selectedText)
            intent.putExtra("isFromCheckin", false)
        } else {
            intent.putExtra("isFromCheckin", true)
        }

        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    private fun navigateToComplete() {
        val selectedText = intent.getStringExtra("selected_text")
        val isFromCheckin = intent.getBooleanExtra("isFromCheckin", false)
        val patientId = intent.getStringExtra("patient_id")  // null 가능성 고려

        val intent = Intent(this, GuidanceCompleteActivity::class.java).apply {
            putExtra("selected_text", selectedText)
            putExtra("isFromCheckin", isFromCheckin)
            if (patientId != null) putExtra("patient_id", patientId)
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }


    private fun startWebSocket() {
        webSocketClient = RobotStatusWebSocketClient(
            url = "ws://192.168.0.10:3000/?client_type=gui",
            targetRobotId = "3"
        ) { status ->
            Log.d("WS", "📩 상태 수신: $status")
            if (status == "navigating_complete" && !isCompleted) {
                runOnUiThread {
                    isCompleted = true
                    navigateToComplete()
                }
            }
        }
        webSocketClient?.connect()
    }

    private fun stopWebSocket() {
        webSocketClient?.disconnect()
        webSocketClient = null
    }


    override fun onResume() {
        super.onResume()
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_FULLSCREEN

        isCompleted = false
        startWebSocket()
    }

    override fun onPause() {
        super.onPause()
        stopWebSocket()
    }

    override fun onDestroy() {
        stopWebSocket()
        super.onDestroy()
    }

    // ✅ 로봇 정지 상태 전송 (IF-08)
    private fun sendRobotStopStatus() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val json = JSONObject().apply {
                    put("robot_id", 3)
                    put("patient_id", patientId?.takeIf { it.isNotBlank() } ?: "unknown")
                }

                val request = Request.Builder()
                    .url(NetworkConfig.getPauseRobotUrl())  // ✅ 올바른 IF-08 URL 사용
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                if (response.isSuccessful) {
                    Log.d("RobotStatus", "✅ 정지 명령 성공: $body")
                } else {
                    Log.e("RobotStatus", "❌ 정지 명령 실패: code=${response.code}, body=$body")
                }
            } catch (e: Exception) {
                Log.e("RobotStatus", "❌ 네트워크 오류로 정지 명령 실패", e)
            }
        }

    }
}

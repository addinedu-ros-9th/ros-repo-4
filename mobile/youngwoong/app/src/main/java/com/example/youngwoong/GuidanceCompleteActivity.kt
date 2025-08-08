package com.example.youngwoong

import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
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
import android.os.Handler
import android.os.Looper
import android.content.Intent
import android.view.View


class GuidanceCompleteActivity : AppCompatActivity() {

    private val timeoutHandler = Handler(Looper.getMainLooper())
    private val timeoutRunnable = Runnable {
        Log.d("GuidanceComplete", "🕒 30초 타임아웃 발생 → 복귀 요청 전송")
        sendRobotReturnCommand()
        navigateToMain(fromTimeout = true) // ✅ 수정: 타임아웃에서 true 넘기기
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_complete)

        val textView = findViewById<TextView>(R.id.text_guidance_message)
        val confirmButton = findViewById<ImageView>(R.id.btn_confirm_on)

        val selectedText = intent.getStringExtra("selected_text") ?: "해당 센터"
        val message = "${selectedText}에 도착했습니다.\n확인을 누르면 영웅이가 복귀합니다."

        // ✅ 강조 색 적용
        val spannable = SpannableString(message)
        val start = message.indexOf(selectedText)
        if (start >= 0) {
            spannable.setSpan(
                ForegroundColorSpan(Color.parseColor("#00696D")),
                start,
                start + selectedText.length,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
        }
        textView.text = spannable

        // ✅ 확인 버튼 클릭 → 복귀 명령 전송
        confirmButton.setOnClickListener {
            timeoutHandler.removeCallbacks(timeoutRunnable)
            navigateToMain(fromTimeout = true)// 🔒 타임아웃 중단
            sendRobotReturnCommand()
        }
    }

    private fun resetTimeoutTimer() {
        timeoutHandler.removeCallbacks(timeoutRunnable)
        timeoutHandler.postDelayed(timeoutRunnable, 30_000) // 30초
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

    private fun navigateToMain(fromTimeout: Boolean = false) {
        Log.d("GuidanceComplete", "navigateToMain 호출됨, fromTimeout=$fromTimeout")
        val intent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            putExtra("from_timeout", fromTimeout) // ✅ 복귀중 여부 전달
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    private fun sendRobotReturnCommand() {
        val isFromCheckin = intent.getBooleanExtra("isFromCheckin", false)
        val rawPatientId = intent.getStringExtra("patient_id")
        val patientId = rawPatientId?.takeIf { it.isNotBlank() } ?: "unknown"  // ✅ 무조건 보냄
        val url = if (isFromCheckin) {
            NetworkConfig.getRobotReturnAuthUrl()       // IF-06
        } else {
            NetworkConfig.getRobotReturnWithoutAuthUrl() // IF-07
        }

        val json = JSONObject().apply {
            put("robot_id", 3)
            put("patient_id", patientId)  // ✅ 항상 포함
        }

        Log.d("RobotReturn", "📤 복귀 요청: $json → $url")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val request = Request.Builder()
                    .url(url)
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                if (response.isSuccessful) {
                    Log.d("RobotReturn", "✅ 복귀 요청 성공: $body")
                } else {
                    Log.e("RobotReturn", "❌ 복귀 요청 실패: code=${response.code}, body=$body")
                }
            } catch (e: Exception) {
                Log.e("RobotReturn", "❌ 복귀 요청 중 예외 발생", e)
            }
        }
    }
}

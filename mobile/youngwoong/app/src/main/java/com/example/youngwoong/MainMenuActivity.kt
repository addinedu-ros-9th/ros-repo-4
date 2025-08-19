package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.widget.ImageView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import android.util.Log
import android.view.View

class MainMenuActivity : AppCompatActivity() {

    private val handler = Handler(Looper.getMainLooper())
    private val idleTimeout = 30_000L // 30초

    private val returnRunnable = Runnable {
        sendTimeoutAlert() // ✅ 타임아웃 발생 시 서버에 알림 전송

        val intent = Intent(this, MainActivity::class.java).apply {
            putExtra("from_timeout", true) // ✅ 복귀중 상태 전달
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }

        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_menu)

        startIdleTimer()

        val checkInButton = findViewById<ImageView>(R.id.button_check_in)
        checkInButton.setOnClickListener {
            applyAlphaEffect(checkInButton)
            checkInButton.postDelayed({
                val intent = Intent(this, AuthenticationActivity::class.java)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        val guidanceButton = findViewById<ImageView>(R.id.button_guidance)
        guidanceButton.setOnClickListener {
            applyAlphaEffect(guidanceButton)
            guidanceButton.postDelayed({
                val intent = Intent(this, GuidanceActivity::class.java)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        val voiceGuideButton = findViewById<ImageView>(R.id.button_voice)
        voiceGuideButton.setOnClickListener {
            applyAlphaEffect(voiceGuideButton)
            voiceGuideButton.postDelayed({
                val intent = Intent(this, VoiceGuideActivity::class.java)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                val intent = Intent(this@MainMenuActivity, MainActivity::class.java).apply {
                    addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                }
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }
        })
    }

    private fun startIdleTimer() {
        handler.postDelayed(returnRunnable, idleTimeout)
    }

    private fun resetIdleTimer() {
        handler.removeCallbacks(returnRunnable)
        handler.postDelayed(returnRunnable, idleTimeout)
    }

    override fun dispatchTouchEvent(ev: MotionEvent?): Boolean {
        resetIdleTimer()
        return super.dispatchTouchEvent(ev)
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(returnRunnable)
    }

    override fun onResume() {
        super.onResume()
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_FULLSCREEN
    }

    private fun applyAlphaEffect(view: ImageView) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }

    // ✅ 30초 타임아웃 발생 시 서버에 알림 전송
    private fun sendTimeoutAlert() {
        val json = JSONObject().apply { put("robot_id", 3) }
        val body = json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())
        val request = Request.Builder()
            .url(NetworkConfig.getTimeoutAlertUrl())
            .post(body)
            .build()

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val response = OkHttpClient().newCall(request).execute()
                val statusCode = response.code
                Log.d("TimeoutAlert", "✅ /alert_timeout 호출 결과: $statusCode")
            } catch (e: Exception) {
                Log.e("TimeoutAlert", "❌ /alert_timeout 호출 실패: ${e.message}")
            }
        }
    }
}

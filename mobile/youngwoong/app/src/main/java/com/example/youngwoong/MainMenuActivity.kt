package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.widget.ImageView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity

class MainMenuActivity : AppCompatActivity() {

    private val handler = Handler(Looper.getMainLooper())
    private val idleTimeout = 10_000L // 10초

    private val returnRunnable = Runnable {
        val intent = Intent(this, MainActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_menu)

        startIdleTimer()

        // check-in 버튼 클릭
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

        // 길안내 버튼 클릭
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

        // 🔊 음성안내 버튼 클릭
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

        // AndroidX OnBackPressedDispatcher 사용 (Android 14 호환)
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

    // 버튼 알파 효과 함수
    private fun applyAlphaEffect(view: ImageView) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }
}

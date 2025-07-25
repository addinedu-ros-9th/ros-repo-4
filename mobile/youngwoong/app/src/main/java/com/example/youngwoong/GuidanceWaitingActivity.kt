package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class GuidanceWaitingActivity : AppCompatActivity() {

    private lateinit var leftEye: ImageView
    private lateinit var rightEye: ImageView
    private lateinit var guidingText: TextView
    private lateinit var backButton: ImageView
    private lateinit var touchHintText: TextView

    private val handler = Handler(Looper.getMainLooper())
    private var angle = 0.0

    private var currentOffsetAngle = 0.0
    private var targetOffsetAngle = 0.0

    private var currentVerticalOffset = 0.0
    private var targetVerticalOffset = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_waiting)

        leftEye = findViewById(R.id.left_eye)
        rightEye = findViewById(R.id.right_eye)
        guidingText = findViewById(R.id.text_guiding)
        backButton = findViewById(R.id.btn_cancel)
        touchHintText = findViewById(R.id.text_touch_hint) // 👈 하단 텍스트 연결

        startEyeAnimation()
        startTouchHintBlink() // 👈 깜빡이기 시작

        // 취소 버튼 (현재는 안 쓰지만 남겨둠)
        backButton.setOnClickListener {
            applyAlphaEffect(backButton)
            backButton.postDelayed({
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

    // ✅ 터치 시 안내 확인 화면으로 이동
    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (event?.action == MotionEvent.ACTION_DOWN) {
            navigateToConfirm()
            return true
        }
        return super.onTouchEvent(event)
    }

    private fun navigateToConfirm() {
        val intent = Intent(this, GuidanceConfirmActivity::class.java)
        intent.putExtra("isFromCheckin", true) // ✅ 명시적으로 true
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

}

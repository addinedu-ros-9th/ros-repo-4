package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        leftEye = findViewById(R.id.left_eye)
        rightEye = findViewById(R.id.right_eye)
        tapPrompt = findViewById(R.id.tap_prompt)
        testButton = findViewById(R.id.testButton)
        
        // 디버깅: 버튼이 제대로 찾아졌는지 확인
        if (testButton == null) {
            println("ERROR: testButton이 null입니다!")
        } else {
            println("SUCCESS: testButton을 찾았습니다!")
            // 버튼을 강제로 보이게 설정
            testButton?.visibility = android.view.View.VISIBLE
        }
        
        // 테스트 버튼이 null이 아닌지 확인 후 클릭 이벤트 설정
        testButton?.setOnClickListener {
            val intent = Intent(this, TestRobotSystemActivity::class.java)
            startActivity(intent)
        }

        startEyeAnimation()
        startPromptBlink()
    }

    private fun startEyeAnimation() {
        val radiusX = 30f
        val radiusY = 15f
        val baseYOffset = 20f  // ⭐ 아래로 내리기

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

    private fun startPromptBlink() {
        val blinkHandler = Handler(Looper.getMainLooper())

        val blinkRunnable = object : Runnable {
            var visible = true
            override fun run() {
                tapPrompt.animate()
                    .alpha(if (visible) 1f else 0f)
                    .setDuration(500)
                    .start()

                visible = !visible
                blinkHandler.postDelayed(this, 700)
            }
        }

        blinkHandler.post(blinkRunnable)
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        if (event?.action == MotionEvent.ACTION_DOWN) {
            val intent = Intent(this, MainMenuActivity::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()
            return true
        }
        return super.onTouchEvent(event)
    }
}
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

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechIntent: Intent
    private val wakeWords = listOf("영웅아", "영화", "영아","영우아", "영웅이")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        leftEye = findViewById(R.id.left_eye)
        rightEye = findViewById(R.id.right_eye)
        tapPrompt = findViewById(R.id.tap_prompt)
        testButton = findViewById(R.id.testButton)

        testButton?.apply {
            visibility = android.view.View.VISIBLE
            setOnClickListener {
                startActivity(Intent(this@MainActivity, TestRobotSystemActivity::class.java))
            }
        }

        setupSTT()
        checkPermissions()

        startEyeAnimation()
        startPromptBlink()
    }

    private fun checkPermissions() {
        val permissions = listOf(Manifest.permission.RECORD_AUDIO)
        val notGranted = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (notGranted.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, notGranted.toTypedArray(), 1001)
        } else {
            startListening()
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
                // 재시작
                speechRecognizer.cancel()
                startListening()
            }

            override fun onResults(results: Bundle?) {
                val result = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.getOrNull(0) ?: return

                for (keyword in wakeWords) {
                    if (result.contains(keyword)) {
                        val intent = Intent(this@MainActivity, VoiceGuideActivity::class.java)
                        intent.putExtra("voice_triggered", true)
                        startActivity(intent)
                        finish()
                        return
                    }
                }

                // 계속 듣기
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

    private fun startPromptBlink() {
        val blinkHandler = Handler(Looper.getMainLooper())
        val blinkRunnable = object : Runnable {
            var visible = true
            override fun run() {
                tapPrompt.animate().alpha(if (visible) 1f else 0f).setDuration(500).start()
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

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
    }
}

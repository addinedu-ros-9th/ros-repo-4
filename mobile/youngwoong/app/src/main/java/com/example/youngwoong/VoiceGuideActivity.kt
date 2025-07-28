package com.example.youngwoong

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.View
import android.view.animation.AlphaAnimation
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.airbnb.lottie.LottieAnimationView

class VoiceGuideActivity : AppCompatActivity() {

    private var isListening = false
    private lateinit var voiceAnimation: LottieAnimationView
    private lateinit var dimView: View
    private lateinit var textPrompt: TextView
    private lateinit var textUserMessage: TextView
    private var blinkAnimation: AlphaAnimation? = null

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechIntent: Intent

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_guide)

        checkPermissions()
        setupSTT()

        voiceAnimation = findViewById(R.id.voice_animation)
        dimView = findViewById(R.id.dim_view)
        textPrompt = findViewById(R.id.text_prompt)
        textUserMessage = findViewById(R.id.text_user_message)

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
                Log.e("STT", "❌ 음성 인식 오류: $error")

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
                val result = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.getOrNull(0)

                result?.let {
                    textUserMessage.text = it
                    textPrompt.text = "듣고 있어요..."
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
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
    }
}

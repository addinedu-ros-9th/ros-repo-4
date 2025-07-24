package com.example.youngwoong

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.view.View
import android.view.animation.AlphaAnimation
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.airbnb.lottie.LottieAnimationView
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.media.MediaScannerConnection
import android.util.Log

class VoiceGuideActivity : AppCompatActivity() {

    private var isRecording = false
    private lateinit var voiceAnimation: LottieAnimationView
    private lateinit var dimView: View
    private lateinit var textPrompt: TextView
    private var blinkAnimation: AlphaAnimation? = null

    private var mediaRecorder: MediaRecorder? = null
    private var audioFilePath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_guide)

        // 권한 확인
        checkPermissions()

        // 뷰 초기화
        voiceAnimation = findViewById(R.id.voice_animation)
        dimView = findViewById(R.id.dim_view)
        textPrompt = findViewById(R.id.text_prompt)

        findViewById<ImageView>(R.id.btn_back).setOnClickListener {
            applyAlphaEffect(it)
            returnToMainMenu()
        }

        findViewById<ImageView>(R.id.btn_voice).setOnClickListener {
            applyAlphaEffect(it)
            toggleRecording(it as ImageView)
        }
    }

    private fun checkPermissions() {
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)

        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            // Android 9 이하일 경우 저장소 권한도 요청
            permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

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
            Toast.makeText(this, "필수 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
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

    private fun toggleRecording(button: ImageView) {
        isRecording = !isRecording

        if (isRecording) {
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
                text = "녹음중입니다!"
                setTextColor(Color.parseColor("#FFA000"))
                startBlinking(this)
            }

            startRecording()
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

            stopRecording()
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

    private fun startRecording() {
        try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "recording_$timeStamp.m4a"

            val outputDir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                "recordings"
            )
            if (!outputDir.exists()) outputDir.mkdirs()

            val audioFile = File(outputDir, fileName)
            audioFilePath = audioFile.absolutePath

            mediaRecorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setOutputFile(audioFilePath)
                prepare()
                start()
            }

            Log.d("VoiceGuide", "녹음 파일 경로: $audioFilePath")

        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "녹음을 시작할 수 없습니다.", Toast.LENGTH_SHORT).show()
            isRecording = false
        }
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                reset()
                release()
            }
            mediaRecorder = null

            // 미디어 스캐너에 등록
            audioFilePath?.let {
                MediaScannerConnection.scanFile(
                    this,
                    arrayOf(it),
                    arrayOf("audio/*"),
                    null
                )
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}

package com.example.youngwoong

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan

class GuidanceResumeActivity : AppCompatActivity() {

    private val robotLocationUrl = NetworkConfig.getRobotLocationUrl()

    private val rosCoords = mapOf(
        "초음파 검사실" to Pair(-4.9f, -1.96f),
        "CT 검사실"     to Pair(-6.3f, -2.04f),
        "X-ray 검사실" to Pair(-5.69f, 4.34f),
        "대장암 센터"  to Pair(0.93f, -2.3f),
        "위암 센터"    to Pair(3.84f, -2.3f),
        "폐암 센터"    to Pair(5.32f, -2.27f),
        "유방암 센터"  to Pair(7.17f, 1.77f),
        "뇌종양 센터"  to Pair(5.45f, 1.69f)
    )

    // ─── 30초 타임아웃 핸들러 ─────────────────────────────────
    private val timeoutHandler = Handler(Looper.getMainLooper())
    private val timeoutRunnable = Runnable {
        Log.d("GuidanceResume", "🕒 30초 타임아웃 발생")
        disableInteraction()
        sendTimeoutAlert()
        navigateToMainWithTimeout()
    }
    // ─────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_resume)

        // 뷰 바인딩
        val resumeMessage     = findViewById<TextView>(R.id.text_resume_message)
        val mapView           = findViewById<ImageView>(R.id.image_map)
        val destinationMarker = findViewById<ImageView>(R.id.image_destination_marker)
        val robotMarker       = findViewById<ImageView>(R.id.image_robot_marker)
        val btnResume         = findViewById<ImageView>(R.id.btn_resume_guidance)
        val btnEnd            = findViewById<ImageView>(R.id.btn_end_guidance)

        // 인텐트 데이터
        val selectedText  = intent.getStringExtra("selected_text") ?: "알 수 없음"
        val isFromCheckin = intent.getBooleanExtra("isFromCheckin", false)
        val patientId     = intent.getStringExtra("patient_id") ?: ""

        // 안내 일시정지 메시지 + 강조
        val highlightColor = Color.parseColor("#00696D")
        val message = "안내가 일시정지 되었습니다\n현재 목적지는 $selectedText 입니다"
        resumeMessage.text = SpannableString(message).apply {
            val start = message.indexOf(selectedText)
            if (start >= 0) {
                setSpan(
                    ForegroundColorSpan(highlightColor),
                    start, start + selectedText.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
            Log.d("Destination", "📍 목적지=$selectedText, 좌표=${rosCoords[selectedText]}")
        }

        // 도착지 마커 표시
        rosCoords[selectedText]?.let { (x, y) ->
            mapView.post {
                destinationMarker.visibility = View.VISIBLE
                destinationMarker.post {
                    val (px, py) = mapToPixelRobot(x, y, mapView)
                    destinationMarker.x = mapView.x + px - destinationMarker.width / 2f
                    destinationMarker.y = mapView.y + py - destinationMarker.height / 2f
                }
            }
        }

        // 로봇 위치 표시
        fetchRobotPosition(robotMarker, mapView)

        // 안내 재개
        btnResume.setOnClickListener {
            applyAlphaEffect(it)
            it.postDelayed({
                sendRestartNavigationRequest()
                Intent(this, GuidanceWaitingActivity::class.java).apply {
                    putExtra("selected_text", selectedText)
                    putExtra("isFromCheckin", isFromCheckin)
                    putExtra("patient_id", patientId)
                }.also { intent ->
                    startActivity(intent)
                    overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                    finish()
                }
            }, 100)
        }

        // 안내 종료 → MainActivity 복귀 (from_timeout=true)
        btnEnd.setOnClickListener {
            applyAlphaEffect(it)
            it.postDelayed({
                sendStopNavigatingRequest()
                Intent(this, MainActivity::class.java).apply {
                    putExtra("from_timeout", true)
                    addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                }.also { intent ->
                    startActivity(intent)
                    overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                    finish()
                }
            }, 100)
        }

        // ─── 타이머 시작 ───────────────────────────────
        resetTimeoutTimer()
        // ───────────────────────────────────────────────
    }

    // ─── 사용자 상호작용 시 타이머 리셋 ───────────────────────────
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
    override fun onDestroy() {
        super.onDestroy()
        timeoutHandler.removeCallbacks(timeoutRunnable)
    }
    // ────────────────────────────────────────────────────────────

    /** 타이머 (30초) 리셋 */
    private fun resetTimeoutTimer() {
        timeoutHandler.removeCallbacks(timeoutRunnable)
        timeoutHandler.postDelayed(timeoutRunnable, 30_000)
    }

    /** 버튼/터치 비활성화 */
    private fun disableInteraction() {
        runOnUiThread {
            findViewById<ImageView>(R.id.btn_resume_guidance).isEnabled = false
            findViewById<ImageView>(R.id.btn_end_guidance).isEnabled    = false
            // 필요시 mapView 터치 리스너 제거 등 추가
        }
    }

    /** 서버에 타임아웃 알림 (IF-09) */
    private fun sendTimeoutAlert() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val url  = NetworkConfig.getTimeoutAlertUrl()
                val json = JSONObject().put("robot_id", 3)
                val body = json.toString()
                    .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())
                val resp = OkHttpClient().newCall(
                    Request.Builder().url(url).post(body).build()
                ).execute()
                Log.d("TimeoutAlert", "✅ code=${resp.code}")
            } catch (e: Exception) {
                Log.e("TimeoutAlert", "❌ ${e.message}")
            }
        }
    }

    /** 타임아웃 플래그와 함께 MainActivity 실행 */
    private fun navigateToMainWithTimeout() {
        Intent(this, MainActivity::class.java).apply {
            putExtra("from_timeout", true)
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }.also { intent ->
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()
        }
    }

    // ─── 이하 기존 API 호출 메서드들 ─────────────────────────────

    /** 안내 중지 명령(IF-08) */
    private fun sendStopNavigatingRequest() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val safePatientId = intent.getStringExtra("patient_id")
                    ?.takeIf { it.isNotBlank() } ?: "unknown"

                val json = JSONObject().apply {
                    put("robot_id", 3)
                    put("patient_id", safePatientId)  // ✅ 항상 포함
                }

                val request = Request.Builder()
                    .url(NetworkConfig.getStopNavigatingUrl())
                    .post(json.toString()
                        .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val resp = OkHttpClient().newCall(request).execute()
                val body = resp.body?.string()

                if (resp.isSuccessful) {
                    Log.d("StopNav", "✅ success: $body")
                } else {
                    Log.e("StopNav", "❌ fail: code=${resp.code}, body=$body")
                }
            } catch (e: Exception) {
                Log.e("StopNav", "❌ network error", e)
            }
        }
    }

    /** 안내 재개 명령(IF-08) */
    private fun sendRestartNavigationRequest() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val safePatientId = intent.getStringExtra("patient_id")
                    ?.takeIf { it.isNotBlank() } ?: "unknown"

                val json = JSONObject().apply {
                    put("robot_id", "3")
                    put("patient_id", safePatientId)  // ✅ 항상 포함
                }

                val request = Request.Builder()
                    .url(NetworkConfig.getRestartNavigationUrl())
                    .post(json.toString()
                        .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val resp = OkHttpClient().newCall(request).execute()
                val body = resp.body?.string()

                if (resp.isSuccessful) {
                    Log.d("RestartNav", "✅ success: $body")
                } else {
                    Log.e("RestartNav", "❌ fail: code=${resp.code}, body=$body")
                }
            } catch (e: Exception) {
                Log.e("RestartNav", "❌ network error", e)
            }
        }
    }

    /** 로봇 위치 받아와서 마커 표시 */
    private fun fetchRobotPosition(robotMarker: ImageView, mapView: ImageView) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val json    = JSONObject().put("robot_id", 3)
                val request = Request.Builder()
                    .url(robotLocationUrl)
                    .post(json.toString()
                        .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()
                val resp = OkHttpClient().newCall(request).execute()
                val body = resp.body?.string()
                Log.d("RobotPos", "📥 $body")
                if (!body.isNullOrEmpty()) {
                    val data = JSONObject(body)
                    val x = data.optDouble("x", Double.NaN).toFloat()
                    val y = data.optDouble("y", Double.NaN).toFloat()
                    if (!x.isNaN() && !y.isNaN()) {
                        val (px, py) = mapToPixelRobot(x, y, mapView)
                        withContext(Dispatchers.Main) {
                            robotMarker.visibility = View.VISIBLE
                            robotMarker.post {
                                robotMarker.x = mapView.x + px - robotMarker.width / 2f
                                robotMarker.y = mapView.y + py - robotMarker.height / 2f
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("RobotPos", "❌ fail", e)
            }
        }
    }

    /** ROS 좌표 → 화면 픽셀 */
    private fun mapToPixelRobot(x: Float, y: Float, mapView: ImageView): Pair<Float, Float> {
        val xMin = -10f; val xMax = 10f
        val yMin = -5f;  val yMax = 5f
        val w = mapView.width.toFloat()
        val h = mapView.height.toFloat()
        val scaleX = w / (xMax - xMin)
        val scaleY = h / (yMax - yMin)
        return (w/2f + x*scaleX) to (h/2f - y*scaleY)
    }

    /** 버튼 터치 효과 */
    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }
}

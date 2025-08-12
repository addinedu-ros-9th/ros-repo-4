package com.example.youngwoong

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class GuidanceCompleteActivity : AppCompatActivity() {

    private lateinit var webSocketClient: RobotStatusWebSocketClient
    private var hasNavigated = false

    // ✅ 로봇 위치 조회 URL
    private val robotLocationUrl = NetworkConfig.getRobotLocationUrl()

    // ✅ 목적지(센터) ROS 좌표
    private val rosCoords = mapOf(
        "초음파 검사실" to Pair(-4.9f, -1.96f),
        "CT 검사실" to Pair(-6.3f, -2.04f),
        "X-ray 검사실" to Pair(-5.69f, 4.34f),
        "대장암 센터" to Pair(0.93f, -2.3f),
        "위암 센터" to Pair(3.84f, -2.3f),
        "폐암 센터" to Pair(5.32f, -2.27f),
        "유방암 센터" to Pair(7.17f, 1.77f),
        "뇌종양 센터" to Pair(5.45f, 1.69f)
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_complete)

        val textView = findViewById<TextView>(R.id.text_guidance_message)
        val confirmButton = findViewById<ImageView>(R.id.btn_confirm_on)

        // ✅ 지도 및 마커
        val mapView = findViewById<ImageView>(R.id.image_map)
        val destinationMarker = findViewById<ImageView>(R.id.image_destination_marker)
        val robotMarker = findViewById<ImageView>(R.id.image_robot_marker)

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

        // ✅ 선택된 목적지 마커 표시 (ROS -> 픽셀)
        rosCoords[selectedText]?.let { (dx, dy) ->
            mapView.post {
                destinationMarker.visibility = View.VISIBLE
                destinationMarker.post {
                    val (px, py) = mapToPixelRobot(dx, dy, mapView)
                    destinationMarker.x = mapView.x + px - destinationMarker.width / 2f
                    destinationMarker.y = mapView.y + py - destinationMarker.height / 2f
                    Log.d("GC-DestMarker", "🧷 목적지 마커: $selectedText @ px=($px,$py)")
                }
            }
        } ?: run {
            Log.w("GC-DestMarker", "⚠️ 목적지 ROS 좌표 매핑 없음: $selectedText")
        }

        // ✅ 로봇 현재 위치 조회해서 마커 표시
        fetchRobotPosition(robotMarker, mapView)

        // ✅ 확인 버튼 클릭 → 복귀 명령 전송 후 메인으로
        confirmButton.setOnClickListener {
            sendRobotReturnCommand()
            safeNavigateToMain()
        }

        // ✅ 중앙서버 WebSocket에서 return_command 수신 시 자동 이동
        webSocketClient = RobotStatusWebSocketClient(
            url = NetworkConfig.getGuiWebSocketUrl(),
            targetRobotId = "3"
        ) { status ->
            if (status == "return_command") {
                Log.d("GuidanceComplete", "📨 return_command 수신 → 메인으로 이동")
                runOnUiThread { navigateToMain(fromTimeout = true) }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_FULLSCREEN
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            if (this::webSocketClient.isInitialized) webSocketClient.disconnect()
        } catch (_: Exception) {}
    }

    private fun safeNavigateToMain() {
        if (hasNavigated) return
        hasNavigated = true
        navigateToMain(fromTimeout = true)
    }

    private fun navigateToMain(fromTimeout: Boolean = false) {
        Log.d("GuidanceComplete", "navigateToMain 호출됨, fromTimeout=$fromTimeout")
        val intent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            putExtra("from_timeout", fromTimeout) // 메인에서 '복귀중입니다' 표시
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    // ======================
    // 좌표/마커 관련 유틸
    // ======================
    private fun mapToPixelRobot(x: Float, y: Float, mapView: ImageView): Pair<Float, Float> {
        val xMin = -10f; val xMax = 10f
        val yMin = -5f;  val yMax = 5f

        val imageWidth = mapView.width.toFloat()
        val imageHeight = mapView.height.toFloat()

        val scaleX = imageWidth / (xMax - xMin)
        val scaleY = imageHeight / (yMax - yMin)

        val pixelX = imageWidth / 2f + x * scaleX
        val pixelY = imageHeight / 2f - y * scaleY // Y축 반전

        Log.d("GC-Map", "🧭 ROS($x,$y) → PX($pixelX,$pixelY) w=$imageWidth h=$imageHeight")
        return pixelX to pixelY
    }

    private fun fetchRobotPosition(robotMarker: ImageView, mapView: ImageView) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val json = JSONObject().apply { put("robot_id", 3) }
                val request = Request.Builder()
                    .url(robotLocationUrl)
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                Log.d("GC-RobotPos", "📥 서버 응답: $body")

                if (!body.isNullOrEmpty()) {
                    val obj = JSONObject(body)
                    val x = obj.optDouble("x", Double.NaN).toFloat()
                    val y = obj.optDouble("y", Double.NaN).toFloat()

                    if (!x.isNaN() && !y.isNaN()) {
                        val (px, py) = mapToPixelRobot(x, y, mapView)
                        withContext(Dispatchers.Main) {
                            robotMarker.visibility = View.VISIBLE
                            robotMarker.post {
                                robotMarker.x = mapView.x + px - robotMarker.width / 2f
                                robotMarker.y = mapView.y + py - robotMarker.height / 2f
                            }
                        }
                    } else {
                        Log.e("GC-RobotPos", "❌ 좌표 파싱 실패: x=$x, y=$y")
                    }
                } else {
                    Log.e("GC-RobotPos", "❌ 응답 body 없음")
                }
            } catch (e: Exception) {
                Log.e("GC-RobotPos", "❌ 위치 요청 실패", e)
            }
        }
    }

    // ======================
    // 복귀 API
    // ======================
    private fun sendRobotReturnCommand() {
        val isFromCheckin = intent.getBooleanExtra("isFromCheckin", false)
        val rawPatientId = intent.getStringExtra("patient_id")
        val patientId = rawPatientId?.takeIf { it.isNotBlank() } ?: "unknown"
        val url = if (isFromCheckin) {
            NetworkConfig.getRobotReturnAuthUrl()       // IF-06
        } else {
            NetworkConfig.getRobotReturnWithoutAuthUrl() // IF-07
        }

        val json = JSONObject().apply {
            put("robot_id", 3)
            put("patient_id", patientId)
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

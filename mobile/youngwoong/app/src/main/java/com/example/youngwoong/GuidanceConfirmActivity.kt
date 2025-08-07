package com.example.youngwoong

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class GuidanceConfirmActivity : AppCompatActivity() {

    private var isFromCheckin: Boolean = false
    private val robotLocationUrl = NetworkConfig.getRobotLocationUrl()

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

    private fun pixelToRosCoord(px: Float, py: Float, mapView: ImageView): Pair<Float, Float> {
        val imageWidth = mapView.width.toFloat()
        val imageHeight = mapView.height.toFloat()

        val centerX = imageWidth / 2f
        val centerY = imageHeight / 2f

        val rosX = ((px - centerX) / imageWidth) * 20f  // 전체 가로: -10 ~ +10
        val rosY = -((py - centerY) / imageHeight) * 10f  // 전체 세로: +5 ~ -5 (Y축 반전)

        Log.d("MapDebug", "📐 중심 기준 px=$px, py=$py → ROS: x=$rosX, y=$rosY")
        return rosX to rosY
    }



    private val uiCoords = mapOf(
        "초음파 검사실" to Pair(-0.4028f, 16.5230f),
        "CT 검사실" to Pair(-1.8587f, 16.6412f),
        "X-ray 검사실" to Pair(-1.8373f, -7.4504f),
        "대장암 센터" to Pair(5.2334f, 18.8963f),
        "위암 센터" to Pair(8.0922f, 18.9370f),
        "폐암 센터" to Pair(9.3855f, 19.0064f),
        "유방암 센터" to Pair(11.6292f, 1.1619f),
        "뇌종양 센터" to Pair(9.8731f, 1.1293f)
    )


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_confirm)

        val cancelButton = findViewById<ImageView>(R.id.btn_cancel)
        val confirmButton = findViewById<ImageView>(R.id.btn_start_guidance)
        val textView = findViewById<TextView>(R.id.text_guidance_message)
        val highlightColor = Color.parseColor("#00696D")
        val mapView = findViewById<ImageView>(R.id.image_map)
        val destinationMarker = findViewById<ImageView>(R.id.image_destination_marker)
        val robotMarker = findViewById<ImageView>(R.id.image_robot_marker)

        val userName = intent.getStringExtra("user_name")
        val department = intent.getStringExtra("department")
        val selectedText = intent.getStringExtra("selected_text")
        val patientId = intent.getStringExtra("patient_id") ?: ""

        isFromCheckin = intent.getBooleanExtra("isFromCheckin", false)
                || (userName != null && department != null)

        val targetName = selectedText ?: department
        val message = when {
            isFromCheckin && userName != null && department != null ->
                "${userName}님 ${department} 접수가 완료되었습니다.\n안내를 시작할까요?"
            selectedText != null ->
                "${selectedText}를 선택하셨습니다.\n안내를 시작할까요?"
            else -> "안내를 시작할까요?"
        }

        textView.text = SpannableString(message).apply {
            listOfNotNull(userName, department, selectedText).forEach { value ->
                val start = message.indexOf(value)
                if (start >= 0) {
                    setSpan(
                        ForegroundColorSpan(highlightColor),
                        start, start + value.length,
                        Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                    )
                }
            }
        }

        targetName?.let { name ->
            rosCoords[name]?.let { (x, y) ->
                mapView.post {
                    destinationMarker.visibility = View.VISIBLE

                    // 보장된 상태에서 위치 설정
                    destinationMarker.post {
                        val (px, py) = mapToPixelRobot(x, y, mapView)

                        destinationMarker.x = mapView.x + px - destinationMarker.width / 2f
                        destinationMarker.y = mapView.y + py - destinationMarker.height / 2f

                        Log.d(
                            "mapmarker",
                            "🧷 최종 마커 위치: x=${destinationMarker.x}, y=${destinationMarker.y}"
                        )
                    }
                }
            }
        }

        mapView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val px = event.x
                val py = event.y

                val imageWidth = mapView.width.toFloat()
                val imageHeight = mapView.height.toFloat()

                val centerX = imageWidth / 2f
                val centerY = imageHeight / 2f

                val rosX = ((px - centerX) / imageWidth) * 20f    // 전체 x범위: -10 ~ 10
                val rosY = -((py - centerY) / imageHeight) * 10f  // 전체 y범위: +5 ~ -5 (Y축 반전)

                Log.d("MapDebug", "🖱️ 터치 위치: px=$px, py=$py")
                Log.d("MapDebug", "📐 중심 기준 px=$px, py=$py → ROS: x=$rosX, y=$rosY")
            }
            true
        }



        fetchRobotPosition(robotMarker, mapView)

        cancelButton.setOnClickListener {
            applyAlphaEffect(it)
            it.postDelayed({
                val intent = if (isFromCheckin)
                    Intent(this, AuthenticationActivity::class.java)
                else
                    Intent(this, GuidanceActivity::class.java)
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        confirmButton.setOnClickListener {
            applyAlphaEffect(it)
            it.postDelayed({
                val centerName = selectedText ?: department ?: "해당 센터"
                val stationId = stationNameToId(centerName)
                if (stationId != null) {
                    sendDirectionRequest(patientId, stationId)
                } else {
                    Log.e("DirectionAPI", "❌ 목적지 매핑 실패: $centerName")
                }

                val intent = Intent(this, GuidanceWaitingActivity::class.java).apply {
                    putExtra("selected_text", centerName)
                    putExtra("isFromCheckin", isFromCheckin)
                    putExtra("patient_id", patientId)
                }
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }
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

                Log.d("RobotPosition", "📥 서버 응답: $body")

                if (!body.isNullOrEmpty()) {
                    val jsonObj = JSONObject(body)
                    val x = jsonObj.optDouble("x", Double.NaN).toFloat()
                    val y = jsonObj.optDouble("y", Double.NaN).toFloat()

                    if (!x.isNaN() && !y.isNaN()) {
                        val (px, py) = mapToPixelRobot(x, y, mapView)
                        withContext(Dispatchers.Main) {
                            robotMarker.visibility = View.VISIBLE
                            robotMarker.post {
                                robotMarker.x = mapView.x + px - robotMarker.width / 2
                                robotMarker.y = mapView.y + py - robotMarker.height / 2
                            }
                        }
                    } else {
                        Log.e("RobotPosition", "❌ 좌표 파싱 실패: x=$x, y=$y")
                    }
                } else {
                    Log.e("RobotPosition", "❌ 응답 body 없음")
                }
            } catch (e: Exception) {
                Log.e("RobotPosition", "❌ 위치 요청 실패", e)
            }
        }
    }

    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }



    private fun mapToPixelRobot(x: Float, y: Float, mapView: ImageView): Pair<Float, Float> {
        val xMin = -10f
        val xMax = 10f
        val yMin = -5f
        val yMax = 5f

        val imageWidth = mapView.width.toFloat()
        val imageHeight = mapView.height.toFloat()

        val scaleX = imageWidth / (xMax - xMin)
        val scaleY = imageHeight / (yMax - yMin)

        // ROS 좌표 기준 변환 (보정 없이 바로 픽셀로)
        val pixelX = imageWidth / 2f + x * scaleX
        val pixelY = imageHeight / 2f - y * scaleY  // Y축 반전

        Log.d("mapmarker", "🧭 [ROS] x=$x, y=$y")
        Log.d("mapmarker", "🖼️ 이미지 크기: width=$imageWidth, height=$imageHeight")
        Log.d("mapmarker", "🖼️ [픽셀 좌표] x=$pixelX, y=$pixelY (ROS: x=$x, y=$y)")

        return pixelX to pixelY
    }


    private fun mapToPixelDirect(x: Float, y: Float): Pair<Float, Float> {
        val xMin = -5.4995f
        val yMin = -10.0572f
        val xMax = 5.1066f
        val yMax = 9.8559f
        val imageWidth = 1020f
        val imageHeight = 530f

        val scaleX = imageWidth / (xMax - xMin)
        val scaleY = imageHeight / (yMax - yMin)

        val pixelX = (x - xMin) * scaleX
        val pixelY = (y - yMin) * scaleY

        return pixelX to pixelY
    }

    private fun pixelToMapCoord(px: Float, py: Float): Pair<Float, Float> {
        val xMin = -5.4995f
        val yMin = -10.0572f
        val xMax = 5.1066f
        val yMax = 9.8559f
        val imageWidth = 1020f
        val imageHeight = 530f

        val scaleX = imageWidth / (xMax - xMin)
        val scaleY = imageHeight / (yMax - yMin)

        val x = px / scaleX + xMin
        val y = py / scaleY + yMin

        return x to y
    }

    private fun stationNameToId(name: String?): Int? = when (name) {
        "초음파 검사실" -> 1
        "CT 검사실" -> 2
        "X-ray 검사실" -> 3
        "대장암 센터" -> 4
        "위암 센터" -> 5
        "폐암 센터" -> 6
        "유방암 센터" -> 7
        "뇌종양 센터" -> 8
        else -> null
    }

    private fun sendDirectionRequest(patientId: String?, stationId: Int) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val url = if (isFromCheckin)
                    NetworkConfig.getAuthDirectionUrl()
                else
                    NetworkConfig.getWithoutAuthDirectionUrl()

                val json = JSONObject().apply {
                    put("robot_id", 3)
                    put("department_id", stationId)
                    if (isFromCheckin && !patientId.isNullOrBlank()) {
                        put("patient_id", patientId.toIntOrNull() ?: return@launch)
                    }
                }

                Log.d("DirectionAPI", "📤 안내 요청: $json → $url")

                val request = Request.Builder()
                    .url(url)
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()

                Log.d("DirectionAPI", "📥 응답 코드: ${response.code}")
                Log.d("DirectionAPI", "📥 응답 바디: ${response.body?.string()}")
            } catch (e: Exception) {
                Log.e("DirectionAPI", "❌ 안내 API 호출 실패", e)
            }
        }
    }
}
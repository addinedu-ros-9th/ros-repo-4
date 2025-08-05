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

class GuidanceConfirmActivity : AppCompatActivity() {

    private var isFromCheckin: Boolean = false
    private val robotLocationUrl = NetworkConfig.getRobotLocationUrl()

    private val stationCoords = mapOf(
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

        isFromCheckin = intent.getBooleanExtra(
            "isFromCheckin",
            false
        ) || (userName != null && department != null)

        // 📍 안내 문구 구성
        if (isFromCheckin) {
            val message = "${userName}님 ${department} 접수가 완료되었습니다.\n안내를 시작할까요?"
            val spannable = SpannableString(message)
            listOf(userName, department).forEach { value ->
                value?.let {
                    val start = message.indexOf(it)
                    if (start >= 0) {
                        spannable.setSpan(
                            ForegroundColorSpan(highlightColor),
                            start, start + it.length,
                            Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                        )
                    }
                }
            }
            textView.text = spannable
        } else if (selectedText != null) {
            val message = "${selectedText}를 선택하셨습니다.\n안내를 시작할까요?"
            val spannable = SpannableString(message)
            val start = message.indexOf(selectedText)
            if (start >= 0) {
                spannable.setSpan(
                    ForegroundColorSpan(highlightColor),
                    start, start + selectedText.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
            textView.text = spannable
        } else {
            textView.text = "안내를 시작할까요?"
        }

        // 📍 목적지 마커 표시
        val targetName = selectedText ?: department
        stationCoords[targetName]?.let { (x, y) ->
            val (px, py) = mapToPixelDirect(x, y)
            destinationMarker.visibility = View.VISIBLE
            destinationMarker.post {
                destinationMarker.x = mapView.x + px - destinationMarker.width / 2
                destinationMarker.y = mapView.y + py - destinationMarker.height / 2
            }
        }

        fetchRobotPosition(robotMarker, mapView)

        cancelButton.setOnClickListener {
            applyAlphaEffect(cancelButton)
            cancelButton.postDelayed({
                val intent = if (isFromCheckin) {
                    Intent(this, AuthenticationActivity::class.java)
                } else {
                    Intent(this, GuidanceActivity::class.java)
                }
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        confirmButton.setOnClickListener {
            applyAlphaEffect(confirmButton)
            confirmButton.postDelayed({
                val centerName = selectedText ?: department ?: "해당 센터"
                val stationId = stationNameToId(centerName)

                Log.d(
                    "DirectionAPI",
                    "🔹 isFromCheckin: $isFromCheckin, patientId: $patientId, selectedText: $selectedText"
                )

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

                val requestBody = json.toString()
                    .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())
                val request = Request.Builder()
                    .url(robotLocationUrl)
                    .post(requestBody)
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                if (!body.isNullOrEmpty()) {
                    val responseJson = JSONObject(body)
                    val x = responseJson.optDouble("x", Double.NaN).toFloat()
                    val y = responseJson.optDouble("y", Double.NaN).toFloat()

                    if (!x.isNaN() && !y.isNaN()) {
                        val (px, py) = mapToPixelDirect(x, y)
                        withContext(Dispatchers.Main) {
                            robotMarker.visibility = View.VISIBLE
                            robotMarker.x = mapView.x + px - robotMarker.width / 2
                            robotMarker.y = mapView.y + py - robotMarker.height / 2
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

    private fun stationNameToId(name: String?): Int? {
        return when (name) {
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
    }

    private fun sendDirectionRequest(patientId: String?, stationId: Int) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val url = if (isFromCheckin) {
                    NetworkConfig.getAuthDirectionUrl()
                } else {
                    NetworkConfig.getWithoutAuthDirectionUrl()
                }

                val json = JSONObject().apply {
                    put("robot_id", 3)
                    put("department_id", stationId)
                    if (isFromCheckin && !patientId.isNullOrBlank()) {
                        try {
                            put("patient_id", patientId.toInt())  // Int로 변환
                        } catch (e: NumberFormatException) {
                            Log.e("DirectionAPI", "❌ patient_id 변환 실패: $patientId", e)
                        }
                    }
                }

                Log.d("DirectionAPI", "📤 안내 요청: $json → $url")

                val request = Request.Builder()
                    .url(url)
                    .post(
                        json.toString()
                            .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())
                    )
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("DirectionAPI", "📥 응답 코드: ${response.code}")
                Log.d("DirectionAPI", "📥 응답 바디: $responseBody")
            } catch (e: Exception) {
                Log.e("DirectionAPI", "❌ 안내 API 호출 실패", e)
            }
        }
    }
}

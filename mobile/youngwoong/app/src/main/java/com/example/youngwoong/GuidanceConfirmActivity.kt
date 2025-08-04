package com.example.youngwoong

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import android.util.Log


class GuidanceConfirmActivity : AppCompatActivity() {

    private var isFromCheckin: Boolean = false
    private val robotLocationUrl = NetworkConfig.getRobotLocationUrl()

    // ✅ 실제 측정된 Android 기준 좌표
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

        isFromCheckin = intent.getBooleanExtra("isFromCheckin", false) ||
                (userName != null && department != null)

        // ✅ 안내 문구
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

        // ✅ 목적지 마커 표시
        val targetName = selectedText ?: department
        stationCoords[targetName]?.let { (x, y) ->
            val (px, py) = mapToPixelDirect(x, y)
            destinationMarker.visibility = View.VISIBLE
            destinationMarker.post {
                destinationMarker.x = mapView.x + px - destinationMarker.width / 2
                destinationMarker.y = mapView.y + py - destinationMarker.height / 2
            }
        }
        // ✅ 로봇 위치 받아오기
        fetchRobotPosition(robotMarker, mapView)

        // ✅ 버튼 처리
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
                val intent = Intent(this, GuidanceWaitingActivity::class.java)
                val centerName = selectedText ?: department ?: "해당 센터"
                intent.putExtra("selected_text", centerName)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }
    }
    private fun fetchRobotPosition(robotMarker: ImageView, mapView: ImageView) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val json = JSONObject().apply {
                    put("robot_id", "3")
                }

                Log.d("RobotPosition", "📤 전송 JSON: $json")
                val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())
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
                    Log.e("RobotPosition", "❌ 서버 응답 body 비어있음")
                }
            } catch (e: Exception) {
                Log.e("RobotPosition", "❌ 로봇 위치 가져오기 실패", e)
            }
        }
    }

    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }

    // ✅ 지도 좌표 → 픽셀 변환
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
}

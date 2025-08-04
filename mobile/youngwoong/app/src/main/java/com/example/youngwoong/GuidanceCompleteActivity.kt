package com.example.youngwoong

import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class GuidanceCompleteActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_complete)

        val textView = findViewById<TextView>(R.id.text_guidance_message)
        val confirmButton = findViewById<ImageView>(R.id.btn_confirm_on)

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

        // ✅ 확인 버튼 클릭 → 복귀 명령 전송
        confirmButton.setOnClickListener {
            sendRobotReturnCommand()
            finish()
        }
    }

    // ✅ 로봇 복귀 명령 (IF-06 / IF-07)
    private fun sendRobotReturnCommand() {
        val isFromCheckin = intent.getBooleanExtra("isFromCheckin", false)
        val patientId = intent.getStringExtra("patient_id") ?: ""
        val url = if (isFromCheckin) {
            NetworkConfig.getRobotReturnAuthUrl()
        } else {
            NetworkConfig.getRobotReturnWithoutAuthUrl()
        }

        val json = JSONObject().apply {
            if (isFromCheckin) put("patient_id", patientId)
            put("robot_id", 3)
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

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

class GuidanceConfirmActivity : AppCompatActivity() {

    private var isFromCheckin: Boolean = false // 경로 구분용

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_confirm)

        val cancelButton = findViewById<ImageView>(R.id.btn_cancel)
        val confirmButton = findViewById<ImageView>(R.id.btn_start_guidance)
        val textView = findViewById<TextView>(R.id.text_guidance_message)
        val highlightColor = Color.parseColor("#00696D")

        val userName = intent.getStringExtra("user_name")
        val department = intent.getStringExtra("department")
        val waitingNumber = intent.getStringExtra("waiting_number")
        val selectedText = intent.getStringExtra("selected_text")

        isFromCheckin = intent.getBooleanExtra("isFromCheckin", false) ||
                (userName != null && department != null)


        // ✅ 취소 버튼
        cancelButton.setOnClickListener {
            applyAlphaEffect(cancelButton)
            cancelButton.postDelayed({
                if (isFromCheckin) {
                    val intent = Intent(this, AuthenticationActivity::class.java)
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    startActivity(intent)
                } else {
                    val intent = Intent(this, GuidanceActivity::class.java)
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    startActivity(intent)
                }
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        // ✅ 확인 버튼
        confirmButton.setOnClickListener {
            applyAlphaEffect(confirmButton)
            confirmButton.postDelayed({
                val intent = Intent(this, GuidanceWaitingActivity::class.java)

                // 👉 null 대비: selectedText 없으면 department 사용
                val centerName = selectedText ?: department ?: "해당 센터"
                intent.putExtra("selected_text", centerName)

                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        // ✅ 안내 문구 표시
        if (isFromCheckin) {
            val message = "${userName}님 ${department} 접수가 완료되었습니다.\n안내를 시작할까요?"
            val spannable = SpannableString(message)
            listOf(userName, department).forEach { it ->
                it?.let { value ->
                    val start = message.indexOf(value)
                    if (start >= 0) {
                        spannable.setSpan(
                            ForegroundColorSpan(highlightColor),
                            start,
                            start + value.length,
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
                    start,
                    start + selectedText.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
            textView.text = spannable
        } else {
            textView.text = "안내를 시작할까요?"
        }
    }

    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }
}

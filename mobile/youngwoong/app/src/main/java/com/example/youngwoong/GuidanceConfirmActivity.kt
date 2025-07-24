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

        // 취소 버튼
        val cancelButton = findViewById<ImageView>(R.id.btn_cancel)
        cancelButton.setOnClickListener {
            applyAlphaEffect(cancelButton)
            cancelButton.postDelayed({
                if (isFromCheckin) {
                    // 접수 경로에서 왔다면 메인으로 이동
                    val intent = Intent(this, AuthenticationActivity::class.java)
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    startActivity(intent)
                } else {
                    // 길안내 경로에서 왔다면 다시 길안내 페이지로 이동
                    val intent = Intent(this, GuidanceActivity::class.java)
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    startActivity(intent)
                }
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                finish()
            }, 100)
        }

        val textView = findViewById<TextView>(R.id.text_guidance_message)
        val highlightColor = Color.parseColor("#00696D")

        val userName = intent.getStringExtra("user_name")
        val department = intent.getStringExtra("department")
        val waitingNumber = intent.getStringExtra("waiting_number")
        val selectedText = intent.getStringExtra("selected_text")

        isFromCheckin = (userName != null && department != null && waitingNumber != null)

        if (isFromCheckin) {
            // 진료 접수 경로
            val message = "${userName}님 ${department} 접수가 완료되었습니다.\n현재 대기번호는 ${waitingNumber}번입니다."
            val spannable = SpannableString(message)

            listOf(userName, department, waitingNumber).forEach { it ->
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
            // 길안내 경로
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

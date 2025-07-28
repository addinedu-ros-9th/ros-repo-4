package com.example.youngwoong

import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class GuidanceCompleteActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance_complete)

        val textView = findViewById<TextView>(R.id.text_guidance_message)
        val confirmButton = findViewById<ImageView>(R.id.btn_confirm_on)

        val selectedText = intent.getStringExtra("selected_text") ?: "해당 센터"
        val message = "${selectedText}에 도착했습니다.\n확인을 누르면 영웅이가 복귀합니다."

        // ✅ 강조 색 적용 (선택 영역만 색 변경)
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

        // ✅ 확인 버튼 클릭 처리
        confirmButton.setOnClickListener {
            // TODO: 여기에 복귀 시 NavigationActivity 등으로 이동하도록 확장 가능
            finish()
        }
    }
}

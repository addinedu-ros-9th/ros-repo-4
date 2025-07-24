package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity

class GuidanceActivity : AppCompatActivity() {

    private lateinit var confirmButton: ImageView
    private var selectedButton: ImageView? = null
    private var selectedId: Int? = null
    private var selectedName: String? = null  // 선택된 이름 저장

    private val buttonMap by lazy {
        mapOf(
            R.id.btn_colon to Pair(R.drawable.btn_colon_default, R.drawable.btn_colon_selected),
            R.id.btn_stomach to Pair(R.drawable.btn_stomach_default, R.drawable.btn_stomach_selected),
            R.id.btn_lung to Pair(R.drawable.btn_lung_default, R.drawable.btn_lung_selected),
            R.id.btn_breast to Pair(R.drawable.btn_breast_default, R.drawable.btn_breast_selected),
            R.id.btn_brain to Pair(R.drawable.btn_brain_default, R.drawable.btn_brain_selected),
            R.id.btn_xray to Pair(R.drawable.btn_xray_default, R.drawable.btn_xray_selected),
            R.id.btn_ct to Pair(R.drawable.btn_ct_default, R.drawable.btn_ct_selected),
            R.id.btn_ultrasound to Pair(R.drawable.btn_ultrasound_default, R.drawable.btn_ultrasound_selected)
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_guidance)

        confirmButton = findViewById(R.id.btn_confirm)
        confirmButton.setImageResource(R.drawable.confirm_btn)

        // 버튼 클릭 리스너 설정
        buttonMap.keys.forEach { id ->
            val button = findViewById<ImageView>(id)
            button.setOnClickListener { view ->
                applyAlphaEffect(view)
                handleSelection(button)
            }
        }

        // 뒤로가기 버튼
        findViewById<ImageView>(R.id.btn_back).setOnClickListener {
            applyAlphaEffect(it)
            returnToMainMenu()
        }

        // 확인 버튼 클릭 시 다음 화면으로 이동
        confirmButton.setOnClickListener {
            if (selectedId != null && selectedName != null) {
                applyAlphaEffect(it)
                val intent = Intent(this, GuidanceConfirmActivity::class.java)
                intent.putExtra("selected_text", selectedName)
                startActivity(intent)
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            }
        }

        // 물리 뒤로가기 대응
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                returnToMainMenu()
            }
        })
    }

    private fun returnToMainMenu() {
        val intent = Intent(this, MainMenuActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    private fun handleSelection(button: ImageView) {
        val id = button.id

        // 이전 선택 초기화
        selectedButton?.let {
            val (defaultRes, _) = buttonMap[it.id] ?: return@let
            it.setImageResource(defaultRes)
        }

        // 현재 선택 적용
        val (_, selectedRes) = buttonMap[id] ?: return
        button.setImageResource(selectedRes)
        selectedButton = button
        selectedId = id

        // 선택된 이름 설정
        selectedName = when (id) {
            R.id.btn_colon -> "대장암 센터"
            R.id.btn_stomach -> "위암 센터"
            R.id.btn_lung -> "폐암 센터"
            R.id.btn_breast -> "유방암 센터"
            R.id.btn_brain -> "뇌종양 센터"
            R.id.btn_xray -> "X-ray 검사실"
            R.id.btn_ct -> "CT 검사실"
            R.id.btn_ultrasound -> "초음파 검사실"
            else -> null
        }

        // 확인 버튼 활성화
        confirmButton.setImageResource(R.drawable.btn_confirm_on)
    }

    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }
}

package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity

class AuthenticationActivity : AppCompatActivity() {

    private val inputBoxes = mutableListOf<TextView>()
    private var currentIndex = 0
    private var maxLength = 13 // 기본값: 주민번호
    private lateinit var dashView: View
    private lateinit var confirmBtn: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_authentication)

        // 하드웨어 뒤로가기 대응
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                goToMainMenu()
            }
        })

        dashView = findViewById(R.id.dash_view)
        val tabJumin = findViewById<ImageView>(R.id.tab_jumin)
        val tabMember = findViewById<ImageView>(R.id.tab_member)
        val inputBg = findViewById<ImageView>(R.id.input_bg)
        confirmBtn = findViewById(R.id.btn_confirm)

        // 입력 박스 초기화
        for (i in 0..12) {
            val resId = resources.getIdentifier("box_$i", "id", packageName)
            inputBoxes.add(findViewById(resId))
        }

        // 뒤로가기 버튼
        findViewById<ImageView>(R.id.btn_back).setOnClickListener {
            applyAlphaEffect(it)
            goToMainMenu()
        }

        // 탭: 회원번호
        tabMember.setOnClickListener {
            applyAlphaEffect(it)
            tabMember.setImageResource(R.drawable.member_tab_active)
            tabJumin.setImageResource(R.drawable.jumin_tab_inactive)
            inputBg.setImageResource(R.drawable.number_input_bg1)
            maxLength = 8
            clearInputs()
            updateBoxVisibility()
        }

        // 탭: 주민번호
        tabJumin.setOnClickListener {
            applyAlphaEffect(it)
            tabJumin.setImageResource(R.drawable.jumin_tab_active)
            tabMember.setImageResource(R.drawable.member_tab_inactive)
            inputBg.setImageResource(R.drawable.number_input_bg)
            maxLength = 13
            clearInputs()
            updateBoxVisibility()
        }

        // 숫자 키패드 버튼
        for (i in 0..9) {
            val btnId = resources.getIdentifier("btn_$i", "id", packageName)
            val button = findViewById<ImageView>(btnId)
            button.setOnClickListener {
                applyAlphaEffect(it)
                if (currentIndex < maxLength) {
                    inputBoxes[currentIndex].text = i.toString()
                    currentIndex++
                    updateConfirmButtonState()
                }
            }
        }

        // 삭제 버튼
        findViewById<ImageView>(R.id.btn_delete).setOnClickListener {
            applyAlphaEffect(it)
            if (currentIndex > 0) {
                currentIndex--
                inputBoxes[currentIndex].text = ""
                updateConfirmButtonState()
            }
        }

        // 확인 버튼
        confirmBtn.setOnClickListener {
            applyAlphaEffect(it)
            if (currentIndex == maxLength) {
                val inputNumber = inputBoxes
                    .take(maxLength)
                    .joinToString("") { it.text.toString() }

                // 이름과 부서를 DB 연동하거나 하드코딩으로 설정
                val userName = "김00" // 나중에 서버 연동 시 값 교체
                val department = "정형외과 접수"

                val popup = CheckinPopupDialog(userName, department) {
                    // 안내 시작(예 버튼 클릭) 처리
                    val intent = Intent(this, GuidanceActivity::class.java)
                    startActivity(intent)
                    finish()
                }

                popup.show(supportFragmentManager, "CheckinPopup")
            }
        }

        updateBoxVisibility()
        updateConfirmButtonState()
    }

    private fun clearInputs() {
        inputBoxes.forEach { it.text = "" }
        currentIndex = 0
        updateConfirmButtonState()
    }

    private fun updateBoxVisibility() {
        for (i in inputBoxes.indices) {
            inputBoxes[i].visibility = if (i < maxLength) View.VISIBLE else View.GONE
        }
        dashView.visibility = if (maxLength == 13) View.VISIBLE else View.GONE
    }

    private fun updateConfirmButtonState() {
        if (::confirmBtn.isInitialized) {
            confirmBtn.setImageResource(
                if (currentIndex == maxLength) R.drawable.btn_confirm_on
                else R.drawable.confirm_btn
            )
        }
    }

    private fun goToMainMenu() {
        val intent = Intent(this, MainMenuActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }

    private fun applyAlphaEffect(view: View) {
        view.alpha = 0.6f
        view.postDelayed({ view.alpha = 1.0f }, 100)
    }
}

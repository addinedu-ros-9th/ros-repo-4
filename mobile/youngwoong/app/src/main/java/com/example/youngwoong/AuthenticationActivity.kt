package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject

class AuthenticationActivity : AppCompatActivity() {

    private val inputBoxes = mutableListOf<TextView>()
    private val realInput = mutableListOf<String>() // 실제 숫자 저장용
    private var currentIndex = 0
    private var maxLength = 13
    private lateinit var dashView: View
    private lateinit var confirmBtn: ImageView

    private val handler = Handler(Looper.getMainLooper())
    private val esp32Url = "http://192.168.0.29/uid" // ESP32 주소
    private var lastUid: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_authentication)

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

        for (i in 0..12) {
            val resId = resources.getIdentifier("box_$i", "id", packageName)
            inputBoxes.add(findViewById(resId))
        }

        findViewById<ImageView>(R.id.btn_back).setOnClickListener {
            applyAlphaEffect(it)
            goToMainMenu()
        }

        tabMember.setOnClickListener {
            applyAlphaEffect(it)
            tabMember.setImageResource(R.drawable.member_tab_active)
            tabJumin.setImageResource(R.drawable.jumin_tab_inactive)
            inputBg.setImageResource(R.drawable.number_input_bg1)
            maxLength = 8
            clearInputs()
            updateBoxVisibility()
        }

        tabJumin.setOnClickListener {
            applyAlphaEffect(it)
            tabJumin.setImageResource(R.drawable.jumin_tab_active)
            tabMember.setImageResource(R.drawable.member_tab_inactive)
            inputBg.setImageResource(R.drawable.number_input_bg)
            maxLength = 13
            clearInputs()
            updateBoxVisibility()
        }

        for (i in 0..9) {
            val btnId = resources.getIdentifier("btn_$i", "id", packageName)
            val button = findViewById<ImageView>(btnId)
            button.setOnClickListener {
                applyAlphaEffect(it)
                if (currentIndex < maxLength) {
                    val digit = i.toString()
                    realInput.add(digit)

                    if (maxLength == 13 && currentIndex >= 7) {
                        inputBoxes[currentIndex].text = "*"
                    } else {
                        inputBoxes[currentIndex].text = digit
                    }

                    currentIndex++
                    updateConfirmButtonState()
                }
            }
        }

        findViewById<ImageView>(R.id.btn_delete).setOnClickListener {
            applyAlphaEffect(it)
            if (currentIndex > 0) {
                currentIndex--
                realInput.removeAt(realInput.size - 1)
                inputBoxes[currentIndex].text = ""
                updateConfirmButtonState()
            }
        }

        confirmBtn.setOnClickListener {
            applyAlphaEffect(it)
            if (currentIndex == maxLength) {
                val inputNumber = realInput.joinToString("")
                showUidPopup(inputNumber)
            }
        }

        updateBoxVisibility()
        updateConfirmButtonState()
        startPollingUid()
    }

    private fun clearInputs() {
        realInput.clear()
        inputBoxes.forEach { it.text = "" }
        currentIndex = 0
        updateConfirmButtonState()
    }

    private fun updateBoxVisibility() {
        inputBoxes.forEachIndexed { i, box ->
            box.visibility = if (i < maxLength) View.VISIBLE else View.GONE
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

    private fun startPollingUid() {
        val intervalMs = 1000L
        val pollingHandler = Handler(Looper.getMainLooper())
        val pollingRunnable = object : Runnable {
            override fun run() {
                pollUidFromESP32()
                pollingHandler.postDelayed(this, intervalMs)
            }
        }
        pollingHandler.post(pollingRunnable)
    }

    private fun pollUidFromESP32() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val client = OkHttpClient()
                val request = Request.Builder().url(esp32Url).build()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                if (!body.isNullOrEmpty()) {
                    val uid = JSONObject(body).optString("uid")
                    if (uid.isNotBlank()) {
                        Log.d("ESP32_UID", "\uD83D\uDCE5 UID 수신: $uid")
                        withContext(Dispatchers.Main) {
                            showUidPopup(uid)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("ESP32_UID", "\u274C UID 요청 실패: ${e.message}")
            }
        }
    }

    private fun showUidPopup(uid: String) {
        val userName = "김00"
        val department = "정형외과"
        val waitingNumber = "001"
        val reservationTime = "00시 00분"

        val popup = CheckinPopupDialog(
            userName = userName,
            department = department,
            reservationTime = reservationTime,
            waitingNumber = waitingNumber
        ) {
            val intent = Intent(this, GuidanceConfirmActivity::class.java).apply {
                putExtra("user_name", userName)
                putExtra("department", department)
                putExtra("waiting_number", waitingNumber)
            }
            startActivity(intent)
            finish()
        }

        popup.show(supportFragmentManager, "CheckinPopup")
    }
}

package com.example.youngwoong

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class AuthenticationActivity : AppCompatActivity() {

    private val inputBoxes = mutableListOf<TextView>()
    private val realInput = mutableListOf<String>()
    private var currentIndex = 0
    private var maxLength = 13
    private lateinit var dashView: View
    private lateinit var confirmBtn: ImageView

    private val esp32Url = NetworkConfig.getEsp32Url()
    private var lastUid: String? = null

    private val timeoutHandler = Handler(Looper.getMainLooper())
    private val timeoutRunnable = Runnable {
        sendTimeoutAlert()
        goToIdleMainActivity()
    }

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

                    inputBoxes[currentIndex].text = if (maxLength == 13 && currentIndex >= 7) "*" else digit
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

                if (maxLength == 13) {
                    val ssnWithDash = inputNumber.substring(0, 6) + "-" + inputNumber.substring(6)
                    verifySSNWithServer(ssnWithDash) // 주민번호로 조회
                } else {
                    verifyPatientIdWithServer(inputNumber) // 회원번호로 조회
                }
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
                    if (uid.isNotBlank() && uid != lastUid) {
                        lastUid = uid
                        Log.d("ESP32_UID", "\uD83D\uDCE5 UID 수신: $uid")
                        verifyRFIDWithServer(uid)
                    }
                }
            } catch (e: Exception) {
                Log.e("ESP32_UID", "❌ UID 요청 실패: ${e.message}")
            }
        }
    }

    private fun verifyRFIDWithServer(rfid: String) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val json = JSONObject().apply {
                    put("robot_id", 3)
                    put("rfid", rfid)
                }

                val request = Request.Builder()
                    .url(NetworkConfig.getRfidAuthUrl())
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                if (!body.isNullOrEmpty()) {
                    val data = JSONObject(body)
                    val name = data.optString("name")
                    val reservation = data.optString("reservation")
                    val reservationTime = data.optString("datetime")
                    val department = data.optString("department")
                    val status = data.optString("status")
                    val patientId = data.optString("patient_id")  // 🔥 추가됨

                    Log.d("AUTH_RFID", "✅ 파싱된 데이터: name=$name, department=$department, time=$reservationTime, patientId=$patientId")

                    if (name.isNotBlank() && department.isNotBlank()) {
                        withContext(Dispatchers.Main) {
                            showUidPopup(name, department, reservationTime, status, patientId)  // 🔥 전달
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@AuthenticationActivity, "❌ 등록된 정보 없음", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@AuthenticationActivity, "❌ 서버 응답 없음", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e("AUTH_RFID", "❌ RFID 본인확인 실패: ${e.message}")
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@AuthenticationActivity, "⚠ 네트워크 오류", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun verifySSNWithServer(ssn: String) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val json = JSONObject().apply {
                    put("robot_id", 3)
                    put("ssn", ssn)
                }

                val request = Request.Builder()
                    .url(NetworkConfig.getSsnAuthUrl())
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                if (!body.isNullOrEmpty()) {
                    val data = JSONObject(body)
                    val name = data.optString("name")
                    val department = data.optString("department")
                    val reservationTime = data.optString("datetime")
                    val status = data.optString("status")
                    val patientId = data.optString("patient_id")  // 🔥 추가된 부분

                    Log.d("AUTH_SSN", "✅ 파싱된 데이터: name=$name, department=$department, time=$reservationTime, patientId=$patientId")

                    if (name.isNotBlank() && department.isNotBlank()) {
                        withContext(Dispatchers.Main) {
                            showUidPopup(name, department, reservationTime, status, patientId)  // 🔥 patientId 넘김
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@AuthenticationActivity, "❗등록된 예약 정보가 없습니다.", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@AuthenticationActivity, "❌ 서버 응답 없음", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e("AUTH_SSN", "❌ 본인 확인 실패", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@AuthenticationActivity, "⚠️ 네트워크 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun verifyPatientIdWithServer(patientId: String) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val patientIdInt = patientId.toIntOrNull()
                if (patientIdInt == null) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@AuthenticationActivity, "❗회원번호는 숫자만 입력해야 합니다.", Toast.LENGTH_SHORT).show()
                    }
                    return@launch
                }

                val json = JSONObject().apply {
                    put("robot_id", 3)
                    put("patient_id", patientIdInt)
                }
                Log.d("AUTH_PATIENT_ID", "📤 전송 데이터: $json")

                val request = Request.Builder()
                    .url(NetworkConfig.getPatientIdAuthUrl())
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val body = response.body?.string()

                Log.d("AUTH_PATIENT_ID", "📥 서버 응답: $body")

                if (!body.isNullOrEmpty()) {
                    val data = JSONObject(body)
                    val name = data.optString("name")
                    val department = data.optString("department")
                    val reservationTime = data.optString("datetime")
                    val status = data.optString("status")
                    val returnedPatientId = data.optString("patient_id")  // 🔥 추가됨

                    Log.d("AUTH_PATIENT_ID", "✅ 파싱된 데이터: name=$name, department=$department, time=$reservationTime, status=$status, patientId=$returnedPatientId")

                    if (name.isNotBlank() && department.isNotBlank()) {
                        withContext(Dispatchers.Main) {
                            showUidPopup(name, department, reservationTime, status, returnedPatientId)  // 🔥 전달
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@AuthenticationActivity, "❌ 등록된 정보 없음", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@AuthenticationActivity, "❌ 서버 응답 없음", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e("AUTH_PATIENT_ID", "❌ 회원번호 확인 실패", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@AuthenticationActivity, "⚠ 네트워크 오류", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun startTimeoutTimer() {
        timeoutHandler.postDelayed(timeoutRunnable, 30_000L)
    }

    private fun resetTimeoutTimer() {
        timeoutHandler.removeCallbacks(timeoutRunnable)
        timeoutHandler.postDelayed(timeoutRunnable, 30_000L)
    }

    override fun dispatchTouchEvent(ev: android.view.MotionEvent?): Boolean {
        resetTimeoutTimer()
        return super.dispatchTouchEvent(ev)
    }

    override fun onResume() {
        super.onResume()
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_FULLSCREEN
        startTimeoutTimer()
    }

    override fun onPause() {
        super.onPause()
        timeoutHandler.removeCallbacks(timeoutRunnable)
    }

    private fun sendTimeoutAlert() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val json = JSONObject().apply { put("robot_id", 3) }
                val request = Request.Builder()
                    .url(NetworkConfig.getTimeoutAlertUrl())
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull()))
                    .build()

                val client = OkHttpClient()
                val response = client.newCall(request).execute()
                val statusCode = response.code

                Log.d("TimeoutAlert", "✅ /alert_timeout 호출 결과: $statusCode")
            } catch (e: Exception) {
                Log.e("TimeoutAlert", "❌ /alert_timeout 호출 실패: ${e.message}")
            }
        }
    }

    private fun goToIdleMainActivity() {
        val intent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }
        startActivity(intent)
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        finish()
    }



    private fun showUidPopup(
        userName: String,
        department: String,
        reservationTime: String,
        status: String,
        patientId: String  // 🔼 파라미터 추가
    ) {
        val popup = CheckinPopupDialog(
            userName = userName,
            department = department,
            reservationTime = reservationTime,
            status = status,
            onConfirm = {
                val intent = Intent(this, GuidanceConfirmActivity::class.java).apply {
                    putExtra("user_name", userName)
                    putExtra("department", department)
                    putExtra("isFromCheckin", true)
                    putExtra("patient_id", patientId)  // 🔥 무조건 전달
                }
                startActivity(intent)
                finish()
            }
        )
        popup.show(supportFragmentManager, "CheckinPopup")
    }
}


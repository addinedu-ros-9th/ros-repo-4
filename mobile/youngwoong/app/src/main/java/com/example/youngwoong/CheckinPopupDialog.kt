package com.example.youngwoong

import android.app.Dialog
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.text.Spannable
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.view.View
import android.view.Window
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.DialogFragment

class CheckinPopupDialog(
    private val userName: String,
    private val department: String,
    private val reservationTime: String = "",
    private val status: String = "", // ✅ 상태 추가
    private val patientId: String = "",  // ✅ 추가
    private val waitingNumber: String = "",
    private val onConfirm: ((patientId: String) -> Unit)? = null
) : DialogFragment() {

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val dialog = Dialog(requireContext())
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
        dialog.setContentView(R.layout.dialog_checkin_popup)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.setCanceledOnTouchOutside(false)

        val textView = dialog.findViewById<TextView>(R.id.text_message)
        val btnYes = dialog.findViewById<ImageView>(R.id.btn_yes)
        val btnNo = dialog.findViewById<ImageView>(R.id.btn_no)

        // ✅ 안내 메시지 구성
        val messageBuilder = StringBuilder()
        messageBuilder.append("${userName}님 반갑습니다.\n")

        if (status == "접수") {
            messageBuilder.append("이미 ${department}에 접수되어 있습니다.\n")
        } else {
            if (reservationTime.isNotBlank() && department.isNotBlank()) {
                messageBuilder.append("오늘 ${reservationTime} ${department} 예약접수되었습니다.\n")
            } else if (department.isNotBlank()) {
                messageBuilder.append("${department} 예약접수되었습니다.\n")
            }
        }

        messageBuilder.append("안내해드릴까요?")
        val message = messageBuilder.toString()

        // ✅ 강조 색상 처리
        val spannable = SpannableString(message)
        val highlightColor = Color.parseColor("#00696D")
        listOf(userName, reservationTime, department).forEach {
            val start = message.indexOf(it)
            if (start >= 0) {
                spannable.setSpan(
                    ForegroundColorSpan(highlightColor),
                    start,
                    start + it.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
        }

        textView.text = spannable

        // ✅ 버튼 클릭 처리
        btnYes.setOnClickListener {
            applyAlphaEffect(it) {
                onConfirm?.invoke(patientId)  // ✅ patientId 전달
                activity?.overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                dismiss()
            }
        }

        btnNo.setOnClickListener {
            applyAlphaEffect(it) {
                dismiss()
            }
        }

        return dialog
    }

    private fun applyAlphaEffect(view: View, after: () -> Unit) {
        view.alpha = 0.6f
        view.postDelayed({
            view.alpha = 1.0f
            after()
        }, 100)
    }
}

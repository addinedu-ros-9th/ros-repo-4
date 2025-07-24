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
    private val reservationTime: String,
    private val waitingNumber: String = "001",  // 기본 대기번호
    private val onConfirm: (() -> Unit)? = null // 확인 콜백
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

        // 메시지 구성
        val message = "${userName}님 반갑습니다.\n오늘 ${reservationTime} ${department} 예약되어있습니다.\n접수해드릴까요?"
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

        // 확인 버튼
        btnYes.setOnClickListener {
            applyAlphaEffect(it) {
                onConfirm?.invoke()
                activity?.overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                dismiss()
            }
        }

        // 취소 버튼
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

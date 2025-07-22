package com.example.youngwoong

import android.app.Dialog
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.text.Spannable
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.view.Window
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.DialogFragment

class CheckinPopupDialog(
    private val userName: String,
    private val department: String,
    private val onConfirm: () -> Unit
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

        // 안내 메시지 구성
        val message = "${userName}님 반갑습니다.\n이미 ${department} 접수되었습니다.\n안내해드릴까요?"
        val spannable = SpannableString(message)

        // 이름 강조
        userName.takeIf { it.isNotBlank() }?.let {
            val start = message.indexOf(it)
            if (start >= 0) {
                spannable.setSpan(
                    ForegroundColorSpan(Color.parseColor("#00696D")),
                    start,
                    start + it.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
        }

        // 병원명 강조
        department.takeIf { it.isNotBlank() }?.let {
            val start = message.indexOf(it)
            if (start >= 0) {
                spannable.setSpan(
                    ForegroundColorSpan(Color.parseColor("#00696D")),
                    start,
                    start + it.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            }
        }

        textView.text = spannable

        // 버튼 이벤트
        btnYes.setOnClickListener {
            onConfirm()
            dismiss()
        }

        btnNo.setOnClickListener {
            dismiss()
        }

        return dialog
    }
}

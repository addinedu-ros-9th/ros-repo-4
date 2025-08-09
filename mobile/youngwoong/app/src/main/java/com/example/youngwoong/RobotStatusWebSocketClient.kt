package com.example.youngwoong

import android.util.Log
import okhttp3.*
import org.json.JSONObject

class RobotStatusWebSocketClient(
    private val url: String,
    private var targetRobotId: String? = null,
    private val onStatusReceived: (String) -> Unit
) {
    private lateinit var webSocket: WebSocket
    private val client = OkHttpClient()

    init {
        connect() // ✅ MainActivity 등에서 따로 호출하지 않아도 자동 연결
    }

    fun connect() {
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d("WebSocket", "📨 수신 메시지: $text")
                try {
                    val json = JSONObject(text)

                    // 1) 기본 파라미터 추출
                    val typeField = json.optString("type")              // GUI or arrived_to_call/return_command ...
                    val statusField = json.optString("status")          // 있을 수도, 없을 수도
                    val robotIdStr = when {
                        json.has("robot_id") -> try { json.getString("robot_id") } catch (_: Exception) { json.optInt("robot_id").toString() }
                        else -> null
                    }

                    // 2) 표준 포맷 처리: type == GUI
                    if (typeField.equals("GUI", ignoreCase = true)) {
                        if (robotIdStr == null) return  // 기존 로직 유지
                        if (targetRobotId == null) {
                            targetRobotId = robotIdStr
                            Log.d("WebSocket", "🎯 로봇 ID 설정됨: $targetRobotId")
                        }
                        if (robotIdStr != targetRobotId) return

                        val status = statusField
                        when (status) {
                            "alert_occupied"  -> onStatusReceived("occupied")
                            "alert_idle"      -> onStatusReceived("idle")
                            "return_command"  -> onStatusReceived("return_command")
                            "arrived_to_call" -> onStatusReceived("arrived_to_call")
                            else              -> Log.w("WebSocket", "⚠️ 알 수 없는 상태: $status")
                        }
                        return
                    }

                    // 3) 레거시/간소 포맷 처리: type에 이벤트명이 직접 들어오는 경우
                    // ex) { "type": "arrived_to_call" } 또는 { "type": "return_command" }
                    val legacyEvent = typeField
                    when (legacyEvent) {
                        "arrived_to_call" -> onStatusReceived("arrived_to_call")
                        "return_command"  -> onStatusReceived("return_command")
                        "alert_idle"      -> onStatusReceived("idle")
                        "alert_occupied"  -> onStatusReceived("occupied")
                        else              -> Log.w("WebSocket", "⚠️ 지원하지 않는 메시지 형식: $legacyEvent")
                    }
                } catch (e: Exception) {
                    Log.e("WebSocket", "❌ JSON 파싱 오류: ${e.message}")
                }
            }


            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d("WebSocket", "🚪 종료 요청됨: $reason")
                webSocket.close(1000, null)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("WebSocket", "❌ 연결 실패: ${t.message}")
            }
        })
    }

    fun disconnect() {
        if (::webSocket.isInitialized) {
            webSocket.close(1000, null)
        }
    }
}

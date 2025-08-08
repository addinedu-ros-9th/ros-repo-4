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
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("WebSocket", "✅ 연결됨: $url")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d("WebSocket", "📨 수신 메시지: $text")
                try {
                    val json = JSONObject(text)

                    val robotId: String = when {
                        json.has("robot_id") -> try {
                            json.getString("robot_id")
                        } catch (e: Exception) {
                            json.optInt("robot_id").toString()
                        }
                        else -> return
                    }

                    val status = json.optString("status")
                    val type = json.optString("type")

                    if (type != "GUI") return

                    if (targetRobotId == null) {
                        targetRobotId = robotId
                        Log.d("WebSocket", "🎯 로봇 ID 설정됨: $targetRobotId")
                    }

                    if (robotId != targetRobotId) return

                    when (status) {
                        "alert_occupied" -> onStatusReceived("occupied")
                        "alert_idle" -> onStatusReceived("idle")
                        else -> Log.w("WebSocket", "⚠️ 알 수 없는 상태: $status")
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

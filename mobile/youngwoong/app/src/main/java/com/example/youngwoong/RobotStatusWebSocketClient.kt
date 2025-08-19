package com.example.youngwoong

import android.os.Handler
import android.os.Looper
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import kotlin.math.min
import kotlin.random.Random

class RobotStatusWebSocketClient(
    private val url: String,
    private var targetRobotId: String? = null,
    private val onStatusReceived: (String) -> Unit
) {

    // === 연결 상태 ===
    private var webSocket: WebSocket? = null
    private var isManuallyClosed = false
    private var isConnecting = false
    private var reconnectAttempts = 0

    // === 워치독(무소식 타임아웃) ===
    private val mainHandler = Handler(Looper.getMainLooper())
    private val heartbeatTimeoutMs = 25_000L // ⏱️ 이 시간 동안 어떤 메시지도 안 오면 재연결
    private var lastInboundAt = System.currentTimeMillis()
    private val heartbeatCheck = object : Runnable {
        override fun run() {
            if (!isManuallyClosed) {
                val idle = System.currentTimeMillis() - lastInboundAt
                if (webSocket != null && idle > heartbeatTimeoutMs) {
                    Log.w("WS", "⏱️ heartbeat timeout ($idle ms) → force reconnect")
                    try { webSocket?.cancel() } catch (_: Exception) {}
                    // onFailure로 떨어지면서 자동 재연결 스케줄됨
                }
                // 계속 감시
                mainHandler.postDelayed(this, heartbeatTimeoutMs / 2)
            }
        }
    }

    // ✅ pong 의존 안 함 (pingInterval 미사용)
    private val client: OkHttpClient = OkHttpClient.Builder()
        .retryOnConnectionFailure(true)
        .build()

    // === 외부 API ===
    fun connect() {
        if (isConnecting || isConnected()) return
        isManuallyClosed = false
        isConnecting = true

        val request = Request.Builder().url(url).build()
        Log.d("WS", "🔌 connect(): $url")
        webSocket = client.newWebSocket(request, listener)

        // 워치독 시작(중복 방지 후 시작)
        mainHandler.removeCallbacks(heartbeatCheck)
        mainHandler.postDelayed(heartbeatCheck, heartbeatTimeoutMs / 2)
    }

    fun disconnect() {
        isManuallyClosed = true
        isConnecting = false
        try { webSocket?.close(1000, "client close") } catch (_: Exception) {}
        webSocket = null
        mainHandler.removeCallbacks(heartbeatCheck)
        Log.d("WS", "🧹 disconnect(): closed by client")
    }

    fun isConnected(): Boolean = webSocket != null

    // === 내부: 재연결(지수 백오프 + 지터) ===
    private fun scheduleReconnect(trigger: String) {
        if (isManuallyClosed) return
        val base = min(30_000, (1000 * (1 shl reconnectAttempts))) // 1s,2s,4s… 최대 30s
        val jitter = Random.nextInt(-300, 300)
        val delay = (base + jitter).coerceAtLeast(500)
        Log.d("WS", "⏳ scheduleReconnect($trigger) in ${delay}ms (attempt=$reconnectAttempts)")
        mainHandler.postDelayed({
            if (isManuallyClosed) return@postDelayed
            reconnectAttempts++
            isConnecting = false
            connect()
        }, delay.toLong())
    }

    // === WebSocket 콜백 ===
    private val listener = object : WebSocketListener() {

        override fun onOpen(ws: WebSocket, response: Response) {
            Log.d("WS", "✅ onOpen http=${response.code}")
            reconnectAttempts = 0
            isConnecting = false
            lastInboundAt = System.currentTimeMillis() // 새 연결 기준으로 타임아웃 초기화
        }

        override fun onMessage(ws: WebSocket, text: String) {
            lastInboundAt = System.currentTimeMillis() // 메시지 수신마다 타임아웃 리셋
            Log.d("WS", "📨 onMessage: $text")
            try {
                val json = JSONObject(text)

                // robot_id 필터링(문자/숫자 모두)
                val robotIdStr: String? = when {
                    json.has("robot_id") -> try { json.getString("robot_id") }
                    catch (_: Exception) { json.optLong("robot_id", -1L).takeIf { it >= 0 }?.toString() }
                    else -> null
                }
                if (!robotIdStr.isNullOrBlank()) {
                    if (targetRobotId.isNullOrBlank()) {
                        targetRobotId = robotIdStr
                        Log.d("WS", "🎯 targetRobotId set: $targetRobotId")
                    } else if (robotIdStr != targetRobotId) {
                        Log.d("WS", "🔎 ignore other robot: $robotIdStr")
                        return
                    }
                }

                val type   = json.optString("type")    // "GUI" 또는 이벤트명
                val status = json.optString("status")  // GUI 포맷
                val event  = json.optString("event")   // 일반/레거시 포맷

                // GUI 포맷
                if (type.equals("GUI", ignoreCase = true)) {
                    val s = status.ifBlank { event }
                    when (s) {
                        "alert_occupied"   -> onStatusReceived("occupied")
                        "alert_idle"       -> onStatusReceived("idle")
                        "return_command"   -> onStatusReceived("return_command")
                        "arrived_to_call"  -> onStatusReceived("arrived_to_call")
                        "user_appear"      -> onStatusReceived("user_appear")
                        "user_disappear"   -> onStatusReceived("user_disappear")
                        "stop_tracking"    -> onStatusReceived("stop_tracking")
                        "navigating_complete",
                        "arrived_to_target"-> onStatusReceived(s)
                        else               -> Log.w("WS", "⚠️ unknown GUI status: $s")
                    }
                    return
                }

                // 일반/레거시 포맷
                val e = event.ifBlank { type }
                when (e) {
                    "arrived_to_call"     -> onStatusReceived("arrived_to_call")
                    "return_command"      -> onStatusReceived("return_command")
                    "alert_idle"          -> onStatusReceived("idle")
                    "alert_occupied"      -> onStatusReceived("occupied")
                    "navigating_complete" -> onStatusReceived("navigating_complete")
                    "arrived_to_target"   -> onStatusReceived("arrived_to_target")
                    "user_appear"         -> onStatusReceived("user_appear")
                    "user_disappear"      -> onStatusReceived("user_disappear")
                    "stop_tracking"       -> onStatusReceived("stop_tracking")
                    "", "null"            -> Log.w("WS", "⚠️ no event name: $text")
                    else                  -> Log.w("WS", "⚠️ unsupported event: $e")
                }
            } catch (t: Throwable) {
                Log.e("WS", "❌ parse error: ${t.message}", t)
            }
        }

        override fun onClosing(ws: WebSocket, code: Int, reason: String) {
            Log.d("WS", "🚪 onClosing code=$code reason=$reason")
            try { ws.close(1000, null) } catch (_: Exception) {}
        }

        override fun onClosed(ws: WebSocket, code: Int, reason: String) {
            Log.d("WS", "🔚 onClosed code=$code reason=$reason manual=$isManuallyClosed")
            webSocket = null
            isConnecting = false
            if (!isManuallyClosed) {
                // 코드에 따라 바로 재시도 or 백오프
                val immediate = (code == 1006 /*abnormal*/ || code == 1012 /*service restart*/ || code == 1011 /*server error*/)
                if (immediate) reconnectAttempts = 0
                scheduleReconnect(if (immediate) "closed-immediate" else "closed")
            }
        }

        override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
            Log.e(
                "WS",
                "💥 onFailure: ${t::class.java.simpleName} ${t.message} http=${response?.code} manual=$isManuallyClosed",
                t
            )
            webSocket = null
            isConnecting = false
            if (!isManuallyClosed) scheduleReconnect("failure")
        }
    }
}

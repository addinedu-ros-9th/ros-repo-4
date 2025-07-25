package com.example.youngwoong

import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * 안드로이드용 실시간 토큰 스트리밍 클래스
 * Python의 CustomStreamer를 안드로이드 환경에 맞게 포팅
 */
class AndroidStreamer(
    private val skipPrompt: Boolean = true,
    private val skipSpecialTokens: Boolean = true,
    private val fastMode: Boolean = false,
    private val debugMode: Boolean = false
) {
    
    private val tokenCache = mutableListOf<Int>()
    private val printLen = AtomicInteger(0)
    private val currentLength = AtomicInteger(0)
    private val isStreamingActive = AtomicBoolean(false)
    
    // 스트리밍 결과를 위한 Flow
    private val _streamingFlow = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 1000
    )
    val streamingFlow: SharedFlow<String> = _streamingFlow.asSharedFlow()
    
    // 완료 상태를 위한 Flow
    private val _completionFlow = MutableSharedFlow<String>(
        replay = 0,
        extraBufferCapacity = 1
    )
    val completionFlow: SharedFlow<String> = _completionFlow.asSharedFlow()
    
    companion object {
        private const val TAG = "AndroidStreamer"
    }
    
    /**
     * 새로운 토큰을 받을 때 호출됨 (Python의 put 함수 포팅)
     */
    suspend fun put(tokens: IntArray) = withContext(Dispatchers.Default) {
        try {
            if (!isStreamingActive.get()) {
                debugPrint("⚠️ 스트리밍이 비활성화된 상태에서 토큰 수신 무시됨")
                return@withContext
            }
            
            if (debugMode) {
                debugPrint("🔍 스트리머 입력: ${tokens.size}개 토큰")
            }
            
            // 빈 토큰 배열 체크
            if (tokens.isEmpty()) {
                if (debugMode) {
                    debugPrint("⚠️ 빈 토큰 배열 무시됨")
                }
                return@withContext
            }
            
            // 프롬프트 건너뛰기 로직 (첫 번째 토큰만 건너뛰기)
            if (skipPrompt && tokens.size == 1 && tokenCache.isEmpty() && tokens[0] == 0) {
                if (debugMode) {
                    debugPrint("🔍 첫 번째 토큰 건너뛰기: ${tokens.contentToString()}")
                }
                return@withContext
            }
            
            // 새로운 토큰 처리
            if (tokens.isNotEmpty()) {
                // 모든 토큰을 캐시에 추가
                tokenCache.addAll(tokens.toList())
                
                if (debugMode) {
                    debugPrint("🔍 토큰 캐시에 추가: ${tokens.size}개 토큰")
                    debugPrint("🔍 총 캐시 길이: ${tokenCache.size}")
                }
                
                // 토큰을 텍스트로 변환 (간단한 구현)
                val text = tokens.joinToString("") { it.toString() }
                
                // 출력 가능한 새로운 부분만 추출
                val currentPrintLen = printLen.get()
                if (text.length > currentPrintLen) {
                    val newText = text.substring(currentPrintLen)
                    printLen.set(text.length)
                    
                    if (debugMode) {
                        debugPrint("🔍 출력할 텍스트: '$newText'")
                    }
                    
                    // 실시간 스트리밍 전송
                    _streamingFlow.emit(newText)
                    
                    // 빠른 모드에 따른 지연시간 조정
                    if (fastMode) {
                        delay(1) // 더 빠른 출력
                    } else {
                        delay(2) // 일반 속도
                    }
                }
            }
            
        } catch (e: Exception) {
            // 전체 처리 실패 시 무시하고 계속 진행
            if (debugMode) {
                debugPrint("⚠️ 스트리머 오류 (무시됨): ${e.message}")
            }
        }
    }
    
    /**
     * 스트리밍 시작
     */
    fun startStreaming() {
        isStreamingActive.set(true)
        debugPrint("🚀 실시간 스트리밍 시작됨")
    }
    
    /**
     * 생성이 끝났을 때 호출됨 (Python의 end 함수 포팅)
     */
    suspend fun end() = withContext(Dispatchers.Default) {
        try {
            isStreamingActive.set(false)
            
            // 최종 텍스트 완성
            val finalText = if (tokenCache.isNotEmpty()) {
                tokenCache.joinToString("") { it.toString() }
            } else {
                ""
            }
            
            // 완료 신호 전송
            _completionFlow.emit(finalText)
            
            debugPrint("✅ 스트리밍 완료됨 (총 ${tokenCache.size}개 토큰)")
            
        } catch (e: Exception) {
            debugPrint("❌ 스트리밍 종료 중 오류: ${e.message}")
        }
    }
    
    /**
     * 스트리밍 상태 초기화
     */
    fun reset() {
        tokenCache.clear()
        printLen.set(0)
        currentLength.set(0)
        isStreamingActive.set(false)
        debugPrint("🔄 스트리머 초기화됨")
    }
    
    /**
     * 현재 누적된 텍스트 반환
     */
    fun getCurrentText(): String {
        return if (tokenCache.isNotEmpty()) {
            tokenCache.joinToString("") { it.toString() }
        } else {
            ""
        }
    }
    
    /**
     * 스트리밍 활성 상태 확인
     */
    fun isActive(): Boolean = isStreamingActive.get()
    
    /**
     * 토큰 캐시 크기 반환
     */
    fun getCacheSize(): Int = tokenCache.size
    
    private fun debugPrint(message: String) {
        if (debugMode) {
            Log.d(TAG, message)
        }
    }
}

 
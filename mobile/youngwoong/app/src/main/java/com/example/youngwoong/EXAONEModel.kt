package com.yourpackage

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class EXAONEModel(private val context: Context) {
    private var interpreter: Interpreter? = null
    private lateinit var tokenizer: OriginalTokenizer
    private var config: JSONObject? = null
    
    fun loadModel() {
        try {
            // 모델 파일 로드
            val modelFile = context.assets.open("exaone_model.tflite")
            val modelBytes = modelFile.readBytes()
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelBuffer.order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            
            // 인터프리터 설정
            val options = Interpreter.Options()
            interpreter = Interpreter(modelBuffer, options)
            
            // 원본 토크나이저 초기화
            tokenizer = OriginalTokenizer(context)
            
            // 설정 파일 로드
            val configFile = context.assets.open("model_config.json")
            val configString = configFile.bufferedReader().use { it.readText() }
            config = JSONObject(configString)
            
            println("✅ EXAONE 모델 로드 완료")
            
        } catch (e: Exception) {
            e.printStackTrace()
            println("❌ 모델 로드 실패: ${e.message}")
        }
    }
    
    fun generateResponse(inputText: String): String {
        try {
            // 텍스트 토크나이징
            val inputIds = tokenizer.encode(inputText)
            
            // 입력 버퍼 생성 (모델이 기대하는 형태: [1, 1])
            val inputBuffer = ByteBuffer.allocateDirect(4) // 1 * 1 * 4 bytes
            inputBuffer.order(ByteOrder.nativeOrder())
            inputBuffer.putInt(inputIds.firstOrNull() ?: 0)
            
            // 출력 버퍼 생성 (vocab_size = 102400)
            val vocabSize = 102400
            val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4)
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // 추론 실행
            interpreter?.run(inputBuffer, outputBuffer)
            
            // 결과 디코딩
            return tokenizer.decodeFromBuffer(outputBuffer)
            
        } catch (e: Exception) {
            e.printStackTrace()
            return "오류가 발생했습니다: ${e.message}"
        }
    }
    
    fun generateResponseWithContext(context: String, userInput: String): String {
        try {
            // 컨텍스트와 사용자 입력을 결합
            val fullInput = "$context\n사용자: $userInput\n어시스턴트:"
            
            // 텍스트 토크나이징
            val inputIds = tokenizer.encode(fullInput)
            
            // 입력 버퍼 생성
            val inputBuffer = ByteBuffer.allocateDirect(4)
            inputBuffer.order(ByteOrder.nativeOrder())
            inputBuffer.putInt(inputIds.firstOrNull() ?: 0)
            
            // 출력 버퍼 생성
            val vocabSize = 102400
            val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4)
            outputBuffer.order(ByteOrder.nativeOrder())
            
            // 추론 실행
            interpreter?.run(inputBuffer, outputBuffer)
            
            // 결과 디코딩
            return tokenizer.decodeFromBuffer(outputBuffer)
            
        } catch (e: Exception) {
            e.printStackTrace()
            return "오류가 발생했습니다: ${e.message}"
        }
    }
    
    fun isModelLoaded(): Boolean {
        return interpreter != null
    }
    
    fun getModelInfo(): String {
        return config?.toString() ?: "모델 정보를 불러올 수 없습니다."
    }
    
    fun close() {
        interpreter?.close()
    }
}

// 사용 예시를 위한 확장 함수들
fun EXAONEModel.generateHospitalResponse(userQuery: String): String {
    val hospitalContext = """
        당신은 병원 안내 로봇 영웅이입니다. 
        환자들에게 친절하고 정확한 정보를 제공해주세요.
        병원 위치, 진료과목, 예약 방법 등에 대해 답변해주세요.
    """.trimIndent()
    
    return generateResponseWithContext(hospitalContext, userQuery)
}

fun EXAONEModel.generateSimpleResponse(userInput: String): String {
    return generateResponse(userInput)
} 
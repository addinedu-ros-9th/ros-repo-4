package com.example.youngwoong

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.util.*

class OriginalTokenizer(private val context: Context) {
    private val vocab = mutableMapOf<String, Int>()
    private val reverseVocab = mutableMapOf<Int, String>()
    private val merges = mutableListOf<Pair<String, String>>()
    private val specialTokens = mutableMapOf<String, Int>()
    
    init {
        loadTokenizer()
    }
    
    private fun loadTokenizer() {
        try {
            // 1. 어휘 사전 로드 (vocab.json)
            val vocabStream = context.assets.open("tokenizer_info/vocab.json")
            val vocabReader = BufferedReader(InputStreamReader(vocabStream))
            val vocabJson = JSONObject(vocabReader.readText())
            
            vocabJson.keys().forEach { token ->
                val tokenId = vocabJson.getInt(token)
                vocab[token] = tokenId
                reverseVocab[tokenId] = token
            }
            
            // 2. BPE 병합 규칙 로드 (merges.txt)
            val mergesStream = context.assets.open("tokenizer_info/merges.txt")
            val mergesReader = BufferedReader(InputStreamReader(mergesStream))
            
            var line: String?
            while (mergesReader.readLine().also { line = it } != null) {
                if (line!!.startsWith("#")) continue // 주석 무시
                val parts = line!!.split(" ")
                if (parts.size == 2) {
                    merges.add(Pair(parts[0], parts[1]))
                }
            }
            
            // 3. 특수 토큰 로드 (special_tokens_map.json)
            val specialTokensStream = context.assets.open("tokenizer_info/special_tokens_map.json")
            val specialTokensReader = BufferedReader(InputStreamReader(specialTokensStream))
            val specialTokensJson = JSONObject(specialTokensReader.readText())
            
            specialTokensJson.keys().forEach { tokenType ->
                val token = specialTokensJson.getJSONObject(tokenType).getString("content")
                val tokenId = vocab[token] ?: 0
                specialTokens[tokenType] = tokenId
            }
            
            println("✅ 원본 토크나이저 로드 완료")
            println("  - 어휘 크기: ${vocab.size}")
            println("  - 병합 규칙: ${merges.size}")
            println("  - 특수 토큰: ${specialTokens.size}")
            
        } catch (e: Exception) {
            e.printStackTrace()
            println("❌ 토크나이저 로드 실패: ${e.message}")
        }
    }
    
    fun encode(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        
        try {
            // 특수 토큰 처리
            val processedText = processSpecialTokens(text)
            
            // BPE 토크나이징
            val words = processedText.split(" ")
            for (word in words) {
                val wordTokens = bpeEncode(word)
                tokens.addAll(wordTokens)
            }
            
            // 빈 리스트인 경우 기본 토큰 추가
            if (tokens.isEmpty()) {
                tokens.add(getPadTokenId())
            }
            
        } catch (e: Exception) {
            e.printStackTrace()
            // 오류 시 기본 토큰 반환
            tokens.add(getPadTokenId())
        }
        
        return tokens
    }
    
    private fun processSpecialTokens(text: String): String {
        var processedText = text
        
        // 특수 토큰들을 실제 토큰으로 변환
        specialTokens.forEach { (tokenType, tokenId) ->
            when (tokenType) {
                "user" -> processedText = processedText.replace("[|user|]", reverseVocab[tokenId] ?: "[|user|]")
                "assistant" -> processedText = processedText.replace("[|assistant|]", reverseVocab[tokenId] ?: "[|assistant|]")
                "system" -> processedText = processedText.replace("[|system|]", reverseVocab[tokenId] ?: "[|system|]")
                "tool" -> processedText = processedText.replace("[|tool|]", reverseVocab[tokenId] ?: "[|tool|]")
                "endofturn" -> processedText = processedText.replace("[|endofturn|]", reverseVocab[tokenId] ?: "[|endofturn|]")
            }
        }
        
        return processedText
    }
    
    private fun bpeEncode(word: String): List<Int> {
        if (word.isEmpty()) return listOf()
        
        // 단어를 문자 단위로 분할
        val chars = word.toCharArray().map { it.toString() }
        var pairs = mutableListOf<Pair<String, String>>()
        
        // 초기 쌍 생성
        for (i in 0 until chars.size - 1) {
            pairs.add(Pair(chars[i], chars[i + 1]))
        }
        
        // BPE 병합 규칙 적용
        for (merge in merges) {
            var i = 0
            while (i < pairs.size) {
                if (pairs[i] == merge) {
                    // 병합 수행
                    val merged = pairs[i].first + pairs[i].second
                    pairs[i] = Pair(merged, "")
                    
                    // 다음 쌍과 병합
                    if (i + 1 < pairs.size) {
                        pairs[i] = Pair(merged, pairs[i + 1].second)
                        pairs.removeAt(i + 1)
                    }
                }
                i++
            }
        }
        
        // 토큰 ID로 변환
        val tokens = mutableListOf<Int>()
        for (pair in pairs) {
            val token = pair.first + pair.second
            val tokenId = vocab[token] ?: vocab[token.lowercase()]
            if (tokenId != null) {
                tokens.add(tokenId)
            } else {
                // 알 수 없는 토큰은 기본값
                tokens.add(getPadTokenId())
            }
        }
        
        return tokens
    }
    
    fun decode(tokenIds: List<Int>): String {
        return tokenIds.joinToString("") { tokenId ->
            reverseVocab[tokenId] ?: "?"
        }
    }
    
    fun decodeFromBuffer(buffer: ByteBuffer): String {
        val tokens = mutableListOf<Int>()
        buffer.rewind()
        
        while (buffer.hasRemaining()) {
            val tokenId = buffer.int
            if (tokenId > 0) {
                tokens.add(tokenId)
            }
        }
        
        return decode(tokens)
    }
    
    fun getVocabSize(): Int {
        return vocab.size
    }
    
    fun getBosTokenId(): Int {
        return specialTokens["bos_token"] ?: 1
    }
    
    fun getEosTokenId(): Int {
        return specialTokens["eos_token"] ?: 361
    }
    
    fun getPadTokenId(): Int {
        return specialTokens["pad_token"] ?: 0
    }
    
    fun getUserTokenId(): Int {
        return specialTokens["user"] ?: 360
    }
    
    fun getAssistantTokenId(): Int {
        return specialTokens["assistant"] ?: 359
    }
    
    /**
     * 정수 배열을 디코딩 (AndroidStreamer용)
     */
    fun decode(tokenIds: IntArray, skipSpecialTokens: Boolean = true): String {
        return tokenIds.map { tokenId ->
            if (skipSpecialTokens && isSpecialToken(tokenId)) {
                ""
            } else {
                reverseVocab[tokenId] ?: ""
            }
        }.joinToString("")
    }
    
    /**
     * 특수 토큰 여부 확인
     */
    private fun isSpecialToken(tokenId: Int): Boolean {
        return specialTokens.values.contains(tokenId)
    }
    
    /**
     * 텍스트를 정수 배열로 인코딩
     */
    fun encodeToIntArray(text: String): IntArray {
        return encode(text).toIntArray()
    }
    
    /**
     * 토큰 정보 출력 (디버그용)
     */
    fun getTokenInfo(): String {
        return """
            토크나이저 정보:
            - 어휘 크기: ${vocab.size}
            - 병합 규칙: ${merges.size}개
            - 특수 토큰: ${specialTokens.size}개
            - BOS 토큰 ID: ${getBosTokenId()}
            - EOS 토큰 ID: ${getEosTokenId()}
            - PAD 토큰 ID: ${getPadTokenId()}
        """.trimIndent()
    }
    
    /**
     * 어휘에 토큰이 존재하는지 확인
     */
    fun hasToken(token: String): Boolean {
        return vocab.containsKey(token)
    }
    
    /**
     * 토큰 ID로 토큰 문자열 조회
     */
    fun getTokenById(tokenId: Int): String? {
        return reverseVocab[tokenId]
    }
    
    /**
     * 토큰 문자열로 토큰 ID 조회
     */
    fun getIdByToken(token: String): Int? {
        return vocab[token]
    }
} 
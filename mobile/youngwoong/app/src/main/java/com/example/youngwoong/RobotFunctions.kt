package com.example.youngwoong

import android.content.Context
import org.json.JSONObject

class RobotFunctions(private val context: Context) {
    
    // 병원 시설 데이터베이스 (실제 환경에서는 SQLite DB 연동)
    private val facilityDatabase = mapOf(
        "CT" to "왼쪽 중앙 영상의학과",
        "X-ray" to "왼쪽 상단 영상의학과", 
        "엑스레이" to "왼쪽 상단 영상의학과",
        "초음파" to "왼쪽 하단 영상의학과",
        "뇌종양" to "오른쪽 상단 암센터",
        "유방암" to "오른쪽 상단 암센터",
        "대장암" to "오른쪽 하단 암센터",
        "위암" to "오른쪽 하단 암센터",
        "폐암" to "오른쪽 하단 암센터"
    )
    
    /**
     * 시설 위치 조회 함수
     */
    fun queryFacility(facility: String): Map<String, Any> {
        return try {
            val normalizedFacility = facility.uppercase().trim()
            val location = facilityDatabase[facility] ?: facilityDatabase[normalizedFacility]
            
            if (location != null) {
                mapOf(
                    "success" to true,
                    "result" to location,
                    "facility" to facility
                )
            } else {
                mapOf(
                    "success" to false,
                    "result" to "error: 시설을 찾을 수 없습니다",
                    "facility" to facility
                )
            }
        } catch (e: Exception) {
            mapOf(
                "success" to false,
                "result" to "error: ${e.message}",
                "facility" to facility
            )
        }
    }
    
    /**
     * 네비게이션 함수
     */
    fun navigate(target: String): Map<String, Any> {
        return try {
            val location = facilityDatabase[target] ?: facilityDatabase[target.uppercase()]
            
            if (location != null) {
                mapOf(
                    "success" to true,
                    "result" to "네비게이션 시작: $target (위치: $location)",
                    "target" to target,
                    "location" to location
                )
            } else {
                mapOf(
                    "success" to false,
                    "result" to "error: 목적지를 찾을 수 없습니다",
                    "target" to target
                )
            }
        } catch (e: Exception) {
            mapOf(
                "success" to false,
                "result" to "error: ${e.message}",
                "target" to target
            )
        }
    }
    
    /**
     * 일반 응답 함수
     */
    fun generalResponse(message: String): Map<String, Any> {
        return try {
            val response = when {
                message.contains("안녕", true) -> "안녕하세요! 저는 병원 안내 로봇 영웅이입니다."
                message.contains("고마", true) || message.contains("감사", true) -> "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요."
                message.contains("누구", true) || message.contains("이름", true) -> "저는 병원 안내 로봇 영웅이입니다. 병원 시설 안내와 길찾기를 도와드려요!"
                else -> "무엇을 도와드릴까요? 병원 시설 안내나 위치 조회를 도와드릴 수 있어요."
            }
            
            mapOf(
                "success" to true,
                "result" to response,
                "message" to message
            )
        } catch (e: Exception) {
            mapOf(
                "success" to false,
                "result" to "error: ${e.message}",
                "message" to message
            )
        }
    }
    
    /**
     * 사용 가능한 시설 목록 조회
     */
    fun getAvailableFacilities(): List<String> {
        return facilityDatabase.keys.toList()
    }
    
    /**
     * 시설 검색 (부분 일치)
     */
    fun searchFacilities(query: String): List<String> {
        return facilityDatabase.keys.filter { 
            it.contains(query, ignoreCase = true) 
        }
    }
} 
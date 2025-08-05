package com.example.youngwoong

object NetworkConfig {
    // 서버 IP 주소들 - 여기서만 변경하면 됩니다
    private const val CENTRAL_SERVER_IP = "192.168.0.37"
    private const val LLM_SERVER_IP = "192.168.0.37"  // LLM 서버 IP (별도 설정 가능)
    
    // 포트 설정
    private const val CENTRAL_SERVER_PORT = "8080"
    private const val LLM_SERVER_PORT = "5000"
    
    // ESP32 IP 주소
    private const val ESP32_IP = "192.168.0.2"
    
    // URL 생성 함수들
    fun getCentralServerUrl(): String = "http://$CENTRAL_SERVER_IP:$CENTRAL_SERVER_PORT"
    fun getLlmServerUrl(): String = "http://$LLM_SERVER_IP:$LLM_SERVER_PORT"
    fun getEsp32Url(): String = "http://$ESP32_IP/uid"
    
    // 개별 엔드포인트 URL들
    fun getRfidAuthUrl(): String = "${getCentralServerUrl()}/auth/rfid"
    fun getSsnAuthUrl(): String = "${getCentralServerUrl()}/auth/ssn"
    fun getPatientIdAuthUrl(): String = "${getCentralServerUrl()}/auth/patient_id"
    fun getRobotLocationUrl(): String = "${getCentralServerUrl()}/get/robot_location"

    // 본인 인증 후 안내 시작 요청 URL (IF-04)
    fun getAuthDirectionUrl(): String = "${getCentralServerUrl()}/auth/direction"
    // 비인증 안내 시작 요청 URL (IF-05)
    fun getWithoutAuthDirectionUrl(): String = "${getCentralServerUrl()}/without_auth/direction"
    // 본인 인증 후 복귀 요청 URL (IF-06)
    fun getRobotReturnAuthUrl(): String = "${getCentralServerUrl()}/auth/robot_return"
    // 비인증 후 복귀 요청 URL (IF-07)
    fun getRobotReturnWithoutAuthUrl(): String = "${getCentralServerUrl()}/without_auth/robot_return"

    // 로봇 상태 변경 (IF-08)
    fun getChangeRobotStatusUrl(): String = "${getCentralServerUrl()}/change/robot_status"

    // 서버 IP들만 반환 (network_security_config.xml용)
    fun getCentralServerIp(): String = CENTRAL_SERVER_IP
    fun getLlmServerIp(): String = LLM_SERVER_IP
} 
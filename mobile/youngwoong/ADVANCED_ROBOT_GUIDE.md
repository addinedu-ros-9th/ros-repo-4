# 🤖 EXAONE 병원 안내 로봇 - 고도화 시스템 가이드

## 📋 개요

Python의 고도화된 프롬프트 엔지니어링 시스템을 안드로이드 환경에 완전히 포팅한 병원 안내 로봇입니다.

### ✨ 주요 특징

- **🧠 자동 모드 전환**: 질문 복잡도에 따라 Reasoning ↔ Non-reasoning 모드 자동 전환
- **🔗 맥락 인식**: 이전 대화 맥락을 고려한 연속 대화 지원
- **⚡ 실시간 스트리밍**: 토큰 단위 실시간 응답 출력
- **🛠️ 함수 호출**: Tool calling을 통한 시설 조회, 네비게이션 등
- **📚 대화 히스토리**: 최근 24개 대화 자동 관리
- **🎯 프롬프트 엔지니어링**: EXAONE 4.0 공식 권장값 기반 최적화

## 🏗️ 시스템 아키텍처

```
RobotSystem (메인 시스템)
├── EXAONEModel (확장된 모델 클래스)
│   ├── 대화 히스토리 관리
│   ├── 맥락 인식 엔진
│   ├── 자동 모드 전환
│   └── 함수 호출 파서
├── AndroidStreamer (실시간 스트리밍)
│   ├── 토큰 캐시 관리
│   ├── Flow 기반 UI 업데이트
│   └── 스트리밍 상태 관리
├── RobotFunctions (로봇 기능들)
│   ├── queryFacility (시설 조회)
│   ├── navigate (네비게이션)
│   └── generalResponse (일반 대화)
└── StreamingManager (스트리밍 관리자)
    ├── 세션 관리
    ├── UI 업데이트 콜백
    └── 리소스 정리
```

## 🚀 빠른 시작

### 1. 기본 사용법

```kotlin
// 1. 로봇 시스템 생성
val robot = RobotSystemBuilder(context)
    .useRealModel(false) // 시뮬레이션 모드
    .enableFastMode(true)
    .build()

// 2. 초기화
robot.initialize()

// 3. 질문하기
val response = robot.ask("CT 어디야?")
println("🤖 $response")

// 4. 리소스 정리
robot.close()
```

### 2. 스트리밍 모드 사용법

```kotlin
// 스트리밍 응답 관찰
lifecycleScope.launch {
    robot.responseFlow.collect { newText ->
        textView.append(newText) // 실시간 업데이트
    }
}

// 스트리밍으로 질문 처리
robot.processUserInputWithStreaming("왜 CT와 X-ray가 다른가요?")
```

### 3. 고급 설정

```kotlin
val robot = RobotSystemBuilder(context)
    .useRealModel(true)          // 실제 EXAONE 모델 사용
    .useReasoning(true)          // Reasoning 모드 활성화
    .enableDebugMode(true)       // 디버그 모드
    .enableFastMode(false)       // 일반 속도 모드
    .build()
```

## 🎯 핵심 기능 상세

### 1. 자동 모드 전환

시스템이 질문의 복잡도를 자동으로 판단하여 적절한 모드를 선택합니다.

**Non-reasoning 모드 (빠른 응답)**
- 간단한 질문: "CT 어디야?", "안녕하세요"
- Temperature: 0.1 (EXAONE 4.0 권장값)
- 함수 호출 우선 사용

**Reasoning 모드 (복잡한 사고)**
- 복잡한 질문: "왜 CT와 X-ray가 다른가요?"
- Temperature: 0.6, top_p: 0.95 (EXAONE 4.0 권장값)
- `<think>` 블록 사용 (계획된 기능)

### 2. 맥락 인식 시스템

```kotlin
// 첫 번째 질문
"CT 검사 받으려면 어떻게 해야 해요?"
// 응답: "CT는 왼쪽 중앙 영상의학과에 있어요..."

// 연속 질문 (맥락 인식)
"거기로 안내해줘"  // "CT로 안내해줘"로 자동 해석
// 응답: "좋아요! CT로 안내해드릴게요. 저를 따라오세요! 🚀"
```

### 3. 함수 호출 시스템

시스템이 사용자 의도를 분석하여 적절한 함수를 자동 호출합니다.

**지원하는 함수들:**
- `query_facility`: 시설 위치 조회
- `navigate`: 네비게이션 시작
- `general_response`: 일반 대화 처리

```kotlin
// 사용자: "초음파실 어디야?"
// → query_facility("초음파") 자동 호출
// → "네! 초음파는 왼쪽 하단 영상의학과에 있어요. 😊"

// 사용자: "CT실로 데려다줘"
// → navigate("CT") 자동 호출  
// → "좋아요! CT로 안내해드릴게요. 저를 따라오세요! 🚀"
```

## 📱 안드로이드 액티비티 통합

### 완전한 액티비티 예제

```kotlin
class MyRobotActivity : AppCompatActivity() {
    private lateinit var robotSystem: RobotSystem
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // 시스템 초기화
        lifecycleScope.launch {
            robotSystem = RobotSystemBuilder(this@MyRobotActivity)
                .useRealModel(true)
                .enableFastMode(true)
                .build()
            
            robotSystem.initialize()
            
            // 스트리밍 응답 관찰
            robotSystem.responseFlow.collect { newText ->
                responseTextView.append(newText)
            }
        }
        
        // 버튼 클릭 처리
        sendButton.setOnClickListener {
            val userInput = inputEditText.text.toString()
            lifecycleScope.launch {
                robotSystem.processUserInputWithStreaming(userInput)
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        robotSystem.close()
    }
}
```

## 🔧 설정 옵션

### 런타임 설정 변경

```kotlin
// 설정 즉시 적용
robot.updateSettings(
    useReasoning = true,     // Reasoning 모드 전환
    fastMode = false,        // 일반 속도 모드
    debugMode = true         // 디버그 로그 활성화
)
```

### 디버그 모드

```kotlin
// 디버그 모드 활성화 시 로그 출력 예시
🔍 맥락 분석: 'CT 어디야?'
🔍 이전 질문: ''
🔍 맥락 연관성: false
💬 일반 질문 감지! Non-reasoning 모드로 자동 전환
🔧 실행할 함수: query_facility({"facility": "CT"})
```

## 📊 성능 최적화

### 메모리 관리

- **대화 히스토리**: 최대 24개 엔트리 자동 관리
- **토큰 캐시**: 스트리밍 완료 후 자동 정리
- **리소스 정리**: `close()` 호출로 모든 리소스 해제

### 응답 속도 최적화

```kotlin
// 빠른 모드 (권장)
.enableFastMode(true)
// - 최대 토큰: 1024 (vs 2048)
// - 스트리밍 지연: 1ms (vs 2ms)
// - EXAONE 4.0 공식 권장값 유지
```

## 🧪 테스트 시나리오

### 1. 시설 찾기 테스트

```kotlin
val testCases = listOf(
    "CT 어디야?" to "CT는 왼쪽 중앙 영상의학과에 있어요.",
    "엑스레이 위치는?" to "X-ray는 왼쪽 상단 영상의학과에 있어요.",
    "초음파실 찾아줘" to "초음파는 왼쪽 하단 영상의학과에 있어요."
)
```

### 2. 맥락 연속 대화 테스트

```kotlin
// 시나리오: CT 관련 연속 질문
"CT 검사는 어떻게 받나요?"
"그곳이 어디에 있나요?"  // 맥락 인식: CT 위치 질문으로 해석
"거기로 안내해주세요"    // 맥락 인식: CT로 안내 요청으로 해석
```

### 3. 복잡한 질문 테스트

```kotlin
// Reasoning 모드 자동 전환되는 질문들
"왜 CT와 X-ray의 원리가 다른가요?"
"암센터에서 가장 중요한 검사는 무엇인가요?"
"만약 뇌종양 검사를 받는다면 어떤 절차를 거쳐야 하나요?"
```

## 🔄 Python 대비 개선사항

| 기능 | Python 버전 | Android 버전 | 개선점 |
|------|-------------|--------------|--------|
| 스트리밍 | TextIteratorStreamer | AndroidStreamer + Flow | UI 친화적, 메모리 효율적 |
| 함수 호출 | JSON 파싱 | 타입 안전 파싱 | 런타임 안정성 향상 |
| 설정 관리 | 전역 변수 | Builder 패턴 | 객체지향적, 확장 가능 |
| 리소스 관리 | Manual | 라이프사이클 연동 | 메모리 누수 방지 |
| UI 업데이트 | Print 출력 | Flow + LiveData | 반응형 UI |

## 🚨 주의사항

### 1. 모델 파일 확인

```
app/src/main/assets/
├── exaone_model.tflite     ✅ 필수
├── model_config.json       ✅ 필수  
└── tokenizer_info/         ✅ 필수
    ├── chat_template.jinja
    ├── merges.txt
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json
```

### 2. 의존성 추가

```kotlin
// build.gradle.kts (Module: app)
dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.6.2")
    implementation("org.tensorflow:tensorflow-lite:2.13.0")
}
```

### 3. 권한 설정

```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

## 📚 추가 리소스

### 사용 예제 실행

```kotlin
// 콘솔에서 테스트 실행
RobotUsageExamples.simpleUsageExample(context)
RobotUsageExamples.advancedUsageExample(context)  
RobotUsageExamples.scenarioTests(context)
```

### 커스터마이징

1. **새로운 함수 추가**: `RobotFunctions.kt`에 함수 추가
2. **프롬프트 수정**: `EXAONEModel.kt`의 `buildSystemPrompt` 수정
3. **스트리밍 설정**: `AndroidStreamer.kt`의 지연시간 조정

## 🎉 완성된 기능 목록

✅ **핵심 시스템**
- [x] EXAONEModel 확장 (대화 히스토리, 맥락 인식)
- [x] RobotFunctions (시설 조회, 네비게이션, 일반 대화)
- [x] AndroidStreamer (실시간 스트리밍)
- [x] RobotSystem (통합 시스템)

✅ **고급 기능**
- [x] 자동 모드 전환 (Reasoning ↔ Non-reasoning)
- [x] 맥락 인식 및 연속 대화
- [x] 함수 호출 (Tool calling) 시스템
- [x] 실시간 토큰 스트리밍
- [x] Flow 기반 UI 업데이트

✅ **사용성**
- [x] Builder 패턴으로 간편한 설정
- [x] 확장 함수로 편리한 사용법
- [x] 완전한 액티비티 예제
- [x] 종합적인 테스트 시나리오

이제 Python의 고도화된 프롬프트 엔지니어링 시스템이 안드로이드에서 완벽하게 동작합니다! 🚀 
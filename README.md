[![표지](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%91%9C%EC%A7%80.png?raw=true)](https://docs.google.com/presentation/d/1oV-mY3bv-sJK2Gfmo4MlbgOfs4dTnHREZe68CmfjYtQ/edit?slide=id.g3735f5e2a27_0_0#slide=id.g3735f5e2a27_0_0)
[ㄴ 클릭시 PPT 이동](https://docs.google.com/presentation/d/1oV-mY3bv-sJK2Gfmo4MlbgOfs4dTnHREZe68CmfjYtQ/edit?slide=id.g3735f5e2a27_0_0#slide=id.g3735f5e2a27_0_0)

## 주제 : 대형병원 안내 로봇 시스템 [ROS2 프로젝트]
![예시 이미지](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%98%88%EC%8B%9C%20%EC%9D%B4%EB%AF%B8%EC%A7%80.png?raw=true)

### 프로젝트 기간
**25.7.09 ~ 25.8.13 약 5주간 진행** <br/>
**Sprint1** : 기획 및 요구사항 정의 <br/>
**Sprint2** : 설계 및 기술조사  <br/>
**Sprint3** : 구현 및 1차 연동테스트 <br/>
**Sprint4** : 구현 및 2차 연동테스트 <br/>
**Sprint5** : 구현 및 3차 연동테스트 <br/>
**Sprint6** : 안정화 및 발표

### 활용 기술
|분류|기술|
|---|---|
|**개발환경**|<img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white"/> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=Ubuntu&logoColor=white"/> <img src="https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"/> |
|**언어**|<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white"/> 
|**UI**|<img src="https://img.shields.io/badge/C++QT-28c745?style=for-the-badge&logo=C++QT&logoColor=white"/> <img src="https://img.shields.io/badge/Android Studio-1AC?style=for-the-badge&logo=C++QT&logoColor=white"/>|
|**DBMS**| <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>|
|**딥러닝**| <img src="https://img.shields.io/badge/YOLOv8-FFBB00?style=for-the-badge&logo=YOLO&logoColor=white" alt="YOLOv8"/> <img src="https://img.shields.io/badge/LSTM-006400?style=for-the-badge&logo=OpenAI&logoColor=white" alt="LSTM"/> <img src="https://img.shields.io/badge/ST--GCN-1E90FF?style=for-the-badge&logo=GraphQL&logoColor=white" alt="ST-GCN"/>|
|**자율주행**| <img src="https://img.shields.io/badge/ROS2-225?style=for-the-badge&logo=ROS2&logoColor=white" alt="ROS2"/> <img src="https://img.shields.io/badge/Slam&Nav-595?style=for-the-badge&logo=Slam&Nav&logoColor=white" alt="ST-GCN"/>|
|**협업**|<img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/> <img src="https://img.shields.io/badge/SLACK-4A154B?style=for-the-badge&logo=slack&logoColor=white"/> <img src="https://img.shields.io/badge/Confluence-172B4D?style=for-the-badge&logo=confluence&logoColor=white"/> <img src="https://img.shields.io/badge/JIRA-0052CC?style=for-the-badge&logo=jira&logoColor=white"/> |



### 목차
- [01. 프로젝트 소개](#01-프로젝트-소개)
- [02. 프로젝트 설계](#02-프로젝트-설계)
- [03. 프로젝트 기능 구현](#03-프로젝트-기능-구현)
- [04. 프로젝트 기술 소개](#04-프로젝트-기술-소개)
- [05. 트러블 슈팅](#05-트러블-슈팅)
- [00. 에필로그](#00-에필로그)
- [마무리 : 소감](#마무리-소감)

# 01. 프로젝트 소개
### 주제 선정 배경
![주제 선정 배경](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%A3%BC%EC%A0%9C%20%EC%84%A0%EC%A0%95%20%EB%B0%B0%EA%B2%BD.png?raw=true)

대형병원 안내 로봇을 주제로 선정한 이유 <br/>
- 복잡한 대형병원 <br/>
- 초고령 사회 & 인력 감소 <br/>
- 자동화 수요 증가 <br/>
- 업무 효율화  <br/>
  
위의 이유로 **대형병원 로봇 안내 시스탬 수요**가 증가하고 있는 추세입니다.

그렇기에 이를 주제로 선정하였습니다.

### 사용자 요구사항 (User Requirements)
| 번호 | 설명 |
|------|----------------|
| 1 | 환자는 필요할 때 로봇을 쉽게 찾을 수 있어야 하며 로봇은 환자가 자주 다니는 구역에서 자율적으로 이동하거나 대기한다|
| 2 | 환자가 서비스를 이용하는 동안 스크린 터치나 음성으로 상호작용한다. |
| 3 | 환자가 열화상 카메라에 가까이 가면 체온 측정을 해주고 체온을 알려준다. |
| 4 | 환자의 손이 감지되면 손소독을 해준다. |
| 5 | 초진 환자의 경우, 접수처로 안내해준다.|
| 6 | 재진 환자의 경우, 본인 확인을 통해 진료과 접수를 도와준다. |
| 7 | 환자의 당일 진료 일정과 방문 경로에 대한 안내를 해준다.|
| 8 | 환자의 현재 몸 상태나 불편 사항을 기록하고 관련 진료과를 접수해준다. |
| 9 | 환자나 보호자의 병원 내 목적지까지 안내와 동행을 해준다. |
| 10 | 환자나 보호자가 이동하는 동안 짐을 수납해준다. |
| 11 | 환자의 진료 이후, 수납처까지 안내해준다. |
| 12 | 야간에도 병원 내 이상 상황을 탐지할 수 있도록 순찰한다. |

[ **요약** ] <br/>
![사용자 요구사항](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%82%AC%EC%9A%A9%EC%9E%90%20%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD.png?raw=true)

사용자 요구사항을 크게 3가지로 요약하면, <br/>
'사용자 인터페이스 / 관리자 GUI / 자율주행' 이렇게 3가지로 요약할 수 있습니다.



# 02. 프로젝트 설계
## System Requirements
| SR_ID | Category | Name | Description | Priority |
|-------|----------|------|-------------|----------|
| SR_01 | 사용자 접수 및 인증 | 자동 복귀 기능 | 행동이 끝나면 지정된 위치(입구, 진료실 등)로 로봇이 자동 복귀하여 대기 상태를 유지합니다. | R |
| SR_02 | 사용자 접수 및 인증 | 자동 충전 기능 | 지정된 위치에서 대기 상태일 경우 자동 충전이 활성화 됩니다.  | R |
| SR_03 | 사용자 접수 및 인증 | 대기 상태 기능 | 대기 상태일 때 자율주행으로 병원 내 지정된 구역을 순회하며 돌아다니거나 대기소에서 대기합니다. | R |
| SR_04 | 사용자 접수 및 인증 | 사용자 접근 감지 기능 | 사람이 일정 거리 내에 접근하면 로봇이 대기 상태에서 사용자 이용 상태로 변경되며 사용자 환경이 활성화 됩니다. | R |
| SR_05 | 사용자 접수 및 인증| 사용자 인터랙션 기능 | 사용자의 명시적 작동 요청을 인식하여 로봇을 활성화합니다. (음성 호출 / 스크린 터치 / 손동작 호출) | R |
| SR_06 | 사용자 접수 및 인증 | 본인 확인 기능  | 본인 인증 절차를 통해 환자를 정확히 식별하고, 각 사용자를 개별적으로 인식합니다. (환자 카드(RFID) / 이름+주민번호 / 환자 번호 )| R |
| SR_07 | 내부 이동 지원 | 진료/검사 일정 확인 기능 | 사용자 인식 후 병원 EMR 시스템 연동을 통해 일정을 안내합니다. | R |
| SR_08 | 내부 이동 지원 | 길 안내 기능 | 현재 위치에서 진료과, 검사실, 부대시설 등 원하는 목적지까지 경로를 안내합니다. | R |
| SR_09 | 내부 이동 지원 | 길 동행 기능  | 진료과, 검사실, 부대시설 등 목적지까지 동행합니다 (다른 층 이동 시 엘리베이터 탑승을 모의 시나리오) | R |
| SR_10 | 사용자 인터페이스 | 표정 출력 기능 | 대기상태 및 동행 시 상황에 맞는 표정을 출력합니다. | O |
| SR_11 | 사용자 인터페이스 | 대화 기능 | 챗봇 기능을 제공합니다. | R |
| SR_12 | 사용자 인터페이스 | 다국어 지원 기능 | 외국인 사용자를 위한 언어를 제공합니다. (영어) | O |
| SR_13 | 진료 정보 연동 (EMR) | 진단 목록 조회 | 환자의 주요 진단 정보를 로봇이 조회하여, 진단을 기반으로 진료실 또는 검사실 경로 안내합니다 | R |
| SR_14 | 진료 정보 연동 (EMR) | 검사 결과 조회 | 검사 결과 데이터를 바탕으로 로봇이 검사 완료 여부를 확인합니다. 검사 후 다음 진료 장소까지 최적의 경로를 안내하거나 검사 결과에 따라 이동 우선순위를 조정합니다. | R |
| SR_15 | 로봇 운영 및 유지 관리 | 장애물 감지 및 회피 기능 | 이동 중 전방/측면의 사람이나 장애물을 감지하여 자동으로 속도 조절 또는 경로 우회합니다. | R |
| SR_16 | 로봇 운영 및 유지 관리 | 로봇 위치 추적 기능 | 실시간으로 로봇의 위치를 지도 상에서 확인합니다. | R |
| SR_17 | 로봇 운영 및 유지 관리 | 상태 알림 기능 | 배터리 부족, 네트워크 장애 발생 등 로봇의 상태를 관리시스템에 알림 전송합니다. | R |
| SR_18 | 로봇 운영 및 유지 관리 | 배터리/네트워크 모니터링 기능 | 배터리 잔량, 네트워크 상태 실시간 확인합니다. | R |
| SR_19 | 로봇 운영 및 유지 관리 | 원격 제어 기능 | 로봇 정지, 재시작, 복귀 등 로봇 제어가 가능합니다. | R |
| SR_20 | 로봇 운영 및 유지 관리 | 오류 로그 기록 기능 | 장애 발생 시 로그 저장, 원인 분석을 지원합니다. | R |
| SR_21 | 데이터 통계 및 분석 | 이동 기록 기능 | 로봇의 일별 이동 경로 저장 및 분석을 지원합니다. | O |
| SR_22 | 데이터 통계 및 분석 | 응대 수 통계 | 안내, 동행 등 기능별 응대한 사용자 수 기록을 지원합니다. | O |
| SR_23 | 데이터 통계 및 분석 | 사용자 만족도 수집 | 안내 후 만족도 조사 제공, 간단한 평점 또는 선택형 질문을 통해 만족도 통계를 수집합니다. | O |

![System Requirements](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/System%20Requirements.png?raw=true)

기능 리스트를 요약하면 크게 3가지로 나눌 수 있습니다. <br/>
자율주행 및 길안내 / 사용자 인터페이스 / 관리자 GUI

### 서비스 흐름 : 사용자 호출
![서비스 흐름1](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%84%9C%EB%B9%84%EC%8A%A4%20%ED%9D%90%EB%A6%84%20%EC%82%AC%EC%9A%A9%EC%9E%90%20%ED%98%B8%EC%B6%9C.png?raw=true)

### 서비스 흐름 : 길 동행
![서비스 흐름2](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%84%9C%EB%B9%84%EC%8A%A4%20%ED%9D%90%EB%A6%84%20%EA%B8%B8%20%EB%8F%99%ED%96%89.png?raw=true)

### System Architecture
![System Architecture](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/System%20Architecture.png?raw=true)

### 시퀀스 다이어그램

<details>
<summary>SC-01 : 사용자 호출 시나리오 [클릭] </summary>
SC-01-01 스크린 터치로 로봇을 호출할 경우

![SC-01-01](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-01-01.png?raw=true)

SC-01-02 손동작으로 호출할 경우

![SC-01-02](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-01-02.png?raw=true)

SC-01-03  음성으로 호출할 경우

![SC-01-03](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-01-03.png?raw=true)

</details>

<details>
<summary> SC-02 : 사용자 인증 및 접수 [클릭] </summary>

SC-02-01 초진/미접수 환자 – 접수 절차

![SC-02-01](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-02-01.png?raw=true)


SC-02-02 이미 접수 완료된 환자 – 다음 일정/경로안내 위주

![SC-02-02](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-02-02.png?raw=true)

</details>

<details>
<summary> SC-03 : 길 동행 시나리오 [클릭] </summary>

![SC-03_1](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-03_1.png?raw=true)
![SC-03_2](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-03_2.png?raw=true)

</details>

<details>
<summary> SC-04 : 자동 복귀 시나리오 [클릭] </summary>

![SC-04](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/sc-04.png?raw=true)

</details>

### ERD
![ERD1](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/erd1.png?raw=true)
![ERD2](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/erd2.png?raw=true)
![ERD3](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/erd3.png?raw=true)

### Interface Specification
#### status code 
| 요청 결과                           | 코드 |
|-------------------------------------|------|
| 정상 요청, 데이터 응답 성공         | 200  |
| 정상 요청, 정보없음 or 응답 실패    | 401  |
| 잘못된 요청                        | 404  |
| 서버 내부 오류                     | 500  |

<details>
<summary>User Gui <-> Central Server [클릭] </summary>

| Interface ID | Function/Description         | Sender      | Receiver    | Endpoint                   | Method | Request Data                                              | Response Data / Status Code / Description                |
|--------------|-----------------------------|-------------|-------------|----------------------------|--------|----------------------------------------------------------|----------------------------------------------------------|
| IF-01        | 관리자 사용중 블락           | central     | GUI         | /alert_occupied            | POST   | {"robot_id": "3"}                                        | status code                                              |
| IF-02        | 사용 가능한 상태 알림        | central     | GUI         | /alert_idle                | POST   | {"robot_id": "3"}                                        | status code                                              |
| IF-03        | 길안내 완료                  | central     | GUI         | /navigating_complete       | POST   | {"robot_id": "3"}                                        |                                                          |
| IF-04        | 30초 동안 터치 없을 경우     | GUI         | central     | /alert_timeout             | POST   | {"robot_id": "3"}                                        | status code                                              |
| IF-05        | 호출 명령 (음성)             | GUI         | central     | /call_with_voice           | POST   | {"robot_id": "3"}                                        | status code                                              |
| IF-06        | 호출 명령 (스크린)           | GUI         | central     | /call_with_screen          | POST   | {"robot_id": "3"}                                        | status code                                              |
| IF-07        | 본인확인 (주민번호)          | GUI         | central     | /auth/ssn                  | POST   | {"robot_id":"3", "ssn":"000000-0000000"}                 | {"name":"임영웅", ...} or {"error":"Reservation not found"}|
| IF-08        | 본인확인 (회원번호)          | GUI         | central     | /auth/patient_id           | POST   | {"robot_id":"3", "patient_id":"00000000"}                | {"name":"임영웅", ...}                                   |
| IF-09        | 본인확인 (RFID 카드)         | GUI         | central     | /auth/rfid                 | POST   | {"robot_id":"3", "rfid":"A1B2C3D4E5"}                    | {"name":"임영웅", ...}                                   |
| IF-10        | 본인 인증 후 길 안내         | GUI         | central     | /auth/direction            | POST   | {"patient_id":"00000000", "robot_id":3, "station_id":5}  | status_code                                              |
| IF-11        | 본인 인증 없이 길 안내       | GUI         | central     | /without_auth/direction    | POST   | {"robot_id":3, "station_id":5}                           | status_code                                              |
| IF-12        | 본인인증 후 사용완료 복귀명령| GUI         | central     | /auth/robot_return         | POST   | {"patient_id":"00000000","robot_id":"3"}                 | status_code                                              |
| IF-13        | 본인인증 없이 사용완료 복귀명령| GUI       | central     | /without_auth/robot_return | POST   | {"robot_id":"3"}                                         |                                                          |
| IF-14        | 로봇 일시정지 명령           | GUI         | central     | /pause_request             | POST   | {"robot_id":"3"}                                         |                                                          |
| IF-15        | 로봇 길안내 재개             | GUI         | central     | /restart_navigation        | POST   | {"robot_id":"3"}                                         |                                                          |
| IF-16        | pause 중 로봇 작업 종료 요청 | GUI         | central     | /stop_navigating           | POST   | {"robot_id":"3"}                                         |                                                          |
| IF-17        | 맵 로봇 위치                 | GUI         | central     | /get/robot_location        | POST   | {"robot_id":"3"}                                         | {"x":5.0, "y":-1.0, "yaw":-80} (1초에 1번 요청)          |
</details>

<details>
<summary>Admin Gui <-> Central Server [클릭] </summary>

| Interface ID | Function/Description         | Sender      | Receiver    | Endpoint                   | Method | Request Data                                              | Response Data / Status Code / Description                |
|--------------|-----------------------------|-------------|-------------|----------------------------|--------|----------------------------------------------------------|----------------------------------------------------------|
| IF-01        | 환자 사용중 블락             | central     | GUI         | /alert_occupied            | POST   | {"robot_id": "3"}                                        | status code                                              |
| IF-02        | 사용 가능한 상태 알림        | central     | GUI         | /alert_idle                | POST   | {"robot_id": "3"}                                        | status code                                              |
| IF-03        | 로그인                       | GUI         | central     | /auth/login                | POST   | {"admin_id":"00000000ssn", "password":"0000"}            | status code                                              |
| IF-04        | 세부 정보                    | GUI         | central     | /auth/detail               | POST   | {"admin_id":"00000000"}                                  | {"name":"임영웅","email":"hero@mail.com","hospital_name":"서울아산병원"} |
| IF-05        | 맵 로봇 위치                 | GUI         | central     | /get/robot_location        | POST   | {"robot_id":3, "admin_id":"admin1"}                      | {"x":5.0, "y":-1.0, "yaw":-0.532151} (1초에 1번 요청)    |
| IF-06        | 로봇 상태                    | GUI         | central     | /get/robot_status          | POST   | {"robot_id":3, "admin_id":"admin1"}                      | {"status":"navigating", "orig":0, "dest":3, "battery":70, "network":4} |
| IF-07        | 이용중인 환자 정보           | GUI         | central     | /get/patient_info          | POST   | {"robot_id":3, "admin_id":"admin1"}                      | {"patient_id":"00000000", "phone":"010-1111-1111", "rfid":"33F7ADEC", "name":"김환자"} |
| IF-08        | 원격 제어 요청               | GUI         | central     | /control_by_admin          | POST   | {"robot_id":3, "admin_id":"admin1"}                      |                                                          |
| IF-09        | 원격 제어 취소               | GUI         | central     | /return_command            | POST   | {"robot_id":3, "admin_id":"admin1"}                      | status code (대기장소로 이동)                            |
| IF-10        | 수동제어 요청                | GUI         | central     | /teleop_request            | POST   | {"robot_id":3, "admin_id":"admin1"}                      | status code                                              |
| IF-11        | 수동제어 명령 완수           | GUI         | central     | /teleop_complete           | POST   | {"robot_id":3, "admin_id":"admin1"}                      | status code                                              |
| IF-12        | teleop 이동 명령             | GUI         | central     | /command/move_teleop       | POST   | {"robot_id":3, "teleop_key":1, "admin_id":"admin1"}       | status code (123=uio, 456=jkl, 789=m,.)                  |
| IF-13        | 목적지 이동 명령             | GUI         | central     | /command/move_dest         | POST   | {"robot_id":3, "dest":0, "admin_id":"admin1"}            | status code                                              |
| IF-14        | 길안내 취소                  | GUI         | central     | /cancel_navigating         | POST   | {"robot_id":3, "admin_id":"admin1"}                      | status code                                              |
| IF-15        | 로봇 로그                    | GUI         | central     | /get/log_data              | POST   | {"period":"today", "start_date":"YYYY-MM-DD", "end_date":"YYYY-MM-DD", "admin_id":"admin1"} | [ { "patient_id":"00000000", "orig":0, "dest":3, "datetime":"YYYY-MM-DD HH:MM:SS" }, ... ] |

</details>

<details>
<summary>Robot Controller <-> Central Server [클릭] </summary>

| Interface ID | Function/Description         | Type        | Topic Name           | Direction         | Message Type             | Message Data / Description                                 |
|--------------|-----------------------------|-------------|----------------------|-------------------|--------------------------|------------------------------------------------------------|
| IF-01        | 로봇 목적지 전송             | ROS2 Topic  | /navigation_command  | Central → Robot   | std_msgs/String          | waypoint_name: 특정 웨이포인트로 주행<br>go_start: 시작점 복귀<br>stop/cancel: 주행 정지<br>status: 현재 주행 상태<br>list: 호출 가능한 waypoint 목록 |
| IF-02        | 로봇의 현재 위치             | ROS2 Topic  | /pose                | Robot → Central   | geometry_msgs/Pose       | 로봇의 현재 위치 및 각도 반환 (/amcl_pose 구독 결과)        |
| IF-03        | 로봇의 주행 시작점           | ROS2 Topic  | /start_point         | Robot → Central   | std_msgs/String          | 로봇의 주행 시작점 반환                                    |
| IF-04        | 로봇의 주행 목적지           | ROS2 Topic  | /target              | Robot → Central   | std_msgs/String          | 로봇의 주행 목적지 반환                                    |
| IF-05        | 로봇의 네트워크 상태         | ROS2 Topic  | /net_level           | Robot → Central   | std_msgs/Int32           | 로봇의 네트워크 상태(신호 세기) 반환                       |
| IF-06        | 로봇의 배터리 잔량           | ROS2 Topic  | /battery             | Robot → Central   | std_msgs/Int32           | 로봇의 배터리 잔량 반환                                    |
| IF-07        | 로봇 주행 상태               | ROS2 Topic  | /nav_status          | Robot → Central   | std_msgs/String          | 현재 로봇의 주행 상태 반환                                 |
| IF-08        | 실시간 추적 목표 전송        | ROS2 Topic  | /trackingGoal        | Central → Robot   | geometry_msgs/Point      | AI서버 트래킹 결과를 좌표 변환 후 로봇에 전달, 동적 목표 제공|
| IF-09        | 장애물 정보 연동             | ROS2 Topic  | /detected_obstacle   | Robot → Central   | float32 x, y, yaw        | 라이다로 감지한 장애물 위치 정보 반환                      |

</details>

<details>
<summary>AI Server <-> Central Server [클릭] </summary>

| Interface ID | Function/Description         | Sender      | Receiver    | Endpoint                | Method | Request Data                                                                 | Response Data         | Description                                                                                      |
|--------------|-----------------------------|-------------|-------------|-------------------------|--------|------------------------------------------------------------------------------|-----------------------|--------------------------------------------------------------------------------------------------|
| IF-01        | 장애물 감지 정보 송신        | Central     | AI          | /obstacle/detected      | POST   | { "robot_id": 3, "left_angle":"10.0", "right_angle":"30.0", "timestamp": 1722601200 } | { "status_code": 200 } | 로봇 라이다에서 장애물 감지 시, 중앙서버가 라이다 좌표(또는 상대좌표) 포함해 AI서버에 전송           |
| IF-02        | 추적 시작 명령               | Central     | AI          | /start_tracking         | POST   | { "robot_id": 3 }                                                            | { "status_code": 200 } | patient_navigating이나 unknown_navigating action 중에 AI서버에 트래킹 시작 명령 송신                |
| IF-03        | 손동작(come) 인식 이벤트 송신| AI          | Central     | /gesture/come           | POST   | { "robot_id": 3, "left_angle":"10.0", "right_angle":"30.0", "timestamp": 1722601202 } | { "status_code": 200 } | AI서버가 카메라에서 손동작(come) 인식 시 angle 송신, 중앙서버는 좌표 변환 후 로봇에 목표 위치 전송  |
| IF-04        | 길안내 중 사람 사라짐         | AI          | Central     | /user_disappear         | POST   | { "robot_id": 3 }                                                             | { "status_code": 200 } | 추적하던 person id가 사라지면 송신, 10초간 user_appear 없으면 중앙서버가 return_command 송신        |
| IF-05        | 사라졌던 사람 다시 나타남     | AI          | Central     | /user_appear            | POST   | { "robot_id": 3 }                                                             | { "status_code": 200 } | 3초 전에 사라졌던 사람이 다시 나타나면 송신, 중앙은 restart_navigating, AI서버는 계속 트래킹         |

</details>

<details>
<summary>Admin Gui <-> AI Server [클릭] </summary>

### HTTP
| Interface ID | Function/Description         | Sender      | Receiver    | Endpoint           | Method | Request Data                                 | Response Data | Description                                                                                  |
|--------------|-----------------------------|-------------|-------------|--------------------|--------|----------------------------------------------|---------------|----------------------------------------------------------------------------------------------|
| IF-01        | 카메라 전후면 변경           | GUI/Central | Robot       | /change/camera     | POST   | {"robot_id":3, "camera":"front"}             | status_code   | UDP로 송신 중인 카메라 이미지가 전면이면 후면, 후면이면 전면 이미지로 전환하여 보내야함.      |

### UDP
| Interface ID | Function/Description         | IP/Port             | Transmission Cycle | Header                                                                 | Payload                | Description                                                                                  |
|--------------|-----------------------------|---------------------|-------------------|-----------------------------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------|
| IF-01        | 실시간 카메라 이미지 전송    | 192.168.0.27:8888   | 30fps 등 설정 주기 | 1byte: Start(0xAB)<br>1byte: 카메라 타입(0x00=front/0x01=rear)<br>4byte: 시퀀스번호<br>4byte: 타임스탬프 | JPEG 등 바이너리 이미지 | 설정된 주기로 연속 송신, 헤더+이미지 데이터로 구성                                             |

</details>

### Hardware
![hardware](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/hardware.png?raw=true)

### 환자 Gui
![환자 Gui 접수 화면](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90GUI%20%EC%A0%91%EC%88%98.png?raw=true)
![환자 Gui 길안내 화면](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90Gui%20%EA%B8%B8%EC%95%88%EB%82%B4.png?raw=true)
![환자 Gui 음성안내 화면](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90Gui%20%EC%9D%8C%EC%84%B1%EC%95%88%EB%82%B4.png?raw=true)

### 관리자 Gui
![관리자 Gui 로그인 화면](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EA%B4%80%EB%A6%AC%EC%9E%90Gui%20%EB%A1%9C%EA%B7%B8%EC%9D%B8.png?raw=true)
![관리자 Gui 대쉬보드 화면](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EA%B4%80%EB%A6%AC%EC%9E%90Gui%20%EB%8C%80%EC%89%AC%EB%B3%B4%EB%93%9C.png?raw=true)
![관리자 Gui 로그 화면](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EA%B4%80%EB%A6%AC%EC%9E%90Gui%20%EB%A1%9C%EA%B7%B8.png?raw=true)

# 03. 프로젝트 기능 구현
### 환자(User) GUI - 예약 접수 / RFID 회원 인식 
![환자 RFID 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90%20GUI%20RFID.gif?raw=true)

### 환자(User) GUI - 길안내
![환자 길안내 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90%20GUI%20%EA%B8%B8%EC%95%88%EB%82%B4.gif?raw=true)

### 환자(User) GUI - LLM 대화 
![환자 LLM 대화 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90%20GUI%20LLM%EB%8C%80%ED%99%94.gif?raw=true)

### 환자(User) GUI - LLM 부가 기능(시설 정보, 위치 조회 등)
![환자 LLM 부가기능 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90%20GUI%20LLM%20%EB%B6%80%EA%B0%80%EA%B8%B0%EB%8A%A5.gif?raw=true)

### 환자(User) GUI - 네비게이션 LLM 에이전트 
![환자 LLM 네비게이션 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90%20GUI%20LLM%20%EB%84%A4%EB%B9%84%EA%B2%8C%EC%9D%B4%EC%85%98.gif?raw=true)

### 환자(User) GUI - 자동 복귀
![환자 로비복귀 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%99%98%EC%9E%90%20GUI%20%EB%A1%9C%EB%B9%84%20%EB%B3%B5%EA%B7%80.gif?raw=true)

### 네비게이션 - 좁은 길 통과 
![좁은길 통과 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%A2%81%EC%9D%80%EB%AC%B8%20%ED%86%B5%EA%B3%BC.gif?raw=true)

### AI 서버 - 손동작 감지
![손동작 감지 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/AI%20%EC%84%9C%EB%B2%84%20%EC%86%90%EB%8F%99%EC%9E%91%20%EA%B0%90%EC%A7%80.gif?raw=true)

### AI 서버 - 객체 트래킹
![객체 트레킹 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/AI%20%EC%84%9C%EB%B2%84%20%ED%8A%B8%EB%9E%98%ED%82%B9.gif?raw=true)

### 관리자(Admin) GUI - 로그인 기능
![관리자GUI 로그인 구현](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EA%B4%80%EB%A6%AC%EC%9E%90Gui%20%EB%A1%9C%EA%B7%B8%EC%9D%B8%20%EA%B5%AC%ED%98%84.png?raw=true)

### 관리자(Admin) GUI - 대쉬보드 [맵 로봇 위치 / 카메라 전후면 전환 ] 
![관리자 대쉬보드 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EA%B4%80%EB%A6%AC%EC%9E%90%20GUI%20%EC%A0%84%ED%9B%84%EB%A9%B4%20%EC%B9%B4%EB%A9%94%EB%9D%BC.gif?raw=true)

### 관리자(Admin) GUI - 네비게이션
![관리자 원격제어 네비게이션 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EA%B4%80%EB%A6%AC%EC%9E%90%20GUI%20%EB%84%A4%EB%B9%84%EA%B2%8C%EC%9D%B4%ED%8C%85.gif?raw=true)

### 관리자(Admin) GUI - 대쉬보드 : 수동제어
![관리자 수동제어 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%88%98%EB%8F%99%EC%A0%9C%EC%96%B4%20GUI%20%EB%A1%9C%EB%B4%87.gif?raw=true)

### 관리자(Admin) GUI - 로그 [로봇 사용 로그 테이블 / 출발지 통계 차트 / 목적지 통계 차트] 
![관리자 로그 영상](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EA%B4%80%EB%A6%AC%EC%9E%90%20GUi%20%EB%A1%9C%EA%B7%B8.gif?raw=true)

# 04. 프로젝트 기술 소개
### 자율주행 및 장애물 회피 [Dijkstra 알고리즘]
![Dijkstra](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/dijkstra.png?raw=true)

### 자율주행 및 장애물 회피 [Vector Pursuit]
![Vector Pursuit](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/vector%20pursuit.png?raw=true)

### 사람 재매칭 & 손동작 호출 [Vision / Deep Learning]
#### 사람 재매칭
![사람 재매칭1](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%82%AC%EB%9E%8C%20%EC%9E%AC%EB%A7%A4%EC%B9%AD1.png?raw=true)
![사람 재매칭2](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%82%AC%EB%9E%8C%20%EC%9E%AC%EB%A7%A4%EC%B9%AD2.png?raw=true)

#### 손동작 호출
![손동작 호출](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%86%90%EB%8F%99%EC%9E%91%20%ED%98%B8%EC%B6%9C.png?raw=true)

### 음성 인식 및 음성 안내 [LLM 에이전트 / Deep Learning]
![LLM 에이전트](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/llm%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8.png?raw=true)


# 05. 트러블 슈팅
![트러블 슈팅1](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%8A%B8%EB%9F%AC%EB%B8%94%20%EC%8A%88%ED%8C%851.png?raw=true)
![트러블 슈팅2](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%8A%B8%EB%9F%AC%EB%B8%94%20%EC%8A%88%ED%8C%852.png?raw=true)
![트러블 슈팅3](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%8A%B8%EB%9F%AC%EB%B8%94%20%EC%8A%88%ED%8C%853.png?raw=true)
![트러블 슈팅4](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%8A%B8%EB%9F%AC%EB%B8%94%20%EC%8A%88%ED%8C%854.png?raw=true)
![트러블 슈팅5](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%8A%B8%EB%9F%AC%EB%B8%94%20%EC%8A%88%ED%8C%855.png?raw=true)

# 00. 에필로그
### 팀소개
![팀소개](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%8C%80%EC%86%8C%EA%B0%9C.jpg?raw=true)

| 이름 | 주요 역할 | 포폴 사이트 |
|:---:|---|---|
| 김채연 (팀장) | 설계 문서 / AI Server / 비젼 딥러닝  | [클릭](https://ckimzll.github.io/portfolio_website/) |
| 김범진 (팀원) | 환자 GUI / HW 제작 / ROS 통신 | [클릭](https://jbjj0708.github.io/portfolio/) | 
| 구민제 (팀원) | LLM 딥러닝 / Central Server / 자율주행 알고리즘 | [클릭](https://koo4802.github.io/portfolio/) |
| 최원호 (팀원) | 관리자 GUI / 시뮬레이션 맵 세팅 / HW 제작 / 3D 프린팅 / 디자인 / HTTP 통신 | [클릭](https://wonho9188.github.io/portfolio/) |
| 김태호 (팀원) | 자율주행 알고리즘 / 장애물 회피 | [클릭](https://kth-amg.github.io/portfolio/) |

### 프로젝트 관리
#### 컨플루언스 - 문서 관리
![컨플루언스](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%BB%A8%ED%94%8C%EB%A3%A8%EC%96%B8%EC%8A%A4.png?raw=true)

#### 지라 - 일정 관리
![지라](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%EC%A7%80%EB%9D%BC.png?raw=true)

#### HW 제작 - 3D 프린팅 + 후가공(퍼티 / 샌딩 / 시트지)
![하드웨어 제작1](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%911.png?raw=true)
![하드웨어 제작2](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%912.png?raw=true)
![하드웨어 제작3](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%913.png?raw=true)
![하드웨어 제작4](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%914.png?raw=true)
![하드웨어 제작5](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%915.png?raw=true)
![하드웨어 제작6](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%916.png?raw=true)
![하드웨어 제작7](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%917.png?raw=true)
![하드웨어 제작8](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%918.png?raw=true)
![하드웨어 제작9](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%919.png?raw=true)
![하드웨어 제작10](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%9110.png?raw=true)
![하드웨어 제작11](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%9111.png?raw=true)
![하드웨어 제작12](https://github.com/addinedu-ros-9th/ros-repo-4/blob/dev/readme_image/%ED%95%98%EB%93%9C%EC%9B%A8%EC%96%B4%20%EC%A0%9C%EC%9E%9112.png?raw=true)


# 마무리
## 소감
| 이름 | 소감 |
|:---:|---|
| 김채연 | | 
| 김범진 | |
| 구민제 | 중앙 서버, 주행 파트 글로벌 플래너, LLM 등 프로젝트 도중에도 맡은 임무가 계속해서 바뀌고 추가되었지만 그 덕에 많은 것을 배울 수 있었습니다. 결국 프로젝트에서 마지막까지 일관성 있게 남았던 것은 팀원들과의 소통이었네요. 우리팀 최고! | 
| 최원호 | 3D 프린팅을 통해 하드웨어 제작과 UI 디자인에 더해 C++을 이용해 QT 화면을 작업하면서 약간은 서버와도 소통도 하고 ros2 통신 세팅을 위한 rviz맵 세팅도 하고 다양한 경험을 조금씩 맛볼 수 있어서 재밌었습니다. 수강생 때 해볼 수 있는 과감한 시도를 다 해본 거 같아 뿌듯합니다. | 
| 김태호 | | 
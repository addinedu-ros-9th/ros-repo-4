# 로봇 컨트롤러에서 중앙서버로 보내는 피드백
int32 robot_id
string status  # "moving", "standby" (로봇이 스스로 판단하는 상태만)
geometry_msgs/PoseWithCovarianceStamped current_pose
int32 battery_percent
string network_status  # "connected", "disconnected", "weak"
float32 current_speed  # 현재 이동 속도 (m/s)
string[] detected_objects  # 감지된 객체들 ["person", "obstacle", "chair", ...]
string error_message  # 오류 발생시 메시지 (정상시 빈 문자열) 
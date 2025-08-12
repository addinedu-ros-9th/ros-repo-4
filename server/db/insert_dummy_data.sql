-- =====================================
-- 더미 데이터 삽입 스크립트 (backup.sql 스키마에 맞게 수정됨)
-- =====================================

USE `HeroDB`;

-- 기존 데이터 삭제 (있다면)
DELETE FROM `navigating_log`;
DELETE FROM `robot_log`;
DELETE FROM `series`;
DELETE FROM `reservations`;
DELETE FROM `robot`;
DELETE FROM `department`;
DELETE FROM `patient`;
DELETE FROM `admin`;
DELETE FROM `log_type`;
DELETE FROM `hospital`;

-- 병원 데이터
INSERT INTO `hospital` (`hospital_id`, `hospital_name`) VALUES
(1, '서울아산병원'),
(2, '히어로병원');

-- 관리자 데이터 (hospital_id 추가)
INSERT INTO `admin` (`admin_id`, `password`, `name`, `email`, `hospital_id`) VALUES
('admin1', '0000', '김관리', 'admin1@hero.com', 1),
('admin2', '0000', '박운영', 'admin2@hero.com', 2);

-- 환자 데이터 (hospital_id 추가)
INSERT INTO `patient` (`patient_id`, `name`, `ssn`, `phone`, `rfid`, `hospital_id`) VALUES
(10011001, '김환자', '900101-1234567', '010-1234-5678', '33F7AD2C', 1),
(10021002, '이환자', '950203-2345678', '010-2345-6789', '8ADBC901', 1),
(10031003, '박환자', '880304-1345679', '010-3456-7890', NULL, 1),
(10041004, '최환자', '920405-2456780', '010-4567-8901', NULL, 1);

-- 로봇 데이터 (hospital_id 추가)
INSERT INTO `robot` (`robot_id`, `hospital_id`) VALUES 
(3, 1);

-- 부서/스테이션 데이터 (hospital_id 추가)
INSERT INTO `department` (`department_id`, `department_name`, `location_x`, `location_y`, `yaw`, `hospital_id`) VALUES
(0, 'CT 검사실', -5.79, -1.88, 90.0, 1),        -- 90도
(1, '초음파 검사실', -4.9, -1.96, 90.0, 1),     -- 90도
(2, 'X-ray 검사실', -5.69, 4.34, 180.0, 1),     -- 180도
(3, '대장암 센터', 0.93, -2.3, 0.0, 1),         -- 0도
(4, '위암 센터', 3.84, -2.3, 0.0, 1),           -- 0도
(5, '폐암 센터', 5.32, -2.27, 0.0, 1),          -- 0도
(6, '뇌종양 센터', 5.97, 1.46, 180.0, 1),       -- 180도
(7, '유방암 센터', 7.64, 1.63, 180.0, 1),       -- 180도
(8, '병원 로비', 9.53, -1.76, 90.0, 1);         -- 90도

-- 로그 타입 데이터 (새로운 이벤트 로그 타입들 추가)
INSERT INTO `log_type` (`type`, `description`) VALUES
-- 호출 관련
('call_wtih_gesture', '손동작 호출'),
('call_with_voice', '음성 호출'),
('call_with_screen', '스크린터치로 호출'),
('control_by_admin', '관리자 원격제어 호출'),
('arrived_to_call', '호출 위치 도착'),

-- 수동제어 관련
('teleop_request', '수동제어 요청'),
('teleop_complete', '수동제어 명령 완수'),

-- 길안내 관련
('patient_navigating', '환자 길안내'),
('unknown_navigating', '알 수 없는 사용자 길안내'),
('admin_navigating', '관리자 길안내'),
('moving_by_patient', '환자 요청 길안내'),
('moving_by_unknown', '알 수 없는 사용자 요청 길안내'),
('pause_request', '일시정지 요청'),
('restart_navigating', '안내 재개 요청'),
('stop_navigation', '길안내 중단'),
('navigating_complete', '길안내 완료'),

-- 반환 관련
('patient_return', '환자 반환 요청'),
('unknow_return', '알 수 없는 사용자 반환 요청'),
('admin_return', '관리자 반환 요청'),
('return_by_patient', '환자 요청 반환'),
('return_by_unknown', '알 수 없는 사용자 요청 반환'),
('return_command', '제한 시간 종료 후 서버에서 자동 반환 명령'),
('arrived_to_station', '대기 장소 도착'),

-- 충전 관련
('charging_request', '충전 요청'),
('charging_complete', '충전 완료');

-- 예약 데이터
INSERT INTO `reservations` (`patient_id`, `reservation_date`) VALUES
(10011001, '2025-01-25'),
(10021002, '2025-01-25'),
(10031003, '2025-01-25'),
(10041004, '2025-01-25'),
(10011001, '2025-08-01');

-- 시리즈 데이터 (status를 VARCHAR로 수정)
INSERT INTO `series` (`series_id`, `department_id`, `dttm`, `status`, `patient_id`, `reservation_date`) VALUES
(0, 0, '2025-01-25 09:00:00', '예약', 10011001, '2025-01-25'),  -- 김환자: CT
(1, 0, '2025-01-25 10:30:00', '접수', 10021002, '2025-01-25'),  -- 이환자: CT
(2, 3, '2025-01-25 11:00:00', '접수', 10021002, '2025-01-25'),  -- 이환자: 대장암센터
(3, 1, '2025-01-25 14:00:00', '완료', 10031003, '2025-01-25'),  -- 박환자: 초음파
(4, 2, '2025-01-25 15:30:00', '접수', 10041004, '2025-01-25'),  -- 최환자: X-ray
(5, 0, '2025-08-01 09:00:00', '예약', 10011001, '2025-08-01');  -- 김환자: CT

-- 로봇 로그 샘플 데이터 (새로운 구조에 맞게 수정)
INSERT INTO `robot_log` (`robot_id`, `patient_id`, `dttm`, `type`, `admin_id`) VALUES
-- 손동작 호출
(3, 10011001, '2025-01-25 08:45:00', 'call_with_gesture', NULL),
(3, NULL, '2025-01-25 08:50:00', 'arrived_to_call', NULL),

-- 환자 길안내
(3, 10011001, '2025-01-25 08:55:00', 'patient_navigating', NULL),
(3, 10011001, '2025-01-25 09:05:00', 'navigating_complete', NULL),

-- 음성 호출
(3, 10021002, '2025-01-25 10:15:00', 'call_with_voice', NULL),
(3, NULL, '2025-01-25 10:20:00', 'arrived_to_call', NULL),

-- 환자 길안내
(3, 10021002, '2025-01-25 10:25:00', 'patient_navigating', NULL),
(3, 10021002, '2025-01-25 10:35:00', 'navigating_complete', NULL),

-- 관리자 길안내
(3, NULL, '2025-01-25 10:55:00', 'admin_navigating', 'admin1'),
(3, NULL, '2025-01-25 11:05:00', 'navigating_complete', 'admin1'),

-- 스크린터치 호출
(3, 10031003, '2025-01-25 13:45:00', 'call_with_screen', NULL),
(3, NULL, '2025-01-25 13:50:00', 'arrived_to_call', NULL),

-- 환자 길안내
(3, 10031003, '2025-01-25 13:55:00', 'patient_navigating', NULL),
(3, 10031003, '2025-01-25 14:05:00', 'navigating_complete', NULL),

-- 환자 반환
(3, 10011001, '2025-01-25 09:10:00', 'patient_return', NULL),
(3, 10011001, '2025-01-25 09:15:00', 'arrived_to_station', NULL),

-- 관리자 반환
(3, NULL, '2025-01-25 11:10:00', 'admin_return', 'admin1'),
(3, NULL, '2025-01-25 11:15:00', 'arrived_to_station', 'admin1'),

-- 수동제어
(3, NULL, '2025-01-25 12:00:00', 'teleop_request', 'admin1'),
(3, NULL, '2025-01-25 12:05:00', 'teleop_complete', 'admin1'),

-- 충전
(3, NULL, '2025-01-25 16:00:00', 'charging_request', NULL),
(3, NULL, '2025-01-25 16:30:00', 'charging_complete', NULL);

-- 네비게이션 로그 데이터 (새로운 테이블)
INSERT INTO `navigating_log` (`robot_id`, `dttm`, `orig`, `dest`) VALUES
-- 김환자: 로비 → CT (길안내 시작)
(3, '2025-01-25 08:55:00', 8, 0),

-- 이환자: 로비 → CT (길안내 시작)
(3, '2025-01-25 10:25:00', 8, 0),

-- 관리자: CT → 대장암센터 (길안내 시작)
(3, '2025-01-25 10:55:00', 0, 3),

-- 박환자: 로비 → 초음파 (길안내 시작)
(3, '2025-01-25 13:55:00', 8, 1),

-- 김환자 반환: CT → 로비 (반환 시작)
(3, '2025-01-25 09:10:00', 0, 8),

-- 관리자 반환: 대장암센터 → 로비 (반환 시작)
(3, '2025-01-25 11:10:00', 3, 8),

-- 김환자 반환 완료: 로비 도착
(3, '2025-01-25 09:15:00', 0, 8),

-- 관리자 반환 완료: 로비 도착
(3, '2025-01-25 11:15:00', 3, 8);


-- =====================================
-- 더미 데이터 삽입 스크립트 (문제점 수정됨)
-- =====================================

USE `HeroDB`;

-- 기존 데이터 삭제 (있다면)
DELETE FROM `robot_log`;
DELETE FROM `series`;
DELETE FROM `reservations`;
DELETE FROM `robot`;
DELETE FROM `station`;  -- department가 아닌 station 테이블 사용
DELETE FROM `patient`;
DELETE FROM `Admin`;    -- 대문자 A로 시작
DELETE FROM `type`;     -- log_type이 아닌 type 테이블

-- 관리자 데이터 (테이블명 수정: admin → Admin)
INSERT INTO `Admin` (`admin_id`, `password`, `name`, `email`, `hospital_name`) VALUES
('admin1', '0000', '김관리', 'admin1@hero.com', '서울아산병원'),
('admin2', '0000', '박운영', 'admin2@hero.com', '히어로병원');

-- 환자 데이터 (지정된 RFID 2개만 사용)
INSERT INTO `patient` (`patient_id`, `name`, `ssn`, `phone`, `rfid`) VALUES
(10011001, '김환자', '900101-1234567', '010-1234-5678', '33F7AD2C'),
(10021002, '이환자', '950203-2345678', '010-2345-6789', '8ADBC901'),
(10031003, '박환자', '880304-1345679', '010-3456-7890', NULL),
(10041004, '최환자', '920405-2456780', '010-4567-8901', NULL);

-- 로봇 데이터
INSERT INTO `robot` (`robot_id`) VALUES (3);

-- 로그 타입 데이터 (테이블명 수정: log_type → type, 컬럼명: type → log_type)
INSERT INTO `type` (`log_type`, `description`) VALUES
('moving_by_patient', '환자 이동'),
('moving_by_robot', '로봇 이동'),
('moving_by_admin', '관리자 이동'),
('moving_by_unknown', '본인 인증 없이 이동'),
('return_by_patient', '환자 반환'),
('return_by_robot', '로봇 반환'),
('return_by_admin', '관리자 반환'),
('return_by_unknown', '본인 인증 없이 반환'),
('assigned_by_unknown', '사용자 호출'),
('interruption', '장애물 감지 중단');

-- 부서/스테이션 데이터 (department → station 테이블 사용)
INSERT INTO `station` (`station_id`, `station_name`, `location_x`, `location_y`, `yaw`) VALUES
(0, 'CT 검사실', -5.79, -1.88, 1.57),        -- 90도 = 1.57 라디안
(1, '초음파 검사실', -4.9, -1.96, 1.57),     -- 90도
(2, 'X-ray 검사실', -5.69, 4.34, 3.14),      -- 180도 = 3.14 라디안
(3, '대장암 센터', 0.93, -2.3, 0.0),         -- 0도
(4, '위암 센터', 3.84, -2.3, 0.0),           -- 0도
(5, '폐암 센터', 5.32, -2.27, 0.0),          -- 0도
(6, '뇌종양 센터', 5.97, 1.46, 3.14),        -- 180도
(7, '유방암 센터', 7.64, 1.63, 3.14),        -- 180도
(8, '병원 로비', 9.53, -1.76, 1.57);         -- 90도

-- department 테이블에도 데이터 삽입 (series에서 참조하므로 필요)
INSERT INTO `department` (`department_id`, `department_name`, `location_x`, `location_y`, `yaw`) VALUES
(0, 'CT 검사실', -5.79, -1.88, 1.57),
(1, '초음파 검사실', -4.9, -1.96, 1.57),
(2, 'X-ray 검사실', -5.69, 4.34, 3.14),
(3, '대장암 센터', 0.93, -2.3, 0.0),
(4, '위암 센터', 3.84, -2.3, 0.0),
(5, '폐암 센터', 5.32, -2.27, 0.0),
(6, '뇌종양 센터', 5.97, 1.46, 3.14),
(7, '유방암 센터', 7.64, 1.63, 3.14);

-- 예약 데이터 (reservations 테이블)
INSERT INTO `reservations` (`patient_id`, `reservation_date`) VALUES
(10011001, '2025-01-25'),
(10021002, '2025-01-25'),
(10031003, '2025-01-25'),
(10041004, '2025-01-25'),
(10011001, '2025-08-01');

-- 시리즈 데이터 (status를 TINYINT로 수정)
INSERT INTO `series` (`series_id`, `department_id`, `dttm`, `status`, `patient_id`, `reservation_date`) VALUES
(0, 0, '2025-01-25 09:00:00', 0, 10011001, '2025-01-25'),  -- 김환자: CT (0=예약)
(1, 0, '2025-01-25 10:30:00', 1, 10021002, '2025-01-25'),  -- 이환자: CT (1=접수)
(2, 3, '2025-01-25 11:00:00', 1, 10021002, '2025-01-25'),  -- 이환자: 대장암센터 (1=접수)
(3, 1, '2025-01-25 14:00:00', 2, 10031003, '2025-01-25'),  -- 박환자: 초음파 (2=완료)
(4, 2, '2025-01-25 15:30:00', 1, 10041004, '2025-01-25'),  -- 최환자: X-ray (1=접수)
(5, 0, '2025-08-01 09:00:00', 0, 10011001, '2025-08-01');  -- 김환자: CT (0=예약)

-- 로봇 로그 샘플 데이터 (dttm 필드 포함, station_id 참조)
INSERT INTO `robot_log` (`robot_id`, `patient_id`, `dttm`, `orig`, `dest`, `log_type`) VALUES
(3, 10011001, '2025-01-25 08:50:00', 8, 0, 'moving_by_patient'),  -- 김환자: 로비 → CT
(3, 10021002, '2025-01-25 10:20:00', 8, 0, 'moving_by_patient'),  -- 이환자: 로비 → CT  
(3, 10021002, '2025-01-25 11:00:00', 0, 3, 'moving_by_patient'),  -- 이환자: CT → 대장암센터
(3, 10031003, '2025-01-25 13:50:00', 8, 1, 'moving_by_patient');  -- 박환자: 로비 → 초음파
(3, 10031003, '2025-01-25 13:50:00', 8, 1, 'moving_by_patient');  -- 박환자: 로비 → 초음파

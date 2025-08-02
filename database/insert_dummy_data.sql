-- =====================================
-- 더미 데이터 삽입 스크립트 (구문 오류 수정됨)
-- =====================================

USE `HeroDB`;

-- 기존 데이터 삭제 (있다면)
DELETE FROM `robot_log`;
DELETE FROM `series`;
DELETE FROM `reservations`;
DELETE FROM `robot`;
DELETE FROM `department`;
DELETE FROM `patient`;
DELETE FROM `admin`;
DELETE FROM `log_type`;

-- 관리자 데이터
INSERT INTO `admin` (`admin_id`, `password`, `name`, `email`, `hospital_name`) VALUES
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

-- 로그 타입 데이터
INSERT INTO `log_type` (`type`, `description`) VALUES
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

-- 부서 데이터 (department 테이블)
-- 번호: 0=CT, 1=초음파, 2=X-ray, 3=대장암, 4=위암, 5=폐암, 6=뇌종양, 7=유방암
INSERT INTO `department` (`department_id`, `department_name`, `location_x`, `location_y`, `yaw`) VALUES
(0, 'CT 검사실', -5.79, -1.88, 1.57),        -- CT (90도 = 1.57 라디안)
(1, '초음파 검사실', -4.9, -1.96, 1.57),     -- Echography 
(2, 'X-ray 검사실', -5.69, 4.34, 3.14),      -- X-ray (180도 = 3.14 라디안)
(3, '대장암 센터', 0.93, -2.3, 0.0),         -- Colon Cancer
(4, '위암 센터', 3.84, -2.3, 0.0),           -- Stomach Cancer
(5, '폐암 센터', 5.32, -2.27, 0.0),          -- Lung Cancer
(6, '뇌종양 센터', 5.97, 1.46, 3.14),        -- Brain Tumor (180도 = 3.14 라디안)
(7, '유방암 센터', 7.64, 1.63, 3.14),        -- Breast Cancer (180도 = 3.14 라디안)
(8, '병원 로비', 9.53, -1.76, 1.57);         -- Lobby Station (90도 = 1.57 라디안)

-- 예약 데이터 (reservations 테이블)
INSERT INTO `reservations` (`patient_id`, `reservation_date`, `department_id`, `status`) VALUES
(10011001, '2025-01-25', 0, 'scheduled'),
(10021002, '2025-01-25', 0, 'scheduled'),
(10031003, '2025-01-25', 1, 'scheduled'),
(10041004, '2025-01-25', 2, 'scheduled'),
(10011001, '2025-08-01', 0, 'scheduled');

-- 시리즈 데이터 (series 테이블)
INSERT INTO `series` (`department_id`, `dttm`, `status`, `patient_id`, `reservation_date`) VALUES
(0, '2025-01-25 09:00:00', '예약', 10011001, '2025-01-25'),  -- 김환자: CT 예약
(0, '2025-01-25 10:30:00', '접수', 10021002, '2025-01-25'),  -- 이환자: CT 접수
(3, '2025-01-25 11:00:00', '접수', 10021002, '2025-01-25'),  -- 이환자: 대장암센터 접수
(1, '2025-01-25 14:00:00', '완료', 10031003, '2025-01-25'),  -- 박환자: 초음파 완료
(2, '2025-01-25 15:30:00', '접수', 10041004, '2025-01-25'),  -- 최환자: X-ray 접수
(0, '2025-08-01 09:00:00', '예약', 10011001, '2025-08-01');  -- 김환자: CT 예약 (다른 날짜)

-- 로봇 로그 샘플 데이터 (컬럼명 확인 후 수정 필요)
-- DESCRIBE robot_log; 결과에 따라 아래 중 하나를 사용:

-- 옵션 1: log_type 컬럼인 경우
-- INSERT INTO `robot_log` (`robot_id`, `patient_id`, `orig`, `dest`, `log_type`) VALUES

-- 옵션 2: type 컬럼인 경우  
-- INSERT INTO `robot_log` (`robot_id`, `patient_id`, `orig`, `dest`, `type`) VALUES

-- 옵션 3: 다른 컬럼명인 경우 (예: action_type)
INSERT INTO `robot_log` (`robot_id`, `patient_id`, `orig`, `dest`) VALUES
(3, 10011001, 8, 0),  -- 김환자: 로비 → CT
(3, 10021002, 8, 0),  -- 이환자: 로비 → CT  
(3, 10021002, 0, 3),  -- 이환자: CT → 대장암센터
(3, 10031003, 8, 1);  -- 박환자: 로비 → 초음파

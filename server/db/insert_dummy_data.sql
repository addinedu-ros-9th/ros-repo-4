-- =====================================
-- 더미 데이터 삽입 스크립트
-- =====================================

USE `HeroDB`;

-- 기존 데이터 삭제 (있다면)
DELETE FROM `robot_log`;
DELETE FROM `reservations`;
DELETE FROM `robot`;
DELETE FROM `station`;
DELETE FROM `patient`;
DELETE FROM `Admin`;

-- 관리자 데이터
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

-- 정류장 데이터 (waypoints 좌표 참고)
-- 번호: 0=CT, 1=초음파, 2=X-ray, 3=대장암, 4=위암, 5=폐암, 6=뇌종양, 7=유방암
INSERT INTO `station` (`station_id`, `station_name`, `location_x`, `location_y`, `yaw`) VALUES
(0, 'CT 검사실', -5.79, -1.88, 1.57),        -- CT (90도 = 1.57 라디안)
(1, '초음파 검사실', -4.9, -1.96, 1.57),     -- Echography 
(2, 'X-ray 검사실', -5.69, 4.34, 3.14),     -- X-ray (180도 = 3.14 라디안)
(3, '대장암 센터', 0.93, -2.3, 0.0),        -- Colon Cancer
(4, '위암 센터', 3.84, -2.3, 0.0),          -- Stomach Cancer
(5, '폐암 센터', 5.32, -2.27, 0.0),         -- Lung Cancer
(6, '뇌종양 센터', 5.97, 1.46, 3.14),       -- Brain Tumor (180도)
(7, '유방암 센터', 7.64, 1.63, 3.14),       -- Breast Cancer (180도)
(8, '병원 로비', 9.53, -1.76, 1.57),        -- Lobby Station (90도)
(9, '통로 A', 0.09, 4.0, 3.14),             -- Gateway A (180도)
(10, '통로 B', -2.6, 4.18, 0.0);           -- Gateway B (0도)

-- 예약 데이터 (reservation 번호 시스템)
-- 형식: [진료실번호][상태] + [추가진료실번호][상태]
-- 상태: 0=예약만, 1=접수완료, 2=도착/완료
INSERT INTO `reservations` (`patient_id`, `datetime`, `reservation`) VALUES
-- 김환자 (RFID: 33F7AD2C): CT 검사실만 예약된 상태 (00)
(10011001, '2025-01-25 09:00:00', '00'),
-- 이환자 (RFID: 8ADBC901): CT 접수완료 + 대장암센터 추가접수 (2013)  
(10021002, '2025-01-25 10:30:00', '2013'),
-- 박환자: 초음파 검사실 도착/완료 (21)
(10031003, '2025-01-25 14:00:00', '21'),
-- 최환자: X-ray 접수 상태 (12)
(10041004, '2025-01-25 15:30:00', '12');

-- 로봇 로그 샘플 데이터
INSERT INTO `robot_log` (`robot_id`, `patient_id`, `datetime`, `orig`, `dest`) VALUES
(3, 10011001, '2025-01-25 08:50:00', 8, 0),  -- 김환자: 로비 → CT
(3, 10021002, '2025-01-25 10:20:00', 8, 0),  -- 이환자: 로비 → CT  
(3, 10021002, '2025-01-25 11:00:00', 0, 3),  -- 이환자: CT → 대장암센터
(3, 10031003, '2025-01-25 13:50:00', 8, 1);  -- 박환자: 로비 → 초음파

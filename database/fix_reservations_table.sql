-- reservations 테이블 수정 및 외래 키 문제 해결

USE `HeroDB`;

-- 기존 외래 키 제약 조건 제거 (있는 경우)
SET FOREIGN_KEY_CHECKS = 0;

-- reservations 테이블 삭제 후 재생성
DROP TABLE IF EXISTS `series`;
DROP TABLE IF EXISTS `reservations`;

-- reservations 테이블 올바르게 재생성
CREATE TABLE `reservations` (
  `patient_id` INT NOT NULL,
  `reservation_date` DATE NOT NULL,
  `department_id` INT NOT NULL,
  `status` VARCHAR(16) NOT NULL DEFAULT 'scheduled',
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`patient_id`, `reservation_date`),
  INDEX `fk_reservations_patient_idx` (`patient_id` ASC),
  INDEX `fk_reservations_department_idx` (`department_id` ASC),
  CONSTRAINT `fk_reservations_patient`
    FOREIGN KEY (`patient_id`)
    REFERENCES `patient` (`patient_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_reservations_department`
    FOREIGN KEY (`department_id`)
    REFERENCES `department` (`department_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE = InnoDB DEFAULT CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci;

-- series 테이블 재생성
CREATE TABLE `series` (
  `series_id` INT NOT NULL AUTO_INCREMENT,
  `department_id` INT NOT NULL,
  `dttm` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `status` VARCHAR(16) NOT NULL DEFAULT 'pending',
  `patient_id` INT NOT NULL,
  `reservation_date` DATE NOT NULL,
  PRIMARY KEY (`series_id`),
  INDEX `fk_series_department_idx` (`department_id` ASC),
  INDEX `fk_series_reservation_idx` (`patient_id` ASC, `reservation_date` ASC),
  CONSTRAINT `fk_series_reservation`
    FOREIGN KEY (`patient_id`, `reservation_date`)
    REFERENCES `reservations` (`patient_id`, `reservation_date`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_series_department`
    FOREIGN KEY (`department_id`)
    REFERENCES `department` (`department_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
) ENGINE = InnoDB DEFAULT CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci;

SET FOREIGN_KEY_CHECKS = 1;

-- 테스트 데이터 삽입
INSERT INTO `reservations` (`patient_id`, `reservation_date`, `department_id`, `status`) VALUES
(1001, '2025-08-02', 1, 'scheduled'),
(1002, '2025-08-03', 2, 'scheduled');

INSERT INTO `series` (`department_id`, `patient_id`, `reservation_date`, `status`) VALUES
(1, 1001, '2025-08-02', 'pending'),
(2, 1002, '2025-08-03', 'pending');

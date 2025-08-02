-- MySQL Workbench Forward Engineering (문제점 수정됨)

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema HeroDB
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `HeroDB` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci ;
USE `HeroDB` ;

-- -----------------------------------------------------
-- Table `HeroDB`.`Admin` (테이블명 통일)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`admin` (
  `admin_id` VARCHAR(20) NOT NULL,
  `password` VARCHAR(20) NOT NULL,
  `name` VARCHAR(20) NOT NULL,
  `email` VARCHAR(50) NOT NULL,  -- 16에서 50으로 증가
  `hospital_name` VARCHAR(32) NOT NULL,
  PRIMARY KEY (`admin_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`department`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`department` (
  `department_id` INT NOT NULL,
  `department_name` VARCHAR(45) NOT NULL,
  `location_x` FLOAT NOT NULL DEFAULT 0.0,  -- 위치 정보 추가
  `location_y` FLOAT NOT NULL DEFAULT 0.0,
  `yaw` FLOAT NOT NULL DEFAULT 0.0,
  PRIMARY KEY (`department_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`patient`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`patient` (
  `patient_id` INT NOT NULL,
  `name` VARCHAR(20) NOT NULL,
  `ssn` VARCHAR(16) NOT NULL,
  `phone` VARCHAR(16) NOT NULL,
  `rfid` VARCHAR(16) NULL DEFAULT NULL,
  PRIMARY KEY (`patient_id`),
  UNIQUE INDEX `ssn_UNIQUE` (`ssn` ASC) VISIBLE,
  UNIQUE INDEX `phone_UNIQUE` (`phone` ASC) VISIBLE,
  UNIQUE INDEX `rfid_UNIQUE` (`rfid` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`reservations`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`reservations` (
  `patient_id` INT NOT NULL,
  `reservation_date` DATE NOT NULL,
  `department_id` INT NOT NULL,  -- 부서 정보 추가
  `status` VARCHAR(16) NOT NULL DEFAULT 'scheduled',  -- 상태 정보 추가
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`patient_id`, `reservation_date`),
  INDEX `fk_reservations_patient_idx` (`patient_id` ASC) VISIBLE,
  INDEX `fk_reservations_department_idx` (`department_id` ASC) VISIBLE,
  CONSTRAINT `fk_reservations_patient`
    FOREIGN KEY (`patient_id`)
    REFERENCES `HeroDB`.`patient` (`patient_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_reservations_department`
    FOREIGN KEY (`department_id`)
    REFERENCES `HeroDB`.`department` (`department_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`robot`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`robot` (
  `robot_id` INT NOT NULL,
  `status` VARCHAR(16) NOT NULL DEFAULT 'idle',  -- 로봇 상태 추가
  `current_location_x` FLOAT DEFAULT NULL,
  `current_location_y` FLOAT DEFAULT NULL,
  PRIMARY KEY (`robot_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`station` (department와 중복 제거하거나 통합)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`station` (
  `station_id` INT NOT NULL,
  `station_name` VARCHAR(32) NOT NULL,
  `location_x` FLOAT NOT NULL,
  `location_y` FLOAT NOT NULL,
  `yaw` FLOAT NOT NULL,
  PRIMARY KEY (`station_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`log_type` (테이블명 수정)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`log_type` (
  `log_type` VARCHAR(45) NOT NULL,
  `description` VARCHAR(128) NULL,
  PRIMARY KEY (`log_type`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`robot_log`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`robot_log` (
  `log_id` INT NOT NULL AUTO_INCREMENT,  -- AUTO_INCREMENT ID 추가
  `robot_id` INT NOT NULL,
  `patient_id` INT NULL DEFAULT NULL,
  `dttm` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `orig` INT NULL DEFAULT NULL,
  `dest` INT NULL DEFAULT NULL,
  `log_type` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`log_id`),  -- 단일 PRIMARY KEY로 변경
  INDEX `fk_robot_log_robot_idx` (`robot_id` ASC) VISIBLE,
  INDEX `fk_robot_log_patient_idx` (`patient_id` ASC) VISIBLE,
  INDEX `fk_robot_log_orig_idx` (`orig` ASC) VISIBLE,
  INDEX `fk_robot_log_dest_idx` (`dest` ASC) VISIBLE,
  INDEX `fk_robot_log_type_idx` (`log_type` ASC) VISIBLE,
  CONSTRAINT `fk_robot_log_robot`
    FOREIGN KEY (`robot_id`)
    REFERENCES `HeroDB`.`robot` (`robot_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_patient`
    FOREIGN KEY (`patient_id`)
    REFERENCES `HeroDB`.`patient` (`patient_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_orig`
    FOREIGN KEY (`orig`)
    REFERENCES `HeroDB`.`station` (`station_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_dest`
    FOREIGN KEY (`dest`)
    REFERENCES `HeroDB`.`station` (`station_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_type`
    FOREIGN KEY (`log_type`)
    REFERENCES `HeroDB`.`log_type` (`log_type`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- -----------------------------------------------------
-- Table `HeroDB`.`series`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`series` (
  `series_id` INT NOT NULL AUTO_INCREMENT,  -- AUTO_INCREMENT 추가
  `department_id` INT NOT NULL,
  `dttm` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `status` VARCHAR(16) NOT NULL DEFAULT 'pending',  -- TINYINT에서 VARCHAR로 변경
  `patient_id` INT NOT NULL,
  `reservation_date` DATE NOT NULL,
  PRIMARY KEY (`series_id`),  -- 단일 PRIMARY KEY로 변경
  INDEX `fk_series_department_idx` (`department_id` ASC) VISIBLE,
  INDEX `fk_series_reservation_idx` (`patient_id` ASC, `reservation_date` ASC) VISIBLE,
  CONSTRAINT `fk_series_reservation`
    FOREIGN KEY (`patient_id`, `reservation_date`)
    REFERENCES `HeroDB`.`reservations` (`patient_id`, `reservation_date`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_series_department`
    FOREIGN KEY (`department_id`)
    REFERENCES `HeroDB`.`department` (`department_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;

-- 기본 데이터 삽입
INSERT INTO `HeroDB`.`log_type` (`log_type`, `description`) VALUES
('MOVE', '로봇 이동'),
('PICKUP', '환자 픽업'),
('DROPOFF', '환자 드롭오프'),
('ERROR', '오류 발생'),
('MAINTENANCE', '유지보수');

-- ...existing code...

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
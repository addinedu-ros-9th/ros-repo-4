-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema HeroDB
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema HeroDB
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `HeroDB` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci ;
USE `HeroDB` ;

-- -----------------------------------------------------
-- Table `HeroDB`.`hospital`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`hospital` (
  `hospital_id` INT NOT NULL,
  `hospital_name` VARCHAR(32) NULL,
  PRIMARY KEY (`hospital_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `HeroDB`.`admin`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`admin` (
  `admin_id` VARCHAR(20) NOT NULL,
  `password` VARCHAR(20) NOT NULL,
  `name` VARCHAR(20) NOT NULL,
  `email` VARCHAR(16) NOT NULL,
  `hospital_id` INT NOT NULL,
  PRIMARY KEY (`admin_id`),
  INDEX `fk_admin_1_idx` (`hospital_id` ASC) VISIBLE,
  CONSTRAINT `fk_admin_1`
    FOREIGN KEY (`hospital_id`)
    REFERENCES `HeroDB`.`hospital` (`hospital_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `HeroDB`.`department`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`department` (
  `department_id` INT NOT NULL,
  `department_name` VARCHAR(32) NOT NULL,
  `location_x` FLOAT NOT NULL,
  `location_y` FLOAT NOT NULL,
  `yaw` FLOAT NOT NULL,
  `hospital_id` INT NOT NULL,
  PRIMARY KEY (`department_id`),
  UNIQUE INDEX `station_id_UNIQUE` (`department_id` ASC) VISIBLE,
  INDEX `fk_department_1_idx` (`hospital_id` ASC) VISIBLE,
  CONSTRAINT `fk_department_1`
    FOREIGN KEY (`hospital_id`)
    REFERENCES `HeroDB`.`hospital` (`hospital_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
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
  `hospital_id` INT NOT NULL,
  PRIMARY KEY (`patient_id`),
  UNIQUE INDEX `ssn_UNIQUE` (`ssn` ASC) VISIBLE,
  UNIQUE INDEX `phone_UNIQUE` (`phone` ASC) VISIBLE,
  UNIQUE INDEX `patient_id_UNIQUE` (`patient_id` ASC) VISIBLE,
  UNIQUE INDEX `rfid_UNIQUE` (`rfid` ASC) VISIBLE,
  INDEX `fk_patient_1_idx` (`hospital_id` ASC) VISIBLE,
  CONSTRAINT `fk_patient_1`
    FOREIGN KEY (`hospital_id`)
    REFERENCES `HeroDB`.`hospital` (`hospital_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `HeroDB`.`reservations`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`reservations` (
  `patient_id` INT NOT NULL,
  `reservation_date` DATE NOT NULL,
  PRIMARY KEY (`patient_id`, `reservation_date`),
  INDEX `fk_reservations_1_idx` (`patient_id` ASC) VISIBLE,
  INDEX `fk_reservations_2_idx` (`reservation_date` ASC, `patient_id` ASC) VISIBLE,
  CONSTRAINT `fk_reservations_1`
    FOREIGN KEY (`patient_id`)
    REFERENCES `HeroDB`.`patient` (`patient_id`)
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
  `hospital_id` INT NOT NULL,
  PRIMARY KEY (`robot_id`),
  INDEX `fk_robot_1_idx` (`hospital_id` ASC) VISIBLE,
  CONSTRAINT `fk_robot_1`
    FOREIGN KEY (`hospital_id`)
    REFERENCES `HeroDB`.`hospital` (`hospital_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `HeroDB`.`log_type`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`log_type` (
  `type` VARCHAR(45) NOT NULL,
  `description` VARCHAR(128) NULL,
  PRIMARY KEY (`type`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `HeroDB`.`robot_log`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`robot_log` (
  `robot_id` INT NOT NULL,
  `patient_id` INT NULL DEFAULT NULL,
  `dttm` DATETIME NOT NULL,
  `type` VARCHAR(45) NOT NULL,
  `admin_id` VARCHAR(20) NULL,
  PRIMARY KEY (`robot_id`, `dttm`),
  INDEX `fk_robot_log_1_idx` (`patient_id` ASC) VISIBLE,
  INDEX `fk_robot_log_5_idx` (`type` ASC) VISIBLE,
  INDEX `fk_robot_log_3_idx` (`admin_id` ASC) VISIBLE,
  CONSTRAINT `fk_robot_log_1`
    FOREIGN KEY (`patient_id`)
    REFERENCES `HeroDB`.`patient` (`patient_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_2`
    FOREIGN KEY (`robot_id`)
    REFERENCES `HeroDB`.`robot` (`robot_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_5`
    FOREIGN KEY (`type`)
    REFERENCES `HeroDB`.`log_type` (`type`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_3`
    FOREIGN KEY (`admin_id`)
    REFERENCES `HeroDB`.`admin` (`admin_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `HeroDB`.`series`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`series` (
  `series_id` INT NOT NULL,
  `department_id` INT NOT NULL,
  `dttm` DATETIME NOT NULL,
  `status` VARCHAR(16) NOT NULL,
  `patient_id` INT NOT NULL,
  `reservation_date` DATE NOT NULL,
  PRIMARY KEY (`series_id`, `patient_id`, `reservation_date`),
  INDEX `fk_series_1_idx` (`patient_id` ASC, `reservation_date` ASC) VISIBLE,
  INDEX `fk_series_2_idx` (`department_id` ASC) VISIBLE,
  CONSTRAINT `fk_series_1`
    FOREIGN KEY (`patient_id` , `reservation_date`)
    REFERENCES `HeroDB`.`reservations` (`patient_id` , `reservation_date`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_series_2`
    FOREIGN KEY (`department_id`)
    REFERENCES `HeroDB`.`department` (`department_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `HeroDB`.`navigating_log`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`navigating_log` (
  `robot_id` INT NOT NULL,
  `dttm` DATETIME NOT NULL,
  `orig` INT NOT NULL,
  `dest` INT NOT NULL,
  PRIMARY KEY (`robot_id`, `dttm`),
  INDEX `fk_navigating_log_2_idx` (`orig` ASC) VISIBLE,
  INDEX `fk_navigating_log_3_idx` (`dest` ASC) VISIBLE,
  CONSTRAINT `fk_navigating_log_1`
    FOREIGN KEY (`robot_id` , `dttm`)
    REFERENCES `HeroDB`.`robot_log` (`robot_id` , `dttm`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_navigating_log_2`
    FOREIGN KEY (`orig`)
    REFERENCES `HeroDB`.`department` (`department_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_navigating_log_3`
    FOREIGN KEY (`dest`)
    REFERENCES `HeroDB`.`department` (`department_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

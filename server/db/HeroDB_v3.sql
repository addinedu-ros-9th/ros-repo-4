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
-- Table `HeroDB`.`Admin`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`Admin` (
  `admin_id` VARCHAR(20) NOT NULL,
  `password` VARCHAR(20) NOT NULL,
  `name` VARCHAR(20) NOT NULL,
  `email` VARCHAR(16) NOT NULL,
  `hospital_name` VARCHAR(32) NOT NULL,
  PRIMARY KEY (`admin_id`))
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
  UNIQUE INDEX `patient_id_UNIQUE` (`patient_id` ASC) VISIBLE,
  UNIQUE INDEX `rfid_UNIQUE` (`rfid` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `HeroDB`.`reservations`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`reservations` (
  `patient_id` INT NOT NULL,
  `datetime` DATETIME NOT NULL,
  `reservation` VARCHAR(64) NOT NULL,
  PRIMARY KEY (`patient_id`, `datetime`),
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
  PRIMARY KEY (`robot_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `HeroDB`.`station`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`station` (
  `station_id` INT NOT NULL,
  `station_name` VARCHAR(32) NOT NULL,
  `location_x` FLOAT NOT NULL,
  `location_y` FLOAT NOT NULL,
  `yaw` FLOAT NOT NULL,
  PRIMARY KEY (`station_id`),
  UNIQUE INDEX `station_id_UNIQUE` (`station_id` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `HeroDB`.`robot_log`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `HeroDB`.`robot_log` (
  `robot_id` INT NOT NULL,
  `patient_id` INT NULL,
  `datetime` DATETIME NOT NULL,
  `orig` INT NULL DEFAULT NULL,
  `dest` INT NULL DEFAULT NULL,
  PRIMARY KEY (`robot_id`, `datetime`),
  INDEX `fk_robot_log_1_idx` (`patient_id` ASC) VISIBLE,
  INDEX `fk_robot_log_3_idx` (`orig` ASC) VISIBLE,
  INDEX `fk_robot_log_4_idx` (`dest` ASC) VISIBLE,
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
  CONSTRAINT `fk_robot_log_3`
    FOREIGN KEY (`orig`)
    REFERENCES `HeroDB`.`station` (`station_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_robot_log_4`
    FOREIGN KEY (`dest`)
    REFERENCES `HeroDB`.`station` (`station_id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_as_ci;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

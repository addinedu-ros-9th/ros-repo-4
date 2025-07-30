#include "central_server/database_manager.h"
#include <iostream>
#include <fstream>

DatabaseManager::DatabaseManager() 
    : host_("localhost"), username_("root"), password_("heR@491!"), database_("HeroDB"), port_(3306)
{
    driver_ = nullptr;
    connection_ = nullptr;
    loadConnectionConfig();
}

DatabaseManager::~DatabaseManager() {
    disconnect();
}

void DatabaseManager::loadConnectionConfig() {
    host_ = "localhost";
    username_ = "root";
    password_ = "heR@491!"; 
    database_ = "HeroDB";
    port_ = 3306;
}

bool DatabaseManager::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    try {
        driver_ = sql::mysql::get_mysql_driver_instance();

        // config.yaml에서 읽은 host_, port_ 사용
        std::string url = "tcp://" + host_ + ":" + std::to_string(port_);
        connection_.reset(driver_->connect(url, username_, password_));
        connection_->setSchema(database_);

        std::cout << "[DB] MySQL 연결 성공: " << database_ << std::endl;
        return true;
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] MySQL 연결 실패: " << e.what() << std::endl;
        std::cerr << "[DB] Error Code: " << e.getErrorCode() << std::endl;
        return false;
    }
}

void DatabaseManager::disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (connection_) {
        connection_.reset();
        std::cout << "[DB] MySQL 연결 해제됨" << std::endl;
    }
}

bool DatabaseManager::isConnected() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    return connection_ && !connection_->isClosed();
}

bool DatabaseManager::getPatientBySSN(const std::string& ssn, PatientInfo& patient) {
    if (!isConnected()) {
        std::cerr << "[DB] 연결되지 않음" << std::endl;
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT patient_id, name, ssn, phone, rfid FROM patient WHERE ssn = ?")
        );
        pstmt->setString(1, ssn);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            patient.patient_id = res->getInt("patient_id");
            patient.name = res->getString("name");
            patient.ssn = res->getString("ssn");
            patient.phone = res->getString("phone");
            patient.rfid = res->getString("rfid");
            
            std::cout << "[DB] 환자 정보 조회 성공 (SSN): " << patient.name << std::endl;
            return true;
        }
        
        std::cout << "[DB] 환자 정보 없음 (SSN): " << ssn << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 환자 조회 실패 (SSN): " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::getPatientById(int patient_id, PatientInfo& patient) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT patient_id, name, ssn, phone, rfid FROM patient WHERE patient_id = ?")
        );
        pstmt->setInt(1, patient_id);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            patient.patient_id = res->getInt("patient_id");
            patient.name = res->getString("name");
            patient.ssn = res->getString("ssn");
            patient.phone = res->getString("phone");
            patient.rfid = res->getString("rfid");
            
            std::cout << "[DB] 환자 정보 조회 성공 (ID): " << patient.name << std::endl;
            return true;
        }
        
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 환자 조회 실패 (ID): " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::getPatientByRFID(const std::string& rfid, PatientInfo& patient) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT patient_id, name, ssn, phone, rfid FROM patient WHERE rfid = ?")
        );
        pstmt->setString(1, rfid);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            patient.patient_id = res->getInt("patient_id");
            patient.name = res->getString("name");
            patient.ssn = res->getString("ssn");
            patient.phone = res->getString("phone");
            patient.rfid = res->getString("rfid");
            
            std::cout << "[DB] 환자 정보 조회 성공 (RFID): " << patient.name << std::endl;
            return true;
        }
        
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 환자 조회 실패 (RFID): " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::authenticateAdmin(const std::string& admin_id, const std::string& password, AdminInfo& admin) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT admin_id, name, email, hospital_name FROM Admin WHERE admin_id = ? AND password = ?")
        );
        pstmt->setString(1, admin_id);
        pstmt->setString(2, password);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            admin.admin_id = res->getString("admin_id");
            admin.name = res->getString("name");
            admin.email = res->getString("email");
            admin.hospital_name = res->getString("hospital_name");
            
            std::cout << "[DB] 관리자 인증 성공: " << admin.name << std::endl;
            return true;
        }
        
        std::cout << "[DB] 관리자 인증 실패: " << admin_id << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 관리자 인증 오류: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::getStationById(int station_id, StationInfo& station) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT station_id, station_name, location_x, location_y FROM station WHERE station_id = ?")
        );
        pstmt->setInt(1, station_id);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            station.station_id = res->getInt("station_id");
            station.station_name = res->getString("station_name");
            station.location_x = res->getDouble("location_x");
            station.location_y = res->getDouble("location_y");
            
            return true;
        }
        
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 정류장 조회 실패: " << e.what() << std::endl;
        return false;
    }
}

std::vector<StationInfo> DatabaseManager::getAllStations() {
    std::vector<StationInfo> stations;
    
    if (!isConnected()) {
        return stations;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::Statement> stmt(connection_->createStatement());
        std::unique_ptr<sql::ResultSet> res(
            stmt->executeQuery("SELECT station_id, station_name, location_x, location_y FROM station")
        );
        
        while (res->next()) {
            StationInfo station;
            station.station_id = res->getInt("station_id");
            station.station_name = res->getString("station_name");
            station.location_x = res->getDouble("location_x");
            station.location_y = res->getDouble("location_y");
            
            stations.push_back(station);
        }
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 정류장 목록 조회 실패: " << e.what() << std::endl;
    }
    
    return stations;
}

bool DatabaseManager::getReservationByPatientId(int patient_id, ReservationInfo& reservation) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT patient_id, reservation_date FROM reservations WHERE patient_id = ? ORDER BY reservation_date DESC LIMIT 1")
        );
        pstmt->setInt(1, patient_id);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            reservation.patient_id = res->getInt("patient_id");
            // reservation_date를 datetime 형식으로 변환
            std::string date_str = res->getString("reservation_date");
            reservation.datetime = date_str + " 09:00:00"; // 기본 시간 추가
            reservation.reservation = "05"; // 기본값
            
            std::cout << "[DB] 예약 정보 조회 성공 (Patient ID): " << patient_id << std::endl;
            return true;
        }
        
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 예약 조회 실패 (Patient ID): " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::insertRobotLog(int robot_id, int patient_id, const std::string& datetime, float orig, float dest) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("INSERT INTO robot_log (robot_id, patient_id, dttm, orig, dest) VALUES (?, ?, ?, ?, ?)")
        );
        pstmt->setInt(1, robot_id);
        pstmt->setInt(2, patient_id);
        pstmt->setString(3, datetime);
        pstmt->setDouble(4, orig);
        pstmt->setDouble(5, dest);
        
        int affected = pstmt->executeUpdate();
        
        if (affected > 0) {
            std::cout << "[DB] 로봇 로그 삽입 성공" << std::endl;
            return true;
        }
        
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 로봇 로그 삽입 실패: " << e.what() << std::endl;
        return false;
    }
} 

// Series 테이블 관련 메서드들

bool DatabaseManager::getSeriesByPatientAndDate(int patient_id, const std::string& reservation_date, SeriesInfo& series) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT series_id, department_id, dttm, status, patient_id, reservation_date FROM series WHERE patient_id = ? AND reservation_date = ? AND series_id = 0")
        );
        pstmt->setInt(1, patient_id);
        pstmt->setString(2, reservation_date);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            series.series_id = res->getInt("series_id");
            series.department_id = res->getInt("department_id");
            series.dttm = res->getString("dttm");
            series.status = res->getString("status");
            series.patient_id = res->getInt("patient_id");
            series.reservation_date = res->getString("reservation_date");
            
            std::cout << "[DB] Series 정보 조회 성공: Patient " << patient_id << ", Status " << series.status << std::endl;
            return true;
        }
        
        std::cout << "[DB] Series 정보 없음: Patient " << patient_id << ", Date " << reservation_date << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] Series 조회 실패: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::updateSeriesStatus(int patient_id, const std::string& reservation_date, const std::string& new_status) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("UPDATE series SET status = ? WHERE patient_id = ? AND reservation_date = ? AND series_id = 0")
        );
        pstmt->setString(1, new_status);
        pstmt->setInt(2, patient_id);
        pstmt->setString(3, reservation_date);
        
        int affected = pstmt->executeUpdate();
        
        if (affected > 0) {
            std::cout << "[DB] Series 상태 업데이트 성공: Patient " << patient_id << " -> " << new_status << std::endl;
            return true;
        }
        
        std::cout << "[DB] Series 상태 업데이트 실패: Patient " << patient_id << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] Series 상태 업데이트 실패: " << e.what() << std::endl;
        return false;
    }
}

std::string DatabaseManager::getCurrentDate() {
    if (!isConnected()) {
        return "";
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::Statement> stmt(connection_->createStatement());
        std::unique_ptr<sql::ResultSet> res(stmt->executeQuery("SELECT CURDATE() as current_date"));
        
        if (res->next()) {
            return res->getString("current_date");
        }
        
        return "";
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 현재 날짜 조회 실패: " << e.what() << std::endl;
        return "";
    }
} 

bool DatabaseManager::getSeriesWithDepartmentName(int patient_id, const std::string& reservation_date, SeriesInfo& series, std::string& department_name) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("SELECT s.series_id, s.department_id, s.dttm, s.status, s.patient_id, s.reservation_date, d.department_name FROM series s JOIN department d ON s.department_id = d.department_id WHERE s.patient_id = ? AND s.reservation_date = ? AND s.series_id = 0")
        );
        pstmt->setInt(1, patient_id);
        pstmt->setString(2, reservation_date);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            series.series_id = res->getInt("series_id");
            series.department_id = res->getInt("department_id");
            series.dttm = res->getString("dttm");
            series.status = res->getString("status");
            series.patient_id = res->getInt("patient_id");
            series.reservation_date = res->getString("reservation_date");
            department_name = res->getString("department_name");
            
            std::cout << "[DB] Series 정보 조회 성공: Patient " << patient_id << ", Department " << department_name << ", Status " << series.status << std::endl;
            return true;
        }
        
        std::cout << "[DB] Series 정보 없음: Patient " << patient_id << ", Date " << reservation_date << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] Series 조회 실패: " << e.what() << std::endl;
        return false;
    }
} 
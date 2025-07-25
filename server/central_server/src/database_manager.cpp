#include "central_server/database_manager.h"
#include <iostream>
#include <fstream>

DatabaseManager::DatabaseManager() 
    : host_("localhost"), username_("root"), password_("0000"), database_("HeroDB"), port_(3306)
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
    password_ = "0000"; 
    database_ = "HeroDB";
    port_ = 3306;
}

bool DatabaseManager::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    try {
        driver_ = sql::mysql::get_mysql_driver_instance();
        
        std::string url = "unix:///var/run/mysqld/mysqld.sock";  // socket 연결
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
        
        std::cout << "[DB] 정류장 목록 조회: " << stations.size() << "개" << std::endl;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 정류장 목록 조회 실패: " << e.what() << std::endl;
    }
    
    return stations;
}

bool DatabaseManager::insertRobotLog(int robot_id, int patient_id, const std::string& datetime, float orig, float dest) {
    if (!isConnected()) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection_->prepareStatement("INSERT INTO robot_log (robot_id, patient_id, datetime, orig, dest) VALUES (?, ?, ?, ?, ?)")
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
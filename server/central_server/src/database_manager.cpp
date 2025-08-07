#include "central_server/database_manager.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>

// ConnectionGuard 구현
ConnectionGuard::ConnectionGuard(DatabaseManager* db_manager, sql::Connection* connection)
    : db_manager_(db_manager), connection_(connection) {}

ConnectionGuard::~ConnectionGuard() {
    if (db_manager_ && connection_) {
        db_manager_->releaseConnection(connection_);
    }
}

DatabaseManager::DatabaseManager() 
    : host_("localhost"), username_("root"), password_("heR@491!"), database_("HeroDB"), port_(3306), pool_size_(5)
{
    driver_ = nullptr;
    loadConnectionConfig();
}

DatabaseManager::~DatabaseManager() {
    disconnect();
}

void DatabaseManager::loadConnectionConfig() {
    host_ = "localhost";
    username_ = "root";
    password_ = "heR@491!"; 
    // password_ = "0000"; 
    database_ = "HeroDB";
    port_ = 3306;
}

bool DatabaseManager::connect() {
    try {
        driver_ = sql::mysql::get_mysql_driver_instance();
        
        if (!initializeConnectionPool()) {
            std::cerr << "[DB] Connection Pool 초기화 실패" << std::endl;
            return false;
        }
        
        std::cout << "[DB] MySQL Connection Pool 연결 성공: " << database_ << " (Pool Size: " << pool_size_ << ")" << std::endl;
        return true;
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] MySQL 연결 실패: " << e.what() << std::endl;
        std::cerr << "[DB] Error Code: " << e.getErrorCode() << std::endl;
        return false;
    }
}

void DatabaseManager::disconnect() {
    cleanupConnectionPool();
    std::cout << "[DB] MySQL Connection Pool 연결 해제됨" << std::endl;
}

bool DatabaseManager::isConnected() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return !connection_pool_.empty() && !available_connections_.empty();
}

bool DatabaseManager::initializeConnectionPool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    try {
        std::string url = "tcp://" + host_ + ":" + std::to_string(port_);
        
        for (size_t i = 0; i < pool_size_; ++i) {
            auto connection = std::unique_ptr<sql::Connection>(driver_->connect(url, username_, password_));
            connection->setSchema(database_);
            
            // Connection 유효성 검사
            std::unique_ptr<sql::Statement> stmt(connection->createStatement());
            stmt->executeQuery("SELECT 1");
            
            connection_pool_.push_back(std::move(connection));
            available_connections_.push(connection_pool_.back().get());
        }
        
        std::cout << "[DB] Connection Pool 초기화 완료: " << pool_size_ << "개 연결 생성" << std::endl;
        return true;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] Connection Pool 초기화 실패: " << e.what() << std::endl;
        return false;
    }
}

sql::Connection* DatabaseManager::getConnection() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (available_connections_.empty()) {
        std::cerr << "[DB] 사용 가능한 Connection이 없습니다" << std::endl;
        return nullptr;
    }
    
    sql::Connection* connection = available_connections_.front();
    available_connections_.pop();
    
    // Connection 유효성 검사
    try {
        std::unique_ptr<sql::Statement> stmt(connection->createStatement());
        stmt->executeQuery("SELECT 1");
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] Connection 유효성 검사 실패, 재생성 시도: " << e.what() << std::endl;
        
        // 실패한 connection 제거하고 새로 생성
        auto it = std::find_if(connection_pool_.begin(), connection_pool_.end(),
                              [connection](const std::unique_ptr<sql::Connection>& conn) {
                                  return conn.get() == connection;
                              });
        
        if (it != connection_pool_.end()) {
            std::string url = "tcp://" + host_ + ":" + std::to_string(port_);
            *it = std::unique_ptr<sql::Connection>(driver_->connect(url, username_, password_));
            (*it)->setSchema(database_);
            connection = it->get();
        }
    }
    
    return connection;
}

void DatabaseManager::releaseConnection(sql::Connection* connection) {
    if (connection == nullptr) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    available_connections_.push(connection);
}

void DatabaseManager::cleanupConnectionPool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    while (!available_connections_.empty()) {
        available_connections_.pop();
    }
    
    connection_pool_.clear();
}

bool DatabaseManager::getPatientBySSN(const std::string& ssn, PatientInfo& patient) {
    if (!isConnected()) {
        std::cerr << "[DB] 연결되지 않음" << std::endl;
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    // RAII를 사용한 Connection 관리
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT patient_id, name, ssn, phone, rfid FROM patient WHERE ssn = ?")
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
    // ConnectionGuard 소멸자에서 자동으로 connection 반환
}

bool DatabaseManager::getPatientById(int patient_id, PatientInfo& patient) {
    if (!isConnected()) {
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT patient_id, name, ssn, phone, rfid FROM patient WHERE patient_id = ?")
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
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT patient_id, name, ssn, phone, rfid FROM patient WHERE rfid = ?")
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
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT admin_id, name, email, hospital_name FROM Admin WHERE admin_id = ? AND password = ?")
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

bool DatabaseManager::isAdminIdExists(const std::string& admin_id) {
    if (!isConnected()) {
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT COUNT(*) FROM Admin WHERE admin_id = ?")
        );
        pstmt->setString(1, admin_id);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next() && res->getInt(1) > 0) {
            return true; // 관리자 ID가 존재함
        }
        
        return false; // 관리자 ID가 존재하지 않음
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 관리자 ID 존재 여부 확인 오류: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::getAdminById(const std::string& admin_id, AdminInfo& admin) {
    if (!isConnected()) {
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT admin_id, name, email, hospital_name FROM Admin WHERE admin_id = ?")
        );
        pstmt->setString(1, admin_id);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            admin.admin_id = res->getString("admin_id");
            admin.name = res->getString("name");
            admin.email = res->getString("email");
            admin.hospital_name = res->getString("hospital_name");
            
            std::cout << "[DB] 관리자 정보 조회 성공: " << admin.name << std::endl;
            return true;
        }
        
        std::cout << "[DB] 관리자 정보 조회 실패: " << admin_id << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 관리자 정보 조회 오류: " << e.what() << std::endl;
        return false;
    }
}

bool DatabaseManager::getDepartmentById(int department_id, DepartmentInfo& department) {
    if (!isConnected()) {
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT department_id, department_name, location_x, location_y, yaw FROM department WHERE department_id = ?")
        );
        pstmt->setInt(1, department_id);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            department.department_id = res->getInt("department_id");
            department.department_name = res->getString("department_name");
            department.location_x = res->getDouble("location_x");
            department.location_y = res->getDouble("location_y");
            department.yaw = res->getDouble("yaw");
            
            return true;
        }
        
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 부서 조회 실패: " << e.what() << std::endl;
        return false;
    }
}

std::vector<DepartmentInfo> DatabaseManager::getAllDepartments() {
    std::vector<DepartmentInfo> departments;
    
    if (!isConnected()) {
        return departments;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return departments;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::Statement> stmt(connection->createStatement());
        std::unique_ptr<sql::ResultSet> res(
            stmt->executeQuery("SELECT department_id, department_name, location_x, location_y, yaw FROM department")
        );
        
        while (res->next()) {
            DepartmentInfo department;
            department.department_id = res->getInt("department_id");
            department.department_name = res->getString("department_name");
            department.location_x = res->getDouble("location_x");
            department.location_y = res->getDouble("location_y");
            department.yaw = res->getDouble("yaw");
            
            departments.push_back(department);
        }
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 부서 목록 조회 실패: " << e.what() << std::endl;
    }
    
    return departments;
}

bool DatabaseManager::getReservationByPatientId(int patient_id, ReservationInfo& reservation) {
    if (!isConnected()) {
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT patient_id, reservation_date FROM reservations WHERE patient_id = ? ORDER BY reservation_date DESC LIMIT 1")
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
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("INSERT INTO robot_log (robot_id, patient_id, dttm, orig, dest) VALUES (?, ?, ?, ?, ?)")
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
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT series_id, department_id, dttm, status, patient_id, reservation_date FROM series WHERE patient_id = ? AND reservation_date = ? AND series_id = 0")
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
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("UPDATE series SET status = ? WHERE patient_id = ? AND reservation_date = ? AND series_id = 0")
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
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return "";
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        std::unique_ptr<sql::Statement> stmt(connection->createStatement());
        std::unique_ptr<sql::ResultSet> res(stmt->executeQuery("SELECT CURDATE() as `current_date`"));
        
        if (res->next()) {
            return res->getString("current_date");
        }
        
        return "";
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 현재 날짜 조회 실패: " << e.what() << std::endl;
        return "";
    }
}

bool DatabaseManager::insertRobotLogWithType(int robot_id, int* patient_id, const std::string& datetime, 
                                           int orig_department_id, int dest_department_id, const std::string& type) {
    if (!isConnected()) {
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        // dttm -> datetime으로 변경
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("INSERT INTO robot_log (robot_id, patient_id, datetime, orig, dest, type) VALUES (?, ?, ?, ?, ?, ?)")
        );
        pstmt->setInt(1, robot_id);
        
        if (patient_id != nullptr) {
            pstmt->setInt(2, *patient_id);
        } else {
            // NULL 값을 설정하는 대신 기본값 0을 사용
            pstmt->setInt(2, 0);
        }
        
        pstmt->setString(3, datetime);
        pstmt->setInt(4, orig_department_id);
        pstmt->setInt(5, dest_department_id);
        pstmt->setString(6, type);
        
        int affected = pstmt->executeUpdate();
        
        if (affected > 0) {
            std::cout << "[DB] 로봇 로그 삽입 성공: Robot " << robot_id 
                     << ", Patient " << (patient_id ? std::to_string(*patient_id) : "NULL")
                     << ", Orig " << orig_department_id << ", Dest " << dest_department_id 
                     << ", Type " << type << std::endl;
            return true;
        }
        
        std::cout << "[DB] 로봇 로그 삽입 실패: Robot " << robot_id << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 로봇 로그 삽입 실패: " << e.what() << std::endl;
        return false;
    }
}

int DatabaseManager::findNearestDepartment(float x, float y) {
    if (!isConnected()) {
        return -1;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return -1;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        // 모든 부서 정보를 가져와서 가장 가까운 부서 찾기
        std::unique_ptr<sql::Statement> stmt(connection->createStatement());
        std::unique_ptr<sql::ResultSet> res(stmt->executeQuery("SELECT department_id, location_x, location_y FROM department"));
        
        int nearest_department_id = -1;
        float min_distance = std::numeric_limits<float>::max();
        
        while (res->next()) {
            int dept_id = res->getInt("department_id");
            float dept_x = static_cast<float>(res->getDouble("location_x"));
            float dept_y = static_cast<float>(res->getDouble("location_y"));
            
            // 유클리드 거리 계산
            float distance = std::sqrt(std::pow(x - dept_x, 2) + std::pow(y - dept_y, 2));
            
            if (distance < min_distance) {
                min_distance = distance;
                nearest_department_id = dept_id;
            }
        }
        
        if (nearest_department_id != -1) {
            std::cout << "[DB] 가장 가까운 부서 찾음: Department " << nearest_department_id 
                     << " (거리: " << min_distance << ")" << std::endl;
        }
        
        return nearest_department_id;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 가장 가까운 부서 찾기 실패: " << e.what() << std::endl;
        return -1;
    }
} 

bool DatabaseManager::getTodayReservationWithDepartmentName(int patient_id, SeriesInfo& series, std::string& department_name) {
    if (!isConnected()) {
        return false;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return false;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        // 오늘 날짜의 "예약" 상태인 건을 찾는 쿼리
        std::unique_ptr<sql::PreparedStatement> pstmt(
            connection->prepareStatement("SELECT s.series_id, s.department_id, s.dttm, s.status, s.patient_id, s.reservation_date, d.department_name FROM series s JOIN department d ON s.department_id = d.department_id WHERE s.patient_id = ? AND s.reservation_date = CURDATE() AND s.series_id = 0")
        );
        pstmt->setInt(1, patient_id);
        
        std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());
        
        if (res->next()) {
            series.series_id = res->getInt("series_id");
            series.department_id = res->getInt("department_id");
            series.dttm = res->getString("dttm");
            series.status = res->getString("status");
            series.patient_id = res->getInt("patient_id");
            series.reservation_date = res->getString("reservation_date");
            department_name = res->getString("department_name");
            
            std::cout << "[DB] 오늘 예약 정보 조회 성공: Patient " << patient_id << ", Department " << department_name << ", Status " << series.status << ", Date " << series.reservation_date << std::endl;
            return true;
        }
        
        std::cout << "[DB] 오늘 예약 정보 없음: Patient " << patient_id << std::endl;
        return false;
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 오늘 예약 조회 실패: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::map<std::string, std::string>> DatabaseManager::getRobotLogData(const std::string& period, 
                                                                                 const std::string& start_date, 
                                                                                 const std::string& end_date) {
    std::vector<std::map<std::string, std::string>> log_data;
    
    if (!isConnected()) {
        std::cerr << "[DB] 데이터베이스 연결되지 않음" << std::endl;
        return log_data;
    }
    
    sql::Connection* raw_connection = getConnection();
    if (!raw_connection) {
        std::cerr << "[DB] Connection 획득 실패" << std::endl;
        return log_data;
    }
    
    ConnectionGuard connection(this, raw_connection);
    
    try {
        // 먼저 robot_log 테이블이 존재하는지 확인
        std::unique_ptr<sql::Statement> check_stmt(connection->createStatement());
        std::unique_ptr<sql::ResultSet> table_check(check_stmt->executeQuery("SHOW TABLES LIKE 'robot_log'"));
        
        if (!table_check->next()) {
            std::cerr << "[DB] robot_log 테이블이 존재하지 않습니다" << std::endl;
            return log_data;
        }
        
        // 테이블 구조 확인
        std::unique_ptr<sql::ResultSet> structure_check(check_stmt->executeQuery("DESCRIBE robot_log"));
        std::cout << "[DB] robot_log 테이블 구조:" << std::endl;
        while (structure_check->next()) {
            std::cout << "[DB] - " << structure_check->getString("Field") 
                     << " (" << structure_check->getString("Type") << ")" << std::endl;
        }
        
        // 이미지에서 확인한 실제 컬럼명 사용: dateTime
        std::string query = "SELECT rl.robot_id, "
                           "COALESCE(rl.patient_id, 0) as patient_id, "
                           "rl.dateTime, "  // 실제 컬럼명 사용
                           "COALESCE(rl.orig, 0) as orig, "
                           "COALESCE(rl.dest, 0) as dest "
                           "FROM robot_log rl ";
        
        std::string where_clause = "";
        
        // period 파라미터 처리
        if (!period.empty()) {
            if (period == "today") {
                where_clause = "WHERE DATE(rl.dateTime) = CURDATE()";
            } else if (period == "week") {
                where_clause = "WHERE rl.dateTime >= DATE_SUB(NOW(), INTERVAL 7 DAY)";
            } else if (period == "month") {
                where_clause = "WHERE rl.dateTime >= DATE_SUB(NOW(), INTERVAL 1 MONTH)";
            }
        } else if (!start_date.empty() || !end_date.empty()) {
            // start_date와 end_date 파라미터 처리
            where_clause = "WHERE 1=1";
            if (!start_date.empty()) {
                where_clause += " AND DATE(rl.dateTime) >= '" + start_date + "'";
            }
            if (!end_date.empty()) {
                where_clause += " AND DATE(rl.dateTime) <= '" + end_date + "'";
            }
        }
        
        query += where_clause + " ORDER BY rl.dateTime DESC LIMIT 100";
        
        std::cout << "[DB] 실행할 쿼리: " << query << std::endl;
        
        std::unique_ptr<sql::Statement> stmt(connection->createStatement());
        std::unique_ptr<sql::ResultSet> res(stmt->executeQuery(query));
        
        int count = 0;
        while (res->next()) {
            std::map<std::string, std::string> log_entry;
            
            try {
                // patient_id 처리 (NULL 체크 후 문자열 변환)
                if (res->isNull("patient_id")) {
                    log_entry["patient_id"] = "0";
                } else {
                    int patient_id_val = res->getInt("patient_id");
                    log_entry["patient_id"] = std::to_string(patient_id_val);
                }
                
                // orig (출발지)
                log_entry["orig"] = std::to_string(res->getInt("orig"));
                
                // dest (목적지)
                log_entry["dest"] = std::to_string(res->getInt("dest"));
                
                // date (날짜) - dateTime 컬럼 사용
                log_entry["date"] = res->getString("dateTime");
                
                log_data.push_back(log_entry);
                count++;
                
                if (count <= 5) {  // 처음 5개 레코드만 디버그 출력
                    std::cout << "[DB] 로그 엔트리 " << count << ": "
                             << "patient_id=" << log_entry["patient_id"]
                             << ", orig=" << log_entry["orig"]
                             << ", dest=" << log_entry["dest"]
                             << ", date=" << log_entry["date"] << std::endl;
                }
                
            } catch (sql::SQLException& inner_e) {
                std::cerr << "[DB] 레코드 처리 중 오류: " << inner_e.what() << std::endl;
                continue;  // 이 레코드는 건너뛰고 다음으로
            }
        }
        
        std::cout << "[DB] 로봇 로그 데이터 조회 완료: " << log_data.size() << "개 레코드" << std::endl;
        
        // 데이터가 없는 경우 테이블 내용 확인
        if (log_data.empty()) {
            std::unique_ptr<sql::ResultSet> count_check(stmt->executeQuery("SELECT COUNT(*) as total FROM robot_log"));
            if (count_check->next()) {
                int total_records = count_check->getInt("total");
                std::cout << "[DB] robot_log 테이블 총 레코드 수: " << total_records << std::endl;
                
                if (total_records == 0) {
                    std::cout << "[DB] robot_log 테이블이 비어있습니다" << std::endl;
                } else {
                    // 최근 몇 개 레코드 확인 - dateTime 컬럼 사용
                    std::unique_ptr<sql::ResultSet> sample_check(stmt->executeQuery("SELECT * FROM robot_log ORDER BY dateTime DESC LIMIT 3"));
                    std::cout << "[DB] 최근 레코드들:" << std::endl;
                    while (sample_check->next()) {
                        std::cout << "[DB] - robot_id: " << sample_check->getInt("robot_id")
                                 << ", patient_id: " << (sample_check->isNull("patient_id") ? "NULL" : std::to_string(sample_check->getInt("patient_id")))
                                 << ", dateTime: " << sample_check->getString("dateTime") << std::endl;
                    }
                }
            }
        }
        
    } catch (sql::SQLException& e) {
        std::cerr << "[DB] 로봇 로그 데이터 조회 실패: " << e.what() << std::endl;
        std::cerr << "[DB] SQL State: " << e.getSQLState() << std::endl;
        std::cerr << "[DB] Error Code: " << e.getErrorCode() << std::endl;
        
        // 연결 상태 재확인
        try {
            std::unique_ptr<sql::Statement> test_stmt(connection->createStatement());
            std::unique_ptr<sql::ResultSet> test_res(test_stmt->executeQuery("SELECT 1"));
            std::cout << "[DB] 데이터베이스 연결은 정상입니다" << std::endl;
        } catch (sql::SQLException& test_e) {
            std::cerr << "[DB] 데이터베이스 연결에 문제가 있습니다: " << test_e.what() << std::endl;
        }
    }
    
    return log_data;
}
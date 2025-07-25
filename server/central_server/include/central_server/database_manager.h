#ifndef DATABASE_MANAGER_H
#define DATABASE_MANAGER_H

#include <mysql_driver.h>
#include <mysql_connection.h>
#include <cppconn/statement.h>
#include <cppconn/resultset.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/exception.h>

#include <string>
#include <memory>
#include <mutex>

struct PatientInfo {
    int patient_id;
    std::string name;
    std::string ssn;
    std::string phone;
    std::string rfid;
};

struct AdminInfo {
    std::string admin_id;
    std::string name;
    std::string email;
    std::string hospital_name;
};

struct StationInfo {
    int station_id;
    std::string station_name;
    float location_x;
    float location_y;
};

class DatabaseManager
{
public:
    DatabaseManager();
    ~DatabaseManager();
    
    // 연결 관리
    bool connect();
    void disconnect();
    bool isConnected();
    
    // 환자 정보 조회
    bool getPatientBySSN(const std::string& ssn, PatientInfo& patient);
    bool getPatientById(int patient_id, PatientInfo& patient);
    bool getPatientByRFID(const std::string& rfid, PatientInfo& patient);
    
    // 관리자 인증
    bool authenticateAdmin(const std::string& admin_id, const std::string& password, AdminInfo& admin);
    
    // 정류장 정보
    bool getStationById(int station_id, StationInfo& station);
    std::vector<StationInfo> getAllStations();
    
    // 로봇 로그
    bool insertRobotLog(int robot_id, int patient_id, const std::string& datetime, float orig, float dest);

private:
    // MySQL 연결 정보
    std::string host_;
    std::string username_;
    std::string password_;
    std::string database_;
    int port_;
    
    // MySQL 연결 객체들
    sql::mysql::MySQL_Driver* driver_;
    std::unique_ptr<sql::Connection> connection_;
    std::mutex connection_mutex_;
    
    // 내부 유틸리티 함수들
    void loadConnectionConfig();
    bool executeQuery(const std::string& query);
    std::unique_ptr<sql::ResultSet> executeSelect(const std::string& query);
};

#endif // DATABASE_MANAGER_H 
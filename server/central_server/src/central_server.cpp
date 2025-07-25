#include "central_server/central_server.h"
#include <chrono>

CentralServer::CentralServer() : Node("central_server"), running_(false) {
    RCLCPP_INFO(this->get_logger(), "CentralServer 생성자 호출됨");
    
    // DatabaseManager 생성
    db_manager_ = std::make_unique<DatabaseManager>();
    
    // HttpServer 생성 (DatabaseManager를 shared_ptr로 전달)
    auto shared_db_manager = std::shared_ptr<DatabaseManager>(db_manager_.get(), [](DatabaseManager*){});
    http_server_ = std::make_unique<HttpServer>(shared_db_manager, 8080);
}

CentralServer::~CentralServer() {
    stop();
}

void CentralServer::start() {
    if (running_.load()) {
        RCLCPP_WARN(this->get_logger(), "서버가 이미 실행중입니다");
        return;
    }
    
    running_.store(true);
    
    RCLCPP_INFO(this->get_logger(), "1단계: DB 스레드 시작중...");
    db_thread_ = std::thread(&CentralServer::runDatabaseThread, this);
    
    RCLCPP_INFO(this->get_logger(), "2단계: HTTP 스레드 시작중...");
    http_thread_ = std::thread(&CentralServer::runHttpThread, this);
    
    // 잠시 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void CentralServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "서버 종료중...");
    running_.store(false);
    
    // 스레드 종료 대기
    if (db_thread_.joinable()) {
        db_thread_.join();
        RCLCPP_INFO(this->get_logger(), "DB 스레드 종료됨");
    }
    
    if (http_thread_.joinable()) {
        http_thread_.join();
        RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료됨");
    }
}

void CentralServer::runDatabaseThread() {
    RCLCPP_INFO(this->get_logger(), "DB 스레드 시작됨");
    
    // DB 연결 시도
    if (db_manager_->connect()) {
        RCLCPP_INFO(this->get_logger(), "MySQL 데이터베이스 연결 성공!");
    } else {
        RCLCPP_ERROR(this->get_logger(), "MySQL 데이터베이스 연결 실패!");
        RCLCPP_WARN(this->get_logger(), "DB 없이 계속 실행합니다...");
    }
    
    while (running_.load() && rclcpp::ok()) {
        // DB 연결 상태 확인 (5초마다)
        if (!db_manager_->isConnected()) {
            RCLCPP_WARN(this->get_logger(), "DB 연결 끊어짐. 재연결 시도...");
            db_manager_->connect();
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    
    RCLCPP_INFO(this->get_logger(), "DB 스레드 종료중...");
}

void CentralServer::runHttpThread() {
    RCLCPP_INFO(this->get_logger(), "HTTP 스레드 시작됨");
    
    // HTTP 서버 시작
    http_server_->start();
    RCLCPP_INFO(this->get_logger(), "HTTP 서버 시작 완료 (포트: 8080)");
    
    while (running_.load() && rclcpp::ok()) {
        // HTTP 서버 상태 확인
        if (!http_server_->isRunning()) {
            RCLCPP_ERROR(this->get_logger(), "HTTP 서버가 중지됨!");
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // HTTP 서버 정지
    http_server_->stop();
    RCLCPP_INFO(this->get_logger(), "HTTP 스레드 종료중...");
} 
#include "./matplotlib-cpp/matplotlibcpp.h"
#include "log.h"
#include "ui_log.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDebug>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QCheckBox>
#include <QHeaderView>
#include <yaml-cpp/yaml.h>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QDate>

namespace plt = matplotlibcpp;

LogWidget::LogWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_LogWidget)
    , currentToggle(2)  // 초기값: 주간 선택
    , logEntries_({
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "CT 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "X-ray 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "X-ray 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "X-ray 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "X-ray 검사실", "뇌졸중 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "X-ray 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "X-ray 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "위암 센터", "CT 검사실", "25.07.15 15:30:00"},
            {"00 00 00 00", "위암 센터", "CT 검사실", "25.07.15 15:30:00"},
            {"00 00 00 00", "위암 센터", "CT 검사실", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
            {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"}
        })  // 로그 엔트리 초기화
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
    setupConnections();
    setupTableData();
    onToggle2Clicked();
}

LogWidget::~LogWidget()
{
    delete ui;
}

void LogWidget::setupConnections()
{
    // 토글 버튼 클릭 이벤트 연결
    if (ui->toggle1) {
        connect(ui->toggle1, &QPushButton::clicked, this, &LogWidget::onToggle1Clicked);
    }
    if (ui->toggle2) {
        connect(ui->toggle2, &QPushButton::clicked, this, &LogWidget::onToggle2Clicked);
    }
    if (ui->toggle3) {
        connect(ui->toggle3, &QPushButton::clicked, this, &LogWidget::onToggle3Clicked);
    }
    
    // 검색 버튼을 눌렀을 때는 날짜 필드 값을 사용하도록 수정
    if (ui->search_btn) {
        connect(ui->search_btn, &QPushButton::clicked, this, [this]() {
            get_robot_log_data_with_period("");  // 빈 period로 날짜 필드 사용
        });
    }
}

void LogWidget::onToggle1Clicked()
{
    qDebug() << "Toggle1 (오늘) clicked";
    currentToggle = 1;
    updateToggleButtons(1);
    
    // 오늘 날짜로 설정
    QDate today = QDate::currentDate();
    setDateFields(today, today);
    
    refresh();
}

void LogWidget::onToggle2Clicked()
{
    qDebug() << "Toggle2 (주간) clicked";
    currentToggle = 2;
    updateToggleButtons(2);
    
    // 일주일 범위로 설정 (오늘부터 7일 전까지)
    QDate today = QDate::currentDate();
    QDate weekAgo = today.addDays(-7);
    setDateFields(weekAgo, today);
    
    refresh();
}

void LogWidget::onToggle3Clicked()
{
    qDebug() << "Toggle3 (월간) clicked";
    currentToggle = 3;
    updateToggleButtons(3);
    
    // 한 달 범위로 설정 (오늘부터 30일 전까지)
    QDate today = QDate::currentDate();
    QDate monthAgo = today.addDays(-30);
    setDateFields(monthAgo, today);
    
    refresh();
}

void LogWidget::updateToggleButtons(int activeToggle)
{
    // 모든 토글 버튼을 비활성 상태로 설정
    if (ui->toggle1) {
        ui->toggle1->setProperty("class", "toggle");
        ui->toggle1->style()->unpolish(ui->toggle1);
        ui->toggle1->style()->polish(ui->toggle1);
    }
    if (ui->toggle2) {
        ui->toggle2->setProperty("class", "toggle");
        ui->toggle2->style()->unpolish(ui->toggle2);
        ui->toggle2->style()->polish(ui->toggle2);
    }
    if (ui->toggle3) {
        ui->toggle3->setProperty("class", "toggle");
        ui->toggle3->style()->unpolish(ui->toggle3);
        ui->toggle3->style()->polish(ui->toggle3);
    }
    
    // 선택된 토글 버튼을 활성 상태로 설정
    switch (activeToggle) {
        case 1:
            if (ui->toggle1) {
                ui->toggle1->setProperty("class", "toggle active");
                ui->toggle1->style()->unpolish(ui->toggle1);
                ui->toggle1->style()->polish(ui->toggle1);
            }
            break;
        case 2:
            if (ui->toggle2) {
                ui->toggle2->setProperty("class", "toggle active");
                ui->toggle2->style()->unpolish(ui->toggle2);
                ui->toggle2->style()->polish(ui->toggle2);
            }
            break;
        case 3:
            if (ui->toggle3) {
                ui->toggle3->setProperty("class", "toggle active");
                ui->toggle3->style()->unpolish(ui->toggle3);
                ui->toggle3->style()->polish(ui->toggle3);
            }
            break;
    }
}


void LogWidget::setWidgetClasses()
{
    // UI 파일의 위젯들에 CSS 클래스 설정
    if (ui->toggle_bg) {
        ui->toggle_bg->setProperty("class", "radius bg graye");
    }
    // 토글 버튼들
    updateToggleButtons(currentToggle);
    if (ui->toggle1) {
        ui->toggle1->setProperty("class", "toggle");  // 토글 버튼 오늘
    }
    if (ui->toggle2) {
        ui->toggle2->setProperty("class", "toggle active");  // 토글 버튼 주간
    }
    if (ui->toggle3) {
        ui->toggle3->setProperty("class", "toggle");  // 토글 버튼 월간
    }
    // 년/월/일 텍스트 라벨들
    if (ui->text_year1) { // start_date_year
        ui->text_year1->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_year2) { // end_date_year
        ui->text_year2->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_month1) { // start_date_month
        ui->text_month1->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_month2) { // end_date_month
        ui->text_month2->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_date1) { // start_date_day
        ui->text_date1->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_date2) { // end_date_day
        ui->text_date2->setProperty("class", "color-gray6 size12 weight300");
    }
    
    // 스핀박스들
    if (ui->year1) {
        ui->year1->setProperty("class", "spinbox small");  // 연도 스핀박스
    }
    if (ui->year2) {
        ui->year2->setProperty("class", "spinbox small");  // 연도 스핀박스
    }
    if (ui->month1) {
        ui->month1->setProperty("class", "spinbox small");  // 월 스핀박스
    }
    if (ui->month2) {
        ui->month2->setProperty("class", "spinbox small");  // 월 스핀박스
    }
    if (ui->date1) {
        ui->date1->setProperty("class", "spinbox small");  // 일 스핀박스
    }
    if (ui->date2) {
        ui->date2->setProperty("class", "spinbox small");  // 일 스핀박스
    }
    if (ui->range1) {
        ui->range1->setProperty("class", "align-center");  // 일 스핀박스
    }

    // 년/월/일 텍스트 라벨들
    if (ui->title_point1) {
        ui->title_point1->setProperty("class", "title_point");
    }
    if (ui->title_point2) {
        ui->title_point2->setProperty("class", "title_point");
    }

    // 버튼 스타일 추가
    if (ui->search_btn) {
        ui->search_btn->setProperty("class", "btn outlined primary weight700");  // 검색 버튼
    }

    // 테이블 스타일
    if (ui->robot_table) {
        ui->robot_table->setProperty("class", "table");  // 토글 버튼 월간
    }

    // 히트맵 스타일
    if (ui->heatmap_otl) {
        ui->heatmap_otl->setProperty("class", "radius otl-grayc");  // 히트맵 배경
    }
    if (ui->heatmap) {
        ui->heatmap->setProperty("class", "bg grayc");  // 히트맵
    }
    if( ui->heatmap_start) {
        ui->heatmap_start->setProperty("class", "size14 weight700 color-gray6");  // 히트맵 제목
    }
    if( ui->heatmap_end) {
        ui->heatmap_end->setProperty("class", "size14 weight700 color-gray6");  // 히트맵 제목
    }

}   

void LogWidget::setupTableData()
{
    if (ui->robot_table) {
        // 테이블 컬럼 설정
        ui->robot_table->setColumnCount(4);
        QStringList headers = {"환자 번호", "출발지", "도착지", "기록일시"};
        ui->robot_table->setHorizontalHeaderLabels(headers);
        
        // 테이블 설정
        ui->robot_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui->robot_table->setAlternatingRowColors(true);
        ui->robot_table->verticalHeader()->setVisible(false);
        
        // 헤더 설정 - 테이블이 전체 너비를 채우도록 설정
        QHeaderView* header = ui->robot_table->horizontalHeader();
        header->setStretchLastSection(false);  // 마지막 섹션 자동 늘리기 비활성화
        
        // 컬럼 너비 설정
        ui->robot_table->setColumnWidth(0, 130);   // 선택
        ui->robot_table->setColumnWidth(1, 150);  // 환자 번호
        ui->robot_table->setColumnWidth(2, 150);  // 촬영지
        ui->robot_table->setColumnWidth(3, 120);  // 도착지
        
        // 기록일시 컬럼(4번)을 남은 공간에 맞게 늘리기
        header->setSectionResizeMode(0, QHeaderView::Stretch);      // 선택
        header->setSectionResizeMode(1, QHeaderView::Stretch);      // 환자 번호
        header->setSectionResizeMode(2, QHeaderView::Stretch);      // 촬영지
        header->setSectionResizeMode(3, QHeaderView::Stretch);      // 도착지
        
        // 데이터 추가
        get_robot_log_data();
    }
}

void LogWidget::get_robot_log_data()
{
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/get/log_data")
                    .arg(CENTRAL_IP.c_str())
                    .arg(CENTRAL_HTTP_PORT);

    QJsonObject data;
    data["period"] = "month";  // 한 달치 데이터 요청
    data["start_date"] = QJsonValue::Null;
    data["end_date"] = QJsonValue::Null;

    QJsonDocument doc(data);
    QByteArray jsonData = doc.toJson();

    qDebug() << "[로그 데이터 요청 URL]:" << url;
    qDebug() << "[전송 데이터]:" << jsonData;
    
    try
    {
        QNetworkRequest request{QUrl(url)};
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        QNetworkReply* reply = manager->post(request, jsonData);

        connect(reply, &QNetworkReply::finished, this, [this, reply]() {
            int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            qDebug() << "[로그 데이터] HTTP 상태 코드:" << statusCode;
            
            QByteArray responseData = reply->readAll();
            qDebug() << "[로그 데이터] 응답 데이터 길이:" << responseData.length();
            qDebug() << "[로그 데이터] 응답 데이터 일부:" << responseData.left(500);
            
            if (statusCode == 200) {
                if (responseData.isEmpty()) {
                    qDebug() << "[로그 데이터] 응답이 비어있습니다";
                    populateRobotTable();  // 하드코딩된 데이터로 표시
                    reply->deleteLater();
                    return;
                }
                
                QJsonParseError parseError;
                QJsonDocument jsonDoc = QJsonDocument::fromJson(responseData, &parseError);
                
                if (parseError.error != QJsonParseError::NoError) {
                    qDebug() << "[로그 데이터] JSON 파싱 오류:" << parseError.errorString();
                    qDebug() << "[로그 데이터] 오류 위치:" << parseError.offset;
                    populateRobotTable();  // 하드코딩된 데이터로 표시
                    reply->deleteLater();
                    return;
                }
                
                // Check if response is an array
                if (jsonDoc.isArray()) {
                    QJsonArray jsonArray = jsonDoc.array();
                    qDebug() << "[로그 데이터] 배열 크기:" << jsonArray.size();
                    
                    logEntries_.clear();
                    
                    for (const QJsonValue& value : jsonArray) {
                        if (value.isObject()) {
                            QJsonObject obj = value.toObject();
                            LogData logData;
                            
                            // patient_id 처리
                            if (obj.contains("patient_id")) {
                                if (obj["patient_id"].isString()) {
                                    logData.patientId = obj["patient_id"].toString();
                                } else {
                                    logData.patientId = QString::number(obj["patient_id"].toInt());
                                }
                            } else {
                                logData.patientId = "0";
                            }
                            
                            // orig/dest를 부서 이름으로 매핑
                            int orig_id = obj["orig"].toInt();
                            int dest_id = obj["dest"].toInt();
                            logData.source = mapDepartmentIdToName(orig_id);
                            logData.destination = mapDepartmentIdToName(dest_id);
                            
                            // 날짜 형식 변환 (2025-08-01 20:30:33 -> 25.08.01 20:30:33)
                            QString datetime = obj["datetime"].toString();
                            if (datetime.length() >= 19) {
                                logData.timestamp = datetime.mid(2, 2) + "." + datetime.mid(5, 2) + "." + datetime.mid(8, 2) + " " + datetime.mid(11, 8);
                            } else {
                                logData.timestamp = datetime;
                            }
                            
                            logEntries_.append(logData);
                            
                            qDebug() << "[로그 데이터] 추가된 엔트리:" << logData.patientId << logData.source << logData.destination << logData.timestamp;
                        }
                    }
                    
                    qDebug() << "[로그 데이터] 처리 완료, 테이블 업데이트 시작";
                    populateRobotTable();
                    qDebug() << "[로그 데이터] 로드 완료:" << logEntries_.size() << "개 항목";
                } else if (jsonDoc.isObject()) {
                    qDebug() << "[로그 데이터] 응답이 객체 형태입니다. 배열이 아닙니다.";
                    QJsonObject obj = jsonDoc.object();
                    qDebug() << "[로그 데이터] 객체 내용:" << obj;
                    populateRobotTable();  // 하드코딩된 데이터로 표시
                } else {
                    qDebug() << "[로그 데이터] 응답이 배열도 객체도 아닙니다.";
                    populateRobotTable();  // 하드코딩된 데이터로 표시
                }
            } else {
                qDebug() << "[로그 데이터] 요청 실패. 상태 코드:" << statusCode;
                qDebug() << "[로그 데이터] 오류 응답:" << responseData;
                populateRobotTable();  // 하드코딩된 데이터로 표시
            }
            reply->deleteLater();
        });

    } catch (const YAML::BadFile& e) {
        qDebug() << "YAML 파일 로드 실패:" << e.what();
        populateRobotTable();  // 하드코딩된 데이터로 표시
        return;
    } catch (const std::exception& e) {
        qDebug() << "네트워크 요청 중 예외:" << e.what();
        populateRobotTable();  // 하드코딩된 데이터로 표시
        return;
    }
}

void LogWidget::get_robot_log_data_with_period(const QString& period)
{
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/get/log_data")
                    .arg(CENTRAL_IP.c_str())
                    .arg(CENTRAL_HTTP_PORT);

    QJsonObject data;
    
    // 토글 버튼이 아닌 경우, 실제 날짜 필드 값을 사용
    if (period.isEmpty()) {
        // 날짜 필드에서 start_date와 end_date 추출
        QString start_date = getDateFromFields(true);  // 시작 날짜
        QString end_date = getDateFromFields(false);   // 종료 날짜
        
        data["period"] = QJsonValue::Null;
        data["start_date"] = start_date;
        data["end_date"] = end_date;
        
        qDebug() << "[날짜 범위] 시작:" << start_date << ", 종료:" << end_date;
    } else {
        data["period"] = period;
        data["start_date"] = QJsonValue::Null;
        data["end_date"] = QJsonValue::Null;
    }

    QJsonDocument doc(data);
    QByteArray jsonData = doc.toJson();

    qDebug() << "[로그 데이터 요청 URL]:" << url;
    qDebug() << "[전송 데이터]:" << jsonData;
    
    try
    {
        QNetworkRequest request{QUrl(url)};
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        QNetworkReply* reply = manager->post(request, jsonData);

        connect(reply, &QNetworkReply::finished, this, [this, reply, period]() {
            int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            qDebug() << "[로그 데이터] HTTP 상태 코드:" << statusCode << "기간:" << period;
            
            QByteArray responseData = reply->readAll();
            qDebug() << "[로그 데이터] 응답 데이터 길이:" << responseData.length();
            
            if (statusCode == 200) {
                if (responseData.isEmpty()) {
                    qDebug() << "[로그 데이터] 응답이 비어있습니다";
                    logEntries_.clear();
                    populateRobotTable();
                    reply->deleteLater();
                    return;
                }
                
                QJsonParseError parseError;
                QJsonDocument jsonDoc = QJsonDocument::fromJson(responseData, &parseError);
                
                if (parseError.error != QJsonParseError::NoError) {
                    qDebug() << "[로그 데이터] JSON 파싱 오류:" << parseError.errorString();
                    logEntries_.clear();
                    populateRobotTable();
                    reply->deleteLater();
                    return;
                }
                
                // Check if response is an array
                if (jsonDoc.isArray()) {
                    QJsonArray jsonArray = jsonDoc.array();
                    qDebug() << "[로그 데이터] 배열 크기:" << jsonArray.size();
                    
                    logEntries_.clear();
                    
                    for (const QJsonValue& value : jsonArray) {
                        if (value.isObject()) {
                            QJsonObject obj = value.toObject();
                            LogData logData;
                            
                            // patient_id 처리
                            if (obj.contains("patient_id")) {
                                if (obj["patient_id"].isString()) {
                                    logData.patientId = obj["patient_id"].toString();
                                } else {
                                    logData.patientId = QString::number(obj["patient_id"].toInt());
                                }
                            } else {
                                logData.patientId = "0";
                            }
                            
                            // orig/dest를 부서 이름으로 매핑
                            int orig_id = obj["orig"].toInt();
                            int dest_id = obj["dest"].toInt();
                            logData.source = mapDepartmentIdToName(orig_id);
                            logData.destination = mapDepartmentIdToName(dest_id);
                            
                            // 날짜 형식 변환 (2025-08-01 20:30:33 -> 25.08.01 20:30:33)
                            QString datetime = obj["datetime"].toString();
                            if (datetime.length() >= 19) {
                                logData.timestamp = datetime.mid(2, 2) + "." + datetime.mid(5, 2) + "." + datetime.mid(8, 2) + " " + datetime.mid(11, 8);
                            } else {
                                logData.timestamp = datetime;
                            }
                            
                            logEntries_.append(logData);
                        }
                    }
                    
                    qDebug() << "[로그 데이터] 처리 완료, 테이블 업데이트 시작";
                    populateRobotTable();
                    qDebug() << "[로그 데이터] 로드 완료:" << logEntries_.size() << "개 항목";
                } else {
                    qDebug() << "[로그 데이터] 응답이 배열이 아닙니다.";
                    logEntries_.clear();
                    populateRobotTable();
                }
            } else {
                qDebug() << "[로그 데이터] 요청 실패. 상태 코드:" << statusCode;
                logEntries_.clear();
                populateRobotTable();
            }
            reply->deleteLater();
        });

    } catch (const std::exception& e) {
        qDebug() << "네트워크 요청 중 예외:" << e.what();
        logEntries_.clear();
        populateRobotTable();
        return;
    }
}

void LogWidget::setDateFields(const QDate& startDate, const QDate& endDate)
{
    // 시작 날짜 설정
    if (ui->year1) {
        ui->year1->setValue(startDate.year());
    }
    if (ui->month1) {
        ui->month1->setValue(startDate.month());
    }
    if (ui->date1) {
        ui->date1->setValue(startDate.day());
    }
    
    // 종료 날짜 설정
    if (ui->year2) {
        ui->year2->setValue(endDate.year());
    }
    if (ui->month2) {
        ui->month2->setValue(endDate.month());
    }
    if (ui->date2) {
        ui->date2->setValue(endDate.day());
    }
    
    qDebug() << "[날짜 설정] 시작:" << startDate.toString("yyyy-MM-dd") 
             << ", 종료:" << endDate.toString("yyyy-MM-dd");
}

QString LogWidget::getDateFromFields(bool isStartDate)
{
    int year, month, day;
    
    if (isStartDate) {
        // 시작 날짜 필드에서 값 가져오기
        year = ui->year1 ? ui->year1->value() : QDate::currentDate().year();
        month = ui->month1 ? ui->month1->value() : QDate::currentDate().month();
        day = ui->date1 ? ui->date1->value() : QDate::currentDate().day();
    } else {
        // 종료 날짜 필드에서 값 가져오기
        year = ui->year2 ? ui->year2->value() : QDate::currentDate().year();
        month = ui->month2 ? ui->month2->value() : QDate::currentDate().month();
        day = ui->date2 ? ui->date2->value() : QDate::currentDate().day();
    }
    
    QDate date(year, month, day);
    if (!date.isValid()) {
        date = QDate::currentDate();
    }
    
    return date.toString("yyyy-MM-dd");
}

void LogWidget::refresh()
{
    qDebug() << "[Refresh] 데이터 새로고침 시작, 현재 토글:" << currentToggle;
    
    // 현재 선택된 토글에 따라 다른 period 설정
    QString period;
    switch (currentToggle) {
        case 1:
            period = "today";
            qDebug() << "[Refresh] 오늘 데이터 로드";
            break;
        case 2:
            period = "week";
            qDebug() << "[Refresh] 주간 데이터 로드";
            break;
        case 3:
            period = "month";
            qDebug() << "[Refresh] 월간 데이터 로드";
            break;
        default:
            period = "month";
            break;
    }
    
    // 서버에서 새로운 데이터 요청
    get_robot_log_data_with_period(period);
}

// 부서 ID를 이름으로 매핑하는 함수 수정
QString LogWidget::mapDepartmentIdToName(int dept_id) {
    switch(dept_id) {
        case 0: return "CT 검사실";
        case 1: return "초음파 검사실";
        case 2: return "X-ray 검사실";
        case 3: return "대장암 센터";
        case 4: return "위암 센터";
        case 5: return "폐암 센터";
        case 6: return "뇌졸중 센터";
        case 7: return "유방암 센터";
        default: return "알 수 없음 (" + QString::number(dept_id) + ")";
    }
}

void LogWidget::populateRobotTable()
{
    if (!ui->robot_table) {
        qDebug() << "[테이블] robot_table이 null입니다";
        return;
    }
    
    qDebug() << "[테이블] populateRobotTable 시작, 엔트리 수:" << logEntries_.size();
    
    // 테이블 행 수 설정
    ui->robot_table->setRowCount(logEntries_.size());
    
    // 데이터 추가
    for (int row = 0; row < logEntries_.size(); ++row) {
        const LogData& entry = logEntries_[row];
        
        qDebug() << "[테이블] 행" << row << "추가:" << entry.patientId << entry.source << entry.destination;
        
        // 환자 번호
        QTableWidgetItem* patientItem = new QTableWidgetItem(entry.patientId);
        if (entry.patientId == "0") {
            patientItem->setText("관리자 제어");
        }
        patientItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 0, patientItem);
        
        // 출발지 라벨 생성
        QLabel* sourceLabel = new QLabel(entry.source);
        sourceLabel->setAlignment(Qt::AlignCenter);
        sourceLabel->setMinimumSize(100, 21);
        sourceLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

        if (entry.source.contains("검사실")) {
            sourceLabel->setProperty("class", "label error");
        } else if (entry.source.contains("센터")) {
            sourceLabel->setProperty("class", "label primary");
        } else {
            sourceLabel->setProperty("class", "label gray");
        }
        
        sourceLabel->style()->unpolish(sourceLabel);
        sourceLabel->style()->polish(sourceLabel);

        QHBoxLayout* sourceLayout = new QHBoxLayout();
        sourceLayout->addWidget(sourceLabel);
        sourceLayout->setAlignment(Qt::AlignCenter);
        sourceLayout->setContentsMargins(0, 0, 0, 0);

        QWidget* sourceWidget = new QWidget();
        sourceWidget->setLayout(sourceLayout);
        sourceWidget->setAttribute(Qt::WA_TranslucentBackground);
        ui->robot_table->setCellWidget(row, 1, sourceWidget);

        // 도착지 라벨 생성
        QLabel* destLabel = new QLabel(entry.destination);
        destLabel->setAlignment(Qt::AlignCenter);
        destLabel->setMinimumSize(100, 21);
        destLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

        if (entry.destination.contains("검사실")) {
            destLabel->setProperty("class", "label error");
        } else if (entry.destination.contains("센터")) {
            destLabel->setProperty("class", "label primary");
        } else {
            destLabel->setProperty("class", "label gray");
        }

        destLabel->style()->unpolish(destLabel);
        destLabel->style()->polish(destLabel);

        QHBoxLayout* destLayout = new QHBoxLayout();
        destLayout->addWidget(destLabel);
        destLayout->setAlignment(Qt::AlignCenter);
        destLayout->setContentsMargins(0, 0, 0, 0);

        QWidget* destWidget = new QWidget();
        destWidget->setLayout(destLayout);
        destWidget->setAttribute(Qt::WA_TranslucentBackground);
        ui->robot_table->setCellWidget(row, 2, destWidget);
        
        // 기록일시
        QTableWidgetItem* timeItem = new QTableWidgetItem(entry.timestamp);
        timeItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 3, timeItem);
    }
    
    // 테이블 강제 업데이트
    ui->robot_table->update();
    ui->robot_table->repaint();
    
    qDebug() << "[테이블] populateRobotTable 완료, 실제 행 수:" << ui->robot_table->rowCount();
}
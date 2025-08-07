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
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QFont>
#include <QTimer>
#include <QMessageBox>  // 추가

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
    
    // 초기 빈 차트 그리기
    QTimer::singleShot(100, this, [this]() {
        if (ui->chart1) {
            drawChart(ui->chart1, {0, 0, 0, 0}, {});
        }
        if (ui->chart2) {
            drawChart(ui->chart2, {0, 0, 0, 0}, {});
        }
    });
}

LogWidget::~LogWidget()
{
    delete ui;
}

void LogWidget::setupConnections()
{
    // 시그널-슬롯 연결 - 실제 UI 위젯 이름 사용
    if (ui->toggle1) {
        connect(ui->toggle1, &QPushButton::clicked, this, &LogWidget::onToggle1Clicked);
    }
    if (ui->toggle2) {
        connect(ui->toggle2, &QPushButton::clicked, this, &LogWidget::onToggle2Clicked);
    }
    if (ui->toggle3) {
        connect(ui->toggle3, &QPushButton::clicked, this, &LogWidget::onToggle3Clicked);
    }
    
    // 검색 버튼 연결
    if (ui->search_btn) {
        connect(ui->search_btn, &QPushButton::clicked, this, [this]() {
            get_robot_log_data_with_period("");  // 빈 period로 날짜 필드 사용
        });
    }
    
    // 테이블 더블 클릭 시 상세 정보 표시
    if (ui->robot_table) {
        connect(ui->robot_table, &QTableWidget::cellDoubleClicked, this, [this](int row, int column) {
            if (row < 0 || row >= logEntries_.size()) {
                return;
            }
            
            const LogData& entry = logEntries_[row];
            QString details = QString("환자 ID: %1\n출발지: %2\n도착지: %3\n일시: %4")
                                .arg(entry.patientId)
                                .arg(entry.source)
                                .arg(entry.destination)
                                .arg(entry.timestamp);
            
            QMessageBox::information(this, "상세 정보", details);
        });
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
    if (ui->title_point3) {
        ui->title_point3->setProperty("class", "title_point");
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
    if (ui->chart1_otl) {
        ui->chart1_otl->setProperty("class", "radius otl-grayc");  // 히트맵 배경
    }
    if (ui->chart2_otl) {
        ui->chart2_otl->setProperty("class", "radius otl-grayc");  // 히트맵 배경
    }
    if (ui->chart1_point1) {
        ui->chart1_point1->setProperty("class", "chart_point1");  // 히트맵 포인트
    }
    if (ui->chart1_point2) {
        ui->chart1_point2->setProperty("class", "chart_point2");  // 히트맵 포인트
    }
    if (ui->chart1_point3) {
        ui->chart1_point3->setProperty("class", "chart_point3");  // 히트맵 포인트
    }
    if (ui->chart1_point4) {
        ui->chart1_point4->setProperty("class", "chart_point4");  // 히트맵 포인트
    }
    if (ui->chart2_point1) {
        ui->chart2_point1->setProperty("class", "chart_point1");  // 히트맵 포인트
    }
    if (ui->chart2_point2) {
        ui->chart2_point2->setProperty("class", "chart_point2");  // 히트맵 포인트
    }
    if (ui->chart2_point3) {
        ui->chart2_point3->setProperty("class", "chart_point3");  // 히트맵 포인트
    }
    if (ui->chart2_point4) {
        ui->chart2_point4->setProperty("class", "chart_point4");  // 히트맵 포인트
    }
    if (ui->chart1_label1) {
        ui->chart1_label1->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    if (ui->chart1_label2) {
        ui->chart1_label2->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    if (ui->chart1_label3) {
        ui->chart1_label3->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    if (ui->chart1_label4) {
        ui->chart1_label4->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    if (ui->chart2_label1) {
        ui->chart2_label1->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    if (ui->chart2_label2) {
        ui->chart2_label2->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    if (ui->chart2_label3) {
        ui->chart2_label3->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    if (ui->chart2_label4) {
        ui->chart2_label4->setProperty("class", "size12 weight700");  // 히트맵 배경
    }
    // if (ui->chart1) {
    //     ui->chart1->setProperty("class", "radius bg grayc");  // 히트맵 차트
    // }
    // if (ui->chart2) {   
    //     ui->chart2->setProperty("class", "radius bg grayc");  //
    // }
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
        case 8: return "로비";  // 로비 추가
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
        QLabel* patientIdLabel = new QLabel(entry.patientId);
        patientIdLabel->setAlignment(Qt::AlignCenter);
        patientIdLabel->setMinimumSize(100, 21);
        patientIdLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

        if (entry.patientId == "0") {
            patientIdLabel->setProperty("class", "label gray");
            patientIdLabel->setText("관리자 제어");
        } else {
            patientIdLabel->setProperty("class", "label primary");
        }

        patientIdLabel->style()->unpolish(patientIdLabel);
        patientIdLabel->style()->polish(patientIdLabel);

        QHBoxLayout* patientIdLayout = new QHBoxLayout();
        patientIdLayout->addWidget(patientIdLabel);
        patientIdLayout->setAlignment(Qt::AlignCenter);
        patientIdLayout->setContentsMargins(0, 0, 0, 0);

        QWidget* patientIdWidget = new QWidget();
        patientIdWidget->setLayout(patientIdLayout);
        patientIdWidget->setAttribute(Qt::WA_TranslucentBackground);
        ui->robot_table->setCellWidget(row, 0, patientIdWidget);

        // QTableWidgetItem* patientItem = new QTableWidgetItem(entry.patientId);
        // if (entry.patientId == "0") {
        //     patientItem->setText("관리자 제어");
        // }
        // patientItem->setTextAlignment(Qt::AlignCenter);
        // ui->robot_table->setItem(row, 0, patientItem);
        
        // 출발지 라벨 생성
        QTableWidgetItem* sourceItem = new QTableWidgetItem(entry.source);
        sourceItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 1, sourceItem);

        // 도착지 라벨 생성
        QTableWidgetItem* destItem = new QTableWidgetItem(entry.destination);
        destItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 2, destItem);

        // 기록일시
        QTableWidgetItem* timeItem = new QTableWidgetItem(entry.timestamp);
        timeItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 3, timeItem);
    }
    
    // 테이블 강제 업데이트
    ui->robot_table->update();
    ui->robot_table->repaint();
    
    // 테이블 데이터가 업데이트된 후 차트도 업데이트
    populateChartTable();
    
    qDebug() << "[테이블] populateRobotTable 완료, 실제 행 수:" << ui->robot_table->rowCount();
}

void LogWidget::populateChartTable()
{
    if (logEntries_.isEmpty()) {
        qDebug() << "[차트 데이터] 로그 엔트리가 비어있습니다";
        return;
    }

    // 출발지별 카운트 집계
    QMap<QString, int> sourceCountMap;
    for (const LogData& entry : logEntries_) {
        if (!entry.source.isEmpty()) {  // 빈 문자열 체크 추가
            sourceCountMap[entry.source]++;  // += 1 대신 ++ 사용
        }
    }

    if (sourceCountMap.isEmpty()) {
        qDebug() << "[차트 데이터] 유효한 출발지 데이터가 없습니다";
        return;
    }

    // 카운트 순으로 정렬 (내림차순)
    QList<QPair<QString, int>> sortedSources;
    for (auto it = sourceCountMap.constBegin(); it != sourceCountMap.constEnd(); ++it) {
        sortedSources.append(qMakePair(it.key(), it.value()));
    }

    // 카운트 기준으로 정렬 (많은 순서대로)
    std::sort(sortedSources.begin(), sortedSources.end(), 
              [](const QPair<QString, int>& a, const QPair<QString, int>& b) {
                  return a.second > b.second;  // 카운트가 높은 순서
              });

    // 정렬된 데이터로 배열 생성
    QStringList sourceNames;
    QList<int> sourceCounts;
    for (const auto& pair : sortedSources) {
        sourceNames << pair.first;
        sourceCounts << pair.second;
    }

    // 차트 업데이트 (상위 4개만 사용)
    updateChartLabels(sourceNames, sourceCounts);

    // 디버깅 출력
    qDebug() << "[차트 데이터] 총 출발지 종류:" << sourceCountMap.size();
    qDebug() << "[차트 데이터] 총 로그 수:" << logEntries_.size();
    qDebug() << "[차트 데이터] 출발지별 카운트 (정렬됨):" << sourceNames << sourceCounts;
}

void LogWidget::updateChartLabels(const QStringList& sourceNames, const QList<int>& sourceCounts)
{
    // Chart 1 (출발지 통계) 업데이트 - 상위 4개
    QList<QLabel*> chart1Labels = {ui->chart1_label1, ui->chart1_label2, ui->chart1_label3, ui->chart1_label4};
    
    QList<int> chart1Counts;
    for (int i = 0; i < chart1Labels.size(); ++i) {
        if (chart1Labels[i]) {
            if (i < sourceNames.size()) {
                QString labelText = QString("%1 (%2)").arg(sourceNames[i]).arg(sourceCounts[i]);
                chart1Labels[i]->setText(labelText);
                chart1Counts.append(sourceCounts[i]);
            } else {
                chart1Labels[i]->setText("없음 (0)");
                chart1Counts.append(0);
            }
        } else {
            chart1Counts.append(0);
        }
    }

    // 목적지별 카운트도 계산
    QMap<QString, int> destCountMap;
    for (const LogData& entry : logEntries_) {
        if (!entry.destination.isEmpty()) {
            destCountMap[entry.destination]++;
        }
    }

    // 목적지도 정렬
    QList<QPair<QString, int>> sortedDests;
    for (auto it = destCountMap.constBegin(); it != destCountMap.constEnd(); ++it) {
        sortedDests.append(qMakePair(it.key(), it.value()));
    }

    std::sort(sortedDests.begin(), sortedDests.end(), 
              [](const QPair<QString, int>& a, const QPair<QString, int>& b) {
                  return a.second > b.second;
              });

    // Chart 2 (목적지 통계) 업데이트 - 상위 4개
    QList<QLabel*> chart2Labels = {ui->chart2_label1, ui->chart2_label2, ui->chart2_label3, ui->chart2_label4};
    
    QList<int> chart2Counts;
    for (int i = 0; i < chart2Labels.size(); ++i) {
        if (chart2Labels[i]) {
            if (i < sortedDests.size()) {
                QString labelText = QString("%1 (%2)").arg(sortedDests[i].first).arg(sortedDests[i].second);
                chart2Labels[i]->setText(labelText);
                chart2Counts.append(sortedDests[i].second);
            } else {
                chart2Labels[i]->setText("N/A (0)");
                chart2Counts.append(0);
            }
        } else {
            chart2Counts.append(0);
        }
    }

    // 실제 차트 그리기
    drawChart(ui->chart1, chart1Counts, sourceNames.mid(0, 4));
    drawChart(ui->chart2, chart2Counts, QStringList());

    qDebug() << "[차트 업데이트] Chart1(출발지), Chart2(목적지) 라벨 및 차트 업데이트 완료";
}

void LogWidget::drawChart(QLabel* chartLabel, const QList<int>& data, const QStringList& labels)
{
    if (!chartLabel || data.isEmpty()) {
        return;
    }

    // 차트 크기
    int width = chartLabel->width();
    int height = chartLabel->height();
    
    if (width <= 0 || height <= 0) {
        // 초기 크기가 설정되지 않은 경우 기본 크기 사용
        width = 400;
        height = 300;
    }

    // 픽스맵 생성
    QPixmap pixmap(width, height);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);

    // 차트 영역 설정 (여백 포함)
    int margin = 40;
    QRect chartRect(margin, margin, width - 2 * margin, height - 2 * margin);

    // 최대값 계산
    int maxValue = 0;
    for (int value : data) {
        if (value > maxValue) {
            maxValue = value;
        }
    }

    if (maxValue == 0) {
        maxValue = 1; // 0으로 나누기 방지
    }

    // 바 차트 그리기
    int barCount = data.size();
    int barWidth = chartRect.width() / (barCount * 2); // 간격 고려
    int barSpacing = barWidth;

    // 색상 배열
    QList<QColor> colors = {
        QColor("#09C7C7"),  // 파란색
        QColor("#FFB14E"),  // 초록색
        QColor("#4AA8F1"),  // 주황색
        QColor("#F77373")   // 빨간색
    };

    for (int i = 0; i < barCount && i < data.size(); ++i) {
        int barHeight = (data[i] * chartRect.height()) / maxValue;
        int x = chartRect.left() + i * (barWidth + barSpacing) + barSpacing / 2;
        int y = chartRect.bottom() - barHeight;

        // 바 그리기
        QColor barColor = colors[i % colors.size()];
        painter.fillRect(x, y, barWidth, barHeight, barColor);

        // 바 테두리
        // painter.setPen(QPen(barColor.darker(), 2));
        // painter.drawRect(x, y, barWidth, barHeight);

        // 값 표시
        painter.setPen(QPen(Qt::black, 1));
        painter.setFont(QFont("Arial", 10, QFont::Bold));
        QRect textRect(x, y - 20, barWidth, 15);
        painter.drawText(textRect, Qt::AlignCenter, QString::number(data[i]));
    }

    // 축 그리기
    painter.setPen(QPen(Qt::gray, 1));
    
    // X축
    painter.drawLine(chartRect.bottomLeft(), chartRect.bottomRight());
    
    // Y축
    painter.drawLine(chartRect.bottomLeft(), chartRect.topLeft());

    // Y축 눈금 그리기
    int ySteps = 5;
    for (int i = 0; i <= ySteps; ++i) {
        int y = chartRect.bottom() - (i * chartRect.height() / ySteps);
        int value = (i * maxValue / ySteps);
        
        // 눈금선
        painter.drawLine(chartRect.left() - 5, y, chartRect.left(), y);
        
        // 값 표시
        painter.setFont(QFont("Arial", 8));
        QRect labelRect(0, y - 7, margin - 10, 14);
        painter.drawText(labelRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(value));
    }

    // 제목 그리기
    // painter.setPen(QPen(Qt::black, 1));
    // painter.setFont(QFont("Arial", 12, QFont::Bold));
    // QString title = (chartLabel == ui->chart1) ? "출발지별 통계" : "목적지별 통계";
    // QRect titleRect(0, 5, width, 20);
    // painter.drawText(titleRect, Qt::AlignCenter, title);

    // 픽스맵을 라벨에 설정
    chartLabel->setPixmap(pixmap);
    chartLabel->setScaledContents(true);
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
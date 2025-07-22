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

LogWidget::LogWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_LogWidget)
    , currentToggle(2)  // 초기값: 주간 선택
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
    setupConnections();
    setupTableData();
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
    
    // 기타 버튼 연결 (선택사항)
    if (ui->search_btn) {
        connect(ui->search_btn, &QPushButton::clicked, [this]() {
            qDebug() << "Search button clicked";
            // 검색 로직 구현
        });
    }
}

void LogWidget::onToggle1Clicked()
{
    qDebug() << "Toggle1 (오늘) clicked";
    currentToggle = 1;
    updateToggleButtons(1);
    
    // 오늘 데이터 로드 로직
    // 예: 오늘 날짜의 로그만 표시
    refresh();
}

void LogWidget::onToggle2Clicked()
{
    qDebug() << "Toggle2 (주간) clicked";
    currentToggle = 2;
    updateToggleButtons(2);
    
    // 주간 데이터 로드 로직
    // 예: 최근 7일 로그 표시
    refresh();
}

void LogWidget::onToggle3Clicked()
{
    qDebug() << "Toggle3 (월간) clicked";
    currentToggle = 3;
    updateToggleButtons(3);
    
    // 월간 데이터 로드 로직
    // 예: 이번 달 로그 표시
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
    if (ui->text_year1) {
        ui->text_year1->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_year2) {
        ui->text_year2->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_month1) {
        ui->text_month1->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_month2) {
        ui->text_month2->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_date1) {
        ui->text_date1->setProperty("class", "color-gray6 size12 weight300");
    }
    if (ui->text_date2) {
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
    if (ui->delete_btn1) {
        ui->delete_btn1->setProperty("class", "btn outlined gray3 weight700");  // 삭제 버튼
    }
    if (ui->delete_btn2) {
        ui->delete_btn2->setProperty("class", "btn outlined gray3 weight700");  // 삭제 버튼
    }

    // 테이블 스타일
    if (ui->robot_table) {
        ui->robot_table->setProperty("class", "table");  // 토글 버튼 월간
    }
    if (ui->miss_table) {
        ui->miss_table->setProperty("class", "table");  // 토글 버튼 월간
    }

}   

void LogWidget::setupTableData()
{
    if (ui->robot_table) {
        // 테이블 컬럼 설정
        ui->robot_table->setColumnCount(6);
        QStringList headers = {"선택", "환자 번호", "촬영지", "도착지", "기록일시", "상세"};
        ui->robot_table->setHorizontalHeaderLabels(headers);
        
        // 테이블 설정
        ui->robot_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui->robot_table->setAlternatingRowColors(true);
        ui->robot_table->verticalHeader()->setVisible(false);
        
        // 헤더 설정 - 테이블이 전체 너비를 채우도록 설정
        QHeaderView* header = ui->robot_table->horizontalHeader();
        header->setStretchLastSection(false);  // 마지막 섹션 자동 늘리기 비활성화
        
        // 컬럼 너비 설정
        ui->robot_table->setColumnWidth(0, 50);   // 선택
        ui->robot_table->setColumnWidth(1, 120);  // 환자 번호
        ui->robot_table->setColumnWidth(2, 120);  // 촬영지
        ui->robot_table->setColumnWidth(3, 120);  // 도착지
        ui->robot_table->setColumnWidth(5, 50);   // 상세
        
        // 기록일시 컬럼(4번)을 남은 공간에 맞게 늘리기
        header->setSectionResizeMode(0, QHeaderView::Fixed);      // 선택
        header->setSectionResizeMode(1, QHeaderView::Fixed);      // 환자 번호
        header->setSectionResizeMode(2, QHeaderView::Fixed);      // 촬영지
        header->setSectionResizeMode(3, QHeaderView::Fixed);      // 도착지
        header->setSectionResizeMode(4, QHeaderView::Stretch);    // 기록일시 - 남은 공간 채우기
        header->setSectionResizeMode(5, QHeaderView::Fixed);      // 상세
        
        // 데이터 추가
        populateRobotTable();
    }
    
    if (ui->miss_table) {
        // 테이블 컬럼 설정
        ui->miss_table->setColumnCount(4);
        QStringList missHeaders = {"선택", "오류 원인", "기록일시", "삭제"};
        ui->miss_table->setHorizontalHeaderLabels(missHeaders);
        
        // 테이블 설정
        ui->miss_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui->miss_table->setAlternatingRowColors(true);
        ui->miss_table->verticalHeader()->setVisible(false);
        
        // 헤더 설정 - 테이블이 전체 너비를 채우도록 설정
        QHeaderView* missHeader = ui->miss_table->horizontalHeader();
        missHeader->setStretchLastSection(false);  // 마지막 섹션 자동 늘리기 비활성화
        
        // 컬럼 너비 설정
        ui->miss_table->setColumnWidth(0, 50);   // 선택
        ui->miss_table->setColumnWidth(1, 130);  // 오류 원인
        ui->miss_table->setColumnWidth(3, 50);   // 삭제
        
        // 기록일시 컬럼(2번)을 남은 공간에 맞게 늘리기
        missHeader->setSectionResizeMode(0, QHeaderView::Fixed);      // 선택
        missHeader->setSectionResizeMode(1, QHeaderView::Fixed);      // 오류 원인
        missHeader->setSectionResizeMode(2, QHeaderView::Stretch);    // 기록일시 - 남은 공간 채우기
        missHeader->setSectionResizeMode(3, QHeaderView::Fixed);      // 삭제
        
        // 데이터 추가
        populateMissTable();
    }
}

void LogWidget::populateRobotTable()
{
    if (!ui->robot_table) return;
    
    // 샘플 데이터 - 이미지와 동일한 데이터
    struct LogData {
        QString patientId;
        QString source;
        QString destination;
        QString timestamp;
    };
    
    QVector<LogData> logEntries = {
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
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "수술장 검사실", "대장암 센터", "25.07.15 15:30:00"},
    };
    
    // 테이블 행 수 설정
    ui->robot_table->setRowCount(logEntries.size());
    
    // 데이터 추가
    for (int row = 0; row < logEntries.size(); ++row) {
        const LogData& entry = logEntries[row];
        
        // 체크박스 (선택 컬럼)
        QCheckBox* checkBox = new QCheckBox();
        checkBox->setChecked(false);
        checkBox->setProperty("class", "checkbox");
        checkBox->setStyleSheet("background-color: transparent;");  // 체크박스 배경 투명하게 설정
        
        // 체크박스를 가운데 정렬하기 위한 레이아웃 설정
        QHBoxLayout* checkboxLayout = new QHBoxLayout();
        checkboxLayout->addWidget(checkBox);
        checkboxLayout->setAlignment(Qt::AlignCenter);  // 가운데 정렬
        checkboxLayout->setContentsMargins(0, 0, 0, 0);  // 여백 제거
        
        // 체크박스 셀에 레이아웃 설정
        QWidget* checkboxWidget = new QWidget();
        checkboxWidget->setLayout(checkboxLayout);
        checkboxWidget->setAttribute(Qt::WA_TranslucentBackground);  // 투명 배경 속성 설정
        ui->robot_table->setCellWidget(row, 0, checkboxWidget);
        
        // 환자 번호
        QTableWidgetItem* patientItem = new QTableWidgetItem(entry.patientId);
        patientItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 1, patientItem);
        
        // 촬영지
        QTableWidgetItem* sourceItem = new QTableWidgetItem(entry.source);
        sourceItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 2, sourceItem);
        
        // 도착지
        QTableWidgetItem* destItem = new QTableWidgetItem(entry.destination);
        destItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 3, destItem);
        
        // 기록일시
        QTableWidgetItem* timeItem = new QTableWidgetItem(entry.timestamp);
        timeItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 4, timeItem);
        
        // 상세 버튼
        QPushButton* deleteBtn = new QPushButton();
        deleteBtn->setFixedSize(22, 22);
        deleteBtn->setProperty("class", "btn delete");
        
        connect(deleteBtn, &QPushButton::clicked, [this, row]() {
            qDebug() << "Detail button clicked for row:" << row;
            // 상세 정보 표시 또는 삭제 로직
        });
        
        // 버튼을 가운데 정렬하기 위한 레이아웃 설정
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        buttonLayout->addWidget(deleteBtn);
        buttonLayout->setAlignment(Qt::AlignCenter);  // 가운데 정렬
        buttonLayout->setContentsMargins(0, 0, 0, 0);  // 여백 제거
        
        // 셀에 레이아웃 설정
        QWidget* cellWidget = new QWidget();
        cellWidget->setLayout(buttonLayout);
        cellWidget->setAttribute(Qt::WA_TranslucentBackground);  // 투명 배경 속성 설정
        ui->robot_table->setCellWidget(row, 5, cellWidget);
    }
    
    qDebug() << "Robot table populated with" << logEntries.size() << "entries";
}

void LogWidget::populateMissTable()
{
    if (!ui->miss_table) return;
    
    // 샘플 데이터 - 미스 테이블용 데이터
    struct MissData {
        QString errorCause;
        QString timestamp;
    };
    
    QVector<MissData> missEntries = {
        {"통신 끊김", "25.07.15 14:20:00"},
        {"고장", "25.07.15 14:35:00"},
        {"전력 부족", "25.07.15 15:10:00"},
        {"통신 끊김", "25.07.15 15:45:00"},
        {"충돌", "25.07.15 16:00:00"},
        {"통신 끊김", "25.07.15 14:20:00"},
        {"고장", "25.07.15 14:35:00"},
        {"전력 부족", "25.07.15 15:10:00"},
        {"통신 끊김", "25.07.15 15:45:00"},
        {"충돌", "25.07.15 16:00:00"},
        {"통신 끊김", "25.07.15 14:20:00"},
        {"고장", "25.07.15 14:35:00"},
        {"전력 부족", "25.07.15 15:10:00"},
        {"통신 끊김", "25.07.15 15:45:00"},
        {"충돌", "25.07.15 16:00:00"},
        {"통신 끊김", "25.07.15 14:20:00"},
        {"고장", "25.07.15 14:35:00"},
        {"전력 부족", "25.07.15 15:10:00"},
        {"통신 끊김", "25.07.15 15:45:00"},
        {"충돌", "25.07.15 16:00:00"},
        {"통신 끊김", "25.07.15 14:20:00"},
        {"고장", "25.07.15 14:35:00"},
        {"전력 부족", "25.07.15 15:10:00"},
        {"통신 끊김", "25.07.15 15:45:00"},
        {"충돌", "25.07.15 16:00:00"},
    };
    
    // 테이블 행 수 설정
    ui->miss_table->setRowCount(missEntries.size());
    
    // 데이터 추가
    for (int row = 0; row < missEntries.size(); ++row) {
        const MissData& entry = missEntries[row];
        
        // 체크박스 (선택 컬럼 - 0번)
        QCheckBox* checkBox = new QCheckBox();
        checkBox->setChecked(false);
        checkBox->setProperty("class", "checkbox");
        checkBox->setStyleSheet("background-color: transparent;");
        
        QHBoxLayout* checkboxLayout = new QHBoxLayout();
        checkboxLayout->addWidget(checkBox);
        checkboxLayout->setAlignment(Qt::AlignCenter);
        checkboxLayout->setContentsMargins(0, 0, 0, 0);
        
        QWidget* checkboxWidget = new QWidget();
        checkboxWidget->setLayout(checkboxLayout);
        checkboxWidget->setAttribute(Qt::WA_TranslucentBackground);
        ui->miss_table->setCellWidget(row, 0, checkboxWidget);
        
        // 오류 원인 (1번)
        QLabel* errorLabel = new QLabel(entry.errorCause);
        errorLabel->setAlignment(Qt::AlignCenter);
        errorLabel->setMinimumSize(80, 21);  // 최소 크기 설정
        errorLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);  // 크기 정책 설정
        
        // "고장"일 때 label primary 클래스 적용
        if (entry.errorCause == "고장") {
            errorLabel->setProperty("class", "label primary");
        } else if (entry.errorCause == "충돌") {
            errorLabel->setProperty("class", "label error");
        } else if (entry.errorCause == "통신 끊김") {
            errorLabel->setProperty("class", "label therity");
        } else if (entry.errorCause == "전력 부족") {
            errorLabel->setProperty("class", "label secondary");
        } else {
            errorLabel->setProperty("class", "label gray");
        }
        
        // 스타일 강제 적용
        errorLabel->style()->unpolish(errorLabel);
        errorLabel->style()->polish(errorLabel); 
        
        // 오류 원인을 가운데 정렬하기 위한 레이아웃 설정
        QHBoxLayout* errorLayout = new QHBoxLayout();
        errorLayout->addWidget(errorLabel);
        errorLayout->setAlignment(Qt::AlignCenter);
        errorLayout->setContentsMargins(0, 0, 0, 0);
        
        QWidget* errorWidget = new QWidget();
        errorWidget->setLayout(errorLayout);
        errorWidget->setAttribute(Qt::WA_TranslucentBackground);
        ui->miss_table->setCellWidget(row, 1, errorWidget);
        
        // 기록일시 (2번)
        QTableWidgetItem* timeItem = new QTableWidgetItem(entry.timestamp);
        timeItem->setTextAlignment(Qt::AlignCenter);
        ui->miss_table->setItem(row, 2, timeItem);
        
        // 삭제 버튼 (3번)
        QPushButton* deleteBtn = new QPushButton();
        deleteBtn->setFixedSize(22, 22);
        deleteBtn->setProperty("class", "btn delete");
        
        connect(deleteBtn, &QPushButton::clicked, [this, row]() {
            qDebug() << "Miss table delete button clicked for row:" << row;
            // 삭제 로직
        });
        
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        buttonLayout->addWidget(deleteBtn);
        buttonLayout->setAlignment(Qt::AlignCenter);
        buttonLayout->setContentsMargins(0, 0, 0, 0);
        
        QWidget* cellWidget = new QWidget();
        cellWidget->setLayout(buttonLayout);
        cellWidget->setAttribute(Qt::WA_TranslucentBackground);
        ui->miss_table->setCellWidget(row, 3, cellWidget);
    }
    
    qDebug() << "Miss table populated with" << missEntries.size() << "entries";
}

void LogWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void LogWidget::refresh()
{
    qDebug() << "Log widget refresh";
    
    // 현재 선택된 토글에 따라 다른 데이터 로드
    switch (currentToggle) {
        case 1:
            qDebug() << "Loading today's logs";
            // 오늘 로그 로드
            if (ui->robot_table) {
                // 오늘의 로봇 로그 데이터 설정
            }
            if (ui->miss_table) {
                // 오늘의 미스 로그 데이터 설정
            }
            break;
        case 2:
            qDebug() << "Loading weekly logs";
            // 주간 로그 로드
            if (ui->robot_table) {
                // 주간 로봇 로그 데이터 설정
            }
            if (ui->miss_table) {
                // 주간 미스 로그 데이터 설정
            }
            break;
        case 3:
            qDebug() << "Loading monthly logs";
            // 월간 로그 로드
            if (ui->robot_table) {
                // 월간 로봇 로그 데이터 설정
            }
            if (ui->miss_table) {
                // 월간 미스 로그 데이터 설정
            }
            break;
    }
    populateRobotTable();
    populateMissTable();
}
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

namespace plt = matplotlibcpp;

LogWidget::LogWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_LogWidget)
    , currentToggle(2)  // 초기값: 주간 선택
    , logEntries()  // 로그 엔트리 초기화
    , heatmapEntries()  // 히트맵 엔트리 초기화
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
        populateRobotTable();
        populateHeatmap();
    }
    
}

void LogWidget::populateRobotTable()
{
    if (!ui->robot_table) return;
    
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
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
        {"00 00 00 00", "초음파 검사실", "대장암 센터", "25.07.15 15:30:00"},
    };
    
    // 테이블 행 수 설정
    ui->robot_table->setRowCount(logEntries.size());
    
    // 데이터 추가
    for (int row = 0; row < logEntries.size(); ++row) {
        const LogData& entry = logEntries[row];
        
        // 환자 번호
        QTableWidgetItem* patientItem = new QTableWidgetItem(entry.patientId);
        patientItem->setTextAlignment(Qt::AlignCenter);
        ui->robot_table->setItem(row, 0, patientItem);
        
        // 촬영지
        // QTableWidgetItem* sourceItem = new QTableWidgetItem(entry.source);
        // sourceItem->setTextAlignment(Qt::AlignCenter);
        // ui->robot_table->setItem(row, 1, sourceItem);
        
        QLabel* sourceLabel = new QLabel(entry.source);
        sourceLabel->setAlignment(Qt::AlignCenter);
        sourceLabel->setMinimumSize(100, 21);  // 최소 크기 설정
        sourceLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);  // 크기 정책 설정

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

        // 도착지
        // QTableWidgetItem* destItem = new QTableWidgetItem(entry.destination);
        // destItem->setTextAlignment(Qt::AlignCenter);
        // ui->robot_table->setItem(row, 2, destItem);

        QLabel* destLabel = new QLabel(entry.destination);
        destLabel->setAlignment(Qt::AlignCenter);
        destLabel->setMinimumSize(100, 21);  // 최소 크기 설정
        destLabel->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);  // 크기 정책 설정

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
}


void LogWidget::populateHeatmap()
{
    if (!ui->heatmap) return;

    plt::backend("Agg");

    heatmapEntries = {
        {0, 4, 2, 0, 0, 0, 1, 0},
        {2, 0, 3, 0, 1, 0, 0, 0},
        {1, 1, 0, 0, 0, 2, 0, 0},
        {0, 0, 0, 0, 5, 0, 1, 0},
        {0, 0, 0, 3, 0, 2, 0, 1},
        {0, 0, 1, 0, 1, 0, 4, 0},
        {2, 0, 0, 0, 0, 1, 0, 2},
        {0, 0, 0, 0, 0, 0, 1, 0}
    };

    plt::detail::_interpreter::get();  // 인터프리터 초기화
    PyRun_SimpleString(R"(
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list("custom", ["#e5f5f5", "#009999"])
plt.register_cmap(name="custom", cmap=custom_cmap)
)");

    int rows = heatmapEntries.size();
    int cols = heatmapEntries[0].size();
    std::vector<float> flat;
    for (const auto& row : heatmapEntries)
        for (int v : row)
            flat.push_back(static_cast<float>(v));

    std::vector<int> x_indices(cols), y_indices(rows);
    for (int i = 0; i < cols; ++i) x_indices[i] = i;
    for (int i = 0; i < rows; ++i) y_indices[i] = i;

    plt::figure_size(455, 626);
    plt::clf();
    plt::imshow(flat.data(), rows, cols, 1, {
        {"cmap", "custom"}, // 혹은 "BuGn", "Greens" 등
        {"interpolation", "nearest"},
        {"aspect", "auto"}
    });

    // 라벨
    // plt::xticks(x_indices, labels, {{"fontsize", "14"}, {"rotation", "30"}});
    // plt::yticks(y_indices, labels, {{"fontsize", "14"}});
    plt::xticks(x_indices);
    plt::yticks(y_indices);
    plt::margins(0, 0);
    plt::subplots_adjust({{"left", 0.0}, {"right", 1.0}, {"top", 1.0}, {"bottom", 0.0}});

    PyRun_SimpleString(R"(
import matplotlib.pyplot as plt
import numpy as np
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)
rows, cols = plt.gci().get_array().shape
ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
ax.grid(which='minor', color='#fff', linewidth=1)

# 셀 값 표시
data = plt.gci().get_array()
data = plt.gci().get_array()
for i in range(rows):
    for j in range(cols):
        ax.text(j, i, int(data[i, j]), ha='center', va='center', color='#222222', fontsize=12)
)");
    std::string filename = "/tmp/heatmap.png";
    try {
        plt::save(filename);
        qDebug() << "Saved heatmap to" << QString::fromStdString(filename);
    } catch (const std::exception& e) {
        qDebug() << "matplotlib-cpp exception:" << e.what();
    }
    plt::close();

    QPixmap pixmap(QString::fromStdString(filename));
    if (!pixmap.isNull()) {
        ui->heatmap->setPixmap(pixmap.scaled(ui->heatmap->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void LogWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void LogWidget::refresh()
{
    // 현재 선택된 토글에 따라 다른 데이터 로드
    switch (currentToggle) {
        case 1:
            // 오늘 로그 로드
            if (ui->robot_table) {
                // 오늘의 로봇 로그 데이터 설정
            }
            break;
        case 2:
            // 주간 로그 로드
            if (ui->robot_table) {
                // 주간 로봇 로그 데이터 설정
            }
            break;
        case 3:
            // 월간 로그 로드
            if (ui->robot_table) {
                // 월간 로봇 로그 데이터 설정
            }
            break;
    }
    populateRobotTable();
}
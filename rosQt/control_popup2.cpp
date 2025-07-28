#include "control_popup2.h"
#include "ui_control_popup2.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QWidget>
#include <QDebug>
#include <QStyle>

ControlPopup2::ControlPopup2(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui_ControlPopup2)
    , current_status_("대기중")
    , selected_button_(nullptr) 
{
    ui->setupUi(this);
    setWindowProperties();
    setupConnections();
    setWidgetClasses();
}

ControlPopup2::~ControlPopup2()
{
    delete ui;
}

void ControlPopup2::setCurrentStatus(const QString& status)
{
    current_status_ = status;
    qDebug() << "ControlPopup2 상태 설정:" << status;
}

void ControlPopup2::setWindowProperties()
{
    // 창 속성 설정
    setWindowTitle("로봇 제어");
    setFixedSize(800, 414);  // 고정 크기
    
    // 창을 화면 중앙에 위치
    if (parentWidget()) {
        QPoint parentCenter = parentWidget()->geometry().center();
        move(130, 193);
    }
}

void ControlPopup2::setWidgetClasses()
{
    if (ui->title) {
        ui->title->setProperty("class", "size24 weight700 color-gray3");
    }
    if (ui->line) {
        ui->line->setProperty("class", "bg grayc");
    }
    if (ui->closeBtn) {
        ui->closeBtn->setProperty("class", "btn outlined gray6 size20");
    }
    if (ui->startBtn) {
        ui->startBtn->setProperty("class", "btn contained primary size20");
    }
    if (ui->btn1) {
        ui->btn1->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn2) {
        ui->btn2->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn3) {
        ui->btn3->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn4) {
        ui->btn4->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn5) {
        ui->btn5->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn6) {
        ui->btn6->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn7) {
        ui->btn7->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn8) {
        ui->btn8->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->bg) {
        ui->bg->setProperty("class", "bg white radius otl-gray6");
    }
}

void ControlPopup2::setupConnections()
{
    // 닫기 버튼 연결 (UI 파일에 closeBtn이 있다고 가정)
    if (ui->closeBtn) {
        connect(ui->closeBtn, &QPushButton::clicked, this, &ControlPopup2::onCloseButtonClicked);
    }
    // 시작 버튼 연결
    if (ui->startBtn) {
        connect(ui->startBtn, &QPushButton::clicked, this, &ControlPopup2::onStartButtonClicked);
    }
    
    // 목적지 버튼들 연결
    if (ui->btn1) {
        connect(ui->btn1, &QPushButton::clicked, this, &ControlPopup2::onBtn1Clicked);
    }
    if (ui->btn2) {
        connect(ui->btn2, &QPushButton::clicked, this, &ControlPopup2::onBtn2Clicked);
    }
    if (ui->btn3) {
        connect(ui->btn3, &QPushButton::clicked, this, &ControlPopup2::onBtn3Clicked);
    }
    if (ui->btn4) {
        connect(ui->btn4, &QPushButton::clicked, this, &ControlPopup2::onBtn4Clicked);
    }
    if (ui->btn5) {
        connect(ui->btn5, &QPushButton::clicked, this, &ControlPopup2::onBtn5Clicked);
    }
    if (ui->btn6) {
        connect(ui->btn6, &QPushButton::clicked, this, &ControlPopup2::onBtn6Clicked);
    }
    if (ui->btn7) {
        connect(ui->btn7, &QPushButton::clicked, this, &ControlPopup2::onBtn7Clicked);
    }
    if (ui->btn8) {
        connect(ui->btn8, &QPushButton::clicked, this, &ControlPopup2::onBtn8Clicked);
    }
}


// 버튼 클릭 이벤트 핸들러들
void ControlPopup2::onBtn1Clicked()
{
    selected_button_ = ui->btn1;
    updateButtonStyles();
}

void ControlPopup2::onBtn2Clicked()
{
    selected_button_ = ui->btn2;
    updateButtonStyles();
}

void ControlPopup2::onBtn3Clicked()
{
    selected_button_ = ui->btn3;
    updateButtonStyles();
}

void ControlPopup2::onBtn4Clicked()
{
    selected_button_ = ui->btn4;
    updateButtonStyles();
}

void ControlPopup2::onBtn5Clicked()
{
    selected_button_ = ui->btn5;
    updateButtonStyles();
}

void ControlPopup2::onBtn6Clicked()
{
    selected_button_ = ui->btn6;
    updateButtonStyles();
}

void ControlPopup2::onBtn7Clicked()
{
    selected_button_ = ui->btn7;
    updateButtonStyles();
}

void ControlPopup2::onBtn8Clicked()
{
    selected_button_ = ui->btn8;
    updateButtonStyles();
}

void ControlPopup2::onCloseButtonClicked()
{
    qDebug() << "제어 팝업 닫기 버튼 클릭됨";
    close();  // 창 닫기
}

void ControlPopup2::onStartButtonClicked()
{
    qDebug() << "start 버튼 클릭됨";
    close();  // 창 닫기
    // 로봇 정지 명령 전송 로직
}

void ControlPopup2::resetAllButtonStyles()
{
    // 모든 버튼을 기본 스타일로 리셋
    QList<QPushButton*> buttons = {ui->btn1, ui->btn2, ui->btn3, ui->btn4, 
                                   ui->btn5, ui->btn6, ui->btn7, ui->btn8};
    
    for (QPushButton* btn : buttons) {
        if (btn) {
            btn->setProperty("class", "btn contained gray size20 weight700");
            btn->style()->unpolish(btn);
            btn->style()->polish(btn);
        }
    }
}


void ControlPopup2::updateButtonStyles()
{
    // 먼저 모든 버튼을 기본 스타일로 리셋
    resetAllButtonStyles();
    
    // 선택된 버튼만 특별한 스타일 적용
    if (selected_button_) {
        selected_button_->setProperty("class", "btn outlined primary size20 weight700");
        selected_button_->style()->unpolish(selected_button_);
        selected_button_->style()->polish(selected_button_);
        
        qDebug() << "버튼 스타일 변경됨:" << selected_button_->text();
    }
}
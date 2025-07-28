#include "control_popup1.h"
#include "ui_control_popup1.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QWidget>

ControlPopup1::ControlPopup1(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui_ControlPopup1)
{
    ui->setupUi(this);
    setWindowProperties();
    setupConnections();
    setWidgetClasses();
}

ControlPopup1::~ControlPopup1()
{
    delete ui;
}

void ControlPopup1::setWindowProperties()
{
    // 창 속성 설정
    setWindowTitle("로봇 제어");
    setFixedSize(800, 256);  // 고정 크기
    
    // 창을 화면 중앙에 위치
    if (parentWidget()) {
        QPoint parentCenter = parentWidget()->geometry().center();
        move(130, 272);
    }
}

void ControlPopup1::setWidgetClasses()
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
    if (ui->stopBtn) {
        ui->stopBtn->setProperty("class", "btn contained primary size20");
    }
    if (ui->content2) {
        ui->content2->setProperty("class", "size24 weight400 color-gray6");
    }
    if (ui->content1) {
        ui->content1->setProperty("class", "size24 weight700 color-primary_dark");
    }
    if (ui->content3) {
        ui->content3->setProperty("class", "size24 weight400 color-gray6");
    }
    if (ui->content4) {
        ui->content4->setProperty("class", "size24 weight700 color-gray6");
    }
    if (ui->content5) {
        ui->content5->setProperty("class", "size24 weight400 color-gray6");
    }
    if (ui->bg) {
        ui->bg->setProperty("class", "bg white radius otl-gray6");
    }
}

void ControlPopup1::setupConnections()
{
    // 닫기 버튼 연결 (UI 파일에 closeBtn이 있다고 가정)
    if (ui->closeBtn) {
        connect(ui->closeBtn, &QPushButton::clicked, this, &ControlPopup1::onCloseButtonClicked);
    }
    
    // 다른 버튼들도 필요에 따라 연결
    // if (ui->startBtn) {
    //     connect(ui->startBtn, &QPushButton::clicked, this, &ControlPopup1::onStartButtonClicked);
    // }
}

void ControlPopup1::onCloseButtonClicked()
{
    qDebug() << "제어 팝업 닫기 버튼 클릭됨";
    close();  // 창 닫기
}

void ControlPopup1::onStopButtonClicked()
{
    qDebug() << "정지 버튼 클릭됨";
    close();  // 창 닫기
    // 로봇 정지 명령 전송 로직
}
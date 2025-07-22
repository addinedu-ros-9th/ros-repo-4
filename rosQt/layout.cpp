#include "layout.h"
#include "dashboard.h"
#include "log.h"
#include "BasicQtApp_autogen/include/ui_layout.h"
#include "ui_userPopover.h"
#include <QDebug>
#include <QStyle>
#include <QApplication>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>

// 임시 사용자 정보 함수들 추가
QString get_user_id() {
    return "admin";
}

struct UserInfo {
    QString user_id;
    QString name;
    QString email;
    QString store_name;
};

UserInfo get_user_info() {
    UserInfo info;
    info.user_id = "admin01";
    info.name = "관리자";
    info.email = "admin@example.com";
    info.store_name = "가산 서울아산병원";
    return info;
}

// UserPopover 구현
UserPopover::UserPopover(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui_userPopover)
{
    ui->setupUi(this);
    setObjectName("userPopover");
    setWindowFlags(Qt::FramelessWindowHint | Qt::Popup);
    setAttribute(Qt::WA_TranslucentBackground);
    
    // 기본 크기 설정
    setFixedSize(200, 120);

    setWidgetClasses();
    
    // 로그아웃 버튼 연결
    connect(ui->logoutBtn, &QPushButton::clicked, this, &UserPopover::logoutRequested);
    connect(ui->logoutBtn, &QPushButton::clicked, this, &QWidget::close);
}

UserPopover::~UserPopover()
{
    delete ui;
}

void UserPopover::setWidgetClasses()
{
    if (ui->userName) {
        ui->userName->setProperty("class", "weight700 size14 color-gray6");
    }
    if (ui->userEmail) {
        ui->userEmail->setProperty("class", "weight300 size12 color-gray6");
    }
    if (ui->logoutBtn) {
        ui->logoutBtn->setProperty("class", "btn contained secondary");
        ui->logoutBtn->setLayoutDirection(Qt::RightToLeft);
    }
    
    style()->unpolish(this);
    style()->polish(this);
    update();
}

void UserPopover::refresh()
{
    qDebug() << "user popover refresh";
    UserInfo info = get_user_info();
    
    if (ui->userName) {
        ui->userName->setText(info.name);
    }
    if (ui->userEmail) {
        ui->userEmail->setText(info.email);
    }
}

void UserPopover::show_at(const QPoint& pos)
{
    qDebug() << "UserPopover::show_at called with pos:" << pos;
    
    // 크기 설정
    resize(200, 120);
    
    // 위치 설정
    move(pos);
    
    // 표시
    show();
    raise();
    activateWindow();
    
    qDebug() << "UserPopover geometry:" << geometry();
    qDebug() << "UserPopover isVisible:" << isVisible();
}

// LayoutWindow 구현
LayoutWindow::LayoutWindow(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui_LayoutWidget)
{
    ui->setupUi(this);
    
    UserInfo info = get_user_info();
    store_name = info.store_name;
    
    setupWidgets();
    setWidgetClasses();
    setupConnections();
    
    // 초기 상태 설정
    clickMenu1();
}

LayoutWindow::~LayoutWindow()
{
    delete ui;
}

void LayoutWindow::setupWidgets()
{
    // UserPopover 생성
    user_popover = new UserPopover(this);
    connect(user_popover, &UserPopover::logoutRequested, this, &LayoutWindow::handle_logout);
    
    // 위젯들 생성
    dashboard_widget = new DashboardWidget(this);
    log_widget = new LogWidget(this);
    
    // 위젯들을 더 작은 영역에 배치 (사이드바와 헤더 영역 제외)
    int content_x = 190;  // 사이드바 너비
    int content_y = 50;   // 헤더 높이
    int content_width = 1250;  
    int content_height = 850;  
    
    // 위젯들 크기와 위치 설정
    dashboard_widget->setGeometry(content_x, content_y, content_width, content_height);
    log_widget->setGeometry(content_x, content_y, content_width, content_height);
    
    // 배경 확실히 설정
    dashboard_widget->setAutoFillBackground(true);
    log_widget->setAutoFillBackground(true);
    
    // 초기에는 Dashboard만 표시
    log_widget->hide();

    qDebug() << "LayoutWindow widgets setup completed";
    qDebug() << "Dashboard widget geometry:" << dashboard_widget->geometry();
    qDebug() << "Log widget geometry:" << log_widget->geometry();
}

void LayoutWindow::setWidgetClasses()
{
    if (ui->menuBtn1) {
        ui->menuBtn1->setProperty("class", "menuBtn1");
    }
    if (ui->menuBtn2) {
        ui->menuBtn2->setProperty("class", "menuBtn2");
    }
    if (ui->logoBtn) {
        ui->logoBtn->setProperty("class", "logo-btn");
    }
    
    // 구분선 스타일 설정
    if (ui->verLine) {
        ui->verLine->setProperty("class", "bg grayc");
    }
    if (ui->horLine) {
        ui->horLine->setProperty("class", "bg grayc");
    }
    
    // 위치 정보 라벨 스타일 설정
    if (ui->locTitle) {
        ui->locTitle->setProperty("class", "weight700 size16 color-gray9 active");
    }
    if (ui->locDepth1) {
        ui->locDepth1->setProperty("class", "weight700 size16 color-gray9 ");
    }
    if (ui->locDivider) {
        ui->locDivider->setProperty("class", "weight700 size16 color-gray9");
    }

    style()->unpolish(this);
    style()->polish(this);
    update();
}

void LayoutWindow::setupConnections()
{
    if (ui->menuBtn1) {
        connect(ui->menuBtn1, &QPushButton::clicked, this, &LayoutWindow::clickMenu1);
    }
    if (ui->menuBtn2) {
        connect(ui->menuBtn2, &QPushButton::clicked, this, &LayoutWindow::clickMenu2);
    }
    
    if (ui->userBtn) {
        connect(ui->userBtn, &QPushButton::clicked, this, &LayoutWindow::show_user_popover);
        ui->userBtn->setLayoutDirection(Qt::RightToLeft);
    }
}

void LayoutWindow::updateMenuButtons(int activeMenu)
{
    if (ui->menuBtn1) {
        ui->menuBtn1->setProperty("class", "menuBtn1");
        ui->menuBtn1->style()->unpolish(ui->menuBtn1);
        ui->menuBtn1->style()->polish(ui->menuBtn1);
    }
    if (ui->menuBtn2) {
        ui->menuBtn2->setProperty("class", "menuBtn2");
        ui->menuBtn2->style()->unpolish(ui->menuBtn2);
        ui->menuBtn2->style()->polish(ui->menuBtn2);
    }
    
    switch (activeMenu) {
        case 1:
            if (ui->menuBtn1) {
                ui->menuBtn1->setProperty("class", "menuBtn1 active");
                ui->menuBtn1->style()->unpolish(ui->menuBtn1);
                ui->menuBtn1->style()->polish(ui->menuBtn1);
            }
            break;
        case 2:
            if (ui->menuBtn2) {
                ui->menuBtn2->setProperty("class", "menuBtn2 active");
                ui->menuBtn2->style()->unpolish(ui->menuBtn2);
                ui->menuBtn2->style()->polish(ui->menuBtn2);
            }
            break;
    }
}

void LayoutWindow::clickMenu1()
{
    updateMenuButtons(1);
    log_widget->hide();
    if (ui->locDepth1) {
        ui->locDepth1->setText("Dashboard");
    }
    dashboard_widget->show();
    dashboard_widget->refresh();
}

void LayoutWindow::clickMenu2()
{
    updateMenuButtons(2);
    dashboard_widget->hide();
    if (ui->locDepth1) {
        ui->locDepth1->setText("Log");
    }
    log_widget->show();
    log_widget->refresh();
}

void LayoutWindow::handle_logout()
{
    qDebug() << "Logout requested";
    emit logout_successful();
}

void LayoutWindow::show_user_popover()
{
    qDebug() << "show_user_popover() called";  // 디버그 로그 추가
    
    if (user_popover) {
        // 버튼 위치 기준으로 팝오버 위치 계산
        QPoint popoverPos;
        if (ui->userBtn) {
            QPoint buttonPos = ui->userBtn->mapToGlobal(QPoint(0, 0));
            popoverPos = QPoint(buttonPos.x() - 80, buttonPos.y() + ui->userBtn->height() + 5);
        } else {
            popoverPos = QPoint(500, 100);  // 기본 위치
        }
        
        qDebug() << "Showing user popover at:" << popoverPos;
        user_popover->show_at(popoverPos);
        user_popover->refresh();
        user_popover->raise();  // 다른 위젯 위로 올리기
    } else {
        qDebug() << "user_popover is null!";
    }
}

void LayoutWindow::refresh()
{
    qDebug() << "layout refresh";
    user_popover->refresh();
    log_widget->refresh();

    UserInfo info = get_user_info();

    if (ui->locTitle) {
        ui->locTitle->setText(info.store_name);
    }
    if (ui->userBtn) {
        ui->userBtn->setText(info.user_id);
    }

}

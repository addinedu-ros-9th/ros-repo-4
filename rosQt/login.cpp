#include "login.h"
#include "ui_login.h"
#include <QDebug>
#include <QStyle>

LoginWindow::LoginWindow(QWidget *parent)
    : QWidget(parent)  // QWidget 대신 QMainWindow로 되돌리기
    , ui(new Ui_LoginWidget)
{
    ui->setupUi(this);
    setWidgetClasses();
    
    // 로그인 버튼 연결 (login.ui에 login_btn이 있다고 가정)
    if (ui->login_btn) {
        connect(ui->login_btn, &QPushButton::clicked, this, &LoginWindow::handleLogin);
    }
    
    // 테스트용: 다른 버튼이라도 연결해보기
    // login.ui에 있는 실제 버튼 이름을 확인하고 사용
}

LoginWindow::~LoginWindow()
{
    delete ui;
}

void LoginWindow::handleLogin()
{
    emit loginSuccessful();
}

void LoginWindow::setWidgetClasses()
{
    // 버튼 클래스 설정
    if (ui->login_btn) {
        ui->login_btn->setProperty("class", "btn contained primary_dark large");
    }
    if (ui->login_logo_icon) {
        ui->login_logo_icon->setProperty("class", "login_logo_icon");
    }
    // 체크박스 클래스 설정
    if (ui->login_checkbox) {
        ui->login_checkbox->setProperty("class", "checkbox gray9");
    }
    
    // 구분선 클래스 설정
    if (ui->login_div) {
        ui->login_div->setProperty("class", "divider ver dotted");
    }
    if (ui->login_div2) {
        ui->login_div2->setProperty("class", "divider ver dotted");
    }
    
    // 텍스트 필드 클래스 설정
    if (ui->login_textfield1) {
        ui->login_textfield1->setProperty("class", "textfield large");
    }
    if (ui->login_textfield2) {
        ui->login_textfield2->setProperty("class", "textfield large");
    }
    
    // 라벨 클래스 설정
    if (ui->login_greeting) {
        ui->login_greeting->setProperty("class", "size24 weight700");
    }
    if (ui->login_desc1) {
        ui->login_desc1->setProperty("class", "size16 weight300 color-gray9");
    }
    if (ui->login_desc2) {
        ui->login_desc2->setProperty("class", "size16 weight300 color-gray9");
    }
    if (ui->login_id) {
        ui->login_id->setProperty("class", "size14 color-gray3");
    }
    if (ui->login_password) {
        ui->login_password->setProperty("class", "size14 color-gray3");
    }
    if (ui->auth_text1) {
        ui->auth_text1->setProperty("class", "size12 color-white");
    }
    if (ui->auth_text2) {
        ui->auth_text2->setProperty("class", "size12 color-white");
    }
    
    // 스타일 새로고침 (중요!)
    this->style()->unpolish(this);
    this->style()->polish(this);
    this->update();
    
}
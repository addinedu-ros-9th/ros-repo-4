#ifndef LOGINWINDOW_H
#define LOGINWINDOW_H

#include <QWidget>  // QWidget 대신 QMainWindow로 되돌리기
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>

// Include the UI header directly instead of forward declaration
#include "ui_login.h"

class LoginWindow : public QWidget  // QWidget 대신 QMainWindow
{
    Q_OBJECT

public:
    explicit LoginWindow(QWidget *parent = nullptr);
    ~LoginWindow();

signals:
    void loginSuccessful();  // 로그인 성공 시그널

private slots:
    void handleLogin();      // 로그인 버튼 클릭 처리
    void check_login();  // 로그인 정보 확인 함수
    void resetTextFieldStyle();  // 새로 추가된 슬롯

private:
    void setWidgetClasses();
    
    Ui_LoginWidget *ui;  // Revert back to original class name
};

#endif // LOGINWINDOW_H
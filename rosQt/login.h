#ifndef LOGINWINDOW_H
#define LOGINWINDOW_H

#include <QWidget>  // QWidget 대신 QMainWindow로 되돌리기
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>

class Ui_LoginWidget;

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

private:
    void setWidgetClasses();
    
    Ui_LoginWidget *ui;
    void check_login();  // 로그인 정보 확인 함수
};

#endif // LOGINWINDOW_H
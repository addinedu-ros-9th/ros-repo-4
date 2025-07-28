#ifndef CONTROL_POPUP1_H
#define CONTROL_POPUP1_H

#include <QWidget>

class Ui_ControlPopup1;  // UI 클래스 전방 선언

class ControlPopup1 : public QWidget
{
    Q_OBJECT

public:
    explicit ControlPopup1(QWidget *parent = nullptr);
    ~ControlPopup1();

private slots:
    void onCloseButtonClicked();
    void onStopButtonClicked();   // 정지 버튼용

private:
    void setWindowProperties();
    void setupConnections();
    void setWidgetClasses();  // 위젯 클래스 설정 함수
    
private:
    Ui_ControlPopup1 *ui;
};

#endif // CONTROL_POPUP1_H
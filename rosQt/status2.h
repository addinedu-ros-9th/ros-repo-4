#ifndef STATUS2WIDGET_H
#define STATUS2WIDGET_H

#include <QWidget>
#include <QPushButton>

class Ui_Status2Widget;  // UI 클래스 전방 선언

class Status2Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Status2Widget(QWidget *parent = nullptr);
    ~Status2Widget();

    void show_at(const QPoint& pos);
    void refresh();

public slots:
    void setMoveFirstText(const QString& text);

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    QPushButton* arrowBtns[9];  // 화살표 버튼 배열
    void setupKeyButton();  // 키 버튼 설정 함수
    void onClickKey1();  // 키 1 클릭 핸들러
    void onClickKey2();  // 키 2 클릭 핸들러
    void onClickKey3();  // 키 3 클릭 핸들러
    void onClickKey4();  // 키 4 클릭 핸들러
    void onClickKey5();  // 키 5 클릭 핸들러 
    void onClickKey6();  // 키 6 클릭 핸들러
    void onClickKey7();  // 키 7 클릭 핸들러
    void onClickKey8();  // 키 8 클릭 핸들러
    void onClickKey9();  // 키 9 클릭 핸들러
    
private:
    Ui_Status2Widget *ui;  // UI 포인터
};

#endif // STATUS2WIDGET_H
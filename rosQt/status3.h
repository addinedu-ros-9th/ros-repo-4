#ifndef STATUS3WIDGET_H
#define STATUS3WIDGET_H

#include <QWidget>
#include <QPushButton>

class Ui_Status3Widget;  // UI 클래스 전방 선언

class Status3Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Status3Widget(QWidget *parent = nullptr);
    ~Status3Widget();

    void show_at(const QPoint& pos);
    void refresh();

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    
private:
    Ui_Status3Widget *ui;  // UI 포인터
};

#endif // STATUS3WIDGET_H
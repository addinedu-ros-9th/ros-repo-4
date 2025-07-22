#ifndef STATUSWIDGET_H
#define STATUSWIDGET_H

#include <QWidget>

class Ui_StatusWidget;  // UI 클래스 전방 선언

class StatusWidget : public QWidget
{
    Q_OBJECT

public:
    explicit StatusWidget(QWidget *parent = nullptr);
    ~StatusWidget();
    
    void show_at(const QPoint& pos);
    void refresh();

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    
private:
    Ui_StatusWidget *ui;  // UI 포인터
};

#endif // STATUSWIDGET_H
#ifndef MAPWIDGET_H
#define MAPWIDGET_H

#include <QWidget>

class Ui_MapWidget;  // UI 클래스 전방 선언

class MapWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MapWidget(QWidget *parent = nullptr);
    ~MapWidget();
    
    void show_at(const QPoint& pos);
    void refresh();

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    
private:
    Ui_MapWidget *ui;  // UI 포인터
};

#endif // MAPWIDGET_H
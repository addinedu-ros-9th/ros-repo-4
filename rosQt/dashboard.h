#ifndef DASHBOARDWIDGET_H
#define DASHBOARDWIDGET_H

#include <QWidget>
#include <QLabel>

class Ui_DashboardWidget;  // UI 클래스 전방 선언
class StatusWidget;
class MapWidget;

class DashboardWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DashboardWidget(QWidget *parent = nullptr);
    ~DashboardWidget();
    
    void show_at(const QPoint& pos);
    void refresh();

private:
    void setupUI();
    void setWidgetClasses();  // CSS 클래스 설정 함수 추가
    void setupStatusWidget(); 
    void setupMapWidget();
    
private:
    Ui_DashboardWidget *ui;  // UI 포인터로 변경
    StatusWidget *status_widget; 
    MapWidget *map_widget;  // 맵 위젯 포인터 추가
};

#endif // DASHBOARDWIDGET_H
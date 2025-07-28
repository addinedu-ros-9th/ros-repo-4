#ifndef DASHBOARDWIDGET_H
#define DASHBOARDWIDGET_H

#include <QWidget>
#include "udp_image_receiver.h" 

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

private slots:
    void onImageReceived(const QPixmap& pixmap);
    void onConnectionError(const QString& error);
    void onConnectionEstablished();

    
private:
    Ui_DashboardWidget *ui;  // UI 포인터로 변경
    StatusWidget *status_widget; 
    MapWidget *map_widget;  // 맵 위젯 포인터 추가
    UdpImageReceiver *udp_receiver_;  // UDP 이미지 수신기 포인터
    
    void setupUI();
    void setWidgetClasses();
    void setupStatusWidget(); 
    void setupMapWidget();
    void setupCameraWidget();
};

#endif // DASHBOARDWIDGET_H
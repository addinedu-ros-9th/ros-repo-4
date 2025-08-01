#ifndef DASHBOARDWIDGET_H
#define DASHBOARDWIDGET_H

#include <QWidget>
#include <QKeyEvent>  // ← 추가
#include "udp_image_receiver.h" 

class Ui_DashboardWidget;  // UI 클래스 전방 선언
class StatusWidget;
class Status2Widget;
class Status3Widget;
class MapWidget;
class ControlPopup1;
class ControlPopup2;

class DashboardWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DashboardWidget(QWidget *parent = nullptr);
    ~DashboardWidget();
    
    void show_at(const QPoint& pos);
    void refresh();
    
    // 상태 관리 함수들 추가
    void setStatus(const QString& newStatus);
    QString getStatus() const;
    void setControlStatus(const QString& newControlStatus);
    QString getControlStatus() const;
    

private slots:
    void onImageReceived(const QPixmap& pixmap);
    void onConnectionError(const QString& error);
    void onConnectionEstablished();
    void onControlButtonClicked();
    void onDestinationButtonClicked();
    void setStatusToIdle();

private:
    Ui_DashboardWidget *ui;
    StatusWidget *status_widget;
    Status2Widget *status_widget2;
    Status3Widget *status_widget3;
    MapWidget *map_widget;
    UdpImageReceiver *udp_receiver_;
    ControlPopup1 *control_popup1_;
    ControlPopup2 *control_popup2_;

    QString status_;  // 상태 변수
    QString control_status_;  // 제어 상태 변수
    QString camera_toggle_status_;

    void setupUI();
    void setWidgetClasses();
    void setupStatusWidget(); 
    void setupMapWidget();
    void setupCameraWidget();
    void setupControlButton();
    void setCameraToggleStatus();
    void onCameraToggle1Clicked();
    void onCameraToggle2Clicked();
};

#endif // DASHBOARDWIDGET_H
#ifndef DASHBOARDWIDGET_H
#define DASHBOARDWIDGET_H

#include <QWidget>
#include <QKeyEvent>  // ← 추가
#include "udp_image_receiver.h" 
#include <rclcpp/rclcpp.hpp>

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
    
    
private slots:
    void onImageReceived(const QPixmap& pixmap);
    void onConnectionError(const QString& error);
    void onConnectionEstablished();
    void onControlButtonClicked();
    void onDestinationButtonClicked();
    void setControlStatusToAssigned();
    void setControlStatus(const QString& newStatus);
private:
    Ui_DashboardWidget *ui;
    StatusWidget *status_widget;
    Status2Widget *status_widget2;
    Status3Widget *status_widget3;
    MapWidget *map_widget;
    UdpImageReceiver *udp_receiver_;
    ControlPopup1 *control_popup1_;
    ControlPopup2 *control_popup2_;

    std::shared_ptr<rclcpp::Node> ros_node_;
    QTimer *ros_timer_;  // ROS2 스핀용 타이머

    QString status_;    
    QString control_status_;  // 제어 주체 변수
    QString camera_toggle_status_;
    
    double pose_x_;
    double pose_y_;
    double pose_yaw_; 
    double pose_qw_;  

    int orig_;
    int dest_;
    int battery_;
    int network_;


    void getRobotStatus();
    void setupUI();
    void setWidgetClasses();
    void setupStatusWidget(); 
    void setupMapWidget();
    void setupCameraWidget();
    void setupControlButton();
    void setCameraToggleStatus();
    void onCameraToggleClicked();
    void get_robot_location();
    // void amcl_pose_callback();
    void setPose(double x, double y, double yaw);
};

#endif // DASHBOARDWIDGET_H
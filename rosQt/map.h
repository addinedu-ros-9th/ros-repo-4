#ifndef MAPWIDGET_H
#define MAPWIDGET_H

#include <QWidget>
#include <QTimer>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <memory>

class Ui_MapWidget;  // UI 클래스 전방 선언

class MapWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MapWidget(QWidget *parent = nullptr);
    ~MapWidget();
    
    void show_at(const QPoint& pos);
    void refresh();
    void setPose(double x, double y, double yaw);
    
private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    // void get_robot_location();
    void amcl_pose_callback();
    
private:
    Ui_MapWidget *ui;  // UI 포인터
    std::shared_ptr<rclcpp::Node> ros_node_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub_;
    QTimer *ros_timer_;  // ROS2 스핀용 타이머
    double pose_x_;
    double pose_y_;
    double pose_yaw_; 
    double pose_qw_;  
};

#endif // MAPWIDGET_H
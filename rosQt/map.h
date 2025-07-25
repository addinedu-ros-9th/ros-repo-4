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

private slots:
    void amcl_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void spinRos();  // ROS2 스핀용 슬롯

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    void setupRosNode();      // ROS2 노드 설정 함수
    
private:
    Ui_MapWidget *ui;  // UI 포인터
    std::shared_ptr<rclcpp::Node> ros_node_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub_;
    QTimer *ros_timer_;  // ROS2 스핀용 타이머
};

#endif // MAPWIDGET_H
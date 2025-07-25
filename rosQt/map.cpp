#include "map.h"
#include "BasicQtApp_autogen/include/ui_map.h"  // UI 헤더 포함
#include <QDebug>
#include <QLabel>
#include <QTimer>
#include <QPixmap>
#include <QTransform>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

MapWidget::MapWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_MapWidget)
    , ros_timer_(new QTimer(this))
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
    setupRosNode();  // ROS2 노드 설정
    
    // 타이머 설정 - 100ms마다 ROS2 스핀 실행
    connect(ros_timer_, &QTimer::timeout, this, &MapWidget::spinRos);
    ros_timer_->start(100);  // 100ms 간격으로 스핀
    
    qDebug() << "MapWidget initialized with ROS2 timer";
}

MapWidget::~MapWidget()
{
    if (ros_timer_) {
        ros_timer_->stop();
    }
    delete ui;
}

void MapWidget::setWidgetClasses()
{
    if (ui->map_img) {
        ui->map_img->setProperty("class", "map_img");
    }
    if (ui->map_robot) {
        ui->map_robot->setProperty("class", "map_robot");
        ui->map_robot->move(219, 386);

        QPixmap pixmap("/home/wonho/ros-repo-4/rosQt/style/images/map_robot.png");
        ui->map_robot->setPixmap(pixmap);
        ui->map_robot->setAlignment(Qt::AlignCenter);
    }
}

void MapWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void MapWidget::refresh()
{
    // qDebug() << "Map widget refresh";
    
    // ROS2 메시지 스핀 (콜백 처리)
    if (ros_node_) {
        rclcpp::spin_some(ros_node_);
    }
}

void MapWidget::setupRosNode()
{
    // ROS2 노드 초기화
    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }
    
    ros_node_ = rclcpp::Node::make_shared("map_widget_node");
    
    // /amcl_pose 토픽 구독
    amcl_pose_sub_ = ros_node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/amcl_pose", 10, 
        [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
            this->amcl_pose_callback(msg);
        });
    
    // qDebug() << "ROS2 node setup complete. Subscribing to /amcl_pose";
}

void MapWidget::amcl_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    
    double qz = msg->pose.pose.orientation.z;
    double qw = msg->pose.pose.orientation.w;

    // 쿼터니언에서 Yaw 각도 계산 (2D에서 간단한 공식)
    double yaw_radians = 2.0 * atan2(qz, qw);
    
    // 라디안을 도(degree)로 변환
    double yaw_degrees = yaw_radians * 180.0 / M_PI;
    
    // 각도를 -180° ~ 180° 범위로 정규화
    while (yaw_degrees > 180.0) yaw_degrees -= 360.0;
    while (yaw_degrees < -180.0) yaw_degrees += 360.0;
    
    // qDebug() << "AMCL Pose received:";
    // qDebug() << "   Position: x=" << x << ", y=" << y;
    // qDebug() << "   Orientation: z=" << qz << ", w=" << qw;
    // qDebug() << "   Yaw: " << yaw_degrees << "도 (" << yaw_radians << " rad)";

    // map_robot 위젯 위치 업데이트 (좌표 변환 필요할 수 있음)
    if (ui->map_robot) {
        // UI 픽셀 크기
        const int UI_WIDTH = 438;
        const int UI_HEIGHT = 772;
        
        double robot_x = -y + 5;
        double robot_y = -x + 10;

        int pixel_x = robot_x / 10 * UI_WIDTH;
        pixel_x = pixel_x - 29;
        
        int pixel_y = robot_y / 20 * UI_HEIGHT;
        pixel_y = pixel_y - 31;

        // qDebug() << "   UI 좌표: pixel_x=" << pixel_x << ", pixel_y=" << pixel_y;
        
        ui->map_robot->move(pixel_x, pixel_y);
        QPixmap pixmap("/home/wonho/ros-repo-4/rosQt/style/images/map_robot.png");
        QTransform transform;
        transform.rotate(-yaw_degrees);  // 시계 방향 회전
        QPixmap rotatedPixmap = pixmap.transformed(transform, Qt::SmoothTransformation);
        ui->map_robot->setPixmap(rotatedPixmap);
        
    }
}

void MapWidget::spinRos()
{
    if (ros_node_ && rclcpp::ok()) {
        rclcpp::spin_some(ros_node_);
    }
}
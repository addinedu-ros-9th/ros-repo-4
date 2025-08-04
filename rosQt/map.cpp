#include "map.h"
#include "BasicQtApp_autogen/include/ui_map.h"  // UI 헤더 포함
#include <QDebug>
#include <QLabel>
#include <QTimer>
#include <QPixmap>
#include <QTransform>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>  
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>

MapWidget::MapWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_MapWidget)
    , ros_timer_(new QTimer(this))
    , pose_x_(0.0)
    , pose_y_(0.0)
    , pose_yaw_(0.0)
    , pose_qw_(1.0)  // 초기값 설정
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();

    // 5초마다 로봇 위치 가져오기
    connect(ros_timer_, &QTimer::timeout, this, &MapWidget::get_robot_location);
    ros_timer_->start(5000);  // 5초 간격
    
    qDebug() << "MapWidget initialized with ROS2 timer";
}

MapWidget::~MapWidget()
{
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
    
    // 로봇 위치 주기적 업데이트
    get_robot_location();
}

void MapWidget::setPose(double x, double y, double yaw)
{
    pose_x_ = x;
    pose_y_ = y;
    pose_yaw_ = yaw;
    pose_qw_ = sqrt(1 - pose_yaw_ * pose_yaw_);
}

void MapWidget::get_robot_location()
{
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/get/robot_location")
                    .arg(CENTRAL_IP.c_str())
                    .arg(CENTRAL_HTTP_PORT);

    QJsonObject data;
    data["robot_id"] = 3;
    QJsonDocument doc(data);
    QByteArray jsonData = doc.toJson();

    qDebug() << "[로봇 위치 요청 URL]:" << url;
    qDebug() << "[전송 데이터]:" << jsonData;
    try
    {
        QNetworkRequest request{QUrl(url)};
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        QNetworkReply* reply = manager->post(request, jsonData);

        connect(reply, &QNetworkReply::finished, this, [this, reply, CENTRAL_IP, CENTRAL_HTTP_PORT]() {
            int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            if (statusCode == 200) {
                QByteArray responseData = reply->readAll();
                QJsonDocument jsonDoc = QJsonDocument::fromJson(responseData);
                QJsonObject result = jsonDoc.object();
                qDebug() << "[응답 내용]:" << result;

                if (result.contains("x") && result.contains("y") && result.contains("yaw")) {
                    double location_x = result["x"].toDouble();
                    double location_y = result["y"].toDouble();
                    double location_yaw = result["yaw"].toDouble();

                    setPose(location_x, location_y, location_yaw);
                    amcl_pose_callback();  // 위치 업데이트 후 콜백 호출
                } else {
                    qDebug() << "응답에 위치 정보가 없습니다.";
                }
            } else if (statusCode == 400) {
                qDebug() << "잘못된 요청입니다. 400 Bad Request";
            } else if (statusCode == 401) {
                qDebug() << "정상 요청, 정보 없음 or 응답 실패. 401";
            } else if (statusCode == 404) {
                qDebug() << "잘못된 요청 404 Not Found";
            } else if (statusCode == 405) {
                qDebug() << "메소드가 리소스 허용 안됨";
            } else if (statusCode == 500) {
                qDebug() << "서버 내부 오류 500 Internal Server Error";
            } else if (statusCode == 503) {
                qDebug() << "서비스 불가";
            } else {
                qDebug() << "알 수 없는 오류 발생. 상태 코드:" << statusCode;
            }

            reply->deleteLater();
        });

    } catch (const YAML::BadFile& e) {
        qDebug() << "YAML 파일 로드 실패:" << e.what();
        return;
    }
}

void MapWidget::amcl_pose_callback()
{
    double x = pose_x_;
    double y = pose_y_;
    double yaw = pose_yaw_;

    double qz = pose_yaw_;
    double qw = pose_qw_;

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
#include "dashboard.h"
#include "ui_dashboard.h"
#include "status.h"
#include "status2.h"
#include "status3.h"
#include "map.h"
#include <QTimer>
#include "udp_image_receiver.h" 
#include "control_popup1.h" 
#include "control_popup2.h" 
#include <QVBoxLayout>
#include <QLabel>
#include <QDebug>
#include <QStyle>
#include <QPushButton>
#include <yaml-cpp/yaml.h>  
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>

DashboardWidget::DashboardWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_DashboardWidget)
    , ros_timer_(new QTimer(this))
    , status_widget(nullptr) 
    , status_widget2(nullptr) // 추가
    , status_widget3(nullptr) // 추가
    , map_widget(nullptr)
    , udp_receiver_(nullptr) 
    , control_popup1_(nullptr) 
    , control_popup2_(nullptr) 
    , pose_x_(0.0)
    , pose_y_(0.0)
    , pose_yaw_(0.0)
    , pose_qw_(1.0)  // 초기값 설정
    , status_("idle")
    , control_status_("환자사용중")
    , camera_toggle_status_("전면")
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
    setupStatusWidget();
    setupMapWidget();
    setupCameraWidget(); 
    setupControlButton();  // 추가
    setCameraToggleStatus();  // 초기 카메라 상태 설정
    getRobotStatus();

    // 5초마다 로봇 위치 가져오기
    connect(ros_timer_, &QTimer::timeout, this, &DashboardWidget::get_robot_location);
    ros_timer_->start(5000);  // 5초 간격
    
}

DashboardWidget::~DashboardWidget()
{
    if (udp_receiver_) {     
        udp_receiver_->stop();
        delete udp_receiver_;
    }
    if (control_popup1_) {  // 추가
        delete control_popup1_;
    }
    if (control_popup2_) {
        delete control_popup2_;
    }
    delete ui;
}

void DashboardWidget::setStatus(const QString& newStatus)
{
    status_ = newStatus;
}
void DashboardWidget::setControlStatus(const QString& newStatus)
{
    if (control_status_ != newStatus) {
        QString oldStatus = control_status_;
        control_status_ = newStatus;
        
        // 기존 status_widget, status_widget2, status_widget3 모두 삭제
        if (status_widget) {
            status_widget->hide();
            delete status_widget;
            status_widget = nullptr;
        }
        if (status_widget2) {
            status_widget2->hide();
            delete status_widget2;
            status_widget2 = nullptr;
        }
        if (status_widget3) {
            status_widget3->hide();
            delete status_widget3;
            status_widget3 = nullptr;
        }

        // 새 status_widget 생성
        if (control_status_ == "환자사용중") {
            status_widget = new StatusWidget(this);
            status_widget->setGeometry(477, 549, 753, 281);
            status_widget->show();
        } else if (control_status_ == "관리자사용중") {
            status_widget2 = new Status2Widget(this);
            status_widget2->setGeometry(477, 549, 753, 281);
            status_widget2->show();
        } else if (control_status_ == "대기중") {
            status_widget3 = new Status3Widget(this);
            status_widget3->setGeometry(477, 549, 753, 281);  // 16:9 비율로 설정
            status_widget3->show();
        }
        // 위치와 크기 설정
        
        qDebug() << "로봇 상태 변경:" << oldStatus << "→" << newStatus;
        
        // 상태가 변경되면 열려있는 팝업들 닫기
        if (control_popup1_ && control_popup1_->isVisible()) {
            control_popup1_->hide();
        }
        if (control_popup2_ && control_popup2_->isVisible()) {
            control_popup2_->hide();
        }

        if (ui->controlBtn) {
            if(control_status_ == "관리자사용중") {
                ui->controlBtn->setText("제어 중지");
            } else if (control_status_ == "대기중") {
                ui->controlBtn->setText("원격 제어");
            } else if (control_status_ == "환자사용중") {
                ui->controlBtn->setVisible(false);
            } 
        }

        if (ui->status_label) {
            ui->status_label->setText(control_status_);
            ui->status_label->setProperty("class", control_status_ == "환자사용중" ? "label primary" : control_status_ == "관리자사용중" ? "label secondary" : "label gray");
            ui->status_label->style()->unpolish(ui->status_label);
            ui->status_label->style()->polish(ui->status_label);
        } 

        if (ui->destinationBtn) {
            ui->destinationBtn->setVisible(control_status_ == "관리자사용중");
        }
    }
}

void DashboardWidget::setPose(double x, double y, double yaw)
{
    pose_x_ = x;
    pose_y_ = y;
    pose_yaw_ = yaw;
    pose_qw_ = sqrt(1 - pose_yaw_ * pose_yaw_);

    if(map_widget) {
        map_widget->setPose(pose_x_, pose_y_, pose_yaw_);
    }
    if(status_widget2) {
        status_widget2->setRobotLocation(pose_x_, pose_y_, pose_yaw_);

    }
}

void DashboardWidget::get_robot_location()
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
                    // setPose(location_x, location_y, location_yaw);
                    // amcl_pose_callback();  // 위치 업데이트 후 콜백 호출
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

void DashboardWidget::setCameraToggleStatus()
{
    if (ui->camera_toggle1) {
        ui->camera_toggle1->setProperty("class", camera_toggle_status_ == "전면" ? "btn contained white otl-gray9" : "btn contained gray ");
        ui->camera_toggle1->style()->unpolish(ui->camera_toggle1);
        ui->camera_toggle1->style()->polish(ui->camera_toggle1);
    }
    if (ui->camera_toggle2) {
        ui->camera_toggle2->setProperty("class", camera_toggle_status_ == "후면" ? "btn contained white otl-gray9" : "btn contained gray ");
        ui->camera_toggle2->style()->unpolish(ui->camera_toggle2);
        ui->camera_toggle2->style()->polish(ui->camera_toggle2);
    }
}


void DashboardWidget::getRobotStatus()
{
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/get/robot_status")
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

                if (result.contains("status") && result.contains("orig") && result.contains("dest") &&
                    result.contains("battery") && result.contains("network")) {
                    setStatus(result["status"].toString());
                    // if (result["status"].toString() == "unknown") {
                    //     QString robot_status = "대기중";
                    //     setStatus(robot_status);
                    // } else if (result["status"].toString() == "idle") {
                    //     QString robot_status = "대기중";
                    //     setStatus(robot_status);
                    // } else if (result["status"].toString() == "moving") {
                    //     QString robot_status = "환자사용중";
                    //     setStatus(robot_status);
                    // } else if (result["status"].toString() == "assigned") {
                    //     QString robot_status = "관리자사용중";
                    //     setStatus(robot_status);
                    // } else {
                    //     QString robot_status = result["status"].toString();
                    //     setStatus(robot_status);
                    // }
                    int robot_orig = result["orig"].toInt();
                    int robot_dest = result["dest"].toInt();
                    int robot_battery = result["battery"].toInt();
                    int robot_network = result["network"].toInt();
                    
                    // 현재 활성화된 status widget에 따라 적절한 메소드 호출
                    if (status_widget) {
                        status_widget->setRobotInfo(robot_orig, robot_dest, robot_battery, robot_network);
                    } else if (status_widget2) {
                        status_widget2->setRobotInfo(robot_orig, robot_dest, robot_battery, robot_network);
                        // status_widget2에도 setRobotInfo 메소드가 있다면 호출
                        // status_widget2->setRobotInfo(robot_orig, robot_dest, robot_battery, robot_network);
                    } else if (status_widget3) {
                        // status_widget3에도 setRobotInfo 메소드가 있다면 호출
                        // status_widget3->setRobotInfo(robot_orig, robot_dest, robot_battery, robot_network);
                    }
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


void DashboardWidget::setupControlButton()
{
    if (ui->controlBtn) {
        connect(ui->controlBtn, &QPushButton::clicked,
                this, &DashboardWidget::onControlButtonClicked);
    } else {
        qDebug() << "❌ controlBtn을 찾을 수 없습니다!";
    }
    if (ui->destinationBtn) {
        connect(ui->destinationBtn, &QPushButton::clicked,
                this, &DashboardWidget::onDestinationButtonClicked);
    } else {
        qDebug() << "❌ destinationBtn을 찾을 수 없습니다!";
    }

    if (ui->camera_toggle1) {
        connect(ui->camera_toggle1, &QPushButton::clicked,
                this, &DashboardWidget::onCameraToggleClicked);
    }
    if (ui->camera_toggle2) {
        connect(ui->camera_toggle2, &QPushButton::clicked,
                this, &DashboardWidget::onCameraToggleClicked);
    }
}

void DashboardWidget::onCameraToggleClicked()
{
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string AI_SERVER_IP = config["ai_server"]["ip"].as<std::string>();
    int AI_SERVER_PORT = config["ai_server"]["port"].as<int>();

    QString url = QString("http://%1:%2/change/camera")
                    .arg(AI_SERVER_IP.c_str())
                    .arg(AI_SERVER_PORT);

    QJsonObject data;
    data["robot_id"] = 3;
    data["camera"] = camera_toggle_status_ == "전면" ? "back" : "front";  // 전면일 때 후면으로, 후면일 때 전면으로
    QJsonDocument doc(data);
    QByteArray jsonData = doc.toJson();

    qDebug() << "[카메라 전환 요청 URL]:" << url;
    qDebug() << "[전송 데이터]:" << jsonData;
    try
    {
        QNetworkRequest request{QUrl(url)};
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        QNetworkReply* reply = manager->post(request, jsonData);

        connect(reply, &QNetworkReply::finished, this, [this, reply, AI_SERVER_IP, AI_SERVER_PORT]() {
            int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            if (statusCode == 200) {
                qDebug() << "카메라 변경 성공. 200";
                camera_toggle_status_ = camera_toggle_status_ == "전면" ? "후면" : "전면";
                setCameraToggleStatus();
                qDebug() << camera_toggle_status_ << " 카메라로 전환됨";
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

void DashboardWidget::setupStatusWidget()
{
    // StatusWidget 생성
    if (control_status_ == "환자사용중") {
        status_widget = new StatusWidget(this);
        status_widget->setGeometry(477, 549, 753, 281);  // 위치와 크기 조정
        status_widget->show();
    } else if (control_status_ == "관리자사용중") {
        status_widget2 = new Status2Widget(this);
        status_widget2->setGeometry(477, 549, 753, 281);  // 위치와 크기 조정
        status_widget2->show();
    } else if (control_status_ == "대기중") {
        status_widget3 = new Status3Widget(this);
        status_widget3->setGeometry(477, 549, 753, 281);  // 위치와 크기 조정
        status_widget3->show();
    }
}

void DashboardWidget::onControlButtonClicked()
{
    qDebug() << "🎮 Control 버튼 클릭! 현재 상태:" << control_status_;
    
    if (control_status_ == "환자사용중") {
        // 이동 중일 때 - control_popup1 표시
        qDebug() << "이동 중 상태 → ControlPopup1 표시";
        
        // // 다른 팝업이 열려있으면 닫기
        // if (control_popup2_ && control_popup2_->isVisible()) {
        //     control_popup2_->hide();
        // }
        
        // // control_popup1 표시
        // if (control_popup1_ && control_popup1_->isVisible()) {
        //     control_popup1_->raise();
        //     control_popup1_->activateWindow();
        //     return;
        // }
        // if (!control_popup1_) {
        //     control_popup1_ = new ControlPopup1(this);
        //     connect(control_popup1_, &ControlPopup1::stopRequested, this, &DashboardWidget::setStatusToAssigned);
        // }
        
        // control_popup1_->show();
        // control_popup1_->raise();
        // control_popup1_->activateWindow();

        
    } else if (control_status_ == "관리자사용중") {
        std::string config_path = "../../config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
        int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

        QString url;
        if (status_ == "navigating") {
            url = QString("http://%1:%2/cancel_navigating")
                        .arg(CENTRAL_IP.c_str())
                        .arg(CENTRAL_HTTP_PORT);
        } else {
            url = QString("http://%1:%2/return_command")
                            .arg(CENTRAL_IP.c_str())
                            .arg(CENTRAL_HTTP_PORT);
        }

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
                    qDebug() << "원격 제어 취소 명령 전송 성공. 200";
                    setControlStatus("대기중");
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
    } else if (control_status_ == "대기중") {
        std::string config_path = "../../config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
        int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

        QString url = QString("http://%1:%2/control_by_admin")
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
                    qDebug() << "원격 제어 시작 명령 전송 성공. 200";
                    setControlStatus("관리자사용중");
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
}

void DashboardWidget::onDestinationButtonClicked()
{
    qDebug() << "🎮 Destination 버튼 클릭! 현재 상태:" << control_status_;

    // 다른 팝업이 열려있으면 닫기
    if (control_popup1_ && control_popup1_->isVisible()) {
        control_popup1_->hide();
    }
    
    // control_popup2 표시 (임시로 popup1과 동일하게, 나중에 교체)
    if (control_popup2_ && control_popup2_->isVisible()) {
        control_popup2_->raise();
        control_popup2_->activateWindow();
        return;
    }
    
    // ControlPopup2 생성 및 표시
    if (!control_popup2_) {
        control_popup2_ = new ControlPopup2(status_widget2, this);  // ← 실제로 ControlPopup2 생성
    }
    
    control_popup2_->show();
    control_popup2_->raise();
    control_popup2_->activateWindow();
}

void DashboardWidget::setupMapWidget()
{
    map_widget = new MapWidget(this);
    
    map_widget->setGeometry(19, 58, 438, 772);  // 위치와 크기 조정
    
    map_widget->show();
}

void DashboardWidget::setupCameraWidget()
{
    // config.yaml에서 UDP 포트 읽기
    int udp_port = 8888;  // 기본값
    try {
        std::string config_path = "../../config.yaml";
        YAML::Node config = YAML::LoadFile(config_path);
        udp_port = config["ros_gui_client"]["udp_receive_port"].as<int>();
    } catch (const std::exception& e) {
        qDebug() << "config.yaml 로드 실패, 기본 포트 8888 사용:" << e.what();
    }
    
    // UDP 이미지 수신기 생성
    udp_receiver_ = new UdpImageReceiver("127.0.0.1", udp_port, this);
    
    // 시그널 연결
    connect(udp_receiver_, &UdpImageReceiver::imageReceived, 
            this, &DashboardWidget::onImageReceived);
    connect(udp_receiver_, &UdpImageReceiver::connectionError, 
            this, &DashboardWidget::onConnectionError);

    // 연결 성공 시그널 추가 (새로 추가할 예정)
    connect(udp_receiver_, &UdpImageReceiver::connectionEstablished,
            this, &DashboardWidget::onConnectionEstablished);
    
    // 수신 시작
    udp_receiver_->start();
    
    // camera_img에 기본 텍스트 설정
    if (ui->camera_img) {
        ui->camera_img->setText(QString("AI Server 연결 중...\n127.0.0.1:%1").arg(udp_port));
        ui->camera_img->setAlignment(Qt::AlignCenter);
        ui->camera_img->setScaledContents(true);
        
        // 연결 중 스타일
        ui->camera_img->setStyleSheet("background-color: #333; color: yellow; font-size: 14px;");
    }
}

void DashboardWidget::setControlStatusToAssigned() {
    setControlStatus("관리자사용중");
}

void DashboardWidget::onImageReceived(const QPixmap& pixmap)
{
    static bool first_image = true;
    if (first_image) {
        qDebug() << "✅ AI Server 연결 성공! 이미지 수신 시작";
        qDebug() << "📸 첫 이미지 크기:" << pixmap.size();
        first_image = false;
    }

    if (ui->camera_img) {
        qDebug() << "🎥 이미지 수신됨 - 크기:" << pixmap.size() << "camera_img 크기:" << ui->camera_img->size();
        
        // camera_img에 받은 이미지 표시
        QPixmap scaled_pixmap = pixmap.scaled(
            ui->camera_img->size(), 
            Qt::KeepAspectRatio, 
            Qt::SmoothTransformation
        );
        
        qDebug() << "🖼️ 스케일된 이미지 크기:" << scaled_pixmap.size();
        
        ui->camera_img->setPixmap(scaled_pixmap);

        // 연결 성공 시 스타일 초기화
        ui->camera_img->setStyleSheet("");
        
        qDebug() << "✅ 이미지 표시 완료";
    } else {
        qDebug() << "❌ camera_img 위젯이 null입니다!";
    }
}

void DashboardWidget::onConnectionError(const QString& error)
{
    qDebug() << "❌ AI Server 연결 실패:" << error;
    
    if (ui->camera_img) {
        ui->camera_img->setText("카메라 연결 실패\n" + error);
        ui->camera_img->setAlignment(Qt::AlignCenter);
    }
}

// 새로 추가할 슬롯
void DashboardWidget::onConnectionEstablished()
{
    qDebug() << "🔗 AI Server UDP 소켓 연결됨 (127.0.0.1:8888)";
    
    if (ui->camera_img) {
        ui->camera_img->setText("AI Server 연결됨\n이미지 수신 대기 중...");
        ui->camera_img->setStyleSheet("background-color: #333; color: green; font-size: 14px;");
    }
}

void DashboardWidget::setWidgetClasses()
{
    if (ui->title_point1) {
        ui->title_point1->setProperty("class", "title_point");
    }
    if (ui->title_point2) {
        ui->title_point2->setProperty("class", "title_point");
    }
    if (ui->title_point3) {
        ui->title_point3->setProperty("class", "title_point");
    }
    if (ui->title1) {
        ui->title1->setProperty("class", "size16 weight700");
    }
    if (ui->title2) {
        ui->title2->setProperty("class", "size16 weight700");
    }
    if (ui->title3) {
        ui->title3->setProperty("class", "size16 weight700");
    }
    if (ui->destinationBtn) {
        ui->destinationBtn->setProperty("class", "btn outlined primary_dark small");
        ui->destinationBtn->setVisible(control_status_ == "관리자사용중");
    }
    if (ui->controlBtn) {
        ui->controlBtn->setProperty("class", "btn outlined primary_dark small");
        ui->controlBtn->setVisible(control_status_ != "환자사용중");
    }
    if (ui->camera_toggle_bg) {
        ui->camera_toggle_bg->setProperty("class", "bg graye radius");
    }
    if (ui->camera_toggle1) {
        ui->camera_toggle1->setProperty("class", "btn contained white otl-gray9");
    }
    if (ui->camera_toggle2) {
        ui->camera_toggle2->setProperty("class", "btn contained gray");
    }
    if (ui->camera_bg) {
        ui->camera_bg->setProperty("class", "bg green_gray1 radius");
    }
    if (ui->camera_img) {
        ui->camera_img->setProperty("class", "camera_img");
    }
    
    if (ui->status_label) {
        ui->status_label->setProperty("class", control_status_ == "환자사용중" ? "label primary" : control_status_ == "관리자사용중" ? "label secondary" : "label gray");
        ui->status_label->setText(control_status_);
    }
}

void DashboardWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void DashboardWidget::refresh()
{
    if (status_widget) {
        status_widget->refresh();
    }
    if (map_widget) {
        map_widget->refresh();
    }

    if (ros_node_) {
        rclcpp::spin_some(ros_node_);
    }
    
    // 로봇 위치 주기적 업데이트
    get_robot_location();
}

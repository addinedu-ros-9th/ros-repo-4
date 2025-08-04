#include "control_popup1.h"
#include "ui_control_popup1.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QWidget>
#include <yaml-cpp/yaml.h>  
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>

ControlPopup1::ControlPopup1(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui_ControlPopup1)
{
    ui->setupUi(this);
    setWindowProperties();
    setupConnections();
    setWidgetClasses();
}

ControlPopup1::~ControlPopup1()
{
    delete ui;
}

void ControlPopup1::setWindowProperties()
{
    // 창 속성 설정
    setWindowTitle("로봇 제어");
    setFixedSize(800, 256);  // 고정 크기
    
    // 창을 화면 중앙에 위치
    if (parentWidget()) {
        QPoint parentCenter = parentWidget()->geometry().center();
        move(130, 272);
    }
}

void ControlPopup1::setWidgetClasses()
{
    if (ui->title) {
        ui->title->setProperty("class", "size24 weight700 color-gray3");
    }
    if (ui->line) {
        ui->line->setProperty("class", "bg grayc");
    }
    if (ui->closeBtn) {
        ui->closeBtn->setProperty("class", "btn outlined gray6 size20");
    }
    if (ui->stopBtn) {
        ui->stopBtn->setProperty("class", "btn contained primary size20");
    }
    if (ui->content2) {
        ui->content2->setProperty("class", "size24 weight400 color-gray6");
    }
    if (ui->content1) {
        ui->content1->setProperty("class", "size24 weight700 color-primary_dark");
    }
    if (ui->content3) {
        ui->content3->setProperty("class", "size24 weight400 color-gray6");
    }
    if (ui->content4) {
        ui->content4->setProperty("class", "size24 weight700 color-gray6");
    }
    if (ui->content5) {
        ui->content5->setProperty("class", "size24 weight400 color-gray6");
    }
    if (ui->bg) {
        ui->bg->setProperty("class", "bg white radius otl-gray6");
    }
}

void ControlPopup1::setupConnections()
{
    // 닫기 버튼 연결 (UI 파일에 closeBtn이 있다고 가정)
    if (ui->closeBtn) {
        connect(ui->closeBtn, &QPushButton::clicked, this, &ControlPopup1::onCloseButtonClicked);
    }
    if (ui->stopBtn) {
        connect(ui->stopBtn, &QPushButton::clicked, this, &ControlPopup1::onStopButtonClicked);
    }

    
    // 다른 버튼들도 필요에 따라 연결
    // if (ui->startBtn) {
    //     connect(ui->startBtn, &QPushButton::clicked, this, &ControlPopup1::onStartButtonClicked);
    // }
}

void ControlPopup1::onCloseButtonClicked()
{
    qDebug() << "제어 팝업 닫기 버튼 클릭됨";
    close();  // 창 닫기
}

void ControlPopup1::onStopButtonClicked()
{
    qDebug() << "정지 버튼 클릭됨";

    // 원격 제어 취소
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/stop/status_moving")
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
                qDebug() << "명령 전송 성공. 200";
                emit stopRequested(); // 시그널 발생
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

    close();  // 창 닫기
    // 로봇 정지 명령 전송 로직
}
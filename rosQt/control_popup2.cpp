#include "control_popup2.h"
#include "ui_control_popup2.h"
#include "status2.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QWidget>
#include <QDebug>
#include <QStyle>
#include <yaml-cpp/yaml.h>  
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>

ControlPopup2::ControlPopup2(Status2Widget* status2Widget, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui_ControlPopup2)
    , status2_(status2Widget)
    , current_status_("대기중")
    , selected_button_(nullptr) 
{
    ui->setupUi(this);
    selected_destination_ = ""; 
    setWindowProperties();
    setupConnections();
    setWidgetClasses();
}

ControlPopup2::~ControlPopup2()
{
    delete ui;
}

void ControlPopup2::setCurrentStatus(const QString& status)
{
    current_status_ = status;
    qDebug() << "ControlPopup2 상태 설정:" << status;
}

void ControlPopup2::setWindowProperties()
{
    // 창 속성 설정
    setWindowTitle("로봇 제어");
    setFixedSize(800, 414);  // 고정 크기
    
    // 창을 화면 중앙에 위치
    if (parentWidget()) {
        QPoint parentCenter = parentWidget()->geometry().center();
        move(130, 193);
    }
}

void ControlPopup2::setWidgetClasses()
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
    if (ui->startBtn) {
        ui->startBtn->setProperty("class", "btn contained primary size20");
    }
    if (ui->btn1) {
        ui->btn1->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn2) {
        ui->btn2->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn3) {
        ui->btn3->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn4) {
        ui->btn4->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn5) {
        ui->btn5->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn6) {
        ui->btn6->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn7) {
        ui->btn7->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->btn8) {
        ui->btn8->setProperty("class", "btn contained gray size20 weight700");
    }
    if (ui->bg) {
        ui->bg->setProperty("class", "bg white radius otl-gray6");
    }
}

void ControlPopup2::setupConnections()
{
    // 닫기 버튼 연결 (UI 파일에 closeBtn이 있다고 가정)
    if (ui->closeBtn) {
        connect(ui->closeBtn, &QPushButton::clicked, this, &ControlPopup2::onCloseButtonClicked);
    }
    // 시작 버튼 연결
    if (ui->startBtn) {
        connect(ui->startBtn, &QPushButton::clicked, this, &ControlPopup2::onStartButtonClicked);
    }
    
    // 목적지 버튼들 연결
    if (ui->btn1) {
        connect(ui->btn1, &QPushButton::clicked, this, &ControlPopup2::onBtn1Clicked);
    }
    if (ui->btn2) {
        connect(ui->btn2, &QPushButton::clicked, this, &ControlPopup2::onBtn2Clicked);
    }
    if (ui->btn3) {
        connect(ui->btn3, &QPushButton::clicked, this, &ControlPopup2::onBtn3Clicked);
    }
    if (ui->btn4) {
        connect(ui->btn4, &QPushButton::clicked, this, &ControlPopup2::onBtn4Clicked);
    }
    if (ui->btn5) {
        connect(ui->btn5, &QPushButton::clicked, this, &ControlPopup2::onBtn5Clicked);
    }
    if (ui->btn6) {
        connect(ui->btn6, &QPushButton::clicked, this, &ControlPopup2::onBtn6Clicked);
    }
    if (ui->btn7) {
        connect(ui->btn7, &QPushButton::clicked, this, &ControlPopup2::onBtn7Clicked);
    }
    if (ui->btn8) {
        connect(ui->btn8, &QPushButton::clicked, this, &ControlPopup2::onBtn8Clicked);
    }
}


// 버튼 클릭 이벤트 핸들러들
void ControlPopup2::onBtn1Clicked()
{
    selected_button_ = ui->btn1;
    updateButtonStyles();
    selected_destination_ = "대장암 센터"; 
}

void ControlPopup2::onBtn2Clicked()
{
    selected_button_ = ui->btn2;
    updateButtonStyles();
    selected_destination_ = "위암 센터";
}

void ControlPopup2::onBtn3Clicked()
{
    selected_button_ = ui->btn3;
    updateButtonStyles();
    selected_destination_ = "폐암 센터";
}

void ControlPopup2::onBtn4Clicked()
{
    selected_button_ = ui->btn4;
    updateButtonStyles();
    selected_destination_ = "유방암 센터"; 
}

void ControlPopup2::onBtn5Clicked()
{
    selected_button_ = ui->btn5;
    updateButtonStyles();
    selected_destination_ = "뇌종양 센터"; 
}

void ControlPopup2::onBtn6Clicked()
{
    selected_button_ = ui->btn6;
    updateButtonStyles();
    selected_destination_ = "X-ray 검사실"; 
}

void ControlPopup2::onBtn7Clicked()
{
    selected_button_ = ui->btn7;
    updateButtonStyles();
    selected_destination_ = "CT 검사실";
}

void ControlPopup2::onBtn8Clicked()
{
    selected_button_ = ui->btn8;
    updateButtonStyles();
    selected_destination_ = "초음파 검사실"; 
}

void ControlPopup2::onCloseButtonClicked()
{
    qDebug() << "제어 팝업 닫기 버튼 클릭됨";
    close();  // 창 닫기
}

int ControlPopup2::intToMapDepartmentId(QString dept) {
    if (dept == "CT 검사실") return 0;
    else if (dept == "초음파 검사실") return 1;
    else if (dept == "X-ray 검사실") return 2;
    else if (dept == "대장암 센터") return 3;
    else if (dept == "위암 센터") return 4;
    else if (dept == "폐암 센터") return 5;
    else if (dept == "뇌졸중 센터") return 6;
    else if (dept == "유방암 센터") return 7;
    else return -1;
}

void ControlPopup2::onStartButtonClicked()
{
    qDebug() << "start 버튼 클릭됨";

    // 원격 제어 취소
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/command/move_dest")
                    .arg(CENTRAL_IP.c_str())
                    .arg(CENTRAL_HTTP_PORT);

    QJsonObject data;
    data["robot_id"] = 3;
    data["dest"] = intToMapDepartmentId(selected_destination_);
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
                if (selected_destination_ != "") {
                    if (status2_) {
                        status2_->setMoveFirstText(selected_destination_);
                    }
                    close();  // 창 닫기
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

void ControlPopup2::resetAllButtonStyles()
{
    // 모든 버튼을 기본 스타일로 리셋
    QList<QPushButton*> buttons = {ui->btn1, ui->btn2, ui->btn3, ui->btn4, 
                                   ui->btn5, ui->btn6, ui->btn7, ui->btn8};
    
    for (QPushButton* btn : buttons) {
        if (btn) {
            btn->setProperty("class", "btn contained gray size20 weight700");
            btn->style()->unpolish(btn);
            btn->style()->polish(btn);
        }
    }
}


void ControlPopup2::updateButtonStyles()
{
    // 먼저 모든 버튼을 기본 스타일로 리셋
    resetAllButtonStyles();
    
    // 선택된 버튼만 특별한 스타일 적용
    if (selected_button_) {
        selected_button_->setProperty("class", "btn outlined primary size20 weight700");
        selected_button_->style()->unpolish(selected_button_);
        selected_button_->style()->polish(selected_button_);
        
        qDebug() << "버튼 스타일 변경됨:" << selected_button_->text();
    }
}
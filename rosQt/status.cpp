#include "status.h"
#include "BasicQtApp_autogen/include/ui_status.h"  // UI 헤더 포함
#include <QDebug>
#include <yaml-cpp/yaml.h>  
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>

StatusWidget::StatusWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_StatusWidget)
    , orig_("CT 검사실")  // 기본값 설정
    , dest_("초음파 검사실")  // 기본값 설정
    , battery_(80)  // 기본값 설정
    , network_("상")  // 기본값 설정
    , patient_id_("00000000")  // 기본값 설정
    , phone_("010-0000-0000")  // 기본값 설정
    , rfid_("00A0AA00")  // 기본값 설정
    , patient_name_("김00")  // 기본값 설정
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
    getPatientInfo();
    // Remove setRobotInfo() call - default values are already set
}

StatusWidget::~StatusWidget()
{
    delete ui;
}

void StatusWidget::setWidgetClasses()
{
    if (ui->move_bg) {
        ui->move_bg->setProperty("class", "bg green_gray1");
    }
    if (ui->move_first) {
        ui->move_first->setProperty("class", "move_first");
    }
    if (ui->move_last) {
        ui->move_last->setProperty("class", "move_last");
    }
    if (ui->move_arrow) {
        ui->move_arrow->setProperty("class", "move_arrow");
    }
    if (ui->point1) {
        ui->point1->setProperty("class", "point");
    }
    if (ui->point2) {
        ui->point2->setProperty("class", "point");
    }
    if (ui->point3) {
        ui->point3->setProperty("class", "point");
    }
    if (ui->point4) {
        ui->point4->setProperty("class", "point");
    }
    if (ui->title1) {
        ui->title1->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title2) {
        ui->title2->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title3) {
        ui->title3->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title4) {
        ui->title4->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title5) {
        ui->title5->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title5_box){
        ui->title5_box->setProperty("class", "otl gray6 radius");
    }

    if (ui->title5_box_cell1) {
        ui->title5_box_cell1->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell2) {
        ui->title5_box_cell2->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell3) {
        ui->title5_box_cell3->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell4) {
        ui->title5_box_cell4->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell5) {
        ui->title5_box_cell5->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell6) {
        ui->title5_box_cell6->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell7) {
        ui->title5_box_cell7->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell8) {
        ui->title5_box_cell8->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell9) {
        ui->title5_box_cell9->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell10) {
        ui->title5_box_cell10->setProperty("class", "radius bg primary");
    }

    if (ui->title5_status) {
        ui->title5_status->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title6) {
        ui->title6->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title6_status) {
        ui->title6_status->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title6_box) {
        ui->title6_box->setProperty("class", "otl gray6 radius");
    }
    if (ui->title6_box_cell1) {
        ui->title6_box_cell1->setProperty("class", "radius bg primary_25p");
    }
    if (ui->title6_box_cell2) {
        ui->title6_box_cell2->setProperty("class", "radius bg primary_50p");
    }
    if (ui->title6_box_cell3) {
        ui->title6_box_cell3->setProperty("class", "radius bg primary_75p");
    }
    if (ui->title6_box_cell4) {
        ui->title6_box_cell4->setProperty("class", "radius bg primary");
    }
    if (ui->content1) {
        ui->content1->setProperty("class", "size14 weight500 color-gray3");
    }
    if (ui->content2) {
        ui->content2->setProperty("class", "size14 weight500 color-gray3");
    }
    if (ui->content3) {
        ui->content3->setProperty("class", "size14 weight500 color-gray3");
    }
    if (ui->content4) {
        ui->content4->setProperty("class", "size14 weight500 color-gray3"); 
    }
    if (ui->machine_box) {
        ui->machine_box->setProperty("class", "otl primary_dark radius");
    }
    if (ui->battery_img) {
        ui->battery_img->setProperty("class", "battery_img");
    }
    if (ui->network_img) {
        ui->network_img->setProperty("class", "network_img");
    }
    
}

void StatusWidget::getPatientInfo()
{
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/get/patient_info")
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

                if (result.contains("patient_id") && result.contains("phone") && result.contains("rfid") &&
                    result.contains("name")) {
                    QString server_patient_id_ = result["patient_id"].toString();
                    QString server_phone_ = result["phone"].toString();
                    QString server_rfid_ = result["rfid"].toString();
                    QString server_patient_name_ = result["name"].toString();
                    setPatientInfo(server_patient_id_, server_phone_, server_rfid_, server_patient_name_);
                    // setRobotInfo(robot_orig, robot_dest, robot_battery, robot_network);
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

void StatusWidget::setPatientInfo(const QString& patient_id, const QString& phone, const QString& rfid, const QString& patient_name)
{
    patient_id_ = patient_id;
    phone_ = phone;
    rfid_ = rfid;
    patient_name_ = patient_name;

    qDebug() << "Setting patient info - ID:" << patient_id_ << "Name:" << patient_name_ << "Phone:" << phone_ << "RFID:" << rfid_;

    if (ui && ui->content4) {
        ui->content4->setText(patient_name_);
        qDebug() << "Set patient name to content4";
    } else {
        qDebug() << "Warning: content4 is null";
    }
    
    if (ui && ui->content2) {
        ui->content2->setText(phone_);
        qDebug() << "Set phone to content2";
    } else {
        qDebug() << "Warning: content2 is null";
    }
    
    if (ui && ui->content3) {
        ui->content3->setText(rfid_);
        qDebug() << "Set RFID to content3";
    } else {
        qDebug() << "Warning: content3 is null";
    }
    
    if (ui && ui->content1) {
        ui->content1->setText(patient_id_);
        qDebug() << "Set patient ID to content1";
    } else {
        qDebug() << "Warning: content1 is null";
    }
}
void StatusWidget::setRobotInfo(int orig, int dest, int battery, int network)
{
    orig_ = mapDepartmentIdToName(orig);
    dest_ = mapDepartmentIdToName(dest);
    battery_ = battery;
    network_ = mapNetworkStatusToString(network);

    if(ui->move_first) {
        ui->move_first->setText(orig_);
    }
    if (ui->move_last) {
        ui->move_last->setText(dest_);
    }
    if (ui->title5_status) {
        ui->title5_status->setText(QString::number(battery_) + "%");
    }
    if (ui->title6_status) {
        ui->title6_status->setText(network_);
    }
}

QString StatusWidget::mapDepartmentIdToName(int dept_id) {
    switch(dept_id) {
        case 0: return "CT 검사실";
        case 1: return "초음파 검사실";
        case 2: return "X-ray 검사실";
        case 3: return "대장암 센터";
        case 4: return "위암 센터";
        case 5: return "폐암 센터";
        case 6: return "뇌졸중 센터";
        case 7: return "유방암 센터";
        case 8: return "로비";  // 로비 추가
        default: return "알 수 없음 (" + QString::number(dept_id) + ")";
    }
}

QString StatusWidget::mapNetworkStatusToString(int network) {
    if (network == 0) {
        return "없음";
    } else if (network == 1) {
        return "하";
    } else if (network == 2) {
        return "중";
    } else if (network == 3) {
        return "상";
    } else if (network == 4) {
        return "최상";
    } else {
        return "통신 안됨";
    }
}

void StatusWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void StatusWidget::refresh()
{
    qDebug() << "Status widget refresh";
}
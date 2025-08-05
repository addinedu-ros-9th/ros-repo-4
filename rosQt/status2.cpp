#include "status2.h"
#include "BasicQtApp_autogen/include/ui_status2.h"  // UI 헤더 포함
#include <QDebug>
#include <QStyle>
#include <QPushButton>
#include <yaml-cpp/yaml.h>  
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>


Status2Widget::Status2Widget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_Status2Widget)
{
    ui->setupUi(this);  // UI 파일 설정
    arrowBtns[0] = ui->arrowBtn1;
    arrowBtns[1] = ui->arrowBtn2;
    arrowBtns[2] = ui->arrowBtn3;
    arrowBtns[3] = ui->arrowBtn4;
    arrowBtns[4] = ui->arrowBtn5;
    arrowBtns[5] = ui->arrowBtn6;
    arrowBtns[6] = ui->arrowBtn7;
    arrowBtns[7] = ui->arrowBtn8;
    arrowBtns[8] = ui->arrowBtn9;
    setWidgetClasses();
    setupKeyButton();
    onClickKey(5);  // 초기 상태 설정
}

Status2Widget::~Status2Widget()
{
    delete ui;
}

void Status2Widget::setWidgetClasses()
{
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            arrowBtns[i]->setProperty("class", "arrowBtn");
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
    if (ui->arrowBtn_key1) {
        ui->arrowBtn_key1->setProperty("class", "arrowBtn-key1");
    }
    if (ui->arrowBtn_key2) {
        ui->arrowBtn_key2->setProperty("class", "arrowBtn-key2");
    }
    if (ui->arrowBtn_key3) {
        ui->arrowBtn_key3->setProperty("class", "arrowBtn-key3");
    }
    if (ui->arrowBtn_key4) {
        ui->arrowBtn_key4->setProperty("class", "arrowBtn-key4");
    }
    if (ui->arrowBtn_key5) {
        ui->arrowBtn_key5->setProperty("class", "arrowBtn-key5");
    }
    if (ui->arrowBtn_key6) {
        ui->arrowBtn_key6->setProperty("class", "arrowBtn-key6");
    }
    if (ui->arrowBtn_key7) {
        ui->arrowBtn_key7->setProperty("class", "arrowBtn-key7");
    }
    if (ui->arrowBtn_key8) {
        ui->arrowBtn_key8->setProperty("class", "arrowBtn-key8");
    }
    if (ui->arrowBtn_key9) {
        ui->arrowBtn_key9->setProperty("class", "arrowBtn-key9");
    }
    
    if (ui->move_bg) {
        ui->move_bg->setProperty("class", "bg green_gray1");
    }
    if (ui->move_first) {
        ui->move_first->setProperty("class", "move_first");
    }

    if (ui->point1) {
        ui->point1->setProperty("class", "point");
    }
    if (ui->title1) {
        ui->title1->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->content1) {
        ui->content1->setProperty("class", "size14 weight500 color-gray3");
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

void Status2Widget::setMoveFirstText(const QString& text)
{
    if (ui->move_first) {
        if (text == "정지중") {
            ui->move_first->setText("정지중");
        } else {
            ui->move_first->setText(text + " 이동중");
        }
    }
}

void Status2Widget::setupKeyButton()
{
    if (ui->arrowBtn1) {
        connect(ui->arrowBtn1, &QPushButton::clicked,
                this, [this]() { onClickKey(1); });
    } 
    if (ui->arrowBtn2) {
        connect(ui->arrowBtn2, &QPushButton::clicked,
                this, [this]() { onClickKey(2); });
    }
    if (ui->arrowBtn3) {
        connect(ui->arrowBtn3, &QPushButton::clicked,
                this, [this]() { onClickKey(3); });
    }
    if (ui->arrowBtn4) {
        connect(ui->arrowBtn4, &QPushButton::clicked,
                this, [this]() { onClickKey(4); });
    }
    if (ui->arrowBtn5) {
        connect(ui->arrowBtn5, &QPushButton::clicked,
                this, [this]() { onClickKey(5); });
    }
    if (ui->arrowBtn6) {
        connect(ui->arrowBtn6, &QPushButton::clicked,
                this, [this]() { onClickKey(6); });
    }
    if (ui->arrowBtn7) {
        connect(ui->arrowBtn7, &QPushButton::clicked,
                this, [this]() { onClickKey(7); });
    }
    if (ui->arrowBtn8) {
        connect(ui->arrowBtn8, &QPushButton::clicked,
                this, [this]() { onClickKey(8); });
    }
    if (ui->arrowBtn9) {
        connect(ui->arrowBtn9, &QPushButton::clicked,
                this, [this]() { onClickKey(9); });
    }
}

void Status2Widget::onClickKey(int clickedNumber)
{
    qDebug() << "Key " << clickedNumber << " clicked";
    
    std::string config_path = "../../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/command/move_teleop")
                    .arg(CENTRAL_IP.c_str())
                    .arg(CENTRAL_HTTP_PORT);

    QJsonObject data;
    data["robot_id"] = 3;
    data["teleop_key"] = clickedNumber;
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

        connect(reply, &QNetworkReply::finished, this, [this, reply, CENTRAL_IP, CENTRAL_HTTP_PORT, data, clickedNumber]() {
            int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            if (statusCode == 200) {
                qDebug() << "teleop " << clickedNumber << "key 명령 전송 성공. 200";

                for (int i = 0; i < 9; ++i) {
                    if (arrowBtns[i]) {
                        if (i == clickedNumber - 1) {
                            arrowBtns[i]->setProperty("class", "arrowBtn active");
                        } else {
                            arrowBtns[i]->setProperty("class", "arrowBtn");
                        }
                        arrowBtns[i]->style()->unpolish(arrowBtns[i]);
                        arrowBtns[i]->style()->polish(arrowBtns[i]);
                        arrowBtns[i]->update();
                    }
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

void Status2Widget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void Status2Widget::refresh()
{
    qDebug() << "Status2 widget refresh";

}

QString Status2Widget::mapDepartmentIdToName(int dept_id) {
    switch(dept_id) {
        case 0: return "CT 검사실";
        case 1: return "초음파 검사실";
        case 2: return "X-ray 검사실";
        case 3: return "대장암 센터";
        case 4: return "위암 센터";
        case 5: return "폐암 센터";
        case 6: return "뇌졸중 센터";
        case 7: return "유방암 센터";
        default: return "알 수 없음 (" + QString::number(dept_id) + ")";
    }
}

QString Status2Widget::mapNetworkStatusToString(int network) {
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

void Status2Widget::setRobotInfo(int orig, int dest, int battery, int network)
{
    orig_ = mapDepartmentIdToName(orig);
    dest_ = mapDepartmentIdToName(dest);
    battery_ = battery;
    network_ = mapNetworkStatusToString(network);

    if (ui->move_first) {
        ui->move_first->setText(dest_);
    }
    if (ui->title5_status) {
        ui->title5_status->setText(QString::number(battery_) + "%");
    }
    if (ui->title6_status) {
        ui->title6_status->setText(network_);
    }
}

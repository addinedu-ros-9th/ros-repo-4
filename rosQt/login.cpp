#include "login.h"
#include "ui_login.h"
#include <QDebug>
#include <QStyle>
#include <yaml-cpp/yaml.h>  
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include "user_info.h" 
#include <QMessageBox>
#include <QMessageBox>
#include <QString>

LoginWindow::LoginWindow(QWidget *parent)
    : QWidget(parent)  // QWidget 대신 QMainWindow로 되돌리기
    , ui(new Ui_LoginWidget)
{
    ui->setupUi(this);
    setWidgetClasses();
    
    // 로그인 버튼 연결 (login.ui에 login_btn이 있다고 가정)
    if (ui->login_btn) {
        connect(ui->login_btn, &QPushButton::clicked, this, &LoginWindow::check_login);
    }
    if (ui->login_textfield1) {
        connect(ui->login_textfield1, &QLineEdit::returnPressed, this, &LoginWindow::handleLogin);
    }
    if (ui->login_textfield2) {
        connect(ui->login_textfield2, &QLineEdit::returnPressed, this, &LoginWindow::handleLogin);
    }
    
    // 테스트용: 다른 버튼이라도 연결해보기
    // login.ui에 있는 실제 버튼 이름을 확인하고 사용
}

LoginWindow::~LoginWindow()
{
    delete ui;
}

void LoginWindow::handleLogin()
{
    emit loginSuccessful();
}

void LoginWindow::setWidgetClasses()
{
    // 버튼 클래스 설정
    if (ui->login_btn) {
        ui->login_btn->setProperty("class", "btn contained primary_dark large");
    }
    if (ui->login_logo_icon) {
        ui->login_logo_icon->setProperty("class", "login_logo_icon");
    }
    // 체크박스 클래스 설정
    if (ui->login_checkbox) {
        ui->login_checkbox->setProperty("class", "checkbox gray9");
    }
    
    // 구분선 클래스 설정
    if (ui->login_div) {
        ui->login_div->setProperty("class", "divider ver dotted");
    }
    if (ui->login_div2) {
        ui->login_div2->setProperty("class", "divider ver dotted");
    }
    
    // 텍스트 필드 클래스 설정
    if (ui->login_textfield1) {
        ui->login_textfield1->setProperty("class", "textfield large");
    }
    if (ui->login_textfield2) {
        ui->login_textfield2->setProperty("class", "textfield large");
    }
    
    // 라벨 클래스 설정
    if (ui->login_greeting) {
        ui->login_greeting->setProperty("class", "size24 weight700");
    }
    if (ui->login_desc1) {
        ui->login_desc1->setProperty("class", "size16 weight300 color-gray9");
    }
    if (ui->login_desc2) {
        ui->login_desc2->setProperty("class", "size16 weight300 color-gray9");
    }
    if (ui->login_id) {
        ui->login_id->setProperty("class", "size14 color-gray3");
    }
    if (ui->login_password) {
        ui->login_password->setProperty("class", "size14 color-gray3");
    }
    if (ui->auth_text1) {
        ui->auth_text1->setProperty("class", "size12 color-white");
    }
    if (ui->auth_text2) {
        ui->auth_text2->setProperty("class", "size12 color-white");
    }
    
    // 스타일 새로고침 (중요!)
    this->style()->unpolish(this);
    this->style()->polish(this);
    this->update();
    
}

void LoginWindow::check_login()
{
    QString user_id = ui->login_textfield1->text();
    QString user_password = ui->login_textfield2->text();

    std::string config_path = "../config.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    std::string CENTRAL_IP = config["central_server"]["ip"].as<std::string>();
    int CENTRAL_HTTP_PORT = config["central_server"]["http_port"].as<int>();

    QString url = QString("http://%1:%2/auth/login")
                    .arg(CENTRAL_IP.c_str())
                    .arg(CENTRAL_HTTP_PORT);

    QJsonObject data;
    data["user_id"] = user_id;
    data["passwd"] = user_password;
    QJsonDocument doc(data);
    QByteArray jsonData = doc.toJson();

    qDebug() << "[로그인 요청 URL]:" << url;
    qDebug() << "[전송 데이터]:" << jsonData;
    try
    {
        QNetworkRequest request{QUrl(url)};
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        QNetworkReply* reply = manager->post(request, jsonData);

        if (ui->login_checkbox && !ui->login_checkbox->isChecked()) {
            if (ui->login_textfield1) ui->login_textfield1->clear();
            if (ui->login_textfield2) ui->login_textfield2->clear();
        }

        connect(reply, &QNetworkReply::finished, this, [this, reply, CENTRAL_IP, CENTRAL_HTTP_PORT, user_id]() {
            int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            if (statusCode == 200) {
                QByteArray responseData = reply->readAll();
                QJsonDocument jsonDoc = QJsonDocument::fromJson(responseData);
                QJsonObject result = jsonDoc.object();
                qDebug() << "[응답 내용]:" << result;

                QString user_info_url = QString("http://%1:%2/auth/detail")
                            .arg(CENTRAL_IP.c_str())
                            .arg(CENTRAL_HTTP_PORT);

                QJsonObject user_info_data;
                user_info_data["user_id"] = user_id;
                QJsonDocument user_info_doc(user_info_data);
                QByteArray user_info_json = user_info_doc.toJson();

                QNetworkRequest userInfoRequest{QUrl(user_info_url)};
                userInfoRequest.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
                QNetworkAccessManager* userInfoManager = new QNetworkAccessManager(this);
                QNetworkReply* userInfoReply = userInfoManager->post(userInfoRequest, user_info_json);
                connect(userInfoReply, &QNetworkReply::finished, this, [this, userInfoReply]() {
                    int userInfoStatusCode = userInfoReply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
                    if (userInfoStatusCode == 200) {
                        QByteArray userInfoData = userInfoReply->readAll();
                        QJsonDocument userInfoDoc = QJsonDocument::fromJson(userInfoData);
                        QJsonObject userInfoResult = userInfoDoc.object();
                        
                        UserInfo info;
                        info.name = userInfoResult.contains("name") ? userInfoResult["name"].toString().toStdString() : "";
                        info.email = userInfoResult.contains("email") ? userInfoResult["email"].toString().toStdString() : "";
                        info.hospital_name = userInfoResult.contains("hospital_name") ? userInfoResult["hospital_name"].toString().toStdString() : "";

                        // UserInfoManager::set_user_id(userInfoResult["user_id"].toString().toStdString());
                        // UserInfoManager::set_user_info(info);

                        qDebug() << "[사용자 정보 응답]:" << userInfoResult;

                        for (const QString& key : {"name", "email", "hospital_name", "user_id"}) {
                            qDebug() << "Key:" << key << ", exists:" << userInfoResult.contains(key)
                                    << ", isString:" << userInfoResult[key].isString()
                                    << ", value:" << userInfoResult[key];
                        }
                        // 사용자 정보 처리 로직 추가
                    } else {
                        qDebug() << "[사용자 정보 요청 실패]:" << userInfoReply->errorString();
                    }
                    // QString name = QString::fromStdString(UserInfoManager::get_user_info().name);
                    // QMessageBox::information(this, "로그인 성공", QString("%1님 환영합니다!").arg(name));
                    emit loginSuccessful();
                    userInfoReply->deleteLater();
                });
            } else if (statusCode == 401) {
                if (ui->login_textfield1) ui->login_textfield1->clearFocus();
                if (ui->login_textfield2) ui->login_textfield2->clearFocus();

                QByteArray responseData = reply->readAll();
                QJsonDocument jsonDoc = QJsonDocument::fromJson(responseData);
                QJsonObject result = jsonDoc.object();
                QString error_msg = result.value("message").toString("로그인 실패");
                qDebug() << "ID 오류 :" << error_msg;

                if (ui->login_textfield1) {
                    ui->login_textfield1->setProperty("class", "textfield large error");
                    ui->login_textfield1->style()->unpolish(ui->login_textfield1);
                    ui->login_textfield1->style()->polish(ui->login_textfield1);
                    ui->login_textfield1->update();
                }
                if (ui->auth_text1) {
                    ui->auth_text1->setProperty("class", "size12 color-error");
                    ui->auth_text1->style()->unpolish(ui->auth_text1);
                    ui->auth_text1->style()->polish(ui->auth_text1);
                    ui->auth_text1->update();
                }
            } else if (statusCode == 402) {
                if (ui->login_textfield1) ui->login_textfield1->clearFocus();
                if (ui->login_textfield2) ui->login_textfield2->clearFocus();

                QByteArray responseData = reply->readAll();
                QJsonDocument jsonDoc = QJsonDocument::fromJson(responseData);
                QJsonObject result = jsonDoc.object();
                QString error_msg = result.value("message").toString("로그인 실패");
                qDebug() << "ID 오류 :" << error_msg;

                if (ui->login_textfield2) {
                    ui->login_textfield2->setProperty("class", "textfield large error");
                    ui->login_textfield2->style()->unpolish(ui->login_textfield2);
                    ui->login_textfield2->style()->polish(ui->login_textfield2);
                    ui->login_textfield2->update();
                }
                if (ui->auth_text2) {
                    ui->auth_text2->setProperty("class", "size12 color-error");
                    ui->auth_text2->style()->unpolish(ui->auth_text2);
                    ui->auth_text2->style()->polish(ui->auth_text2);
                    ui->auth_text2->update();
                }
            } else if (statusCode == 404) {
                if (ui->login_textfield1 && ui->login_textfield1->text().isEmpty()) {
                    ui->login_textfield1->setProperty("class", "textfield large error");
                    ui->login_textfield1->style()->unpolish(ui->login_textfield1);
                    ui->login_textfield1->style()->polish(ui->login_textfield1);
                    ui->login_textfield1->update();
                    if (ui->auth_text1) {
                        ui->auth_text1->setProperty("class", "size12 color-error");
                        ui->auth_text1->style()->unpolish(ui->auth_text1);
                        ui->auth_text1->style()->polish(ui->auth_text1);
                        ui->auth_text1->update();
                    }
                }
                if (ui->login_textfield2 && ui->login_textfield2->text().isEmpty()) {
                    ui->login_textfield2->setProperty("class", "textfield large error");
                    ui->login_textfield2->style()->unpolish(ui->login_textfield2);
                    ui->login_textfield2->style()->polish(ui->login_textfield2);
                    ui->login_textfield2->update();
                    if (ui->auth_text2) {
                        ui->auth_text2->setProperty("class", "size12 color-error");
                        ui->auth_text2->style()->unpolish(ui->auth_text2);
                        ui->auth_text2->style()->polish(ui->auth_text2);
                        ui->auth_text2->update();
                    }
                }
            }
            reply->deleteLater();
        });
    } catch (const std::exception& e) {
        qDebug() << "[네트워크 예외]:" << e.what();
        QMessageBox::critical(this, "네트워크 오류", e.what());
    }
}
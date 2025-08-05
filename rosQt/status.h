#ifndef STATUSWIDGET_H
#define STATUSWIDGET_H

#include <QWidget>

class Ui_StatusWidget;  // UI 클래스 전방 선언

class StatusWidget : public QWidget
{
    Q_OBJECT

public:
    explicit StatusWidget(QWidget *parent = nullptr);
    ~StatusWidget();
    
    void show_at(const QPoint& pos);
    void refresh();
    void setRobotInfo(int orig, int dest, int battery, int network);

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    
private:
    Ui_StatusWidget *ui;  // UI 포인터
    QString mapDepartmentIdToName(int dept_id);  // 출발지 도착지 매핑 함수
    QString mapNetworkStatusToString(int network);  // 네트워크 상태 매핑 함수
    void getPatientInfo();  // 환자 정보 가져오기
    void setPatientInfo(const QString& patient_id, const QString& phone, const QString& rfid, const QString& patient_name);  // 환자 정보 설정
    
    QString orig_;
    QString dest_;
    int battery_;
    QString network_;
    QString patient_id_;
    QString phone_;
    QString rfid_;
    QString patient_name_;

};

#endif // STATUSWIDGET_H
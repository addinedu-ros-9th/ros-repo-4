#ifndef STATUS2WIDGET_H
#define STATUS2WIDGET_H

#include <QWidget>
#include <QPushButton>

class Ui_Status2Widget;  // UI 클래스 전방 선언

class Status2Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Status2Widget(QWidget *parent = nullptr);
    ~Status2Widget();

    void show_at(const QPoint& pos);
    void refresh();

    void setRobotInfo(int orig, int dest, int battery, int network);

public slots:
    void setMoveFirstText(const QString& text);
    void setRobotLocation(double x, double y, double yaw);

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수
    QPushButton* arrowBtns[9];  // 화살표 버튼 배열
    void setupKeyButton();  // 키 버튼 설정 함수
    void onClickKey(int clickedNumber);  // 키 클릭 핸들러
    
    QString mapDepartmentIdToName(int dept_id);  // 출발지 도착지 매핑 함수
    QString mapNetworkStatusToString(int network);  // 네트워크 상태 매핑 함수
    
private:
    Ui_Status2Widget *ui;  // UI 포인터

    QString orig_;
    QString dest_;
    int battery_;
    QString network_;
};

#endif // STATUS2WIDGET_H
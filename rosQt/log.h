#ifndef LOGWIDGET_H
#define LOGWIDGET_H

#include <QWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QCheckBox>
#include <QDateTime>
#include <QString>
#include <QVector>

struct LogData {
    QString patientId;
    QString source;
    QString destination;
    QString timestamp;
};

class Ui_LogWidget;  // UI 클래스 전방 선언

class LogWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LogWidget(QWidget *parent = nullptr);
    ~LogWidget();
    
    void show_at(const QPoint& pos);
    void refresh();

private slots:
    void onToggle1Clicked();  // 오늘 버튼 클릭
    void onToggle2Clicked();  // 주간 버튼 클릭
    void onToggle3Clicked();  // 월간 버튼 클릭

private:
    void setWidgetClasses();  // CSS 클래스 설정 함수 추가
    void setupConnections();  // 시그널-슬롯 연결 함수
    void updateToggleButtons(int activeToggle);  // 토글 버튼 상태 업데이트
    void setupTableData();  // 테이블 데이터 설정 함수 추가
    void populateRobotTable(); // 로봇 테이블 데이터 추가 함수
    void populateHeatmap();

private:
    Ui_LogWidget *ui;
    int currentToggle;  // 현재 선택된 토글 (1, 2, 3)
    QVector<LogData> logEntries;
    QVector<QVector<int>> heatmapEntries;
};

#endif // LOGWIDGET_H
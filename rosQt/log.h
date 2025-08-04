#ifndef LOGWIDGET_H
#define LOGWIDGET_H

#include <QWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QCheckBox>
#include <QDateTime>
#include <QString>
#include <QVector>

class QLabel;  // QLabel 전방 선언 추가

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
    void populateRobotTable(); 
    void populateChartTable();  // 기존
    void updateChartLabels(const QStringList& sourceNames, const QList<int>& sourceCounts);  // 새로 추가
    void drawChart(QLabel* chartLabel, const QList<int>& data, const QStringList& labels = QStringList());  // 새로 추가
    void get_robot_log_data();
    void get_robot_log_data_with_period(const QString& period);
    QString mapDepartmentIdToName(int dept_id);
    void setDateFields(const QDate& startDate, const QDate& endDate);  // 새로 추가
    QString getDateFromFields(bool isStartDate);  // 새로 추가

private:
    Ui_LogWidget *ui;
    int currentToggle;  // 현재 선택된 토글 (1, 2, 3)
    QVector<LogData> logEntries_;
};

#endif // LOGWIDGET_H
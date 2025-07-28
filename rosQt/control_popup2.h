#ifndef CONTROL_POPUP2_H
#define CONTROL_POPUP2_H

#include <QWidget>
#include <QDebug>
#include <QPushButton>

class Ui_ControlPopup2;  // UI 클래스 전방 선언

class ControlPopup2 : public QWidget
{
    Q_OBJECT

public:
    explicit ControlPopup2(QWidget *parent = nullptr);
    ~ControlPopup2();

    void setCurrentStatus(const QString& status);

private slots:
    void onCloseButtonClicked();
    void onStartButtonClicked();   // 시작 버튼용
    void onBtn1Clicked();  
    void onBtn2Clicked();  
    void onBtn3Clicked();
    void onBtn4Clicked();
    void onBtn5Clicked();
    void onBtn6Clicked();
    void onBtn7Clicked();
    void onBtn8Clicked();

private:
    void setWindowProperties();
    void setupConnections();
    void setWidgetClasses(); 
    void updateButtonStyles();  
    void resetAllButtonStyles(); 
    
private:
    Ui_ControlPopup2 *ui;
    QString current_status_;
    QPushButton* selected_button_;
};

#endif // CONTROL_POPUP2_H